#!/usr/bin/env python3
"""Power/energy monitor for benchmarking runs.

The R6C exposes only a voltage ADC channel (USB-C VBUS), not current, so true
wattage needs an external USB-C power meter or an inline shunt. This script
supports:
  - serial:  power meter streaming lines ("12.1,2.3" = V,A or "27.8" = W)
  - file:    same format, useful for recorded logs or `--input -` (stdin)
  - adc:     raw SARADC voltage only (no power; reported for completeness)
  - synthetic: deterministic smoke test for CI
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nano_llm.bench.metrics import now_iso, summarize, write_json  # noqa: E402
from nano_llm.utils.system import adc_raw_to_voltage, read_adc_raw  # noqa: E402

NUM = r"-?\d+(?:\.\d+)?"


@dataclass
class PowerSample:
    timestamp: float
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    power_w: Optional[float] = None
    raw: str = ""

    def iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finish(sample: PowerSample) -> PowerSample:
    if sample.power_w is None and sample.voltage_v is not None and sample.current_a is not None:
        sample.power_w = sample.voltage_v * sample.current_a
    return sample


def parse_power_line(line: str, timestamp: Optional[float] = None) -> Optional[PowerSample]:
    raw = (line or "").strip()
    if not raw:
        return None
    ts = time.time() if timestamp is None else timestamp
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            lowered = {str(key).lower(): value for key, value in payload.items()}
            voltage = _float(lowered.get("voltage", lowered.get("voltage_v", lowered.get("v"))))
            current = _float(
                lowered.get("current", lowered.get("current_a", lowered.get("a", lowered.get("i", lowered.get("amp")))))
            )
            power = _float(lowered.get("power", lowered.get("power_w", lowered.get("w"))))
            return _finish(PowerSample(ts, voltage, current, power, raw))

    pairs = {
        key.lower(): float(value)
        for key, value in re.findall(
            rf"(voltage|volt|current|amp|power|v|a|i|w)\s*[=:]\s*({NUM})", raw, flags=re.IGNORECASE
        )
    }
    if pairs:
        voltage = _float(pairs.get("voltage", pairs.get("volt", pairs.get("v"))))
        current = _float(pairs.get("current", pairs.get("amp", pairs.get("a", pairs.get("i")))))
        power = _float(pairs.get("power", pairs.get("w")))
        return _finish(PowerSample(ts, voltage, current, power, raw))

    numbers = [float(value) for value in re.findall(NUM, raw)]
    if not numbers:
        return None
    if len(numbers) == 1:
        return PowerSample(ts, None, None, numbers[0], raw)
    if len(numbers) == 2:
        return _finish(PowerSample(ts, numbers[0], numbers[1], None, raw))
    return _finish(PowerSample(ts, numbers[0], numbers[1], numbers[2], raw))


class SyntheticMeter:
    def __init__(self, base_w: float = 6.5, noise_w: float = 0.4, voltage: float = 12.0):
        self.base_w = base_w
        self.noise_w = noise_w
        self.voltage = voltage

    def read(self) -> Optional[PowerSample]:
        power = max(0.0, self.base_w + random.uniform(-self.noise_w, self.noise_w))
        current = power / self.voltage if self.voltage else None
        return PowerSample(time.time(), self.voltage, current, power, f"synthetic:{power:.3f}")


class FileMeter:
    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._stdin = path == "-"

    def _open(self):
        if self._fh is None:
            self._fh = sys.stdin if self._stdin else open(self.path, "r", encoding="utf-8")
        return self._fh

    def read(self) -> Optional[PowerSample]:
        fh = self._open()
        for line in fh:
            sample = parse_power_line(line)
            if sample is not None:
                return sample
        return None

    def close(self) -> None:
        if self._fh is not None and not self._stdin:
            self._fh.close()


class SerialMeter:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for --meter serial: pip install pyserial") from exc
        self.serial = serial.Serial(port, baudrate=baud, timeout=timeout)

    def read(self) -> Optional[PowerSample]:
        line = self.serial.readline()
        if not line:
            return None
        return parse_power_line(line.decode("utf-8", errors="replace"))


class ADCMeter:
    def __init__(self, raw_path: str, vref: float = 1.8, bits: int = 12, scale: float = 1.0):
        self.raw_path = raw_path
        self.vref = vref
        self.bits = bits
        self.scale = scale

    def read(self) -> Optional[PowerSample]:
        raw = read_adc_raw(self.raw_path)
        if raw is None:
            return None
        voltage = adc_raw_to_voltage(raw, self.vref, self.bits, self.scale)
        return PowerSample(time.time(), voltage, None, None, f"adc:{raw}")


def run_monitor(
    meter,
    duration_s: float,
    interval_s: float,
    out_path: str,
    max_samples: Optional[int] = None,
    printer=print,
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    start = time.time()
    samples: List[PowerSample] = []
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "voltage_v", "current_a", "power_w", "raw"])
        while True:
            if duration_s > 0 and time.time() - start >= duration_s:
                break
            if max_samples is not None and len(samples) >= max_samples:
                break
            sample = meter.read()
            if sample is None:
                if isinstance(meter, FileMeter):
                    break
                time.sleep(max(interval_s, 0.05))
                continue
            samples.append(sample)
            writer.writerow(
                [
                    sample.iso(),
                    "" if sample.voltage_v is None else f"{sample.voltage_v:.4f}",
                    "" if sample.current_a is None else f"{sample.current_a:.4f}",
                    "" if sample.power_w is None else f"{sample.power_w:.4f}",
                    sample.raw,
                ]
            )
            fh.flush()
            if printer and len(samples) % 10 == 0:
                printer(
                    f"{sample.iso()} P={sample.power_w if sample.power_w is not None else 'n/a'} W"
                )
            if interval_s > 0 and not isinstance(meter, FileMeter):
                time.sleep(interval_s)

    powers = [sample.power_w for sample in samples if sample.power_w is not None]
    elapsed = max(time.time() - start, 1e-6)
    if powers:
        energy_wh = (sum(powers) / len(powers)) * elapsed / 3600.0
    else:
        energy_wh = 0.0
    note = (
        "power from the configured meter"
        if powers
        else "no wattage readings; the ADC channel reports voltage only. Use a USB-C power meter."
    )
    summary = {
        "timestamp": now_iso(),
        "samples": len(samples),
        "duration_s": round(elapsed, 3),
        "power_w": summarize(powers) if powers else None,
        "energy_wh": round(energy_wh, 4),
        "note": note,
        "csv": os.path.abspath(out_path),
    }
    write_json(out_path.rsplit(".", 1)[0] + ".summary.json", summary)
    return summary


def build_meter(args):
    if args.meter == "synthetic":
        return SyntheticMeter(base_w=args.base_w, noise_w=args.noise_w, voltage=args.voltage)
    if args.meter == "file":
        if not args.input:
            raise ValueError("--input is required for --meter file")
        return FileMeter(args.input)
    if args.meter == "serial":
        return SerialMeter(args.serial_port, args.baud)
    if args.meter == "adc":
        return ADCMeter(args.adc_raw_path, args.adc_vref, args.adc_bits, args.adc_scale)
    if args.meter == "manual":
        return FileMeter("-")
    raise ValueError(f"unknown meter: {args.meter}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record power samples to CSV")
    parser.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    parser.add_argument("--meter", default=None, choices=["synthetic", "file", "serial", "adc", "manual"])
    parser.add_argument("--duration", type=float, default=60.0, help="seconds; 0 = until EOF (file)")
    parser.add_argument("--interval", type=float, default=None, help="sampling interval (s)")
    parser.add_argument("--out", default=None, help="CSV output path")
    parser.add_argument("--input", default=None, help="file path or '-' for stdin")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--base-w", type=float, default=6.5, help="synthetic base power")
    parser.add_argument("--noise-w", type=float, default=0.4, help="synthetic noise amplitude")
    parser.add_argument("--voltage", type=float, default=12.0, help="synthetic voltage")
    parser.add_argument("--adc-raw-path", default=None)
    parser.add_argument("--adc-vref", type=float, default=None)
    parser.add_argument("--adc-bits", type=int, default=None)
    parser.add_argument("--adc-scale", type=float, default=None)
    args = parser.parse_args()

    try:
        from nano_llm.config import load_config

        cfg = load_config(args.config, ROOT)
        power_cfg = cfg.power
        args.meter = args.meter or power_cfg.meter
        args.interval = args.interval if args.interval is not None else power_cfg.interval_s
        args.out = args.out or cfg.resolve(power_cfg.out)
        args.input = args.input if args.input is not None else power_cfg.input_file
        args.serial_port = args.serial_port or power_cfg.serial_port
        args.baud = args.baud or power_cfg.baud
        args.adc_raw_path = args.adc_raw_path or power_cfg.adc_raw_path
        args.adc_vref = args.adc_vref if args.adc_vref is not None else power_cfg.adc_vref
        args.adc_bits = args.adc_bits or power_cfg.adc_bits
        args.adc_scale = args.adc_scale if args.adc_scale is not None else power_cfg.adc_scale
    except Exception:
        args.meter = args.meter or "synthetic"
        args.interval = args.interval if args.interval is not None else 1.0
        args.out = args.out or "data/results/power.csv"
        args.serial_port = args.serial_port or "/dev/ttyUSB0"
        args.baud = args.baud or 115200
        args.adc_raw_path = args.adc_raw_path or "/sys/devices/platform/fec10000.saradc/iio:device0/in_voltage2_raw"
        args.adc_vref = args.adc_vref if args.adc_vref is not None else 1.8
        args.adc_bits = args.adc_bits or 12
        args.adc_scale = args.adc_scale if args.adc_scale is not None else 3.0

    if args.meter == "adc":
        print("WARN: ADC channel 2 on the R6C measures USB-C voltage only; power_w will be empty.")
        print("      Use a USB-C power meter / inline shunt for watts.")

    if args.duration <= 0 and args.meter not in ("file", "manual"):
        print("ERROR: --duration must be > 0 for streaming meters")
        return 2

    meter = build_meter(args)
    try:
        summary = run_monitor(meter, args.duration, args.interval, args.out)
    finally:
        if hasattr(meter, "close"):
            meter.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
