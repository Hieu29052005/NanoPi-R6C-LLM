from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Dict, Optional

NPU_VERSION_PATHS = (
    "/sys/kernel/debug/rknpu/version",
    "/sys/kernel/debug/rknpu/load",
)


def file_size_gb(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    if os.path.isdir(path):
        total = 0
        for dirpath, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    continue
        return total / (1024**3)
    return os.path.getsize(path) / (1024**3)


def read_status_kb(key: str, status_path: str = "/proc/self/status") -> Optional[int]:
    try:
        with open(status_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    match = re.search(r"(\d+)", line)
                    if match:
                        return int(match.group(1))
    except OSError:
        return None
    return None


def read_vm_hwm_gb() -> float:
    kb = read_status_kb("VmHWM")
    return (kb or 0) / (1024**2)


def read_vm_rss_gb() -> float:
    kb = read_status_kb("VmRSS")
    return (kb or 0) / (1024**2)


def read_meminfo_gb() -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, raw = line.split(":", 1)
                match = re.search(r"(\d+)", raw)
                if match:
                    values[key.strip()] = int(match.group(1)) / (1024**2)
    except OSError:
        return values
    return values


def board_model() -> str:
    path = "/proc/device-tree/model"
    try:
        with open(path, "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return platform.machine()


def npu_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    for path in NPU_VERSION_PATHS:
        if not os.path.exists(path):
            continue
        try:
            info[path] = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
    return info


def read_adc_raw(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def adc_raw_to_voltage(raw: int, vref: float = 1.8, bits: int = 12, scale: float = 1.0) -> float:
    if bits <= 0:
        raise ValueError("bits must be positive")
    return (raw / float((1 << bits) - 1)) * vref * scale


def process_open_files() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0
