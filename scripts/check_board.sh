#!/usr/bin/env bash
# Quick pre-flight for NanoPi-R6C: model, NPU driver, RKLLM library, power ADC.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
WARN=0

ok() { echo "PASS: $*"; PASS=$((PASS + 1)); }
warn() { echo "WARN: $*"; WARN=$((WARN + 1)); }

echo "== board =="
MODEL="$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || true)"
[ -n "$MODEL" ] && ok "$MODEL" || warn "cannot read /proc/device-tree/model"
uname -a

MEM_TOTAL="$(awk '/MemTotal/ {printf "%.2f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
MEM_AVAIL="$(awk '/MemAvailable/ {printf "%.2f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
ok "RAM total ${MEM_TOTAL} GB, available ${MEM_AVAIL} GB"

echo "== NPU =="
if [ -r /sys/kernel/debug/rknpu/version ]; then
  ok "rknpu version: $(cat /sys/kernel/debug/rknpu/version)"
else
  warn "cannot read /sys/kernel/debug/rknpu/version (try sudo)"
fi
if command -v dmesg >/dev/null 2>&1; then
  dmesg 2>/dev/null | grep -iE 'rknpu|rknn' | tail -n 3 || warn "no rknpu lines in dmesg (permissions?)"
fi

echo "== RKLLM =="
LIB="$(find "$ROOT/models" "$ROOT/lib" -maxdepth 2 -name 'librkllmrt.so' 2>/dev/null | head -n1)"
if [ -n "$LIB" ]; then
  ok "librkllmrt.so: $LIB"
else
  warn "librkllmrt.so not found in models/ or lib/"
fi

echo "== models =="
shopt -s nullglob
MODELS=("$ROOT"/models/*.rkllm "$ROOT"/models/*.gguf)
if [ ${#MODELS[@]} -gt 0 ]; then
  for model in "${MODELS[@]}"; do
    ok "$(basename "$model") $(du -h "$model" | cut -f1)"
  done
else
  warn "no .rkllm/.gguf models in $ROOT/models"
fi
shopt -u nullglob

echo "== power ADC =="
ADC=/sys/devices/platform/fec10000.saradc/iio:device0/in_voltage2_raw
if [ -r "$ADC" ]; then
  ok "ADC voltage raw: $(cat "$ADC") (voltage only; use a USB-C power meter for W)"
else
  warn "ADC path not readable: $ADC"
fi

echo "== python =="
python3 --version || warn "python3 missing"
python3 - <<'PY' || true
mods = ["yaml", "fastapi", "uvicorn"]
for name in mods:
    try:
        __import__(name)
        print(f"PASS: python module {name}")
    except Exception:
        print(f"WARN: python module {name} missing (pip install -r requirements.txt)")
PY

echo
echo "summary: ${PASS} pass, ${WARN} warn"
echo "Reminder: RK3588S NPU does not support native W4A8; use W8A8 (RKLLM) or W4A4 (rk-llama.cpp)."
