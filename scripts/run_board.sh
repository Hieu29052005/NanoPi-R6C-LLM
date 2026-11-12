#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ulimit -Sn 65536 2>/dev/null || true
ulimit -n 65536 2>/dev/null || echo "note: raise 'ulimit -n 65536' as root for 7B-class models"

export LD_LIBRARY_PATH="${RKLLM_LIB_DIR:-$ROOT/models}:$ROOT/lib:${LD_LIBRARY_PATH:-}"
export RKLLM_LOG_LEVEL="${RKLLM_LOG_LEVEL:-1}"
if [ -n "${RKNPU_DOMAINS:-}" ]; then
  export RKNPU_DOMAINS
fi

GOVERNOR=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
if [ -w "${GOVERNOR}" ]; then
  echo performance | tee "${GOVERNOR}" >/dev/null || true
else
  echo "note: cannot set CPU governor without root (optional)"
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi

exec "$PYTHON" run.py "$@"
