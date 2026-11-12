#!/usr/bin/env bash
# SpinQuant rotation + GPTQ/RTN for W4A8 on a CUDA host (Kaggle T4/L4/A100).
# Target artifact: GGUF for rk-llama.cpp / llama.cpp (Huong A).
#
# LICENSE: QwenSpinQuant / SpinQuant is CC-BY-NC 4.0 (non-commercial).
# This script refuses to run unless --yes is passed after legal review.
set -euo pipefail

QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$HOME/QwenSpinQuant}"
MODEL=""
W_BITS=4
A_BITS=8
KV_BITS=16
OUT_DIR=""
RTN=0
CONFIRMED=0
SKIP_ROTATION=0
SKIP_EVAL=0

usage() {
  cat <<'EOF'
Usage: spinquant_w4a8.sh --model <hf_id_or_path> --yes [options]

Options:
  --model <id|path>     Model to quantize (e.g. Qwen/Qwen2.5-7B-Instruct)
  --w-bits <n>          Weight bits (default: 4)
  --a-bits <n>          Activation bits (default: 8)
  --kv-bits <n>         KV cache bits (default: 16)
  --out <dir>           Copy QwenSpinQuant output/ here
  --rtn                 Use RTN instead of GPTQ (slower quality check, faster run)
  --skip-rotation        Skip 10_optimize_rotation.sh
  --skip-eval            Skip 2_eval_ptq.sh
  --yes                  Confirm you reviewed the CC-BY-NC 4.0 license
  -h, --help

Environment:
  QWEN_SPINQUANT_DIR    Path to github.com/shijiew/QwenSpinQuant (default: ~/QwenSpinQuant)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --w-bits) W_BITS="$2"; shift 2 ;;
    --a-bits) A_BITS="$2"; shift 2 ;;
    --kv-bits) KV_BITS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --rtn) RTN=1; shift ;;
    --skip-rotation) SKIP_ROTATION=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --yes) CONFIRMED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "ERROR: --model is required" >&2
  usage
  exit 2
fi

if [ "$CONFIRMED" != "1" ]; then
  cat >&2 <<'EOF'
REFUSING TO RUN: SpinQuant / QwenSpinQuant is licensed CC-BY-NC 4.0.
It may not be used for commercial products without permission.
Get legal sign-off, then re-run with --yes.
EOF
  exit 3
fi

if [ ! -d "$QWEN_SPINQUANT_DIR" ]; then
  echo "ERROR: QWEN_SPINQUANT_DIR not found: $QWEN_SPINQUANT_DIR" >&2
  echo "  git clone https://github.com/shijiew/QwenSpinQuant" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found" >&2
  exit 2
fi

python - <<'PY' || { echo "ERROR: CUDA is required for SpinQuant/GPTQ" >&2; exit 2; }
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
print(f"CUDA: {torch.cuda.get_device_name(0)}")
PY

ARGS=("$MODEL" "$W_BITS" "$A_BITS" "$KV_BITS")
if [ "$RTN" = "1" ]; then
  ARGS+=(--w_rtn)
fi

cd "$QWEN_SPINQUANT_DIR"

if [ "$SKIP_ROTATION" != "1" ]; then
  echo "==> bash scripts/10_optimize_rotation.sh ${ARGS[*]}"
  bash scripts/10_optimize_rotation.sh "${ARGS[@]}"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  echo "==> bash scripts/2_eval_ptq.sh ${ARGS[*]}"
  bash scripts/2_eval_ptq.sh "${ARGS[@]}"
fi

if [ -n "$OUT_DIR" ] && [ -d "$QWEN_SPINQUANT_DIR/output" ]; then
  mkdir -p "$OUT_DIR"
  cp -a "$QWEN_SPINQUANT_DIR/output/." "$OUT_DIR/"
  echo "OK: copied output -> $OUT_DIR"
fi

echo "Next: convert the rotated checkpoint to GGUF with scripts/export_gguf.sh"
echo "      then deploy the Q4_0/Q4_K_M GGUF to the R6C and run llama.cpp (Huong A)."
