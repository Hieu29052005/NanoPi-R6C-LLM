#!/usr/bin/env bash
# SpinQuant rotation + GPTQ/RTN for W4A8 on a CUDA host (Kaggle T4/L4/A100/RTX 50xx).
# Target artifact: GGUF for rk-llama.cpp / llama.cpp (Huong A).
#
# LICENSE: QwenSpinQuant / SpinQuant is CC-BY-NC 4.0 (non-commercial).
# This script refuses to run unless --yes is passed after legal review.
#
# QwenSpinQuant needs its own Python environment: it imports APIs that were
# removed from newer transformers releases. Use --setup-env once to create it.
set -euo pipefail

expand_home() {
  printf '%s\n' "${1/#\~/$HOME}"
}

abs_path() {
  local p
  p="$(expand_home "$1")"
  if [ -e "$p" ] || [[ "$p" == /* ]] || [[ "$p" == ./* ]] || [[ "$p" == ../* ]] || [ -d "$(dirname "$p")" ]; then
    realpath -m "$p"
  else
    printf '%s\n' "$p"
  fi
}

QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$HOME/QwenSpinQuant}"
SPINQUANT_VENV="${SPINQUANT_VENV:-$QWEN_SPINQUANT_DIR/.venv-spinquant}"
SPINQUANT_PYTHON="${SPINQUANT_PYTHON:-}"
SPINQUANT_REQUIREMENTS="${SPINQUANT_REQUIREMENTS:-$QWEN_SPINQUANT_DIR/requirement.txt}"
SPINQUANT_TORCH_INDEX_URL="${SPINQUANT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SPINQUANT_TORCH_PACKAGES="${SPINQUANT_TORCH_PACKAGES:-torch torchvision}"
SPINQUANT_ROTATION_DIR="${SPINQUANT_ROTATION_DIR:-$QWEN_SPINQUANT_DIR/output_rotation}"
SPINQUANT_OUTPUT_DIR="${SPINQUANT_OUTPUT_DIR:-$QWEN_SPINQUANT_DIR/output}"
SPINQUANT_LOG_DIR="${SPINQUANT_LOG_DIR:-$QWEN_SPINQUANT_DIR/logs}"
SPINQUANT_MAX_STEPS="${SPINQUANT_MAX_STEPS:-100}"
SPINQUANT_MODEL_MAX_LENGTH="${SPINQUANT_MODEL_MAX_LENGTH:-2048}"
SPINQUANT_NPROC="${SPINQUANT_NPROC:-}"
SPINQUANT_EVAL_NPROC="${SPINQUANT_EVAL_NPROC:-1}"
SPINQUANT_BASE_PYTHON="${SPINQUANT_BASE_PYTHON:-python3}"

MODEL=""
W_BITS=4
A_BITS=8
KV_BITS=16
OUT_DIR=""
RTN=0
CONFIRMED=0
SKIP_ROTATION=0
SKIP_EVAL=0
SETUP_ENV=0

usage() {
  cat <<'EOF'
Usage: spinquant_w4a8.sh --model <hf_id_or_path> --yes [options]

Options:
  --model <id|path>       Model to quantize (e.g. Qwen/Qwen2.5-7B-Instruct)
  --w-bits <n>            Weight bits (default: 4)
  --a-bits <n>            Activation bits (default: 8)
  --kv-bits <n>           KV cache bits (default: 16)
  --out <dir>             Copy quantized output + rotation here
  --rtn                   Use RTN instead of GPTQ
  --skip-rotation         Skip rotation optimization
  --skip-eval             Skip PTQ evaluation
  --setup-env             Create/update the dedicated QwenSpinQuant venv
  --spinquant-python <p>  Use this Python interpreter instead of the venv
  --venv <dir>            Venv path (default: $QWEN_SPINQUANT_DIR/.venv-spinquant)
  --nproc <n>             GPUs for rotation (default: detected, capped at 8)
  --torch-index-url <u>   PyTorch wheel index (default: cu128)
  --yes                   Confirm you reviewed the CC-BY-NC 4.0 license
  -h, --help

Environment:
  QWEN_SPINQUANT_DIR          QwenSpinQuant checkout (default: ~/QwenSpinQuant)
  SPINQUANT_PYTHON            Existing compatible Python interpreter
  SPINQUANT_VENV              Dedicated venv path
  SPINQUANT_TORCH_INDEX_URL   PyTorch wheel index (default: cu128)
  SPINQUANT_TORCH_PACKAGES    Packages to install (default: "torch torchvision")
  SPINQUANT_SKIP_TORCH=1      Do not install PyTorch in --setup-env
  SPINQUANT_SKIP_HADAMARD=1   Do not install fast-hadamard-transform
  SPINQUANT_ROTATION_DIR      Rotation output dir
  SPINQUANT_OUTPUT_DIR        Quantized output dir
  SPINQUANT_LOG_DIR           Log dir
  SPINQUANT_MAX_STEPS         Rotation steps (default: 100)
  SPINQUANT_MODEL_MAX_LENGTH  Max sequence length (default: 2048)
  SPINQUANT_NPROC             GPUs for rotation
  SPINQUANT_EVAL_NPROC        GPUs for PTQ eval (default: 1)
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
    --setup-env) SETUP_ENV=1; shift ;;
    --spinquant-python) SPINQUANT_PYTHON="$2"; shift 2 ;;
    --venv) SPINQUANT_VENV="$2"; shift 2 ;;
    --nproc) SPINQUANT_NPROC="$2"; shift 2 ;;
    --torch-index-url) SPINQUANT_TORCH_INDEX_URL="$2"; shift 2 ;;
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

QWEN_SPINQUANT_DIR="$(expand_home "$QWEN_SPINQUANT_DIR")"
if [ ! -d "$QWEN_SPINQUANT_DIR" ]; then
  echo "ERROR: QWEN_SPINQUANT_DIR not found: $QWEN_SPINQUANT_DIR" >&2
  echo "  git clone https://github.com/shijiew/QwenSpinQuant" >&2
  exit 2
fi
QWEN_SPINQUANT_DIR="$(realpath -m "$QWEN_SPINQUANT_DIR")"

SPINQUANT_VENV="$(realpath -m "$(expand_home "$SPINQUANT_VENV")")"
SPINQUANT_ROTATION_DIR="$(realpath -m "$(expand_home "$SPINQUANT_ROTATION_DIR")")"
SPINQUANT_OUTPUT_DIR="$(realpath -m "$(expand_home "$SPINQUANT_OUTPUT_DIR")")"
SPINQUANT_LOG_DIR="$(realpath -m "$(expand_home "$SPINQUANT_LOG_DIR")")"
SPINQUANT_REQUIREMENTS="$(expand_home "$SPINQUANT_REQUIREMENTS")"

MODEL="$(abs_path "$MODEL")"
if [[ "$MODEL" == /* ]] && [ ! -e "$MODEL" ]; then
  echo "ERROR: model path not found: $MODEL" >&2
  exit 2
fi
if [ -n "$OUT_DIR" ]; then
  OUT_DIR="$(realpath -m "$(expand_home "$OUT_DIR")")"
fi

if [ -z "$SPINQUANT_NPROC" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    SPINQUANT_NPROC="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  SPINQUANT_NPROC="${SPINQUANT_NPROC:-1}"
fi
if [ "$SPINQUANT_NPROC" -gt 8 ]; then
  SPINQUANT_NPROC=8
fi
case "$SPINQUANT_NPROC" in
  ''|*[!0-9]*) echo "ERROR: invalid --nproc: $SPINQUANT_NPROC" >&2; exit 2 ;;
esac
case "$SPINQUANT_EVAL_NPROC" in
  ''|*[!0-9]*) echo "ERROR: invalid SPINQUANT_EVAL_NPROC: $SPINQUANT_EVAL_NPROC" >&2; exit 2 ;;
esac
if [ "$SPINQUANT_NPROC" -lt 1 ] || [ "$SPINQUANT_EVAL_NPROC" -lt 1 ]; then
  echo "ERROR: GPU counts must be >= 1" >&2
  exit 2
fi

setup_env() {
  local base_python
  base_python="$(command -v "$SPINQUANT_BASE_PYTHON" || true)"
  if [ -z "$base_python" ]; then
    echo "ERROR: base Python not found: $SPINQUANT_BASE_PYTHON" >&2
    exit 2
  fi
  echo "==> creating QwenSpinQuant venv: $SPINQUANT_VENV"
  "$base_python" -m venv "$SPINQUANT_VENV"
  "$SPINQUANT_VENV/bin/pip" install --upgrade pip wheel setuptools

  if [ "${SPINQUANT_SKIP_TORCH:-0}" != "1" ]; then
    echo "==> installing PyTorch from $SPINQUANT_TORCH_INDEX_URL"
    # shellcheck disable=SC2086
    "$SPINQUANT_VENV/bin/pip" install $SPINQUANT_TORCH_PACKAGES --index-url "$SPINQUANT_TORCH_INDEX_URL"
  else
    echo "==> skipping PyTorch install (SPINQUANT_SKIP_TORCH=1)"
  fi

  if [ -f "$SPINQUANT_REQUIREMENTS" ]; then
    echo "==> installing QwenSpinQuant requirements: $SPINQUANT_REQUIREMENTS"
    "$SPINQUANT_VENV/bin/pip" install -r "$SPINQUANT_REQUIREMENTS"
  else
    echo "==> requirement.txt not found; installing pinned fallback versions"
    "$SPINQUANT_VENV/bin/pip" install \
      "transformers==4.49.0" "accelerate==1.4.0" "datasets==2.20.0" \
      sentencepiece tensorboardX
  fi

  if [ "${SPINQUANT_SKIP_HADAMARD:-0}" != "1" ]; then
    echo "==> installing fast-hadamard-transform"
    if ! command -v git >/dev/null 2>&1; then
      echo "ERROR: git is required to install fast-hadamard-transform" >&2
      exit 2
    fi
    "$SPINQUANT_VENV/bin/pip" install "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
  fi
}

if [ "$SETUP_ENV" = "1" ]; then
  setup_env
fi

if [ -n "$SPINQUANT_PYTHON" ]; then
  PY="$(expand_home "$SPINQUANT_PYTHON")"
elif [ -x "$SPINQUANT_VENV/bin/python" ]; then
  PY="$SPINQUANT_VENV/bin/python"
elif [ -x "$QWEN_SPINQUANT_DIR/.venv/bin/python" ]; then
  PY="$QWEN_SPINQUANT_DIR/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "ERROR: no usable Python interpreter found" >&2
  echo "  Run once with --setup-env to create $SPINQUANT_VENV" >&2
  exit 2
fi
export PATH="$(dirname "$PY"):$PATH"

preflight() {
  local py="$1"
  local require_hadamard="$2"
  echo "==> python: $py"
  "$py" - "$require_hadamard" <<'PY'
import sys

require_hadamard = sys.argv[1] == "1"

try:
    import torch
except Exception as exc:
    sys.exit(f"ERROR: torch import failed: {exc}")

print(f"torch: {torch.__version__} | cuda: {torch.version.cuda} | available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is required for SpinQuant/GPTQ")
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} | compute capability: {cap[0]}.{cap[1]}")
if cap[0] >= 12 and torch.__version__.split(".")[0] == "2" and int(torch.__version__.split(".")[1].split("+")[0]) < 7:
    print("WARN: Blackwell GPUs usually need PyTorch >= 2.7 with CUDA 12.8+", file=sys.stderr)

try:
    import transformers
except Exception as exc:
    sys.exit(f"ERROR: transformers import failed: {exc}")
print(f"transformers: {transformers.__version__}")
try:
    from transformers.utils import is_torch_compile_available  # noqa: F401
except ImportError:
    sys.exit(
        "ERROR: transformers "
        + transformers.__version__
        + " is incompatible with QwenSpinQuant.\n"
        "       QwenSpinQuant expects transformers==4.49.0 (or 4.44.2).\n"
        "       Run with --setup-env to create a dedicated venv, or set SPINQUANT_PYTHON."
    )

try:
    import accelerate  # noqa: F401
    import datasets  # noqa: F401
except Exception as exc:
    sys.exit(f"ERROR: accelerate/datasets import failed: {exc}")

if require_hadamard:
    try:
        import fast_hadamard_transform  # noqa: F401
    except Exception as exc:
        sys.exit(
            f"ERROR: fast_hadamard_transform is not importable: {exc}\n"
            "       Run --setup-env or install fast-hadamard-transform manually."
        )
PY
}

if ! preflight "$PY" "$([ "$SKIP_ROTATION" = "1" ] && echo 0 || echo 1)"; then
  cat >&2 <<EOF

QwenSpinQuant must run with its own pinned dependencies.
Fix (one time):
  bash scripts/spinquant_w4a8.sh --setup-env --model "$MODEL" --out "$OUT_DIR" --yes
Or point at an existing compatible environment:
  export SPINQUANT_PYTHON=/path/to/venv/bin/python
EOF
  exit 2
fi

mkdir -p "$SPINQUANT_ROTATION_DIR" "$SPINQUANT_OUTPUT_DIR" "$SPINQUANT_LOG_DIR"

cd "$QWEN_SPINQUANT_DIR"

run_rotation() {
  local args=(
    --nnodes=1
    --nproc_per_node="$SPINQUANT_NPROC"
    optimize_rotation.py
    --input_model "$MODEL"
    --output_rotation_path "$SPINQUANT_ROTATION_DIR"
    --output_dir "$SPINQUANT_OUTPUT_DIR/"
    --logging_dir "$SPINQUANT_LOG_DIR/"
    --model_max_length "$SPINQUANT_MODEL_MAX_LENGTH"
    --fp16 False
    --bf16 True
    --log_on_each_node False
    --per_device_train_batch_size 1
    --logging_steps 1
    --learning_rate 1.5
    --weight_decay 0.
    --lr_scheduler_type cosine
    --gradient_checkpointing True
    --save_safetensors False
    --max_steps "$SPINQUANT_MAX_STEPS"
    --w_bits "$W_BITS"
    --a_bits "$A_BITS"
    --k_bits "$KV_BITS"
    --v_bits "$KV_BITS"
    --w_clip
    --a_asym
    --k_asym
    --v_asym
    --k_groupsize 128
    --v_groupsize 128
  )
  if [ "$RTN" = "1" ]; then
    args+=(--w_rtn)
  fi
  echo "==> optimize_rotation.py (nproc=$SPINQUANT_NPROC)"
  "$PY" -m torch.distributed.run "${args[@]}"
}

run_eval() {
  local args=(
    --nnodes=1
    --nproc_per_node="$SPINQUANT_EVAL_NPROC"
    ptq.py
    --input_model "$MODEL"
    --do_train False
    --do_eval True
    --per_device_eval_batch_size 4
    --model_max_length "$SPINQUANT_MODEL_MAX_LENGTH"
    --fp16 False
    --bf16 True
    --save_safetensors False
    --w_bits "$W_BITS"
    --a_bits "$A_BITS"
    --k_bits "$KV_BITS"
    --v_bits "$KV_BITS"
    --w_clip
    --a_asym
    --k_asym
    --v_asym
    --k_groupsize 128
    --v_groupsize 128
    --rotate
    --optimized_rotation_path "$SPINQUANT_ROTATION_DIR/R.bin"
  )
  if [ "$RTN" = "1" ]; then
    args+=(--w_rtn)
  fi
  echo "==> ptq.py (nproc=$SPINQUANT_EVAL_NPROC)"
  "$PY" -m torch.distributed.run "${args[@]}"
}

if [ "$SKIP_ROTATION" != "1" ]; then
  run_rotation
  if [ ! -f "$SPINQUANT_ROTATION_DIR/R.bin" ]; then
    echo "WARN: rotation file not found: $SPINQUANT_ROTATION_DIR/R.bin" >&2
  fi
fi

if [ "$SKIP_EVAL" != "1" ]; then
  if [ ! -f "$SPINQUANT_ROTATION_DIR/R.bin" ]; then
    echo "ERROR: missing rotation file: $SPINQUANT_ROTATION_DIR/R.bin" >&2
    echo "  Run without --skip-rotation first, or set SPINQUANT_ROTATION_DIR." >&2
    exit 2
  fi
  run_eval
fi

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  if [ -d "$SPINQUANT_OUTPUT_DIR" ]; then
    cp -a "$SPINQUANT_OUTPUT_DIR/." "$OUT_DIR/"
  fi
  if [ -d "$SPINQUANT_ROTATION_DIR" ]; then
    mkdir -p "$OUT_DIR/rotation"
    cp -a "$SPINQUANT_ROTATION_DIR/." "$OUT_DIR/rotation/"
  fi
  echo "OK: copied output -> $OUT_DIR"
fi

echo "Next: convert the rotated checkpoint to GGUF with scripts/export_gguf.sh"
echo "      then deploy the Q4_0/Q4_K_M GGUF to the R6C and run llama.cpp (Huong A)."
