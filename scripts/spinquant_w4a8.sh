#!/usr/bin/env bash
# SpinQuant rotation + PTQ eval cho Qwen2.5 (CUDA host).
# Luong chay:
#   1. Tao venv (uv neu co, neu khong thi python -m venv)
#   2. Cai requirements.txt
#   3. Build fast-hadamard-transform voi CUDA toolkit khop voi torch
#   4. scripts/10_optimize_rotation.sh
#   5. scripts/2_eval_ptq.sh
#
# LICENSE: QwenSpinQuant / SpinQuant la CC-BY-NC 4.0 (phi thuong mai).
# Script tu choi chay neu khong co --yes.
set -euo pipefail

# ---------------------------------------------------------------- duong dan
PROJECT_DIR="${PROJECT_DIR:-/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM}"
QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$PROJECT_DIR/QwenSpinQuant}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"
FHT_REF="${FHT_REF:-e7706faf8d1c3b9f241e36860640ad1dac644ede}"
FHT_URL="git+https://github.com/Dao-AILab/fast-hadamard-transform.git@${FHT_REF}"

# ------------------------------------------------------------ tham so mac dinh
MODEL=""
W_BITS=4
A_BITS=8
KV_BITS=16
OUT_DIR=""
RTN=0
CONFIRMED=0
SKIP_SETUP=0
SKIP_ROTATION=0
SKIP_EVAL=0
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<'EOF'
Usage: spinquant_w4a8.sh --model <hf_id_or_path> --yes [options]

Options:
  --model <id|path>     Model can quantize (vd: /path/to/Qwen2.5-7B-Instruct)
  --w-bits <n>          Weight bits (mac dinh: 4)
  --a-bits <n>          Activation bits (mac dinh: 8)
  --kv-bits <n>         KV cache bits (mac dinh: 16)
  --out <dir>           Copy output/ va rotation/ vao thu muc nay
  --rtn                 Dung RTN thay vi GPTQ (truyen --w_rtn cho script)
  --python <bin>        Python dung de tao venv (mac dinh: python3.10 hoac python3)
  --skip-setup          Bo qua tao venv / cai package (venv da san sang)
  --skip-rotation       Bo qua scripts/10_optimize_rotation.sh
  --skip-eval           Bo qua scripts/2_eval_ptq.sh
  --yes                 Xac nhan da xem license CC-BY-NC 4.0
  -h, --help

Bien moi truong:
  PROJECT_DIR           (mac dinh: /mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM)
  QWEN_SPINQUANT_DIR    (mac dinh: $PROJECT_DIR/QwenSpinQuant)
  VENV_DIR              (mac dinh: $PROJECT_DIR/.venv)
  REQ_FILE              (mac dinh: $PROJECT_DIR/requirements.txt)
  CUDA_HOME             (mac dinh: tu tim /usr/local/cuda-13.0 roi /usr/local/cuda)
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
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-rotation) SKIP_ROTATION=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --yes) CONFIRMED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ kiem tra
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
  echo "  git clone https://github.com/shijiew/QwenSpinQuant $QWEN_SPINQUANT_DIR" >&2
  exit 2
fi

for f in scripts/10_optimize_rotation.sh scripts/2_eval_ptq.sh; do
  if [ ! -f "$QWEN_SPINQUANT_DIR/$f" ]; then
    echo "ERROR: missing $QWEN_SPINQUANT_DIR/$f" >&2
    exit 2
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN=python3.10
  else
    PYTHON_BIN=python3
  fi
fi

# ------------------------------------------------------------- CUDA toolkit
# torch cu130 can nvcc 13.x. Tu tim toolkit neu CUDA_HOME chua duoc dat.
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do
    if [ -x "$c/bin/nvcc" ]; then
      CUDA_HOME="$c"
      break
    fi
  done
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Cai CUDA toolkit khop voi torch (vd 13.0) va dat CUDA_HOME." >&2
  exit 2
fi
echo "nvcc: $(nvcc --version | tail -n 1)"

export UV_LINK_MODE=copy
export MAX_JOBS="${MAX_JOBS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ----------------------------------------------------------- venv + packages
have_uv() { command -v uv >/dev/null 2>&1; }

pip_install() {
  if have_uv; then
    uv pip install --python "$VENV_DIR/bin/python" "$@"
  else
    "$VENV_DIR/bin/python" -m pip install "$@"
  fi
}

if [ "$SKIP_SETUP" != "1" ]; then
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "==> Creating venv at $VENV_DIR"
    if have_uv; then
      uv venv "$VENV_DIR" --python "$PYTHON_BIN"
    else
      "$PYTHON_BIN" -m venv "$VENV_DIR"
      "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    fi
  else
    echo "==> Reusing venv at $VENV_DIR"
  fi

  if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements file not found: $REQ_FILE" >&2
    exit 2
  fi

  # fast-hadamard-transform phai build SAU torch va voi --no-build-isolation,
  # nen loai no khoi lan cai requirements dau tien.
  TMP_REQ="$(mktemp)"
  trap 'rm -f "$TMP_REQ"' EXIT
  grep -vi "fast[-_]hadamard[-_]transform" "$REQ_FILE" > "$TMP_REQ" || true

  echo "==> Installing $REQ_FILE (without fast-hadamard-transform)"
  pip_install -r "$TMP_REQ"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "ERROR: venv not found at $VENV_DIR (bo --skip-setup de tao)." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: torch chua duoc cai. Them torch vao $REQ_FILE (hoac cai truoc roi chay lai)." >&2
  exit 2
fi

python - <<'PY' || { echo "ERROR: CUDA khong dung duoc trong torch." >&2; exit 2; }
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
n = torch.cuda.device_count()
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {n} GPU(s):")
for i in range(n):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

# nvcc va torch phai cung major CUDA, neu khong torch se tu choi build.
python - <<'PY' || exit 2
import re, subprocess, sys, torch
out = subprocess.check_output(["nvcc", "--version"], text=True)
m = re.search(r"release\s+(\d+)\.(\d+)", out)
nvcc_major = int(m.group(1)) if m else None
torch_major = int(torch.version.cuda.split(".")[0])
if nvcc_major != torch_major:
    print(f"ERROR: nvcc {m.group(0) if m else '?'} khong khop torch CUDA {torch.version.cuda}.", file=sys.stderr)
    print("Dat CUDA_HOME toi toolkit cung major voi torch (vd /usr/local/cuda-13.0).", file=sys.stderr)
    sys.exit(1)
print(f"nvcc {m.group(1)}.{m.group(2)} khop torch CUDA {torch.version.cuda}")
PY

if ! python -c 'import fast_hadamard_transform' >/dev/null 2>&1; then
  ARCH="$(python -c 'import torch; m = torch.cuda.get_device_capability(0); print(f"{m[0]}.{m[1]}")')"
  echo "==> Building fast-hadamard-transform (TORCH_CUDA_ARCH_LIST=$ARCH)"
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}" pip_install --no-build-isolation "$FHT_URL"
fi

python -c 'import fast_hadamard_transform; print("fast_hadamard_transform OK")'

# -------------------------------------------------------------- chay SpinQuant
ARGS=("$MODEL" "$W_BITS" "$A_BITS" "$KV_BITS")
if [ "$RTN" = "1" ]; then
  ARGS+=(--w_rtn)
fi

cd "$QWEN_SPINQUANT_DIR"
LOG_DIR="$QWEN_SPINQUANT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "$SKIP_ROTATION" != "1" ]; then
  echo "==> bash scripts/10_optimize_rotation.sh ${ARGS[*]}"
  bash scripts/10_optimize_rotation.sh "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/rotation_$STAMP.log"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  echo "==> bash scripts/2_eval_ptq.sh ${ARGS[*]}"
  bash scripts/2_eval_ptq.sh "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/eval_$STAMP.log"
fi

# ------------------------------------------------------------------- ket qua
if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  for d in output output_ptq rotation; do
    if [ -d "$QWEN_SPINQUANT_DIR/$d" ]; then
      mkdir -p "$OUT_DIR/$d"
      cp -a "$QWEN_SPINQUANT_DIR/$d/." "$OUT_DIR/$d/"
      echo "OK: copied $d -> $OUT_DIR/$d"
    fi
  done
fi

echo "Done. Logs: $LOG_DIR"
echo "Next: export model da rotate/quantize sang dinh dang runtime tren board (GGUF cho llama.cpp)."
