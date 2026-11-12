#!/usr/bin/env bash
# Convert a HuggingFace checkpoint to GGUF and quantize it with llama.cpp.
# Run on an x86_64 host with enough RAM/disk; the R6C only runs inference.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_DIR=""
OUTDIR="data/gguf"
QUANT="Q4_K_M"
OUTTYPE="f16"
SKIP_F16=0

usage() {
  cat <<'EOF'
Usage: export_gguf.sh --model-dir <dir> [options]

Options:
  --model-dir <dir>     HuggingFace model directory (after SpinQuant if used)
  --llama-cpp-dir <dir> llama.cpp checkout (default: $LLAMA_CPP_DIR or ~/llama.cpp)
  --outdir <dir>        Output directory (default: data/gguf)
  --quant <type>        Q4_0 | Q4_K_M | Q5_K_M | Q8_0 | f16 (default: Q4_K_M)
  --outtype <type>      f16 | bf16 | f32 | q8_0 (default: f16)
  --skip-f16            Reuse an existing <model>-f16.gguf in outdir
  -h, --help

Examples:
  LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
      --model-dir models/Qwen2.5-7B-Instruct --quant Q4_K_M
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --llama-cpp-dir) LLAMA_CPP_DIR="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --outtype) OUTTYPE="$2"; shift 2 ;;
    --skip-f16) SKIP_F16=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: --model-dir must point to an existing HuggingFace directory" >&2
  exit 2
fi

CONVERTER="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [ ! -f "$CONVERTER" ]; then
  echo "ERROR: converter not found: $CONVERTER" >&2
  echo "  git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt" >&2
  exit 2
fi

mkdir -p "$OUTDIR"
NAME="$(basename "$(cd "$MODEL_DIR" && pwd)")"
F16="$OUTDIR/${NAME}-${OUTTYPE}.gguf"
TARGET="$OUTDIR/${NAME}-${QUANT}.gguf"

if [ "$SKIP_F16" != "1" ]; then
  echo "==> convert HF -> $F16 ($OUTTYPE)"
  python "$CONVERTER" "$MODEL_DIR" --outfile "$F16" --outtype "$OUTTYPE"
fi

if [ "$QUANT" = "f16" ] || [ "$QUANT" = "bf16" ]; then
  echo "OK: $F16"
  ls -lh "$F16"
  exit 0
fi

QUANTIZER="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -x "$QUANTIZER" ]; then
  QUANTIZER="$(command -v llama-quantize || true)"
fi
if [ -z "$QUANTIZER" ] || [ ! -x "$QUANTIZER" ]; then
  echo "ERROR: llama-quantize not found; build llama.cpp first" >&2
  exit 2
fi

echo "==> quantize $F16 -> $TARGET ($QUANT)"
"$QUANTIZER" "$F16" "$TARGET" "$QUANT"

echo "OK: $TARGET"
ls -lh "$TARGET"
echo "Copy the GGUF to the R6C and set model.backend=gguf + model.path in config/config.yaml."
