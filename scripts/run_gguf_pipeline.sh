#!/usr/bin/env bash
# So sanh 2 model tren RTX 5080: Qwen goc (bf16) vs Qwen SpinQuant 4-bit (xoay R1,R2 + llama.cpp Q4).
#
#   1. export_rotated_hf.py  : gop R1,R2 vao trong so (xem dong [verify] in ra o buoc nay)
#   2. llama.cpp             : clone + build
#   3. convert_hf_to_gguf.py : 2 model -> GGUF bf16
#   4. llama-quantize        : model xoay -> 4-bit
#   5. bench_llm.py          : size / RAM / VRAM / load / TTFT / prefill / decode / cong suat
#   6. (tuy chon PPL=1)      : perplexity wikitext-2
#
# Chay:  source $PROJECT_DIR/.venv/bin/activate && GPU=1 bash run_gguf_pipeline.sh
# Moi buoc bo qua neu file dau ra da ton tai. Them WITH_ORIG_Q4=1 neu muon them cot "origin_Q4".
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM}"
QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$PROJECT_DIR/QwenSpinQuant}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct}"
ROT_DIR="${ROT_DIR:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct-rot}"
R_BIN="${R_BIN:-$QWEN_SPINQUANT_DIR/rotation/R.bin}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/llama.cpp}"
GGUF_DIR="${GGUF_DIR:-$PROJECT_DIR/gguf}"
BENCH="${BENCH:-$PROJECT_DIR/scripts/eval_performance.py}"
OUT="${OUT:-$PROJECT_DIR/bench_gguf.json}"

GPU="${GPU:-1}"                       # chi so GPU VAT LY (dung cho CUDA_VISIBLE_DEVICES va NVML)
QUANT="${QUANT:-Q4_0}"                # kieu quantize cua llama.cpp (Q4_0, Q4_K_M, ...)
LLAMA_CUDA="${LLAMA_CUDA:-1}"         # 1 = build llama.cpp / llama-cpp-python co CUDA
STOP_AFTER_EXPORT="${STOP_AFTER_EXPORT:-0}"
PPL="${PPL:-0}"
WITH_ORIG_Q4="${WITH_ORIG_Q4:-0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
RUNS="${RUNS:-3}"
CUDA_ARCH="${CUDA_ARCH:-120}"         # RTX 5080 = sm_120
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"     # giam xuong neu model BF16 khong vua VRAM
if [ "$LLAMA_CUDA" = "1" ]; then
  LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_CPP_DIR/build-cuda$CUDA_ARCH}"
else
  LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_CPP_DIR/build}"
fi

step() { printf '\n==> %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "${VIRTUAL_ENV:-}" ] || die "chua activate venv (source $PROJECT_DIR/.venv/bin/activate)"
[ -f "$MODEL_DIR/config.json" ] || die "khong thay model goc: $MODEL_DIR"
[ -f "$R_BIN" ] || die "khong thay R.bin: $R_BIN"
mkdir -p "$GGUF_DIR"

# CUDA toolkit cung major voi torch (cu130 -> 13.x), giong luc build fast-hadamard-transform
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do
    [ -x "$c/bin/nvcc" ] && CUDA_HOME="$c" && break
  done
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export UV_LINK_MODE=copy

if [ "$LLAMA_CUDA" = "1" ]; then
  [ -n "${CUDA_HOME:-}" ] || die "CUDA Toolkit not found; set CUDA_HOME (RTX 5080: export CUDA_HOME=/usr/local/cuda)"
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "khong tim thay nvcc trong CUDA_HOME=$CUDA_HOME"
  if ! python - "$CUDA_HOME/bin/nvcc" "$CUDA_ARCH" <<'PY'
import re
import subprocess
import sys

nvcc, arch = sys.argv[1:]
out = subprocess.check_output([nvcc, "--version"], text=True)
m = re.search(r"release\s+(\d+)\.(\d+)", out)
if not m:
    raise SystemExit(1)
major, minor = map(int, m.groups())
print(f"CUDA Toolkit: {major}.{minor} ({nvcc})")
# Blackwell (sm_120) needs CUDA 12.8 or newer.
if arch == "120" and (major, minor) < (12, 8):
    raise SystemExit(2)
PY
  then
    die "CUDA_ARCH=$CUDA_ARCH requires CUDA Toolkit 12.8+; set CUDA_HOME to a compatible toolkit (RTX 5080: /usr/local/cuda points to CUDA 13.0)"
  fi
fi

# ---------------------------------------------------------------- 1. export
if [ ! -f "$ROT_DIR/config.json" ]; then
  step "1/6 Gop R1,R2 vao trong so -> $ROT_DIR"
  (cd "$QWEN_SPINQUANT_DIR" && CUDA_VISIBLE_DEVICES="$GPU" python export_rotated_hf.py \
      --input_model "$MODEL_DIR" --rotation_path "$R_BIN" --output_dir "$ROT_DIR" --verify)
else
  step "1/6 Da co $ROT_DIR, bo qua export"
fi
if [ "$STOP_AFTER_EXPORT" = "1" ]; then
  echo "Dung sau export (STOP_AFTER_EXPORT=1). Doc ket qua [verify] o tren; neu KL nho va top-1 ~100% thi chay lai khong co bien nay."
  exit 0
fi

# ---------------------------------------------------------------- 2. llama.cpp
if ! command -v cmake >/dev/null 2>&1; then
  die "khong tim thay cmake; cai trong venv bang: uv pip install 'cmake>=3.18,<4'"
fi
if ! python - <<'PY'
import re
import subprocess
import sys

output = subprocess.check_output(["cmake", "--version"], text=True)
match = re.search(r"cmake version (\d+)\.(\d+)", output)
if not match or tuple(map(int, match.groups())) < (3, 18):
    sys.exit(1)
print(f"CMake: {match.group(1)}.{match.group(2)}")
PY
then
  die "llama.cpp CUDA build requires CMake >=3.18; current: $(cmake --version | head -n 1). Run: uv pip install --upgrade 'cmake>=3.18,<4'"
fi
step "2/6 llama.cpp"
[ -d "$LLAMA_CPP_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
if [ ! -x "$LLAMA_BUILD_DIR/bin/llama-quantize" ]; then
  CMAKE_FLAGS=()
  if [ "$LLAMA_CUDA" = "1" ]; then
    CMAKE_FLAGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc" "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_BUILD_DIR" "${CMAKE_FLAGS[@]}"
  cmake --build "$LLAMA_BUILD_DIR" --config Release -j "$(nproc)" --target llama-quantize llama-perplexity
fi

# KHONG cai requirements/requirements-convert_hf_to_gguf.txt: file do ghim torch ban CPU
# va se GHI DE torch cu130 trong venv cua ban. Chi cai thu vien gguf di kem repo.
if ! python -c "import gguf, sentencepiece" >/dev/null 2>&1; then
  uv pip install -e "$LLAMA_CPP_DIR/gguf-py" sentencepiece
fi

# ---------------------------------------------------------------- 3. convert
convert() {  # $1 = thu muc HF, $2 = file gguf dau ra
  if [ -f "$2" ]; then echo "co san: $2"; return; fi
  python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$1" --outtype bf16 --outfile "$2"
}
step "3/6 Convert sang GGUF bf16"
convert "$MODEL_DIR" "$GGUF_DIR/orig-bf16.gguf"
convert "$ROT_DIR" "$GGUF_DIR/rot-bf16.gguf"

# ---------------------------------------------------------------- 4. quantize
quant() {  # $1 = gguf bf16, $2 = gguf quantize
  if [ -f "$2" ]; then echo "co san: $2"; return; fi
  "$LLAMA_BUILD_DIR/bin/llama-quantize" "$1" "$2" "$QUANT"
}
step "4/6 Quantize -> $QUANT"
quant "$GGUF_DIR/rot-bf16.gguf" "$GGUF_DIR/rot-$QUANT.gguf"
if [ "$WITH_ORIG_Q4" = "1" ]; then
  quant "$GGUF_DIR/orig-bf16.gguf" "$GGUF_DIR/orig-$QUANT.gguf"
fi

# ---------------------------------------------------------------- 5. benchmark
step "5/6 Benchmark (llama-cpp-python, GPU $GPU)"
[ -f "$BENCH" ] || die "khong thay $BENCH (chep bench_llm.py vao do hoac dat BENCH=...)"
if ! python -c "import llama_cpp" >/dev/null 2>&1; then
  echo "Cai llama-cpp-python (build tu source, mat vai phut)..."
  if [ "$LLAMA_CUDA" = "1" ]; then
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
      uv pip install llama-cpp-python --no-cache-dir
  else
    uv pip install llama-cpp-python --no-cache-dir
  fi
fi
python - <<'PY'
import llama_cpp
print("llama-cpp-python", llama_cpp.__version__, "| GPU offload:", llama_cpp.llama_supports_gpu_offload())
PY

MODELS=(--model "origin=gguf:$GGUF_DIR/orig-bf16.gguf" --model "spinquant_4bit=gguf:$GGUF_DIR/rot-$QUANT.gguf")
if [ "$WITH_ORIG_Q4" = "1" ]; then
  MODELS+=(--model "origin_$QUANT=gguf:$GGUF_DIR/orig-$QUANT.gguf")
fi
# origin bf16 (~14 GiB) co the khong vua 16 GB: neu OOM thi cot do bi bo qua va bang chi con cot con lai.
CUDA_VISIBLE_DEVICES="$GPU" python "$BENCH" \
  "${MODELS[@]}" \
  --n-gpu-layers "$N_GPU_LAYERS" --no-mmap \
  --prompt-tokens "$PROMPT_TOKENS" --gen-tokens "$GEN_TOKENS" --runs "$RUNS" \
  --power nvml --nvml-index "$GPU" --power-interval 0.05 --idle-seconds 5 \
  --out "$OUT"

# ---------------------------------------------------------------- 6. perplexity
if [ "$PPL" = "1" ]; then
  step "6/6 Perplexity wikitext-2 (llama-perplexity)"
  WIKI="$GGUF_DIR/wikitext-2-raw/wiki.test.raw"
  if [ ! -f "$WIKI" ]; then
    (cd "$GGUF_DIR" && bash "$LLAMA_CPP_DIR/scripts/get-wikitext-2.sh")
  fi
  [ -f "$WIKI" ] || die "khong tai duoc wikitext-2: tai tay roi dat tai $WIKI"
  PPL_MODELS=("orig-bf16" "rot-$QUANT")
  [ "$WITH_ORIG_Q4" = "1" ] && PPL_MODELS+=("orig-$QUANT")
  for m in "${PPL_MODELS[@]}"; do
    echo "--- $m"
    CUDA_VISIBLE_DEVICES="$GPU" "$LLAMA_BUILD_DIR/bin/llama-perplexity" \
      -m "$GGUF_DIR/$m.gguf" -f "$WIKI" -ngl 99 2>&1 | tee "$GGUF_DIR/ppl_$m.log" || echo "[skip] $m"
  done
  echo; echo "== Final estimate =="
  grep -H "Final estimate" "$GGUF_DIR"/ppl_*.log || true
fi

echo; echo "Xong. Ket qua do: $OUT (+ .csv)"
