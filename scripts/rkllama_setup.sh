#!/usr/bin/env bash
# Build rk-llama.cpp (invisiofficial, branch rknpu2) for RK3588S NPU inference.
# This is Huong B: W4A4/W8A8 GGUF on the NPU, subject to the 4 GB IOMMU domain limit.
set -euo pipefail

RKLLAMA_REPO="${RKLLAMA_REPO:-https://github.com/invisiofficial/rk-llama.cpp.git}"
RKLLAMA_DIR="${RKLLAMA_DIR:-$HOME/rk-llama.cpp}"
RKLLAMA_BRANCH="${RKLLAMA_BRANCH:-rknpu2}"
BUILD_DIR="${BUILD_DIR:-$RKLLAMA_DIR/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

usage() {
  cat <<'EOF'
Usage: rkllama_setup.sh [options]

Options:
  --dir <path>       rk-llama.cpp checkout (default: $RKLLAMA_DIR or ~/rk-llama.cpp)
  --branch <name>    branch (default: rknpu2)
  --jobs <n>         build parallelism (default: nproc)
  --no-clone         fail instead of cloning when the checkout is missing
  -h, --help

Notes:
  - Set RKNPU_DOMAINS when a single model exceeds the 4 GB IOMMU domain.
  - Run `ulimit -n 65536` before loading 7B-class models.
  - librkllmrt.so must be discoverable (LD_LIBRARY_PATH).
EOF
}

NO_CLONE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) RKLLAMA_DIR="$2"; shift 2 ;;
    --branch) RKLLAMA_BRANCH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-clone) NO_CLONE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [ ! -d "$RKLLAMA_DIR/.git" ]; then
  if [ "$NO_CLONE" = "1" ]; then
    echo "ERROR: no checkout at $RKLLAMA_DIR" >&2
    exit 2
  fi
  echo "==> cloning $RKLLAMA_REPO ($RKLLAMA_BRANCH)"
  git clone --branch "$RKLLAMA_BRANCH" --depth 1 "$RKLLAMA_REPO" "$RKLLAMA_DIR"
fi

echo "==> cmake configure (LLAMA_RKNPU2=ON)"
cmake -S "$RKLLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_RKNPU2=ON

echo "==> cmake build"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

cat <<EOF

OK: build finished in $BUILD_DIR

Before running on the board:
  ulimit -n 65536
  export LD_LIBRARY_PATH=/path/to/rknn-llm/runtime/lib:\$LD_LIBRARY_PATH
  export RKNPU_DOMAINS=...        # required when model > 4 GB per IOMMU domain
  export RKNPU_HYBRID=...         # e.g. Q8_0:W8A8_LOWRANK (see ggml-rknpu2 README)

Then point the runtime at the built llama-cli/llama-server and use model.backend=gguf.
EOF
