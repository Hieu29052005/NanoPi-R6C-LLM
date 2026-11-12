#!/usr/bin/env bash
set -euo pipefail

BOARD="${BOARD:-pi@NanoPi-R6C}"
BOARD_DIR="${BOARD_DIR:-~/NanoPi-R6C-LLM}"
COPY_MODELS="${COPY_MODELS:-0}"
COPY_RESULTS="${COPY_RESULTS:-1}"

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "deploying ${ROOT} -> ${BOARD}:${BOARD_DIR}"
ssh "${BOARD}" "mkdir -p ${BOARD_DIR}"

EXCLUDES=(
  --exclude=.venv
  --exclude=__pycache__
  --exclude=.pytest_cache
  --exclude=.git
  --exclude=logs
  --exclude=data/gguf
)
if [ "${COPY_MODELS}" != "1" ]; then
  EXCLUDES+=(--exclude=models)
fi
if [ "${COPY_RESULTS}" != "1" ]; then
  EXCLUDES+=(--exclude=data/results)
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -az "${EXCLUDES[@]}" "${ROOT}/" "${BOARD}:${BOARD_DIR}/"
else
  tar "${EXCLUDES[@]}" -czf - . | ssh "${BOARD}" "tar -xzf - -C ${BOARD_DIR}"
fi

echo "done. next steps on the board:"
echo "  ssh ${BOARD}"
echo "  cd ${BOARD_DIR}"
echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "  # copy librkllmrt.so to models/ and the .rkllm/.gguf files to models/"
echo "  bash scripts/check_board.sh"
echo "  bash scripts/run_board.sh --config config/config.yaml"
