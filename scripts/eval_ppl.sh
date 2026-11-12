#!/usr/bin/env bash
# WikiText-2 perplexity with llama.cpp's llama-perplexity.
# Quality gate: PPL delta <= +5% (W4A8) or +10% (W4A4) vs FP16 baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=""
CORPUS="$ROOT/data/corpus/wikitext-2-raw-val.txt"
CTX=2048
CHUNKS=64
THREADS=4
OUT="$ROOT/data/results/ppl.json"
BIN="${LLAMA_PERPLEXITY:-llama-perplexity}"
GPU_LAYERS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: eval_ppl.sh --model <model.gguf> [options]

Options:
  --model <gguf>       GGUF model to evaluate
  --corpus <txt>       WikiText-2 raw validation text (default: data/corpus/wikitext-2-raw-val.txt)
  --ctx <n>            Context size (default: 2048)
  --chunks <n>         Number of chunks (default: 64)
  --threads <n>        CPU threads (default: 4)
  --gpu-layers <n>     Offload layers to GPU (default: 0)
  --out <json>         Summary JSON output
  --bin <path>         llama-perplexity binary (default: $LLAMA_PERPLEXITY or PATH)
  --dry-run            Print the command only
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "ERROR: --model must point to an existing .gguf file" >&2
  exit 2
fi
if [ ! -f "$CORPUS" ]; then
  echo "ERROR: corpus not found: $CORPUS" >&2
  echo "  Download wikitext-2-raw-v1 and save validation text as data/corpus/wikitext-2-raw-val.txt" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
LOG="${OUT%.json}.log"
CMD=("$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -t "$THREADS" -ngl "$GPU_LAYERS")

echo "==> ${CMD[*]}"
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi
if ! command -v "$BIN" >/dev/null 2>&1 && [ ! -x "$BIN" ]; then
  echo "ERROR: llama-perplexity not found: $BIN" >&2
  echo "  Build llama.cpp or set LLAMA_PERPLEXITY=/path/to/llama-perplexity" >&2
  exit 2
fi

set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: llama-perplexity exited with $STATUS" >&2
  exit "$STATUS"
fi

PPL="$(grep -oE 'Final estimate: PPL = [0-9.]+' "$LOG" | tail -n1 | grep -oE '[0-9.]+' || true)"
if [ -z "$PPL" ]; then
  PPL="$(grep -oE 'PPL = [0-9.]+' "$LOG" | tail -n1 | grep -oE '[0-9.]+' || true)"
fi
python3 - "$OUT" "$MODEL" "$CORPUS" "$CTX" "$CHUNKS" "$PPL" "$LOG" <<'PY'
import json, sys, time
out, model, corpus, ctx, chunks, ppl, log = sys.argv[1:8]
payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": model,
    "corpus": corpus,
    "ctx": int(ctx),
    "chunks": int(chunks),
    "ppl": float(ppl) if ppl else None,
    "log": log,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print(f"PPL = {ppl or 'unknown'} -> {out}")
PY
