#!/usr/bin/env bash
# Toi uu rotation (R1, R2) cho Qwen2.5 bang SpinQuant.
# Usage: bash scripts/10_optimize_rotation.sh <model> <w_bits> <a_bits> <kv_bits> [--w_rtn]
#   (tham so thu 5 neu co se bi bo qua, rotation khong dung RTN/GPTQ)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ $# -lt 4 ]; then
  echo "Usage: $0 <model> <w_bits> <a_bits> <kv_bits>" >&2
  exit 2
fi

INPUT_MODEL="$1"
W_BITS="$2"
A_BITS="$3"
KV_BITS="$4"

# ---- co the chinh bang bien moi truong ----
NPROC="${NPROC:-2}"                              # so GPU
MAX_LEN="${MAX_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
LR="${LR:-1.5}"
FSDP_OPTS="${FSDP_OPTS:-full_shard auto_wrap offload}"   # bo "offload" neu gap loi lan device
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-Qwen2DecoderLayer}"
USE_W_CLIP="${USE_W_CLIP:-0}"                    # 1 = bat --w_clip (rat ton VRAM luc train)

OUT_ROT="${OUT_ROT:-$ROOT_DIR/rotation}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$OUT_ROT" "$OUT_DIR" "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA=()
if [ "$USE_W_CLIP" = "1" ]; then
  EXTRA+=(--w_clip)
fi

torchrun --nnodes=1 --nproc_per_node="$NPROC" optimize_rotation.py \
  --input_model "$INPUT_MODEL" \
  --output_rotation_path "$OUT_ROT" \
  --output_dir "$OUT_DIR" \
  --logging_dir "$LOG_DIR" \
  --model_max_length "$MAX_LEN" \
  --fp16 False --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --logging_steps 1 \
  --learning_rate "$LR" \
  --weight_decay 0. \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --save_safetensors False \
  --max_steps "$MAX_STEPS" \
  --w_bits "$W_BITS" --a_bits "$A_BITS" --k_bits "$KV_BITS" --v_bits "$KV_BITS" \
  --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  "${EXTRA[@]}" \
  --fsdp "$FSDP_OPTS" \
  --fsdp_transformer_layer_cls_to_wrap "$FSDP_LAYER_CLS"

echo "Rotation saved to: $OUT_ROT/R.bin"
