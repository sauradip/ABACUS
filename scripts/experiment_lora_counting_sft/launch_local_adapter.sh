#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DUAL-LoRA — train the LOCAL adapter on FSC-147 quadrant/random crops only.
# Cold-start (no init_adapter_from). Used at CARC depth >= 1 to count crops.
# Paired at inference time with the unmodified Variant B adapter (depth 0).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
source unicount/bin/activate

BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="${TRAIN_JSON:-/data/amondal/UniCountData/combined_train/fsc147_crop_augmented.json}"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-/data/amondal/unicount_runs/lora_local_adapter_${STAMP}}"

NGPU="${NGPU:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
N_PER_BUCKET="${N_PER_BUCKET:-1000}"
BUCKET_SEED="${BUCKET_SEED:-42}"

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
mkdir -p "$OUT_DIR"

if [[ ! -f "$TRAIN_JSON" ]]; then
  echo "ERROR: $TRAIN_JSON not found"; exit 1
fi

echo "============================================================"
echo " LOCAL ADAPTER — cold-start LoRA on FSC-147 crops only"
echo "  base model       : $BASE_MODEL"
echo "  train data       : $TRAIN_JSON"
echo "  output           : $OUT_DIR"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch  : $NGPU × $BATCH × $GRAD_ACCUM = $EFF_BATCH"
echo "  bucket_balanced  : True   n_per_bucket=$N_PER_BUCKET   seed=$BUCKET_SEED"
echo "============================================================"

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_lora_counting_sft.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --data_path                       "$TRAIN_JSON" \
        --bucket_balanced                 True \
        --n_per_bucket                    "$N_PER_BUCKET" \
        --bucket_seed                     "$BUCKET_SEED" \
        --output_dir                      "$OUT_DIR" \
        --deepspeed                       "$DS_CFG" \
        --num_train_epochs                "$EPOCHS" \
        --per_device_train_batch_size     "$BATCH" \
        --gradient_accumulation_steps     "$GRAD_ACCUM" \
        --learning_rate                   "$LR" \
        --warmup_ratio                    0.03 \
        --lr_scheduler_type               cosine \
        --weight_decay                    0.05 \
        --bf16                            True \
        --model_max_length                512 \
        --logging_steps                   10 \
        --save_steps                      500 \
        --save_strategy                   steps \
        --save_total_limit                3 \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          2 \
        --report_to                       none

echo "[LOCAL ADAPTER] training complete → $OUT_DIR"
echo " adapter: $OUT_DIR/adapter"
