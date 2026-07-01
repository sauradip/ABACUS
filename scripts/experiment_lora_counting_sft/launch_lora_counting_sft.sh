#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch LoRA SFT — Variant B, 8 × GPU, DeepSpeed ZeRO-2.
#
# Hyperparameters exactly as per ADAPTIVE_TILING_FULL_SPEC.md §A.7.
#
#   effective batch = 8 GPU × per_device(1) × grad_accum(2) = 16  ← spec §A.7
#
# Notes:
#  • ZeRO-2 (not ZeRO-3): UniLIP.__init__ calls SanaTransformer2DModel.from_pretrained()
#    internally, which creates real tensors while ZeRO-3's meta-tensor deferred init is
#    active → NotImplementedError. ZeRO-2 shards gradients + optimizer states only,
#    keeps full params → no conflict.
#  • ZeRO-2 still enables merge-during-training (params are full, not sharded).
#  • bf16 used. Spec uses fp16 for 1-GPU reproduction; bf16 matches rest of 3B pipeline.
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/launch_lora_counting_sft.sh
#
# Env overrides:
#   EPOCHS NGPU BATCH GRAD_ACCUM LR
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

source unicount/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/train/train_counting.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_${STAMP}"

# ── Hyperparameters — spec §A.7 (scaled to 8 GPUs) ────────────────────────
NGPU="${NGPU:-8}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-4e-5}"
BATCH="${BATCH:-1}"           # per_device (spec = 1)
GRAD_ACCUM="${GRAD_ACCUM:-2}" # 8 GPU × 1 × 2 = 16 effective (spec = 16)

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))

echo "============================================================"
echo " Variant B LoRA Counting SFT — ZeRO-2, 8 × GPU"
echo "  base model  : $BASE_MODEL"
echo "  mllm hf     : $MLLM_HF"
echo "  train data  : $TRAIN_JSON"
echo "  deepspeed   : $DS_CFG"
echo "  output      : $OUT_DIR"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch = $NGPU × $BATCH × $GRAD_ACCUM = $EFF_BATCH  (spec target: 16)"
echo "============================================================"

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_lora_counting_sft.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --data_path                       "$TRAIN_JSON" \
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

echo "============================================================"
echo " Training complete → $OUT_DIR"
echo " Adapter : $OUT_DIR/adapter/"
echo " Merged  : $OUT_DIR/merged/   (merged in-place during training)"
echo "============================================================"
