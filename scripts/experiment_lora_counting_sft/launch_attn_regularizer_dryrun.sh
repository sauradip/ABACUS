#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Dry-Run: Dual-Loss Training (CE + ObjectFocusedAttentionLoss)
#
# Tests data pipeline and dual-loss computation without full training.
# Uses first 100 records from attn_regularizer_dataset.
#
# Follows v3s specs (unfrozen connector, LoRA r=64/α=128) but:
#  - 1 epoch only (quick test)
#  - 100 records (fast iteration)
#  - Frequent checkpoints (debugging)
#  - Fresh start (no warm-start)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="data/attn_regularizer_dataset/attn_regularizer_dryrun_train.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"

# Verify paths
[ -d "$BASE_MODEL" ] || { echo "missing $BASE_MODEL"; exit 1; }
[ -f "$TRAIN_JSON" ] || { echo "missing $TRAIN_JSON"; exit 1; }
[ -f "$DS_CFG" ] || { echo "missing $DS_CFG"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="attn_regularizer_dryrun"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs
echo "PRE-FLIGHT timestamp: $STAMP" > "$OUT_DIR/_DRYRUN_METADATA.txt"

# ── Hyperparameters (v3s spec) ─────────────────────────────────────────────
NGPU=8
EPOCHS=1                    # Dry run: 1 epoch only
LR=1e-5
BATCH=2
GRAD_ACCUM=1
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
WARMUP_RATIO=0.06

# 100 records / 16 eff_batch = 6-7 steps/epoch
# Save every 3 steps for frequent checkpoints (debugging)
SAVE_STEPS=3
SAVE_LIMIT=8

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
LOG="logs/${RUN_TAG}_${STAMP}.log"

# ── Dual-Loss parameters ───────────────────────────────────────────────────
LAMBDA_AR=0.1              # Weight for AR loss: total = ce_loss + λ_ar * ar_loss
AR_SIGMA=1.0               # Gaussian spread in patch units
AR_TEMPERATURE=0.1         # Temperature for sharpening (lower = sharper peaks)
AR_USE_SHARPENING=True     # Enable temperature-based sharpening

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo " DRY-RUN: Dual-Loss Training (CE + AR Loss)"
echo "  base model    : $BASE_MODEL"
echo "  train data    : $TRAIN_JSON (100 records)"
echo "  output        : $OUT_DIR"
echo "  log           : $LOG"
echo "  epochs=$EPOCHS lr=$LR per_device=$BATCH ngpu=$NGPU eff_batch=$EFF_BATCH"
echo "  lora r=$LORA_RANK α=$LORA_ALPHA dropout=$LORA_DROPOUT warmup=$WARMUP_RATIO"
echo "  save_steps=$SAVE_STEPS (frequent checkpoints for testing)"
echo ""
echo " Dual-Loss Config:"
echo "  lambda_ar     = $LAMBDA_AR"
echo "  ar_sigma      = $AR_SIGMA"
echo "  ar_temperature= $AR_TEMPERATURE"
echo "  ar_sharpening = $AR_USE_SHARPENING"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --data_path                       "$TRAIN_JSON" \
        --output_dir                      "$OUT_DIR" \
        --deepspeed                       "$DS_CFG" \
        --lora_rank                       "$LORA_RANK" \
        --lora_alpha                      "$LORA_ALPHA" \
        --lora_dropout                    "$LORA_DROPOUT" \
        --num_train_epochs                "$EPOCHS" \
        --per_device_train_batch_size     "$BATCH" \
        --gradient_accumulation_steps     "$GRAD_ACCUM" \
        --learning_rate                   "$LR" \
        --warmup_ratio                    "$WARMUP_RATIO" \
        --lr_scheduler_type               cosine \
        --weight_decay                    0.0 \
        --max_grad_norm                   1.0 \
        --bf16                            True \
        --model_max_length                512 \
        --logging_steps                   1 \
        --save_steps                      "$SAVE_STEPS" \
        --save_strategy                   steps \
        --save_total_limit                "$SAVE_LIMIT" \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          4 \
        --report_to                       none \
        --lambda_ar                       "$LAMBDA_AR" \
        --ar_sigma                        "$AR_SIGMA" \
        --ar_temperature                  "$AR_TEMPERATURE" \
        --ar_use_sharpening               "$AR_USE_SHARPENING" \
    2>&1 | tee "$LOG"

echo "============================================================"
echo " DRY-RUN COMPLETE → $OUT_DIR"
echo ""
echo " Success criteria:"
echo "  ✓ Data loads without errors"
echo "  ✓ CE loss + AR loss both compute"
echo "  ✓ Combined loss = ce_loss + λ*ar_loss"
echo "  ✓ Backward pass completes without NaN"
echo "  ✓ Checkpoint saves with adapter/"
echo ""
echo " Check log for dual-loss outputs:"
echo "  grep 'dual-loss' $LOG"
echo "============================================================"
