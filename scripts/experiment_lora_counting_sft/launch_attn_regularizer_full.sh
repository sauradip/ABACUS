#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full Training: Dual-Loss (CE + ObjectFocusedAttentionLoss) with Best-Checkpoint
#
# Phase 8: Complete training on all 15.8K attn_regularizer data with ShanghaiTech.
# Follows v3s specs with attention regularization + unfrozen connector.
# Uses full dataset (FSC-147: 6.1K, UCount: 9.0K, ShanghaiTech: 700)
#
# Expected improvements over checkpoint-2670:
# - SHA-A: 432→<100 MAE (fixes 0% recursion from missing ShanghaiTech)
# - SHA-B: 123→<50 MAE (density learning from SHA samples)
# - FSC: Incremental improvements with full AR loss coverage
#
# Specs: unfrozen connector, LoRA r=64/α=128, 3 epochs, full 15.8K dataset
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="data/attn_regularizer_dataset/attn_regularizer_train_all.json"
VAL_JSON="data/attn_regularizer_dataset/attn_regularizer_val_all.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"

# Verify paths
[ -d "$BASE_MODEL" ] || { echo "missing $BASE_MODEL"; exit 1; }
[ -f "$TRAIN_JSON" ] || { echo "missing $TRAIN_JSON"; exit 1; }
[ -f "$VAL_JSON" ] || { echo "missing $VAL_JSON"; exit 1; }
[ -f "$DS_CFG" ] || { echo "missing $DS_CFG"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="attn_regularizer_full_best"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs
echo "Full training started: $STAMP" > "$OUT_DIR/_METADATA.txt"

# ── Hyperparameters (v3s spec) ─────────────────────────────────────────────
NGPU=8
EPOCHS=3                    # Full run: 3 epochs
LR=1e-5
BATCH=2
GRAD_ACCUM=1
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
WARMUP_RATIO=0.06

# Train: 13339 records / 16 eff_batch ≈ 834 steps/epoch
# Eval every 400 steps (roughly 6x per epoch) for frequent validation
EVAL_STEPS=400
SAVE_TOTAL_LIMIT=3         # Keep only best 3 checkpoints

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
LOG="logs/${RUN_TAG}_${STAMP}.log"

# ── Dual-Loss parameters ───────────────────────────────────────────────────
LAMBDA_AR=0.1              # Weight for AR loss: total = ce_loss + λ_ar * ar_loss
AR_SIGMA=1.0               # Gaussian spread in patch units
AR_TEMPERATURE=0.1         # Temperature for sharpening (lower = sharper peaks)
AR_USE_SHARPENING=True     # Enable temperature-based sharpening

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export DEEPSPEED_CPU_OFFLOAD=1
export DEEPSPEED_SKIP_CUDA_CHECK=1

echo "============================================================"
echo " FULL TRAINING: Dual-Loss (CE + AR Loss) with ShanghaiTech"
echo "  base model    : $BASE_MODEL"
echo "  train data    : $TRAIN_JSON (13339 records: FSC 3.6K + UCount 9.0K + SHA 0.7K)"
echo "  val data      : $VAL_JSON (1286 FSC-147 val records)"
echo "  output        : $OUT_DIR"
echo "  log           : $LOG"
echo "  epochs=$EPOCHS lr=$LR per_device=$BATCH ngpu=$NGPU eff_batch=$EFF_BATCH"
echo "  lora r=$LORA_RANK α=$LORA_ALPHA dropout=$LORA_DROPOUT warmup=$WARMUP_RATIO"
echo "  eval_steps=$EVAL_STEPS (save best checkpoint by eval_loss)"
echo "  dataloader_workers=8 (max GPU utilization)"
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
        --save_steps                      "$SAVE_TOTAL_LIMIT" \
        --save_strategy                   steps \
        --save_total_limit                "$SAVE_TOTAL_LIMIT" \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          8 \
        --report_to                       none \
        --lambda_ar                       "$LAMBDA_AR" \
        --ar_sigma                        "$AR_SIGMA" \
        --ar_temperature                  "$AR_TEMPERATURE" \
        --ar_use_sharpening               "$AR_USE_SHARPENING" \
    2>&1 | tee "$LOG"

echo "============================================================"
echo " FULL TRAINING COMPLETE → $OUT_DIR"
echo " Best checkpoint saved (by eval_loss)"
echo "============================================================"
