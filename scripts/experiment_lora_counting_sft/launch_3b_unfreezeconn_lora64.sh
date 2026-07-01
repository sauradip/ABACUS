#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Variant B reproduction — UniLIP-3B base, LoRA r=64/alpha=128, connector
# UNFROZEN, full 8×A100-80GB utilisation.
#
# Spec: ADAPTIVE_TILING_FULL_SPEC.md §A (Variant B), with user deltas:
#   • Base = UniLIP-3B (spec already calls for 3B)
#   • LoRA r=32/α=64  →  r=64/α=128                          (user request)
#   • multi_modal_projector ("connector") UNFROZEN            (user request)
#   • All 8 GPUs fully utilised (per_device=2, grad_accum=1)
#       → eff batch = 8 × 2 × 1 = 16  (matches spec §A.7)
#
# Output: /data/amondal/unicount_runs/lora_counting_sft_3b_unfreezeconn_lora64a128_${STAMP}/
# (deliberately distinct from existing lora_counting_sft_variantB_* runs)
#
# Wallclock estimate (3B, r=64, conn unfrozen, 8×A100): ~120-180 min for 10
# epochs over 3,659 train samples at eff_batch=16.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root: /data/amondal/UniCount

# Use the conda base env (has torch / transformers / peft / accelerate /
# deepspeed installed; the local `unicount/` venv is missing deepspeed).
export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/train/train_counting.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="lora_counting_sft_3b_unfreezeconn_lora64a128"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs

# ── Hyperparameters (spec §A.7 + user overrides) ──────────────────────────
NGPU=8
EPOCHS=10
# Empirically: 4e-5 with r=64 LoRA + unfrozen connector caused a NaN/loss
# collapse around epoch ~2.8 on the first attempt (loss → 0.0, grad_norm
# → √2 forever, then NCCL ALLREDUCE timeout).  Halve the LR to 2e-5 for
# the larger trainable surface (LoRA r=64 + lm_head + connector ~85M).
LR=2e-5
BATCH=2            # per_device — 2 fits comfortably on A100-80GB at 3B
GRAD_ACCUM=1       # → eff_batch = 8 * 2 * 1 = 16
SAVE_STEPS=229     # ≈ 1 epoch (3659/16 = 228.7)
SAVE_LIMIT=10      # keep all 10 per-epoch checkpoints
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
LOG="logs/${RUN_TAG}_${STAMP}.log"

# Visibility / cache hygiene
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo " 3B Variant-B LoRA SFT (r=64/α=128, connector unfrozen)"
echo "  base model    : $BASE_MODEL"
echo "  mllm hf       : $MLLM_HF"
echo "  train data    : $TRAIN_JSON"
echo "  deepspeed     : $DS_CFG"
echo "  output        : $OUT_DIR"
echo "  log           : $LOG"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch = $EFF_BATCH"
echo "  lora r=$LORA_RANK α=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  save_steps=$SAVE_STEPS  save_total_limit=$SAVE_LIMIT (per-epoch ckpts)"
echo "  GPUs visible  : $CUDA_VISIBLE_DEVICES"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_lora_counting_sft_3b_unfreezeconn.py \
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
        --warmup_ratio                    0.05 \
        --lr_scheduler_type               cosine \
        --weight_decay                    0.0 \
        --max_grad_norm                   1.0 \
        --bf16                            True \
        --model_max_length                512 \
        --logging_steps                   10 \
        --save_steps                      "$SAVE_STEPS" \
        --save_strategy                   steps \
        --save_total_limit                "$SAVE_LIMIT" \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          4 \
        --report_to                       none \
    2>&1 | tee "$LOG"

echo "============================================================"
echo " Training complete → $OUT_DIR"
echo " Final adapter   : $OUT_DIR/adapter/"
echo " Connector wts   : $OUT_DIR/adapter/multi_modal_projector.bin"
echo " Merged model    : $OUT_DIR/merged/"
echo " Per-epoch ckpts : $OUT_DIR/checkpoint-{229,458,...,2290}/"
echo "============================================================"
