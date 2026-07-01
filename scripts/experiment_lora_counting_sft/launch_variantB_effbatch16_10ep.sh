#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Variant B LoRA SFT — effective batch 16, 10 epochs, per-epoch checkpoints.
#
# Differences vs launch_lora_counting_sft.sh:
#   • per_device=2, grad_accum=1   → eff_batch = 8 * 2 * 1 = 16
#       (original Variant B used per_device=1, grad_accum=2 = 16 — same eff,
#        this halves per-step iter count and gives 1 grad-step per micro-batch)
#   • LR 1e-4 (vs 4e-5) — user request
#   • num_train_epochs=10 (vs 3)
#   • save_steps=229 (≈ 1 epoch with eff_batch=16 on 3,659 train samples)
#   • save_total_limit=10 (keep every epoch's checkpoint)
#
# Repo conventions enforced:
#   • bf16 (NOT fp16 — UniLIP is bf16 throughout)
#   • DeepSpeed ZeRO-2 (NOT a non-existent configs/zero2.yaml)
#   • --model_name_or_path (NOT --base_model)
#   • --mllm_hf_path required
#
# Per-checkpoint adapters do NOT auto-emit (the trainer saves checkpoint-XXX
# as full state_dicts). Use the companion extract+eval script after this run:
#   bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \
#        $OUT_DIR
#
# Wallclock estimate: ~100 min on 8×H100. Safe to nohup overnight.
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
OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_variantB_effbatch16_10ep_${STAMP}"

mkdir -p "$OUT_DIR"
mkdir -p logs

NGPU=8
EPOCHS=10
LR=1e-4
BATCH=2          # per_device
GRAD_ACCUM=1
SAVE_STEPS=229   # ≈ 1 epoch (3659/16=228.7)
SAVE_LIMIT=10    # keep all 10 epoch checkpoints
EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))

LOG="logs/variantB_effbatch16_10ep_${STAMP}.log"

echo "============================================================"
echo " Variant B Effective-Batch-16, 10-epoch LoRA SFT"
echo "  base model    : $BASE_MODEL"
echo "  mllm hf       : $MLLM_HF"
echo "  train data    : $TRAIN_JSON"
echo "  deepspeed     : $DS_CFG"
echo "  output        : $OUT_DIR"
echo "  log           : $LOG"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch = $EFF_BATCH"
echo "  save_steps=$SAVE_STEPS  save_total_limit=$SAVE_LIMIT (per-epoch ckpts)"
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
        --save_steps                      "$SAVE_STEPS" \
        --save_strategy                   steps \
        --save_total_limit                "$SAVE_LIMIT" \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          2 \
        --report_to                       none \
    2>&1 | tee "$LOG"

echo "============================================================"
echo " Training complete → $OUT_DIR"
echo " Final adapter   : $OUT_DIR/adapter/"
echo " Per-epoch ckpts : $OUT_DIR/checkpoint-{229,458,...,2290}/"
echo ""
echo " NEXT: extract per-epoch adapters and eval each on FSC val:"
echo "   bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \\"
echo "        $OUT_DIR"
echo "============================================================"
