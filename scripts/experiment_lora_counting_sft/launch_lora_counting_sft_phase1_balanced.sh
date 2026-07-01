#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE 1 — Variant B continuation w/ bucket-balanced sampling on the
#           consolidated UCount dataset (decontaminated).
#
# Goal of Phase 1: extend Variant B's exposure to broader category breadth
# (1,043 cats from parts 1+2+3) without letting the GT=1-3 spike in the new
# data bias the model toward small numbers. We achieve this by sampling
# uniformly across COUNT BUCKETS instead of uniformly across IMAGES.
#
# Per-epoch sample budget (with default n_per_bucket=2000):
#     bucket           avail   used
#     (0,5)            24939   2000
#     (6,20)            9634   2000
#     (21,50)           4484   2000
#     (51,100)          2578   2000   (only 2578 available → use all 2578)
#     (101,200)         7068   2000
#     (201,inf)         1000   1000
#     -----------------------------
#     total/epoch                   ~10,062 samples
#
#   eff_batch = 8 × 1 × 2 = 16  → ~628 steps/epoch  →  3 epochs ≈ 1,884 steps
#   save_steps=500 → ~3 checkpoints inside the run.
#
# IMPORTANT — paper-side decision gate after Phase 1 (per plan):
#   val MAE ≤ 18.0     →  KEEP   (continue to Phase 2 when dense data lands)
#   val MAE 18.0–19.5  →  NEUTRAL (continue but lowered expectations)
#   val MAE > 19.5     →  REVERT (drop Phase 1, keep Variant B baseline)
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/launch_lora_counting_sft_phase1_balanced.sh
#
# Env overrides:
#   EPOCHS NGPU BATCH GRAD_ACCUM LR N_PER_BUCKET TRAIN_JSON RESUME_FROM
#
#   RESUME_FROM=/path/to/variantB/checkpoint-XXXX  (optional; copies adapter
#   weights into OUT_DIR before starting so HF Trainer resumes from there).
#   If unset, training starts from scratch — the LoRA adapter is re-initialised.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

source unicount/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="${TRAIN_JSON:-/data/amondal/UniCountData/ucount_consolidated/train_counting_clean.json}"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_phase1_balanced_${STAMP}"

# ── Hyperparameters — Phase 1 (mirrors Variant B for fair comparison) ─────
NGPU="${NGPU:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-4e-5}"
BATCH="${BATCH:-1}"           # per_device
GRAD_ACCUM="${GRAD_ACCUM:-2}" # 8 GPU × 1 × 2 = 16 effective
N_PER_BUCKET="${N_PER_BUCKET:-2000}"
BUCKET_SEED="${BUCKET_SEED:-42}"

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))

mkdir -p "$OUT_DIR"

# ── Optional: warm-start from existing Variant B checkpoint ────────────────
if [[ -n "${RESUME_FROM:-}" ]]; then
    if [[ ! -d "$RESUME_FROM" ]]; then
        echo "ERROR: RESUME_FROM=$RESUME_FROM does not exist"; exit 1
    fi
    DEST="$OUT_DIR/$(basename "$RESUME_FROM")"
    echo "[Phase 1] Warm-starting from: $RESUME_FROM"
    echo "[Phase 1]   → copying to:    $DEST"
    cp -r "$RESUME_FROM" "$DEST"
    echo "[Phase 1] HF Trainer will detect this checkpoint and resume from it."
fi

echo "============================================================"
echo " PHASE 1 — Bucket-balanced LoRA Counting SFT (Variant B params)"
echo "  base model       : $BASE_MODEL"
echo "  mllm hf          : $MLLM_HF"
echo "  train data       : $TRAIN_JSON"
echo "  deepspeed        : $DS_CFG"
echo "  output           : $OUT_DIR"
echo "  resume_from      : ${RESUME_FROM:-<scratch>}"
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
        --save_total_limit                6 \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          2 \
        --report_to                       none

echo "============================================================"
echo " PHASE 1 training complete → $OUT_DIR"
echo " Adapter : $OUT_DIR/adapter/"
echo " Merged  : $OUT_DIR/merged/   (ZeRO-2 in-place merge)"
echo ""
echo " NEXT — eval gate:"
echo "   bash crco/launch_eval.sh $OUT_DIR/adapter   # FSC-147 val CARC T=100,d=3,avg_50"
echo "   then check val MAE vs 18.0 / 19.5 thresholds in the Phase-1 plan."
echo "============================================================"
