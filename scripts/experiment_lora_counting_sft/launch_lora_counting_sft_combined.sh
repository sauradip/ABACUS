#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# COMBINED training run — Variant B WARM-STARTED on:
#   FSC-147 train + FSC-147 crop augmentations + UCF-QNRF train + JHU-Crowd train
# with bucket-balanced sampling.
#
# CRITICAL DIFFERENCE vs phase1_balanced:
#   This launcher uses --init_adapter_from to load the LoRA weights from the
#   Variant B run, but starts with a FRESH optimizer + LR schedule. That is
#   different from RESUME_FROM (which copies a checkpoint dir into OUT_DIR
#   and lets HF Trainer's resume_from_checkpoint pull optimizer + scheduler +
#   step counter — wrong for a different dataset).
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/launch_lora_counting_sft_combined.sh
#
# Env overrides:
#   EPOCHS NGPU BATCH GRAD_ACCUM LR N_PER_BUCKET TRAIN_JSON INIT_ADAPTER_FROM
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
source unicount/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="${TRAIN_JSON:-/data/amondal/UniCountData/combined_train/combined_counting_train.json}"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
INIT_ADAPTER_FROM="${INIT_ADAPTER_FROM:-/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_combined_${STAMP}"

# ── Hyperparameters ────────────────────────────────────────────────────────
NGPU="${NGPU:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-4e-5}"
BATCH="${BATCH:-1}"            # per_device_train_batch_size
GRAD_ACCUM="${GRAD_ACCUM:-2}"  # 8 GPU × 1 × 2 = 16 effective
N_PER_BUCKET="${N_PER_BUCKET:-1500}"
BUCKET_SEED="${BUCKET_SEED:-42}"

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))

mkdir -p "$OUT_DIR"

# ── Validate warm-start adapter exists ────────────────────────────────────
if [[ ! -f "$INIT_ADAPTER_FROM/adapter_model.safetensors" ]]; then
    echo "ERROR: $INIT_ADAPTER_FROM/adapter_model.safetensors not found"; exit 1
fi
if [[ ! -f "$TRAIN_JSON" ]]; then
    echo "ERROR: $TRAIN_JSON not found — run data/build_combined/build_combined_dataset.py first"; exit 1
fi

echo "============================================================"
echo " COMBINED — FSC-147 + crops + UCF-QNRF + JHU-Crowd  (warm-start Variant B)"
echo "  base model        : $BASE_MODEL"
echo "  init adapter from : $INIT_ADAPTER_FROM    (fresh optimizer + schedule)"
echo "  train data        : $TRAIN_JSON"
echo "  deepspeed         : $DS_CFG"
echo "  output            : $OUT_DIR"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch   : $NGPU × $BATCH × $GRAD_ACCUM = $EFF_BATCH"
echo "  bucket_balanced   : True   n_per_bucket=$N_PER_BUCKET   seed=$BUCKET_SEED"
echo "============================================================"

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_lora_counting_sft.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --init_adapter_from               "$INIT_ADAPTER_FROM" \
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
echo " COMBINED training complete → $OUT_DIR"
echo " Adapter : $OUT_DIR/adapter/"
echo ""
echo " NEXT — eval gate:"
echo "   bash crco/launch_eval.sh $OUT_DIR/adapter   # FSC-147 val+test CARC T=100,d=3,avg_50"
echo "   Decision gate:"
echo "     val MAE ≤ 17.5     → ADOPT (run cross-dataset SHT-B, CARPK, SHT-A)"
echo "     val MAE 17.5-18.5  → marginal — adopt iff test MAE also improved"
echo "     val MAE > 18.5     → investigate (check 501+ bucket)"
echo "     val MAE > 22       → REVERT"
echo "============================================================"
