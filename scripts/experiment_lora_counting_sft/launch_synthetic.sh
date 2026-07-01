#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Synthetic-counting LoRA SFT — warm-started from Variant B.
#
# Trains the Variant B adapter on 8,400 synthetic dot images with a uniform
# count distribution across [1, 800] to recalibrate the LM-head numerical
# output mapping (see "Counting Circuits", arXiv 2603.18523).
#
# Two runs (sequential, single 8-GPU pool):
#   • 1ep — minimal calibration
#   • 3ep — checkpoint per epoch for selection
#
# Both warm-started from Variant B with FRESH optimizer state.
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/launch_synthetic.sh           # both
#   MODE=1ep bash scripts/experiment_lora_counting_sft/launch_synthetic.sh
#   MODE=3ep bash scripts/experiment_lora_counting_sft/launch_synthetic.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
source unicount/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="data/synthetic_dots_train.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
WARM_INIT="/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter"

NGPU=8
BATCH=2          # per_device
GRAD_ACCUM=1
SAVE_LIMIT=5
LR=2e-5
EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
MODE="${MODE:-both}"

mkdir -p logs

# Steps per epoch ≈ 8400 / 16 = 525 (assuming 8400 samples).
# We compute it dynamically from the JSON so it survives generation drift.
N_SAMPLES=$(python -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))")
STEPS_PER_EPOCH=$(( (N_SAMPLES + EFF_BATCH - 1) / EFF_BATCH ))
echo "[launch_synthetic] N_SAMPLES=${N_SAMPLES}  steps/epoch=${STEPS_PER_EPOCH}"

run_one () {
    local TAG="$1"      # 1ep | 3ep
    local EPOCHS="$2"
    local SAVE_STEPS="$3"
    local STAMP=$(date +%Y%m%d_%H%M%S)
    local OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_synthetic_${TAG}_${STAMP}"
    local LOG="logs/synthetic_${TAG}_${STAMP}.log"

    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo " Synthetic LoRA SFT — ${TAG^^}"
    echo "  base model   : $BASE_MODEL"
    echo "  data         : $TRAIN_JSON  (${N_SAMPLES} samples)"
    echo "  warm-start   : $WARM_INIT"
    echo "  output       : $OUT_DIR"
    echo "  log          : $LOG"
    echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
    echo "  effective batch=$EFF_BATCH  save_steps=$SAVE_STEPS  save_total_limit=$SAVE_LIMIT"
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
            --model_max_length                2048 \
            --logging_steps                   10 \
            --save_steps                      "$SAVE_STEPS" \
            --save_strategy                   steps \
            --save_total_limit                "$SAVE_LIMIT" \
            --gradient_checkpointing          True \
            --remove_unused_columns           False \
            --dataloader_num_workers          2 \
            --report_to                       none \
            --init_adapter_from               "$WARM_INIT" \
        2>&1 | tee "$LOG"

    echo "[$TAG] training done -> $OUT_DIR"
    echo "$OUT_DIR" >> logs/synthetic_runs.txt
}

if [[ "$MODE" == "1ep" || "$MODE" == "both" ]]; then
    run_one 1ep 1 "$STEPS_PER_EPOCH"
fi
if [[ "$MODE" == "3ep" || "$MODE" == "both" ]]; then
    run_one 3ep 3 "$STEPS_PER_EPOCH"
fi

echo "============================================================"
echo " ALL synthetic training runs complete."
echo " See logs/synthetic_runs.txt for output dirs."
echo "============================================================"
