#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# D³T (Divide-and-Discern Dialogue Tuning) LoRA SFT — cold + warm runs.
#
# Pipeline = SFT-only single-turn (3,659) + D3T multi-turn binary-search (464)
#          = 4,123 records. Image-set unchanged (FSC-147 train).
#
# Eff batch = 8 GPU × 2 per-device × 1 grad-accum = 16
# Steps/epoch = 4123/16 ≈ 258.  10 epochs = 2,580 steps.
#
# Multi-turn loss masking is already correct in preprocess_internvl (verified
# offline: every gpt turn — Yes/No + final integer — gets gradient).
#
# Two runs launched SEQUENTIALLY (single GPU pool):
#   • cold:  base UniLIP-3B, LR 1e-4 (matches the 28th experiment baseline)
#   • warm:  init from Variant B adapter, LR 4e-5 (avoid forgetting calibration)
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/launch_d3t.sh           # both
#   MODE=cold bash scripts/experiment_lora_counting_sft/launch_d3t.sh # cold only
#   MODE=warm bash scripts/experiment_lora_counting_sft/launch_d3t.sh # warm only
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
source unicount/bin/activate

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="data/fsc147_sft_plus_d3t.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
WARM_INIT="/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter"

NGPU=8
EPOCHS=10
BATCH=2          # per_device
GRAD_ACCUM=1
SAVE_STEPS=258   # ≈ 1 epoch
SAVE_LIMIT=10
EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
MODE="${MODE:-both}"

mkdir -p logs

run_one () {
    local TAG="$1"     # cold | warm
    local LR="$2"
    local EXTRA="$3"   # additional flags (e.g. --init_adapter_from ...)
    local STAMP=$(date +%Y%m%d_%H%M%S)
    local OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_d3t_${TAG}_${STAMP}"
    local LOG="logs/d3t_${TAG}_${STAMP}.log"

    mkdir -p "$OUT_DIR"
    echo "============================================================"
    echo " D3T LoRA SFT — ${TAG^^}"
    echo "  base model   : $BASE_MODEL"
    echo "  data         : $TRAIN_JSON  (4,123 = 3,659 SFT + 464 D3T)"
    echo "  deepspeed    : $DS_CFG"
    echo "  output       : $OUT_DIR"
    echo "  log          : $LOG"
    echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
    echo "  effective batch = $EFF_BATCH   save_steps=$SAVE_STEPS   save_total_limit=$SAVE_LIMIT"
    [[ -n "$EXTRA" ]] && echo "  extra        : $EXTRA"
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
            $EXTRA \
        2>&1 | tee "$LOG"

    echo "[$TAG] training done → $OUT_DIR"
}

if [[ "$MODE" == "cold" || "$MODE" == "both" ]]; then
    run_one cold 1e-4 ""
fi
if [[ "$MODE" == "warm" || "$MODE" == "both" ]]; then
    run_one warm 4e-5 "--init_adapter_from $WARM_INIT"
fi

echo "============================================================"
echo " ALL D3T training runs complete."
echo " NEXT: per-epoch eval — for each run dir, run:"
echo "   bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \\"
echo "        <RUN_DIR>"
echo "============================================================"
