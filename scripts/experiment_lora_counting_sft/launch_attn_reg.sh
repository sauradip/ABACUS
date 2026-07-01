#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Attention-focus-regularizer LoRA SFT — warm-started from Variant B.
# Implements the KL(attn || Gaussian-point-prior) regularizer from
# arXiv 2603.18523 (Counting Circuits), §8.
#
# Notes:
#   - Forces attn_implementation=eager (sdpa/flash do not expose attn probs).
#   - per_device_batch=1 + grad_accum=2 to absorb the (28, H, T, T) attention
#     tensor materialisation cost.
#   - target_layers chosen to match the paper's Qwen2.5-VL-7B layers (same
#     LLM family + same depth as UniLIP-3B's Qwen2-1.5B: 28 layers).
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."
source unicount/bin/activate

BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/train/train_counting.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
WARM_INIT="/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter"

NGPU=8
BATCH=1
GRAD_ACCUM=2
EPOCHS=3
LR=2e-5
EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
LAMBDA_FOCUS=1.0
TARGET_LAYERS="2,18,19,20,21,22"

mkdir -p logs

N_SAMPLES=$(python -c "import json; print(len(json.load(open('${TRAIN_JSON}'))))")
STEPS_PER_EPOCH=$(( (N_SAMPLES + EFF_BATCH - 1) / EFF_BATCH ))

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/data/amondal/unicount_runs/lora_counting_sft_attn_reg_${STAMP}"
LOG="logs/attn_reg_${STAMP}.log"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Attention-focus regularizer LoRA SFT"
echo "  base model     : $BASE_MODEL"
echo "  data           : $TRAIN_JSON  (${N_SAMPLES} samples)"
echo "  warm-start     : $WARM_INIT"
echo "  output         : $OUT_DIR"
echo "  log            : $LOG"
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU"
echo "  effective batch=$EFF_BATCH   steps/epoch=${STEPS_PER_EPOCH}"
echo "  lambda_focus=$LAMBDA_FOCUS   target_layers=$TARGET_LAYERS"
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
        --save_steps                      "$STEPS_PER_EPOCH" \
        --save_strategy                   steps \
        --save_total_limit                3 \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          2 \
        --report_to                       none \
        --init_adapter_from               "$WARM_INIT" \
        --attention_regularizer           True \
        --lambda_focus                    "$LAMBDA_FOCUS" \
        --target_layers                   "$TARGET_LAYERS" \
        --attn_grid                       16 \
        --attn_sigma                      1.0 \
    2>&1 | tee "$LOG"

echo "$OUT_DIR" > logs/attn_reg_last_run.txt
echo "[done] training complete -> $OUT_DIR"
