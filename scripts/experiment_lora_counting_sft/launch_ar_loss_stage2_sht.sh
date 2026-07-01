#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 2 fine-tuning — SHT-A/B upsampled, conservative AR
#
# Init:   ar_loss_v3s_lambda0001 checkpoint-14500/adapter_extracted
# Data:   balanced_mix_v3s + SHT-A 4× + SHT-B 2× (56,347 records)
# Val:    mixed FSC-147 val + SHT-A/B test[:100] (1,486 records)
# LR:     2e-6  (5× lower than stage 1 — prevents forgetting)
# λ_ar:   0.001 (same as stage 1 — upsampling does the heavy lifting)
# Epochs: 2
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/balanced_mix_v3s/stage2_sht_upsampled_train.json"
VAL_JSON="outputs/experiment_lora_counting_sft/balanced_mix_v3s/stage2_mixed_val.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"

STAGE1_RUN="/data/amondal/unicount_runs/ar_loss_v3s_lambda0001_20260508_154727"
INIT_ADAPTER="${STAGE1_RUN}/checkpoint-14500/adapter_extracted"
INIT_CONN="${INIT_ADAPTER}/multi_modal_projector.bin"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="ar_loss_stage2_sht"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs

# ── Hyperparameters ────────────────────────────────────────────────────────
NGPU=8
EPOCHS=2
LR=2e-6
BATCH=2
GRAD_ACCUM=1
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
WARMUP_RATIO=0.03
EVAL_STEPS=300
SAVE_STEPS=300
SAVE_LIMIT=10

# ── AR loss config ─────────────────────────────────────────────────────────
LAMBDA_AR=0.001
AR_SIGMA=1.0
AR_TEMPERATURE=1.0
AR_USE_SHARPENING=False

LOG="logs/${RUN_TAG}_${STAMP}.log"

# ── Environment ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export INIT_CONNECTOR_FROM="$INIT_CONN"
export WANDB_PROJECT="ar_loss_v3s"

# ── Pre-flight checks ──────────────────────────────────────────────────────
[ -f "$TRAIN_JSON" ] || { echo "ABORT: missing TRAIN_JSON $TRAIN_JSON"; exit 1; }
[ -f "$VAL_JSON" ]   || { echo "ABORT: missing VAL_JSON $VAL_JSON"; exit 1; }
[ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "ABORT: missing stage1 adapter"; exit 1; }
[ -f "$INIT_CONN" ]  || { echo "ABORT: missing connector $INIT_CONN"; exit 1; }

echo "============================================================"
echo " Stage 2 — SHT upsampled  λ=${LAMBDA_AR}  LR=${LR}"
echo "  init adapter  : $INIT_ADAPTER"
echo "  train data    : $TRAIN_JSON  ($(python3 -c "import json; print(len(json.load(open('$TRAIN_JSON'))))" 2>/dev/null) records)"
echo "  val data      : $VAL_JSON    ($(python3 -c "import json; print(len(json.load(open('$VAL_JSON'))))" 2>/dev/null) records)"
echo "  output        : $OUT_DIR"
echo ""
echo "  epochs=$EPOCHS  lr=$LR  eff_batch=$(( NGPU * BATCH * GRAD_ACCUM ))"
echo "  λ_ar=$LAMBDA_AR  eval_steps=$EVAL_STEPS  save_steps=$SAVE_STEPS"
echo "  best-checkpoint: load_best_model_at_end=True  metric=eval_loss (mixed FSC+SHT)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --data_path                       "$TRAIN_JSON" \
        --validation_data_path            "$VAL_JSON" \
        --output_dir                      "$OUT_DIR" \
        --deepspeed                       "$DS_CFG" \
        --init_adapter_from               "$INIT_ADAPTER" \
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
        --logging_steps                   10 \
        --save_strategy                   steps \
        --save_steps                      "$SAVE_STEPS" \
        --save_total_limit                "$SAVE_LIMIT" \
        --eval_strategy                   steps \
        --eval_steps                      "$EVAL_STEPS" \
        --load_best_model_at_end          True \
        --metric_for_best_model           eval_loss \
        --greater_is_better               False \
        --gradient_checkpointing          True \
        --remove_unused_columns           False \
        --dataloader_num_workers          4 \
        --report_to                       wandb \
        --run_name                        "${RUN_TAG}_${STAMP}" \
        --seed                            42 \
        --lambda_ar                       "$LAMBDA_AR" \
        --ar_sigma                        "$AR_SIGMA" \
        --ar_temperature                  "$AR_TEMPERATURE" \
        --ar_use_sharpening               "$AR_USE_SHARPENING" \
    2>&1 | tee "$LOG"

echo "============================================================"
echo " Stage 2 complete → $OUT_DIR"
echo " Best checkpoint: load_best_model_at_end (mixed FSC+SHT val)"
echo "============================================================"
