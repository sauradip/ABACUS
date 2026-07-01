#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AR-Loss v3s reproduction — matches v3s exactly except:
#   - λ_ar=0.01 attention regularization (ObjectFocusedAttentionLoss)
#   - data: balanced_mix_train_with_centers.json (same 49,847 records + object_centers)
#   - trainer: train_dual_loss_3b.py (CE + AR loss)
#
# Everything else is identical to launch_balancedmix_v3s.sh:
#   - Warm-start: BASELINE_BEST adapter + connector
#   - 3 epochs, lr=1e-5, cosine, warmup 0.06, eff batch 16
#   - LoRA r=64 α=128 dropout=0.05, connector unfrozen
#   - accelerate + DeepSpeed ZeRO-2, bf16
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"
BASELINE="/data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect"
INIT_ADAPTER="${BASELINE}/adapter"
INIT_CONN="${BASELINE}/adapter/multi_modal_projector.bin"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="ar_loss_v3s_lambda001"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs

# ── Hyperparameters (identical to v3s) ────────────────────────────────────
NGPU=8
EPOCHS=3
LR=1e-5
BATCH=2
GRAD_ACCUM=1           # eff_batch = 8 * 2 * 1 = 16
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
WARMUP_RATIO=0.06
SAVE_STEPS=1558        # ~once per half-epoch (49847 / 16 ≈ 3115 steps/epoch)
SAVE_LIMIT=8

# ── AR loss config ─────────────────────────────────────────────────────────
LAMBDA_AR=0.01
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
[ -f "$TRAIN_JSON" ]          || { echo "ABORT: missing TRAIN_JSON $TRAIN_JSON"; exit 1; }
[ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "ABORT: missing baseline adapter"; exit 1; }
[ -f "$INIT_CONN" ]           || { echo "ABORT: missing connector $INIT_CONN"; exit 1; }
[ -d "$OUT_DIR" ] && [ -n "$(ls -A $OUT_DIR 2>/dev/null)" ] && { echo "ABORT: OUT_DIR not empty: $OUT_DIR"; exit 1; }

BASELINE_MD5_ADAPTER=$(md5sum "$INIT_ADAPTER/adapter_model.safetensors" | awk '{print $1}')
BASELINE_MD5_CONN=$(md5sum "$INIT_CONN" | awk '{print $1}')

echo "============================================================"
echo " AR-Loss v3s  λ=${LAMBDA_AR}"
echo "  base model    : $BASE_MODEL"
echo "  init adapter  : $INIT_ADAPTER"
echo "  init connector: $INIT_CONN"
echo "  train data    : $TRAIN_JSON  ($(python3 -c "import json; print(len(json.load(open('$TRAIN_JSON'))))" 2>/dev/null) records)"
echo "  output        : $OUT_DIR"
echo "  log           : $LOG"
echo ""
echo "  epochs=$EPOCHS  lr=$LR  per_device=$BATCH  grad_accum=$GRAD_ACCUM  ngpu=$NGPU  eff_batch=$(( NGPU * BATCH * GRAD_ACCUM ))"
echo "  lora r=$LORA_RANK α=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  λ_ar=$LAMBDA_AR  ar_sigma=$AR_SIGMA  ar_temperature=$AR_TEMPERATURE  ar_sharpening=$AR_USE_SHARPENING"
echo "  save_steps=$SAVE_STEPS  save_limit=$SAVE_LIMIT"
echo ""
echo "  baseline adapter md5 : $BASELINE_MD5_ADAPTER"
echo "  baseline connector md5: $BASELINE_MD5_CONN"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# Save pre-flight md5s
echo "$BASELINE_MD5_ADAPTER  $INIT_ADAPTER/adapter_model.safetensors" >  "$OUT_DIR/_PREFLIGHT_MD5.txt"
echo "$BASELINE_MD5_CONN     $INIT_CONN"                              >> "$OUT_DIR/_PREFLIGHT_MD5.txt"

accelerate launch \
    --num_processes="${NGPU}" \
    --mixed_precision=bf16 \
    scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
        --model_name_or_path              "$BASE_MODEL" \
        --mllm_hf_path                    "$MLLM_HF" \
        --data_path                       "$TRAIN_JSON" \
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
        --save_steps                      "$SAVE_STEPS" \
        --save_strategy                   steps \
        --save_total_limit                "$SAVE_LIMIT" \
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
echo " Training complete → $OUT_DIR"
echo "============================================================"
