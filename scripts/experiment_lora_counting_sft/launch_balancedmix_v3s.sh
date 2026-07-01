#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# v3-S CONTINUATION run from BASELINE_BEST_lora64a128_allsplits_countdetect
#
# IMPORTANT — separation of ckpts:
#   * INIT FROM   : /data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/
#                   (the original best baseline; symlink, never mutated)
#   * NOT FROM v2 : v2 had a person-prior shift baked in that hurt SHT-B.
#                   v3-S deliberately starts fresh from BASELINE.
#   * OUTPUT TO   : /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_<STAMP>/
#                   (brand-new dir; no overlap with baseline or v2 artifacts).
#
# Hyperparams (per spec):
#   - LoRA r=64 / α=128 / dropout=0.05 (matches baseline adapter)
#   - lr=1e-5, epochs=3, cosine, warmup 0.06, eff batch 16
#   - bf16 on 8x A100 (per_device=2, accum=1, ngpu=8)
#   - Gradient checkpointing on (lm_head trainable)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
TRAIN_JSON="outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json"
DS_CFG="scripts/experiment_lora_counting_sft/ds_zero2.json"

BASELINE="/data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect"
INIT_ADAPTER="${BASELINE}/adapter"
INIT_CONN="${BASELINE}/adapter/multi_modal_projector.bin"

[ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "missing $INIT_ADAPTER/adapter_model.safetensors"; exit 1; }
[ -f "$INIT_CONN" ] || { echo "missing $INIT_CONN"; exit 1; }
[ -f "$TRAIN_JSON" ] || { echo "missing $TRAIN_JSON"; exit 1; }

# ── Pre-flight: record baseline checksum (must be unchanged after) ─────────
BASELINE_MD5_ADAPTER=$(md5sum "$INIT_ADAPTER/adapter_model.safetensors" | awk '{print $1}')
BASELINE_MD5_CONN=$(md5sum "$INIT_CONN" | awk '{print $1}')
echo "PRE-FLIGHT baseline adapter md5 : $BASELINE_MD5_ADAPTER"
echo "PRE-FLIGHT baseline connector md5: $BASELINE_MD5_CONN"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="lora_counting_sft_3b_balancedmix_v3s_lora64a128"
OUT_DIR="/data/amondal/unicount_runs/${RUN_TAG}_${STAMP}"

mkdir -p "$OUT_DIR" logs
echo "$BASELINE_MD5_ADAPTER  $INIT_ADAPTER/adapter_model.safetensors"  > "$OUT_DIR/_PREFLIGHT_BASELINE_MD5.txt"
echo "$BASELINE_MD5_CONN     $INIT_CONN" >> "$OUT_DIR/_PREFLIGHT_BASELINE_MD5.txt"

# ── Hyperparameters ───────────────────────────────────────────────────────
NGPU=8
EPOCHS=3
LR=1e-5
BATCH=2
GRAD_ACCUM=1                # eff_batch = 8*2*1 = 16
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
WARMUP_RATIO=0.06

# 49847 / 16 = 3115 steps/epoch * 3 = ~9346 steps
# Save ~once per half-epoch (~1558 steps) → keep ~6 ckpts
SAVE_STEPS=1558
SAVE_LIMIT=8

EFF_BATCH=$(( NGPU * BATCH * GRAD_ACCUM ))
LOG="logs/${RUN_TAG}_${STAMP}.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export INIT_CONNECTOR_FROM="$INIT_CONN"

echo "============================================================"
echo " v3-S CONTINUATION (warm-start from BASELINE_BEST, NOT v2)"
echo "  base model    : $BASE_MODEL"
echo "  init adapter  : $INIT_ADAPTER"
echo "  init connector: $INIT_CONN"
echo "  train data    : $TRAIN_JSON"
echo "  output        : $OUT_DIR   (brand new dir)"
echo "  log           : $LOG"
echo "  epochs=$EPOCHS lr=$LR per_device=$BATCH grad_accum=$GRAD_ACCUM ngpu=$NGPU eff_batch=$EFF_BATCH"
echo "  lora r=$LORA_RANK α=$LORA_ALPHA dropout=$LORA_DROPOUT warmup_ratio=$WARMUP_RATIO"
echo "  save_steps=$SAVE_STEPS save_total_limit=$SAVE_LIMIT"
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
        --report_to                       none \
    2>&1 | tee "$LOG"

# ── Post-flight: confirm baseline ckpt unchanged ───────────────────────────
POST_MD5_ADAPTER=$(md5sum "$INIT_ADAPTER/adapter_model.safetensors" | awk '{print $1}')
POST_MD5_CONN=$(md5sum "$INIT_CONN" | awk '{print $1}')
echo "POST-FLIGHT baseline adapter md5 : $POST_MD5_ADAPTER"
echo "POST-FLIGHT baseline connector md5: $POST_MD5_CONN"
if [ "$POST_MD5_ADAPTER" != "$BASELINE_MD5_ADAPTER" ] || [ "$POST_MD5_CONN" != "$BASELINE_MD5_CONN" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! BASELINE WAS MUTATED — bug in training pipeline      !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 2
fi
echo "============================================================"
echo " v3-S done → $OUT_DIR"
echo " Final adapter   : $OUT_DIR/adapter/"
echo " Connector wts   : $OUT_DIR/adapter/multi_modal_projector.bin"
echo " BASELINE preserved (md5 verified) : $BASELINE"
echo " v2 ckpt also preserved untouched : /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v2_cont_lora64a128_20260503_162648"
echo "============================================================"
