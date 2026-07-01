#!/bin/bash
# Evaluate dual-loss trained counter on attn_regularizer_val split
# Compares MAE/RMSE against CE-only v3s baseline

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
CHECKPOINT="/data/amondal/unicount_runs/attn_regularizer_full_best_20260507_144336/checkpoint-2670"
VAL_JSON="data/attn_regularizer_dataset/attn_regularizer_val.json"
OUT_JSON="outputs/experiment_lora_counting_sft/eval/dual_loss_val_mae.json"

# ── Environment ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$OUT_JSON")"

echo "============================================================"
echo " EVALUATION: Dual-Loss Trained Counter"
echo "  checkpoint    : $CHECKPOINT"
echo "  val_data      : $VAL_JSON (1,581 images)"
echo "  output        : $OUT_JSON"
echo "  world_size    : 8 GPUs"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

accelerate launch \
    --num_processes=8 \
    --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_dual_loss_3b.py \
        --base_model "$BASE_MODEL" \
        --mllm_hf "$MLLM_HF" \
        --checkpoint_dir "$CHECKPOINT" \
        --val_json "$VAL_JSON" \
        --out_json "$OUT_JSON" \
        --dataset_name "attention_regularizer" \
        --split_name "attn_regularizer_val"

echo "============================================================"
echo " EVALUATION COMPLETE → $OUT_JSON"
echo "============================================================"
