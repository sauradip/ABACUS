#!/bin/bash
set -euo pipefail

cd /data/amondal/UniCount
source unicount/bin/activate

BEST_CKPT="/data/amondal/unicount_runs/jigsaw_sft_final_20260430_150418/checkpoint-174"
DATA_PATH="/data/amondal/UniCount/outputs/experiment_jigsaw/val/val_jigsaw.jsonl"
OUT_DIR="/data/amondal/UniCount/outputs/evals/jigsaw_val_ckpt174_simple"
NUM_SAMPLES="${1:-0}"

rm -rf "${OUT_DIR}"

accelerate launch --num_processes=8 --mixed_precision=bf16 \
  scripts/experiment_jigsaw/eval_jigsaw_simple_multi_gpu.py \
  --model_path "${BEST_CKPT}" \
  --data_path "${DATA_PATH}" \
  --out_dir "${OUT_DIR}" \
  --num_samples "${NUM_SAMPLES}"

echo ""
echo "Eval summary: ${OUT_DIR}/val_summary.json"
