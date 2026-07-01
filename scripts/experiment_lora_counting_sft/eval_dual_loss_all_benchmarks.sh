#!/bin/bash
# Multi-dataset evaluation: FSC-147 (val/test), SHA-A/B (test), CARPK (test)
# Evaluates dual-loss trained checkpoint across all benchmarks

set -euo pipefail

cd "$(dirname "$0")/../.."

export PATH="/home/nvidia/miniconda3/bin:${PATH}"

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
CHECKPOINT="/data/amondal/unicount_runs/attn_regularizer_full_best_20260507_144336/checkpoint-2670"
OUT_DIR="outputs/experiment_lora_counting_sft/eval/dual_loss_benchmarks"

# ── Environment ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR"

# ── Evaluation datasets ────────────────────────────────────────────────────────
declare -A DATASETS=(
    ["fsc147_val"]="outputs/experiment_lora_counting_sft/cross_eval/fsc147_val_countdetect_counting.json"
    ["fsc147_test"]="outputs/experiment_lora_counting_sft/cross_eval/fsc147_test_countdetect_counting.json"
    ["sha_a_test"]="data/evaluation_datasets/sha_test_a.json"
    ["sha_b_test"]="data/evaluation_datasets/sha_test_b.json"
    ["carpk_test"]="outputs/experiment_lora_counting_sft/cross_eval/carpk_test_counting.json"
)

echo "============================================================"
echo " MULTI-DATASET EVALUATION: Dual-Loss Trained Counter"
echo "  checkpoint    : $CHECKPOINT"
echo "  base_model    : $BASE_MODEL"
echo "  output_dir    : $OUT_DIR"
echo ""
echo " Datasets:"
for name in "${!DATASETS[@]}"; do
    path="${DATASETS[$name]}"
    if [ -f "$path" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$path'))))" 2>/dev/null || echo "?")
        echo "  ✓ $name: $count"
    else
        echo "  ✗ $name: NOT FOUND ($path)"
    fi
done
echo ""
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

echo ""
echo "Starting evaluations..."
echo ""

# ── Run evaluations ───────────────────────────────────────────────────────────
for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"

    if [ ! -f "$dataset_path" ]; then
        echo "[SKIP] $dataset_name: file not found"
        continue
    fi

    output_json="$OUT_DIR/${dataset_name}_mae.json"

    echo "[ Eval ] $dataset_name → $output_json"

    accelerate launch \
        --num_processes=8 \
        --mixed_precision=no \
        scripts/experiment_lora_counting_sft/eval_dual_loss_3b.py \
            --base_model "$BASE_MODEL" \
            --mllm_hf "$MLLM_HF" \
            --checkpoint_dir "$CHECKPOINT" \
            --val_json "$dataset_path" \
            --out_json "$output_json" \
            --dataset_name "$dataset_name" \
            --split_name "$dataset_name"

    # Extract key metrics
    python3 << PYEOF
import json
with open("$output_json") as f:
    result = json.load(f)
print(f"  MAE: {result['MAE']:.2f} | RMSE: {result['RMSE']:.2f} | Parse: {result['parse_rate']:.1%}")
PYEOF

    echo ""
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results summary:"
for dataset_name in fsc147_val fsc147_test sha_a_test sha_b_test carpk_test; do
    output_json="$OUT_DIR/${dataset_name}_mae.json"
    if [ -f "$output_json" ]; then
        python3 << PYEOF
import json
with open("$output_json") as f:
    result = json.load(f)
print(f"  {result['dataset']:20s} | n={result['n']:4d} | MAE={result['MAE']:6.2f} | RMSE={result['RMSE']:6.2f} | Parse={result['parse_rate']:5.1%}")
PYEOF
    fi
done

echo ""
echo "Output directory: $OUT_DIR"
echo "============================================================"
