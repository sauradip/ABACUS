#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="/home/nvidia/miniconda3/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

CHECKPOINT="/data/amondal/unicount_runs/ar_loss_stage2_sht_20260508_203646/checkpoint-600/adapter_extracted"
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
CROSS_EVAL="outputs/experiment_lora_counting_sft/cross_eval"
OUT_DIR="/data/amondal/unicount_runs/ar_loss_stage2_sht_20260508_203646/eval_ctap"
mkdir -p "$OUT_DIR"

CONN="${CHECKPOINT}/multi_modal_projector.bin"

run_eval() {
  local tag=$1 dataset=$2 data=$3 T=$4 max_depth=$5 min_size=$6
  local out="${OUT_DIR}/${tag}.json"
  echo "=== $tag  T=$T d=$max_depth s=$min_size ==="
  accelerate launch --num_processes=8 --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
      --base_model       "$BASE_MODEL" \
      --mllm_hf          "$MLLM_HF" \
      --checkpoint_dir   "$CHECKPOINT" \
      --val_json         "$data" \
      --out_json         "$out" \
      --dataset_name     "$dataset" \
      --T                "$T" \
      --max_depth        "$max_depth" \
      --min_size         "$min_size" \
      --connector_weights "$CONN" \
      --ann_json         /tmp/__no_such_ann__.json
  python3 -c "
import json; r=json.load(open('$out'))
print(f'  MAE={r[\"MAE\"]:.2f}  RMSE={r[\"RMSE\"]:.2f}  recurse={r[\"fraction_recursive\"]:.1%}  n={r[\"n\"]}')
"
}

run_eval fsc147_val   "FSC-147"        "${CROSS_EVAL}/fsc147_val_countdetect_counting.json"    100 3  224
run_eval fsc147_test  "FSC-147"        "${CROSS_EVAL}/fsc147_test_countdetect_counting.json"   100 3  224
run_eval sht_a_test   "ShanghaiTech-A" "${CROSS_EVAL}/sht_a_test_countdetect_counting.json"    50 4   64
run_eval sht_b_test   "ShanghaiTech-B" "${CROSS_EVAL}/sht_b_test_countdetect_counting.json"    50 3   96
run_eval carpk_test   "CARPK"          "${CROSS_EVAL}/carpk_test_countdetect_counting.json"   100 3  224

echo ""
echo "=== SUMMARY ==="
for tag in fsc147_val fsc147_test sht_a_test sht_b_test carpk_test; do
  f="${OUT_DIR}/${tag}.json"
  [ -f "$f" ] && python3 -c "
import json; r=json.load(open('$f'))
print(f'  {r[\"dataset\"]:20s}  MAE={r[\"MAE\"]:7.2f}  RMSE={r[\"RMSE\"]:7.2f}')
"
done
