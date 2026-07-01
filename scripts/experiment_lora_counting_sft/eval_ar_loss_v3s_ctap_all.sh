#!/usr/bin/env bash
# CTAP+NRT eval on ar_loss_v3s_lambda001 checkpoint-9348 across 5 benchmarks:
#   FSC-147 val, FSC-147 test, ShanghaiTech-A, ShanghaiTech-B, CARPK
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

RUN_DIR="/data/amondal/unicount_runs/ar_loss_v3s_lambda001_20260508_121909"
CHECKPOINT="${RUN_DIR}/checkpoint-9348/adapter_extracted"
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
ANN_FSC="/data/amondal/FSC147_hf/annotation_FSC147_384.json"
CROSS_EVAL="outputs/experiment_lora_counting_sft/cross_eval"
OUT_DIR="${RUN_DIR}/eval_ctap"
LOG_DIR="${RUN_DIR}/logs"
STAMP=$(date +%Y%m%d_%H%M%S)

T=100
MAX_DEPTH=3
MIN_SIZE=224

mkdir -p "$OUT_DIR" "$LOG_DIR"

JOBS=(
  "fsc147_val   FSC-147          ${CROSS_EVAL}/fsc147_val_countdetect_counting.json    yes"
  "fsc147_test  FSC-147          ${CROSS_EVAL}/fsc147_test_countdetect_counting.json   yes"
  "sht_a_test   ShanghaiTech-A   ${CROSS_EVAL}/sht_a_test_countdetect_counting.json    no"
  "sht_b_test   ShanghaiTech-B   ${CROSS_EVAL}/sht_b_test_countdetect_counting.json    no"
  "carpk_test   CARPK            ${CROSS_EVAL}/carpk_test_countdetect_counting.json     no"
)

echo "============================================================"
echo " AR-Loss v3s λ=0.01 — CTAP+NRT Evaluation (5 benchmarks)"
echo "  checkpoint : $CHECKPOINT"
echo "  T=${T}  max_depth=${MAX_DEPTH}  min_size=${MIN_SIZE}"
echo "  output_dir : $OUT_DIR"
echo "============================================================"
for row in "${JOBS[@]}"; do
  read -r tag dataset data useann <<<"$row"
  if [ -f "$data" ]; then
    n=$(python3 -c "import json; print(len(json.load(open('$data'))))" 2>/dev/null || echo "?")
    echo "  + $tag ($dataset): $n samples"
  else
    echo "  - $tag: NOT FOUND ($data)"
  fi
done
echo ""
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

for row in "${JOBS[@]}"; do
  read -r tag dataset data useann <<<"$row"

  if [ ! -f "$data" ]; then
    echo "[SKIP] $tag: $data not found"; continue
  fi

  out="${OUT_DIR}/${tag}_ctap_T${T}_d${MAX_DEPTH}.json"
  log="${LOG_DIR}/eval_ctap_arloss_${tag}_${STAMP}.log"

  echo "============================================================"
  echo "[$tag]  dataset=$dataset  T=$T  max_depth=$MAX_DEPTH"
  echo "  out : $out"
  echo "  log : $log"
  echo "============================================================"

  if [[ "$useann" == "yes" ]]; then
    ANN_ARG=( --ann_json "$ANN_FSC" )
  else
    ANN_ARG=( --ann_json /tmp/__no_such_ann__.json )
  fi

  accelerate launch --num_processes=8 --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
      --base_model       "$BASE_MODEL" \
      --mllm_hf          "$MLLM_HF" \
      --checkpoint_dir   "$CHECKPOINT" \
      --val_json         "$data" \
      --out_json         "$out" \
      --dataset_name     "$dataset" \
      --T                "$T" \
      --max_depth        "$MAX_DEPTH" \
      --min_size         "$MIN_SIZE" \
      --connector_weights "${CHECKPOINT}/multi_modal_projector.bin" \
      "${ANN_ARG[@]}" \
    2>&1 | tee "$log" || { echo "[FAIL] $tag — see $log"; tail -30 "$log"; }

  if [ -f "$out" ]; then
    python3 -c "
import json
with open('$out') as f:
    r = json.load(f)
print(f'  >>> MAE={r[\"MAE\"]:.2f}  RMSE={r[\"RMSE\"]:.2f}  frac_recursive={r[\"fraction_recursive\"]:.1%}  n={r[\"n\"]}')
"
  fi
  echo ""
done

echo "============================================================"
echo " RESULTS SUMMARY — AR-Loss v3s λ=0.01 CTAP+NRT"
echo "  checkpoint: $CHECKPOINT"
echo "============================================================"
printf "  %-22s | %5s | %6s | %6s | %6s\n" "Dataset" "n" "MAE" "RMSE" "Recurse%"
printf "  %s\n" "$(python3 -c "print('-'*65)")"
for row in "${JOBS[@]}"; do
  read -r tag dataset data useann <<<"$row"
  out="${OUT_DIR}/${tag}_ctap_T${T}_d${MAX_DEPTH}.json"
  if [ -f "$out" ]; then
    python3 -c "
import json
with open('$out') as f:
    r = json.load(f)
print(f'  {r[\"dataset\"]:22s} | {r[\"n\"]:5d} | {r[\"MAE\"]:6.2f} | {r[\"RMSE\"]:6.2f} | {r[\"fraction_recursive\"]:5.1%}')
"
  else
    printf "  %-22s | %5s | %6s | %6s | %6s\n" "$dataset" "-" "-" "-" "-"
  fi
done
echo "============================================================"
echo "Output dir: $OUT_DIR"
