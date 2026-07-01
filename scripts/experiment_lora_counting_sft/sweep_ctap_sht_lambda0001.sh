#!/usr/bin/env bash
# Sweep CTAP parameters (T, max_depth, min_size) on SHT-A and SHT-B
# for ar_loss_v3s_lambda0001 best checkpoint (checkpoint-14500)
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

RUN_DIR="/data/amondal/unicount_runs/ar_loss_v3s_lambda0001_20260508_154727"
CHECKPOINT="${RUN_DIR}/checkpoint-14500/adapter_extracted"
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
CROSS_EVAL="outputs/experiment_lora_counting_sft/cross_eval"
OUT_DIR="${RUN_DIR}/eval_ctap_sweep"
LOG_DIR="${RUN_DIR}/logs"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Datasets to sweep (tag, dataset_name, data_json)
DATASETS=(
  "sht_a_test ShanghaiTech-A ${CROSS_EVAL}/sht_a_test_countdetect_counting.json"
  "sht_b_test ShanghaiTech-B ${CROSS_EVAL}/sht_b_test_countdetect_counting.json"
)

# Parameter grid: "T max_depth min_size"
CONFIGS=(
  "100 3 224"   # baseline
  "100 3  96"   # lower min_size only
  "100 4  96"   # lower min_size + more depth
  " 50 3  96"   # lower T + lower min_size
  " 50 4  96"   # lower T + more depth + lower min_size
  " 50 4  64"   # aggressive: very small patches
  " 75 4  96"   # moderate T
)

echo "============================================================"
echo " CTAP Sweep — SHT-A / SHT-B"
echo "  checkpoint: $CHECKPOINT"
echo "  configs   : ${#CONFIGS[@]}  datasets: ${#DATASETS[@]}"
echo "============================================================"

for ds_row in "${DATASETS[@]}"; do
  read -r tag dataset data <<<"$ds_row"
  if [ ! -f "$data" ]; then
    echo "[SKIP] $tag: $data not found"; continue
  fi
  n=$(python3 -c "import json; print(len(json.load(open('$data'))))" 2>/dev/null || echo "?")
  echo "  $tag ($dataset): $n samples"
done
echo ""

for ds_row in "${DATASETS[@]}"; do
  read -r tag dataset data <<<"$ds_row"
  [ -f "$data" ] || { echo "[SKIP] $tag: not found"; continue; }

  for cfg in "${CONFIGS[@]}"; do
    read -r T MAX_DEPTH MIN_SIZE <<<"$cfg"
    label="T${T}_d${MAX_DEPTH}_s${MIN_SIZE}"
    out="${OUT_DIR}/${tag}_${label}.json"
    log="${LOG_DIR}/sweep_${tag}_${label}_${STAMP}.log"

    if [ -f "$out" ]; then
      echo "[SKIP] $tag $label — already exists"
      continue
    fi

    echo "------------------------------------------------------------"
    echo "[$tag]  T=$T  max_depth=$MAX_DEPTH  min_size=$MIN_SIZE"
    echo "  out: $out"
    echo "------------------------------------------------------------"

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
        --ann_json         /tmp/__no_such_ann__.json \
      2>&1 | tee "$log" || { echo "[FAIL] $tag $label — see $log"; tail -20 "$log"; continue; }

    if [ -f "$out" ]; then
      python3 -c "
import json
r = json.load(open('$out'))
print(f'  >>> MAE={r[\"MAE\"]:.2f}  RMSE={r[\"RMSE\"]:.2f}  recurse={r[\"fraction_recursive\"]:.1%}  n={r[\"n\"]}')
"
    fi
    echo ""
  done
done

echo "============================================================"
echo " SWEEP RESULTS SUMMARY"
printf "  %-14s | %-16s | %6s | %6s | %7s\n" "Dataset" "Config" "MAE" "RMSE" "Recurse%"
printf "  %s\n" "$(python3 -c "print('-'*65)")"
for ds_row in "${DATASETS[@]}"; do
  read -r tag dataset data <<<"$ds_row"
  for cfg in "${CONFIGS[@]}"; do
    read -r T MAX_DEPTH MIN_SIZE <<<"$cfg"
    label="T${T}_d${MAX_DEPTH}_s${MIN_SIZE}"
    out="${OUT_DIR}/${tag}_${label}.json"
    if [ -f "$out" ]; then
      python3 -c "
import json
r = json.load(open('$out'))
print(f'  {r[\"dataset\"]:14s} | T={\"$T\":>3s} d={\"$MAX_DEPTH\"} s={\"$MIN_SIZE\":>3s}       | {r[\"MAE\"]:6.2f} | {r[\"RMSE\"]:6.2f} | {r[\"fraction_recursive\"]:6.1%}')
"
    fi
  done
done
echo "============================================================"
echo "Output dir: $OUT_DIR"
