#!/usr/bin/env bash
# CTAP+NRT eval on the balanced-mix-v2 continuation ckpt across 8 splits.
# Mirrors eval_ctap_countdetect_alldatasets.sh but points at the new run.
set -euo pipefail

cd "$(dirname "$0")/../.."

export PATH="/home/nvidia/miniconda3/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/amondal/UniCount/.triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/amondal/UniCount/.torch_cache}"
export TOKENIZERS_PARALLELISM=false

CKPT="/data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v2_cont_lora64a128_20260503_162648/adapter"
CONN="${CKPT}/multi_modal_projector.bin"
EVAL_OUT="outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v2"
LOG_DIR="logs"
STAMP=$(date +%Y%m%d_%H%M%S)
ANN_FSC="/data/amondal/FSC147_hf/annotation_FSC147_384.json"

mkdir -p "$EVAL_OUT" "$LOG_DIR"

T=100
MAX_DEPTH=3
MIN_SIZE=224

JOBS=(
  "fsc147_val      FSC-147         val   fsc147_val_countdetect_counting.json   yes"
  "fsc147_test     FSC-147         test  fsc147_test_countdetect_counting.json  yes"
  "carpk_test      CARPK           test  carpk_test_countdetect_counting.json   no"
  "jhu_valid       JHU-Crowd       valid jhu_valid_countdetect_counting.json    no"
  "jhu_test        JHU-Crowd       test  jhu_test_countdetect_counting.json     no"
  "qnrf_test       UCF-QNRF        test  qnrf_test_countdetect_counting.json    no"
  "sht_a_test      ShanghaiTech-A  test  sht_a_test_countdetect_counting.json   no"
  "sht_b_test      ShanghaiTech-B  test  sht_b_test_countdetect_counting.json   no"
)

for row in "${JOBS[@]}"; do
  read -r tag dataset split json useann <<<"$row"
  data="outputs/experiment_lora_counting_sft/cross_eval/${json}"
  out="${EVAL_OUT}/${tag}_ctap_mae.json"
  log="${LOG_DIR}/eval_ctap_balancedmix_v2_${tag}_${STAMP}.log"
  echo "============================================================"
  echo "[$tag]  dataset=$dataset split=$split  T=$T max_depth=$MAX_DEPTH"
  echo "  data : $data"
  echo "  out  : $out"
  echo "  log  : $log"
  echo "============================================================"
  if [[ "$useann" == "yes" ]]; then
    ANN_ARG=( --ann_json "$ANN_FSC" )
  else
    ANN_ARG=( --ann_json /tmp/__no_such_ann__.json )
  fi
  accelerate launch --num_processes=8 --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
      --checkpoint_dir    "$CKPT" \
      --connector_weights "$CONN" \
      --val_json          "$data" \
      --out_json          "$out" \
      --dataset_name      "$dataset" \
      --T                 "$T" \
      --max_depth         "$MAX_DEPTH" \
      --min_size          "$MIN_SIZE" \
      "${ANN_ARG[@]}" \
    > "$log" 2>&1 || { echo "[FAIL] $tag — see $log"; tail -30 "$log"; }
done

echo "============================================================"
echo " Done. Results in $EVAL_OUT/"
echo "============================================================"
ls -la "$EVAL_OUT"
