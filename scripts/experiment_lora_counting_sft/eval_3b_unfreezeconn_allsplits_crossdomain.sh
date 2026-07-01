#!/usr/bin/env bash
# Run cross-domain eval with the all-splits LoRA-r64α128 + unfrozen-connector ckpt.
#
# Datasets (test/val splits only — no eval subset trained on):
#   CARPK            (test, 459 images)              category="cars"
#   JHU-Crowd        (valid, 497 ; test, 1600)       category="people"
#   UCF-QNRF         (test, 334)                     category="people"
#   ShanghaiTech A   (test, 182)                     category="people"
#   ShanghaiTech B   (test, 316)                     category="people"

set -euo pipefail

REPO=/data/amondal/UniCount
cd "$REPO"

CKPT=/data/amondal/unicount_runs/lora_counting_sft_3b_unfreezeconn_lora64a128_allsplits_20260503_121710/adapter
CONN="$CKPT/multi_modal_projector.bin"

EVAL_OUT="$REPO/outputs/experiment_lora_counting_sft/cross_eval"
LOG_DIR="$REPO/logs"
mkdir -p "$EVAL_OUT" "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)

declare -a JOBS=(
  "carpk_test    CARPK             test  carpk_test_counting.json"
  "jhu_valid     JHU-Crowd         valid jhu_valid_counting.json"
  "jhu_test      JHU-Crowd         test  jhu_test_counting.json"
  "qnrf_test     UCF-QNRF          test  qnrf_test_counting.json"
  "sht_a_test    ShanghaiTech-A    test  sht_a_test_counting.json"
  "sht_b_test    ShanghaiTech-B    test  sht_b_test_counting.json"
)

for entry in "${JOBS[@]}"; do
  read -r tag dataset split jsonfile <<<"$entry"
  data="$EVAL_OUT/$jsonfile"
  out="$EVAL_OUT/${tag}_mae.json"
  log="$LOG_DIR/eval_xdom_${tag}_${STAMP}.log"
  echo "============================================================"
  echo "[${tag}]  dataset=${dataset} split=${split}"
  echo "  data : $data"
  echo "  out  : $out"
  echo "  log  : $log"
  echo "============================================================"

  accelerate launch --num_processes=8 --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_lora_counting_sft.py \
    --checkpoint_dir   "$CKPT" \
    --connector_weights "$CONN" \
    --val_json         "$data" \
    --out_json         "$out" \
    --dataset_name     "$dataset" \
    --split_name       "$split" \
    > "$log" 2>&1
done

echo
echo "============================================================"
echo " ALL CROSS-DOMAIN EVALS DONE → $EVAL_OUT"
ls -la "$EVAL_OUT"/*_mae.json
echo "============================================================"
