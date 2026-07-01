#!/usr/bin/env bash
# Wait for synthetic 3ep training to finish, then run CARC eval on:
#   - 1ep adapter
#   - each 3ep checkpoint (per-epoch)
# Writes per-run JSONs to outputs/experiment_lora_counting_sft/eval/.
set -uo pipefail

cd /data/amondal/UniCount
source unicount/bin/activate

LOG_DIR=logs
EVAL_DIR=outputs/experiment_lora_counting_sft/eval
VAL_JSON=outputs/experiment_lora_counting_sft/val/val_counting.json
mkdir -p "$EVAL_DIR" "$LOG_DIR"

# Wait until no train_lora_counting_sft.py procs remain.
echo "[wait] polling for training to end ..."
while pgrep -f train_lora_counting_sft.py > /dev/null; do
    sleep 60
done
echo "[wait] training done at $(date)"

ONE_EP_DIR=$(ls -d /data/amondal/unicount_runs/lora_counting_sft_synthetic_1ep_*/ 2>/dev/null | tail -1)
THREE_EP_DIR=$(ls -d /data/amondal/unicount_runs/lora_counting_sft_synthetic_3ep_*/ 2>/dev/null | tail -1)

echo "[paths] 1ep=$ONE_EP_DIR"
echo "[paths] 3ep=$THREE_EP_DIR"

# 1ep: use the saved adapter dir
ONE_EP_ADAPTER="${ONE_EP_DIR}adapter"
echo "[eval] 1ep -> $ONE_EP_ADAPTER"
accelerate launch --num_processes=8 --mixed_precision=no \
    scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
    --T 100 --max_depth 3 \
    --checkpoint_dir "$ONE_EP_ADAPTER" \
    --val_json "$VAL_JSON" \
    --out_json "$EVAL_DIR/val_recursive_T100_d3_avg_synthetic_1ep.json" \
    > "$LOG_DIR/synth_eval_1ep.log" 2>&1
echo "[eval] 1ep done"

# 3ep: enumerate every checkpoint-N dir
for CKPT in $(ls -d ${THREE_EP_DIR}checkpoint-*/ 2>/dev/null | sort -t- -k2 -n); do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    echo "[eval] 3ep step=$STEP -> $CKPT"
    accelerate launch --num_processes=8 --mixed_precision=no \
        scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
        --T 100 --max_depth 3 \
        --checkpoint_dir "$CKPT" \
        --val_json "$VAL_JSON" \
        --out_json "$EVAL_DIR/val_recursive_T100_d3_avg_synthetic_3ep_step${STEP}.json" \
        > "$LOG_DIR/synth_eval_3ep_step${STEP}.log" 2>&1
    echo "[eval] 3ep step=$STEP done"
done

echo "[done] all evals complete at $(date)"
ls -la "$EVAL_DIR"/val_recursive_T100_d3_avg_synthetic*.json
