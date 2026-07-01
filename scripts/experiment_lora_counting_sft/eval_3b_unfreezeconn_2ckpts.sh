#!/usr/bin/env bash
# Run FSC-147 val + test evaluation on the two unfreeze-connector LoRA runs:
#   1) train-only        : lora_counting_sft_3b_unfreezeconn_lora64a128_20260503_115133
#   2) train+val+test    : lora_counting_sft_3b_unfreezeconn_lora64a128_allsplits_20260503_121710
#
# Loads the LoRA adapter AND the trained multi_modal_projector.bin connector
# weights. Greedy decoding (spec §A.9). Multi-GPU (8×A100) via accelerate.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

export PATH="/home/nvidia/miniconda3/bin:${PATH}"
export HF_HOME="${HF_HOME:-/data/amondal/UniCount/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPU=8
EVAL_OUT=/data/amondal/UniCount/outputs/experiment_lora_counting_sft/eval
mkdir -p "$EVAL_OUT" logs

declare -A CKPT
CKPT[trainonly]=/data/amondal/unicount_runs/lora_counting_sft_3b_unfreezeconn_lora64a128_20260503_115133
CKPT[allsplits]=/data/amondal/unicount_runs/lora_counting_sft_3b_unfreezeconn_lora64a128_allsplits_20260503_121710

declare -A VALJ
VALJ[val]=outputs/experiment_lora_counting_sft/val/val_counting.json
VALJ[test]=outputs/experiment_lora_counting_sft/test/test_counting.json

STAMP=$(date +%Y%m%d_%H%M%S)

for tag in trainonly allsplits; do
  CK="${CKPT[$tag]}"
  for split in val test; do
    OUT="$EVAL_OUT/${tag}_${split}_mae.json"
    LOG="logs/eval_${tag}_${split}_${STAMP}.log"
    echo "============================================================"
    echo "[$tag / $split]"
    echo "  adapter   : $CK/adapter"
    echo "  connector : $CK/adapter/multi_modal_projector.bin"
    echo "  data      : ${VALJ[$split]}"
    echo "  out       : $OUT"
    echo "  log       : $LOG"
    echo "============================================================"
    accelerate launch \
        --num_processes="$NGPU" \
        --mixed_precision=no \
        scripts/experiment_lora_counting_sft/eval_lora_counting_sft.py \
            --checkpoint_dir    "$CK/adapter" \
            --connector_weights "$CK/adapter/multi_modal_projector.bin" \
            --val_json          "${VALJ[$split]}" \
            --out_json          "$OUT" \
        2>&1 | tee "$LOG"
  done
done

echo
echo "============================================================"
echo " ALL EVALS DONE → $EVAL_OUT/"
ls -la "$EVAL_OUT"/{trainonly,allsplits}_{val,test}_mae.json 2>/dev/null
echo "============================================================"
