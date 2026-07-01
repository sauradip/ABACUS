#!/bin/bash
cd /data/amondal/UniCount
source unicount/bin/activate
python3 scripts/experiment_jigsaw/eval_jigsaw.py \
  --model_path /data/amondal/unicount_runs/jigsaw_sft_final_20260430_150418/checkpoint-174 \
  --data_path  /data/amondal/UniCount/outputs/experiment_jigsaw/val/val_jigsaw.jsonl \
  --out_dir    /data/amondal/UniCount/outputs/evals/jigsaw_val_ckpt174 \
  --num_samples 200 2>&1 | tee /tmp/eval_jigsaw_run.log
