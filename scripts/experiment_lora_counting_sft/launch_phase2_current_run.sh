#!/bin/bash
# Phase 2 launch for existing pipeline run: pretext_dual_loss_pipeline_20260508_030427
# Phase 1 already complete; adapter extracted at PHASE1_BEST_ADAPTER

set -e

TIMESTAMP="20260508_030427"
PIPELINE_ROOT="/data/amondal/unicount_runs/pretext_dual_loss_pipeline_${TIMESTAMP}"
PHASE1_BEST_ADAPTER="${PIPELINE_ROOT}/phase1_pretext/adapter_extracted"
PHASE2_CHECKPOINTS="${PIPELINE_ROOT}/phase2_dual_loss/checkpoints"
PHASE2_LOGS="${PIPELINE_ROOT}/phase2_dual_loss/logs"
MLLM_HF_PATH="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"

mkdir -p "$PHASE2_CHECKPOINTS" "$PHASE2_LOGS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Phase 2 from: $PHASE1_BEST_ADAPTER"

cd /data/amondal/UniCount

WANDB_PROJECT="pretext_dual_loss_pipeline" torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
  --model_name_or_path "$PHASE1_BEST_ADAPTER" \
  --mllm_hf_path "$MLLM_HF_PATH" \
  --data_path outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json \
  --output_dir "$PHASE2_CHECKPOINTS" \
  \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --lr_scheduler_type cosine \
  \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --bf16 True \
  \
  --lambda_ar 0.1 \
  --ar_sigma 1.0 \
  --ar_temperature 1.0 \
  --ar_use_sharpening False \
  \
  --save_steps 500 \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --load_best_model_at_end True \
  --metric_for_best_model loss \
  --greater_is_better False \
  --save_total_limit 3 \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --run_name "phase2_dual_loss_${TIMESTAMP}" \
  --seed 42 \
  --logging_first_step \
  2>&1 | tee "$PHASE2_LOGS/train_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 complete. Checkpoints: ${PHASE2_CHECKPOINTS}"
