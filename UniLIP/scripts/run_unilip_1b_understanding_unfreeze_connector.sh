#!/bin/bash
set -euo pipefail

export OUTPUT_FOLDER="${OUTPUT_FOLDER:-work_dirs/1b_fsc147_understanding_sft_unfreeze_connector}"
export STAGE2_CKPT="${STAGE2_CKPT:-../.resolved_models/UniLIP-1B}"
export MLLM_PATH="${MLLM_PATH:-/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B/snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8}"
export MLLM_HF_PATH="${MLLM_HF_PATH:-/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171}"
export DATA_PATH="${DATA_PATH:-/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_sft.json}"

if [[ ! -d "$STAGE2_CKPT" ]]; then
  echo "Checkpoint not found: $STAGE2_CKPT" >&2
  exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Data JSON not found: $DATA_PATH" >&2
  exit 1
fi

: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${RANK:?RANK is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

torchrun --nproc_per_node="$GPUS_PER_NODE" --nnodes="$WORLD_SIZE" --node_rank="$RANK" \
  --master_port="$MASTER_PORT" --master_addr="$MASTER_ADDR" \
  unilip/train/train_understanding.py \
  --deepspeed deepspeed_scripts/zero2.json \
  --model_name_or_path "$STAGE2_CKPT" \
  --mllm_path "$MLLM_PATH" \
  --mllm_hf_path "$MLLM_HF_PATH" \
  --version internvl \
  --data_type mix \
  --data_path "$DATA_PATH" \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir "$OUTPUT_FOLDER" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --learning_rate 4e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --model_max_length 512 \
  --logging_steps 5 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --report_to none \
  --run_name unilip_1b_fsc147_understanding_sft_unfreeze_connector \
  --fix_llm False \
  --fix_vit True \
  --fix_connect False
