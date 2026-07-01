#!/bin/bash
set -euo pipefail

export OUTPUT_FOLDER="${OUTPUT_FOLDER:-../work_dirs/1b_stage3_fsc147_ti2i_understanding}"
export EDIT_IMG_FOLDER="${EDIT_IMG_FOLDER:-/projects/u6bl/myprojects/Datasets/FSC-147/edit_sft_fsc147}"
export STAGE2_CKPT="${STAGE2_CKPT:-../.resolved_models/UniLIP-1B}"
export MLLM_PATH="${MLLM_PATH:-/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B/snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8}"
export MLLM_HF_PATH="${MLLM_HF_PATH:-/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171}"

if [[ ! -d "$EDIT_IMG_FOLDER" ]]; then
  echo "Edit dataset directory not found: $EDIT_IMG_FOLDER" >&2
  exit 1
fi

if [[ ! -d "$STAGE2_CKPT" ]]; then
  echo "Stage-2 checkpoint directory not found: $STAGE2_CKPT" >&2
  exit 1
fi

# Expected distributed env variables are exported by Slurm submit script.
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${RANK:?RANK is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

torchrun --nproc_per_node="$GPUS_PER_NODE" --nnodes="$WORLD_SIZE" --node_rank="$RANK" --master_port="$MASTER_PORT" --master_addr="$MASTER_ADDR" \
  unilip/train/train_stage3.py \
  --deepspeed deepspeed_scripts/zero0.json \
  --model_name_or_path "$STAGE2_CKPT" \
  --unilip_factor 10.6 \
  --mllm_path "$MLLM_PATH" \
  --mllm_hf_path "$MLLM_HF_PATH" \
  --vae_path mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
  --dit_path Efficient-Large-Model/Sana_600M_512px_diffusers \
  --version internvl \
  --data_type mix \
  --edit_image_folder "$EDIT_IMG_FOLDER" \
  --edit_repeat 6 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir "$OUTPUT_FOLDER" \
  --num_train_epochs 8 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --eval_strategy no \
  --save_strategy no \
  --save_steps 1000000 \
  --save_total_limit 2 \
  --learning_rate 8e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr":5e-6}' \
  --model_max_length 1024 \
  --logging_steps 10 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --lazy_preprocess True \
  --n_query 256 \
  --n_und_query 256 \
  --report_to none \
  --run_name unilip_1b_stage3_fsc147_ti2i_understanding \
  --fix_dit False \
  --fix_connect False \
  --fix_llm False
