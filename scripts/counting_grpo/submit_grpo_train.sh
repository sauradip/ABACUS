#!/bin/bash
#SBATCH --job-name=counting_grpo
#SBATCH --output=/projects/u6bl/myprojects/omnicountgen/logs/counting_grpo_%j.log
#SBATCH --error=/projects/u6bl/myprojects/omnicountgen/logs/counting_grpo_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

set -euo pipefail

JANUS_DIR="/projects/u6bl/myprojects/Janus"
WORKSPACE_DIR="/projects/u6bl/myprojects"
LIB_DIR="$JANUS_DIR/python_libs"
HF_CACHE_DIR="$JANUS_DIR/.hf_cache"
TRITON_DIR="$JANUS_DIR/.triton_cache"

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$JANUS_DIR/.cache"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH="$WORKSPACE_DIR/UniLIP/python_libs/lib/python3.10/site-packages:$WORKSPACE_DIR:$WORKSPACE_DIR/VLM-R1/src/open-r1-multimodal/src:$WORKSPACE_DIR/UniLIP:$LIB_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PATH="$LIB_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1

SFT_CHECKPOINT="$WORKSPACE_DIR/UniLIP/work_dirs/1b_fsc147_understanding_sft"
TRAIN_DATA="$WORKSPACE_DIR/omnicountgen/counting_grpo/grpo_data/train.jsonl"
OUTPUT_DIR="$WORKSPACE_DIR/omnicountgen/counting_grpo/checkpoints_numiter2_stable"

# Clear stale checkpoints — prevents auto-resume from a different run
rm -rf "$OUTPUT_DIR"/checkpoint-*
mkdir -p "$OUTPUT_DIR" "$TRITON_DIR" "$PIP_CACHE_DIR" "$HF_CACHE_DIR"

TOK_SNAPSHOT_BASE="$WORKSPACE_DIR/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots"
if ls -d "$TOK_SNAPSHOT_BASE"/* >/dev/null 2>&1; then
  UNILIP_TOKENIZER_PATH=$(ls -d "$TOK_SNAPSHOT_BASE"/* | head -n 1)
else
  UNILIP_TOKENIZER_PATH="OpenGVLab/InternVL3-1B"
fi

INTERNVL_SNAPSHOT_BASE="$HF_CACHE_DIR/hub/models--OpenGVLab--InternVL3-1B/snapshots"
if ls -d "$INTERNVL_SNAPSHOT_BASE"/* >/dev/null 2>&1; then
  UNILIP_INTERNVL_SOURCE=$(ls -d "$INTERNVL_SNAPSHOT_BASE"/* | head -n 1)
else
  UNILIP_INTERNVL_SOURCE="OpenGVLab/InternVL3-1B"
fi

INTERNVL_HF_SNAPSHOT_BASE="$HF_CACHE_DIR/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots"
if ls -d "$INTERNVL_HF_SNAPSHOT_BASE"/* >/dev/null 2>&1; then
  UNILIP_INTERNVL_HF_SOURCE=$(ls -d "$INTERNVL_HF_SNAPSHOT_BASE"/* | head -n 1)
else
  UNILIP_INTERNVL_HF_SOURCE="OpenGVLab/InternVL3-1B-hf"
fi

echo "=== COUNTING GRPO FULL TRAINING ==="
echo "Checkpoint: $SFT_CHECKPOINT"
echo "Train data: $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") examples)"
echo "Output:     $OUTPUT_DIR"
echo ""

apptainer exec --nv \
  --bind /projects:/projects \
  --env PYTHONPATH="$PYTHONPATH" \
  --env PYTHONUSERBASE="$LIB_DIR" \
  --env HF_HOME="$HF_CACHE_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env UNILIP_TOKENIZER_PATH="$UNILIP_TOKENIZER_PATH" \
  --env UNILIP_INTERNVL_SOURCE="$UNILIP_INTERNVL_SOURCE" \
  --env UNILIP_INTERNVL_HF_SOURCE="$UNILIP_INTERNVL_HF_SOURCE" \
  "$JANUS_DIR/pytorch_24.08.sif" \
  bash -lc '
    set -e
    cd /projects/u6bl/myprojects/VLM-R1/src/open-r1-multimodal

    python -m accelerate.commands.launch \
      --dynamo_backend no \
      --num_processes 4 \
      --num_machines 1 \
      --mixed_precision bf16 \
      src/open_r1/grpo_jsonl.py \
      --dataset_name this_is_not_used \
      --use_vllm False \
      --model_name_or_path '"$SFT_CHECKPOINT"' \
      --data_file_paths '"$TRAIN_DATA"' \
      --image_folders / \
      --output_dir '"$OUTPUT_DIR"' \
      --reward_funcs counting \
      --num_generations 8 \
      --max_prompt_length 512 \
      --max_completion_length 20 \
      --learning_rate 5e-7 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --num_train_epochs 5 \
      --warmup_ratio 0.05 \
      --bf16 \
      --beta 0.04 \
      --max_grad_norm 0.5 \
      --num_iterations 2 \
      --logging_steps 10 \
      --save_steps 200 \
      --save_total_limit 3 \
      --task_type counting \
      --attn_implementation eager \
      --gradient_checkpointing True \
      --gradient_checkpointing_kwargs "{\"use_reentrant\": false}" \
      --report_to none \
      --trust_remote_code True \
      --ddp_find_unused_parameters True

    echo "=== Counting GRPO training complete ==="
  '
