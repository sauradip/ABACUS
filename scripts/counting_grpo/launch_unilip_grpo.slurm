#!/bin/bash
#SBATCH --job-name=hf_unilip_grpo_dense
#SBATCH --output=logs/hf_unilip_grpo_dense_%j.log
#SBATCH --error=logs/hf_unilip_grpo_dense_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --exclude=nid010397

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"
mkdir -p logs "$REPO_DIR/.cache/huggingface" "$REPO_DIR/.cache/triton" "$REPO_DIR/.cache/tmp"

# 1. INITIALIZE FROM BEST SFT CHECKPOINT
export BASE_MODEL="${BASE_MODEL:-$REPO_DIR/checkpoints/hf_multi_image_count_sft_fsc147_train_10ep_evalbest/checkpoint-232}"
export DATA_PATH="${DATA_PATH:-$REPO_DIR/outputs/unilip_grpo_adaptive/hf_native_multi_image_grpo_train_dense.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/hf_unilip_grpo_dense_curriculum_from_best232}"

export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export TEMPERATURE="${TEMPERATURE:-1.2}"
export TOP_P="${TOP_P:-0.95}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
export MAX_STEPS="${MAX_STEPS:--1}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export ALLOW_ATTN_FALLBACK="${ALLOW_ATTN_FALLBACK:-0}"

export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/.cache/triton}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.cache/tmp}"
export CONTAINER_PATH="${CONTAINER_PATH:-$REPO_DIR/.cache/images/nvcr_pytorch_24.08-py3_arm64_sandbox}"
export CONTAINER_PYTHONUSERBASE="${CONTAINER_PYTHONUSERBASE:-$REPO_DIR/.cache/container_py310}"
export APPTAINER_WRITABLE="${APPTAINER_WRITABLE:-0}"

echo "=== Stage 3.2.2: Dense Curriculum GRPO ==="
echo "Base Model : $BASE_MODEL"
echo "Dataset    : $DATA_PATH"
echo "Output     : $OUTPUT_DIR"
echo "Processes  : $NUM_PROCESSES"
echo "Batch/G/A  : $PER_DEVICE_TRAIN_BATCH_SIZE / $NUM_GENERATIONS / $GRADIENT_ACCUMULATION_STEPS"

if [[ ! -d "$BASE_MODEL" || ! -f "$BASE_MODEL/config.json" ]]; then
  echo "ERROR: missing SFT base checkpoint: $BASE_MODEL" >&2
  exit 2
fi

run_in_container() {
  if [[ ! -e "$CONTAINER_PATH" ]]; then
    echo "ERROR: missing Apptainer container: $CONTAINER_PATH" >&2
    exit 2
  fi
  local writable_args=()
  if [[ "$APPTAINER_WRITABLE" == "1" ]]; then
    writable_args+=(--writable)
  fi
  apptainer exec "${writable_args[@]}" --nv \
    --bind /projects:/projects \
    --bind /lus:/lus \
    --bind /home:/home \
    --env PYTHONUSERBASE="$CONTAINER_PYTHONUSERBASE" \
    --env PATH="$CONTAINER_PYTHONUSERBASE/bin:$PATH" \
    --env HF_HOME="$HF_HOME" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env TMPDIR="$TMPDIR" \
    --env PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" \
    --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="$PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" \
    "$CONTAINER_PATH" \
    bash -lc 'cd "$0"; export PATH="$PYTHONUSERBASE/bin:$PATH"; export PYTHONPATH="$PWD:$PWD/UniLIP:$PYTHONUSERBASE/lib/python3.10/site-packages:${PYTHONPATH:-}"; exec "$@"' \
    "$REPO_DIR" "$@"
}

# 2. Check if dataset exists, build if not
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Building DENSE GRPO dataset (min_gt=50)..."
  run_in_container python3 scripts/counting_grpo/generate_unilip_grpo_data.py \
    --min_gt 50 \
    --output_jsonl "$DATA_PATH"
fi

rows="$(wc -l < "$DATA_PATH")"
echo "GRPO rows  : $rows"
if [[ "$rows" -lt 1 ]]; then
  echo "ERROR: GRPO dataset is empty: $DATA_PATH" >&2
  exit 2
fi

# 3. Launch Accelerate
run_in_container python3 -m accelerate.commands.launch --num_processes "$NUM_PROCESSES" \
  scripts/counting_grpo/train_unilip_grpo.py \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_generations "$NUM_GENERATIONS" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --attn_implementation "$ATTN_IMPL" \
  --allow_attn_fallback "$ALLOW_ATTN_FALLBACK"
