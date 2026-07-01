#!/bin/bash
#SBATCH --job-name=stage32_grpo
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/stage32_grpo_%j.log
#SBATCH --error=logs/stage32_grpo_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv311/bin/python}"

export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-6144}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"

MODEL_PATH="${MODEL_PATH:-/projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_pca_4371501}"
DATASET_PATH="${DATASET_PATH:-outputs/scaffold_rex_5k_pca/curriculum_grpo_v32.jsonl}"
REWARD_SCRIPT="${REWARD_SCRIPT:-scripts/counting_grpo/grpo_reward_v32.py}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/stage32_grpo_r1rex_tally}"

mkdir -p logs "$OUTPUT_DIR"

echo "=== Stage 3.2 GRPO Launch ==="
echo "Model        : $MODEL_PATH"
echo "Dataset      : $DATASET_PATH"
echo "Reward script: $REWARD_SCRIPT"
echo "Output dir   : $OUTPUT_DIR"
echo "MAX_PROMPT   : $MAX_PROMPT_LEN"
echo "MAX_GEN      : $MAX_GEN_LEN"
echo "BATCH/ACCUM  : $BATCH_SIZE / $GRAD_ACCUM"

$PYTHON_BIN -m accelerate.commands.launch --num_processes 4 \
    scripts/counting_grpo/train_grpo_v32.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_PATH" \
    --reward_script "$REWARD_SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_generations 8 \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_completion_length "$MAX_GEN_LEN" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 5e-6 \
    --save_steps 50 \
    --bf16 True \
    --attn_implementation flash_attention_2
