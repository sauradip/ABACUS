#!/bin/bash
#SBATCH --job-name=grpo_stage2
#SBATCH --output=logs/grpo_stage2_%j.log
#SBATCH --error=logs/grpo_stage2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=workq

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
PYTHON_BIN="${PYTHON_BIN:-}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 20000)))}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}"
CONTAINER_PATH="${CONTAINER_PATH:-}"
APPTAINER_CACHE_DIR="${APPTAINER_CACHE_DIR:-$REPO_DIR/.cache/apptainer_cache}"
APPTAINER_TMP_DIR="${APPTAINER_TMP_DIR:-$REPO_DIR/.cache/apptainer_tmp}"

MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/native_sft_stage1_r64_lr2e4}"
# Base model snapshot for image processor config (checkpoint dir lacks preprocessor_config.json)
PROCESSOR_PATH="${PROCESSOR_PATH:-OpenGVLab/InternVL2-2B}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_DIR/outputs/fsc147_scaffold_full/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/grpo_stage2_r64_lr2e6_b005_${SLURM_JOB_ID}}"

LEARNING_RATE="${LEARNING_RATE:-2e-6}"
KL_BETA="${KL_BETA:-0.05}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:--1}"

mkdir -p "$REPO_DIR/logs" "$OUTPUT_DIR"
mkdir -p "$APPTAINER_CACHE_DIR" "$APPTAINER_TMP_DIR"

cd "$REPO_DIR"

echo "=== GRPO STAGE 2 ==="
echo "Model path:    $MODEL_PATH"
echo "Train data:    $TRAIN_DATA"
echo "Output dir:    $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "KL beta:       $KL_BETA"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "nnodes:                $NNODES"
echo "nproc_per_node:        $NPROC_PER_NODE"
echo "world_size:            $((NNODES * NPROC_PER_NODE))"
echo "master_addr:           $MASTER_ADDR"
echo "master_port:           $MASTER_PORT"
echo "num_generations:        $NUM_GENERATIONS"
echo "max_prompt_length:      $MAX_PROMPT_LENGTH"
echo "max_completion_length:  $MAX_COMPLETION_LENGTH"
echo "per_device_batch_size:  $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "gradient_accum_steps:   $GRADIENT_ACCUMULATION_STEPS"
echo "num_train_epochs:       $NUM_TRAIN_EPOCHS"
echo "num_iterations:         $NUM_ITERATIONS"
echo "max_steps:              $MAX_STEPS"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    "$REPO_DIR/.venv311/bin/python" \
    "$REPO_DIR/.venv/bin/python" \
    "python3" \
    "python"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c "import torch, torch.distributed.run" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Could not find a Python interpreter with torch.distributed.run available."
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=$REPO_DIR/.venv311/bin/python"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, torch.distributed.run" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN does not provide torch.distributed.run: $PYTHON_BIN"
  exit 1
fi

echo "Python bin:            $PYTHON_BIN"

# Machine preflight: validate selected Python+Torch on each allocated node.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -lc '"$PYTHON_BIN" -c "import socket,sys,torch,torch.distributed.run; print(f\"[preflight] host={socket.gethostname()} py={sys.version.split()[0]} torch={torch.__version__}\")"'

export PYTHON_BIN NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export MODEL_PATH PROCESSOR_PATH TRAIN_DATA OUTPUT_DIR
export LEARNING_RATE KL_BETA NUM_GENERATIONS MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_TRAIN_EPOCHS NUM_ITERATIONS
export LOGGING_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT MAX_STEPS
export CONTAINER_PATH APPTAINER_CACHE_DIR APPTAINER_TMP_DIR

# FIX: Redirect torch dynamo/inductor cache to a writable project path.
# Compute nodes may lack write permission to /local/user/<uid> (the default TMPDIR).
_CACHE_BASE="$REPO_DIR/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_BASE/inductor"
export TORCH_COMPILE_CACHE_DIR="$_CACHE_BASE/dynamo"
export XDG_CACHE_HOME="$_CACHE_BASE/xdg"
export TMPDIR="$_CACHE_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR TORCH_COMPILE_CACHE_DIR XDG_CACHE_HOME TMPDIR TEMP TMP

echo "Launching Stage 2 GRPO (InternVL2 standalone)..."
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -lc '
set -euo pipefail
if [[ -n "$CONTAINER_PATH" ]]; then
  export APPTAINER_CACHEDIR="$APPTAINER_CACHE_DIR"
  export APPTAINER_TMPDIR="$APPTAINER_TMP_DIR"
  apptainer exec --nv \
    --bind /projects:/projects \
    --bind /home:/home \
    --bind /lus:/lus \
    "$CONTAINER_PATH" \
    "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/counting_grpo/train_internvl_grpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --processor_name_or_path "$PROCESSOR_PATH" \
    --data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$KL_BETA" \
    --num_generations "$NUM_GENERATIONS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.5 \
    --num_iterations "$NUM_ITERATIONS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --attn_implementation eager \
    --bf16
else
  "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$SLURM_PROCID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/counting_grpo/train_internvl_grpo.py \
  --model_name_or_path "$MODEL_PATH" \
  --processor_name_or_path "$PROCESSOR_PATH" \
  --data_path "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --learning_rate "$LEARNING_RATE" \
  --beta "$KL_BETA" \
  --num_generations "$NUM_GENERATIONS" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --warmup_ratio 0.05 \
  --max_grad_norm 0.5 \
  --num_iterations "$NUM_ITERATIONS" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --attn_implementation eager \
  --bf16
fi
'

echo "=== GRPO Stage 2 launch complete ==="
