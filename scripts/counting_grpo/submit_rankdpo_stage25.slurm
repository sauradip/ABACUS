#!/bin/bash
#SBATCH --job-name=rankdpo_stage25
#SBATCH --output=logs/rankdpo_stage25_%j.log
#SBATCH --error=logs/rankdpo_stage25_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

# Queue-friendly defaults for single-GPU RankDPO.
# Override at submit time if needed, for example:
#   sbatch --time=12:00:00 --cpus-per-task=12 --mem=180G scripts/counting_grpo/submit_rankdpo_stage25.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"
mkdir -p logs

# Prefer project venv for reproducibility.
if [[ -x .venv311/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
fi

export PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv311/bin/python}"
export DATA_JSONL="${DATA_JSONL:-$REPO_DIR/outputs/rankdpo/preference_pairs.jsonl}"
export OUT_DIR="${OUT_DIR:-$REPO_DIR/checkpoints/rankdpo_stage25_gh200_${SLURM_JOB_ID}}"
export SCAFFOLD_JSONL="${SCAFFOLD_JSONL:-$REPO_DIR/outputs/fsc147_scaffold_full/all.jsonl}"
export STAGE1_CKPT="${STAGE1_CKPT:-$REPO_DIR/checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"
export VERIFY_MAX_NEW_TOKENS="${VERIFY_MAX_NEW_TOKENS:-256}"
export VERIFY_ONLY="${VERIFY_ONLY:-0}"
export SKIP_MERGE_VERIFY="${SKIP_MERGE_VERIFY:-0}"
export PURGE_HF_CACHE="${PURGE_HF_CACHE:-1}"
export INIT_MODE="${INIT_MODE:-manual_inject}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ "$PURGE_HF_CACHE" == "1" ]]; then
  rm -rf "$HOME/.cache/huggingface/"
fi

echo "=== RankDPO Stage 2.5 Submit ==="
echo "Job ID:                 ${SLURM_JOB_ID}"
echo "Node list:              ${SLURM_NODELIST:-pending}"
echo "Requested resources:    1 GPU | ${SLURM_CPUS_PER_TASK:-8} CPU | ${SLURM_MEM_PER_NODE:-120G} mem"
echo "Python:                 ${PYTHON_BIN}"
echo "Data JSONL:             ${DATA_JSONL}"
echo "Output dir:             ${OUT_DIR}"
echo "Stage1 checkpoint:      ${STAGE1_CKPT}"
echo "Verify max new tokens:  ${VERIFY_MAX_NEW_TOKENS}"
echo "Verify only:            ${VERIFY_ONLY}"
echo "Skip merge verify:      ${SKIP_MERGE_VERIFY}"
echo "Init mode:              ${INIT_MODE}"
echo "Max length:             ${MAX_LENGTH}"
echo "Prompt/completion:      ${MAX_PROMPT_LENGTH}/${MAX_COMPLETION_LENGTH}"
echo "Batch/grad accum:       ${PER_DEVICE_TRAIN_BATCH_SIZE}/${GRADIENT_ACCUMULATION_STEPS}"

bash scripts/counting_grpo/launch_rankdpo_stage25_gh200.sh

echo "=== RankDPO Stage 2.5 complete ==="
