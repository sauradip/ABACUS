#!/bin/bash
#SBATCH --job-name=scaffold_audit_v2
#SBATCH --output=logs/scaffold_audit_v2_%j.log
#SBATCH --error=logs/scaffold_audit_v2_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"
mkdir -p logs

if [[ -x ".venv311/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
fi

# Writable torch cache dirs on compute nodes.
_CACHE_BASE="$REPO_DIR/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_BASE/inductor"
export TORCH_COMPILE_CACHE_DIR="$_CACHE_BASE/dynamo"
export XDG_CACHE_HOME="$_CACHE_BASE/xdg"
export TMPDIR="$_CACHE_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/scaffold_rex_stage15_4342903}"
export SCAFFOLD_JSONL="${SCAFFOLD_JSONL:-$REPO_DIR/outputs/scaffold_rex_5k/all.jsonl}"
export AUDIT_JSON="${AUDIT_JSON:-$MODEL_PATH/scaffold_rex_audit_v2.json}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export ATTN_IMPL="${ATTN_IMPL:-eager}"
export PRINT_SAMPLES="${PRINT_SAMPLES:-3}"
export START_INDEX="${START_INDEX:-0}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"

echo "=== Submit Scaffold Self-Audit V2 ==="
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST : ${SLURM_NODELIST:-pending}"
echo "MODEL_PATH     : $MODEL_PATH"
echo "SCAFFOLD_JSONL : $SCAFFOLD_JSONL"
echo "AUDIT_JSON     : $AUDIT_JSON"
echo "MAX_SAMPLES    : $MAX_SAMPLES"

echo "Launching..."
bash scripts/counting_grpo/launch_scaffold_rex_audit_v2_gh200.sh

echo "Done. Output: $AUDIT_JSON"
