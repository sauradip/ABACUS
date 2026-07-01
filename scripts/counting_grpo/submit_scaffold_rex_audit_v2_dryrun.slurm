#!/bin/bash
#SBATCH --job-name=scaffold_audit_v2_dryrun
#SBATCH --output=logs/scaffold_audit_v2_dryrun_%j.log
#SBATCH --error=logs/scaffold_audit_v2_dryrun_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

_CACHE_BASE="$REPO_DIR/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_BASE/inductor"
export TORCH_COMPILE_CACHE_DIR="$_CACHE_BASE/dynamo"
export XDG_CACHE_HOME="$_CACHE_BASE/xdg"
export TMPDIR="$_CACHE_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/scaffold_rex_stage15_pca_4360912}"
export SCAFFOLD_JSONL="${SCAFFOLD_JSONL:-$REPO_DIR/outputs/scaffold_rex_5k_pca/cross_density_15.jsonl}"
export AUDIT_JSON="${AUDIT_JSON:-$REPO_DIR/checkpoints/scaffold_rex_stage15_pca_4360912/dry_run_patched_audit.json}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export ATTN_IMPL="${ATTN_IMPL:-eager}"
export PRINT_SAMPLES="${PRINT_SAMPLES:-3}"
export START_INDEX="${START_INDEX:-0}"
export MAX_SAMPLES="${MAX_SAMPLES:-15}"

bash scripts/counting_grpo/launch_scaffold_rex_audit_v2_gh200.sh
