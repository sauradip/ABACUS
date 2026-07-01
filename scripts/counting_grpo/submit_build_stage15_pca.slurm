#!/bin/bash
#SBATCH --job-name=build_stage15_pca
#SBATCH --output=logs/build_stage15_pca_%j.log
#SBATCH --error=logs/build_stage15_pca_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
export PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv311/bin/python}"

export INPUT_JSONL="${INPUT_JSONL:-$REPO_DIR/outputs/scaffold_rex_5k/all.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/scaffold_rex_5k_pca}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export IMG_SIZE="${IMG_SIZE:-448}"
export EXPECTED_ROWS="${EXPECTED_ROWS:-4945}"

# Keep caches writable on compute nodes
_CACHE_BASE="$REPO_DIR/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_BASE/inductor"
export TORCH_COMPILE_CACHE_DIR="$_CACHE_BASE/dynamo"
export XDG_CACHE_HOME="$_CACHE_BASE/xdg"
export TMPDIR="$_CACHE_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"


echo "=== Build Stage15 PCA Dataset (GPU) ==="
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID:-local}"
echo "Python        : $PYTHON_BIN"
echo "Input JSONL   : $INPUT_JSONL"
echo "Output dir    : $OUTPUT_DIR"
echo "Batch size    : $BATCH_SIZE"

"$PYTHON_BIN" scripts/counting_grpo/build_stage15_pca_dataset.py \
  --input_jsonl "$INPUT_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --img_size "$IMG_SIZE" \
  --device cuda \
  --skip_existing 1

"$PYTHON_BIN" scripts/counting_grpo/check_stage15_pca_preflight.py \
  --data_path "$OUTPUT_DIR/all.jsonl" \
  --use_pca_images 1 \
  --expected_rows "$EXPECTED_ROWS" \
  --strict 1

echo "=== PCA build + strict preflight complete ==="
