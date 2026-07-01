#!/usr/bin/env bash
# Local debug gate before Stage 1.5 PCA-SFT sbatch submit.
#
# Usage:
#   bash scripts/counting_grpo/submit_stage15_sft_pca_safe.sh
#
# Optional env:
#   TRAIN_JSONL          (default: outputs/scaffold_rex_5k_pca/all.jsonl)
#   USE_PCA_IMAGES       (default: 1)
#   EXPECTED_ROWS        (default: 4945 for PCA mode, else 0)
#   SUBMIT_TO_SLURM      (default: 1; set 0 for debug-only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv311/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

USE_PCA_IMAGES_RAW="${USE_PCA_IMAGES:-1}"
case "${USE_PCA_IMAGES_RAW,,}" in
  1|true|yes|y|on) USE_PCA_IMAGES="1" ;;
  0|false|no|n|off) USE_PCA_IMAGES="0" ;;
  *)
    echo "ERROR: USE_PCA_IMAGES must be one of 1/0/true/false/yes/no (got '$USE_PCA_IMAGES_RAW')"
    exit 1
    ;;
esac

if [[ "$USE_PCA_IMAGES" == "1" ]]; then
  DEFAULT_JSONL="$REPO_ROOT/outputs/scaffold_rex_5k_pca/all.jsonl"
  EXPECTED_ROWS="${EXPECTED_ROWS:-4945}"
else
  DEFAULT_JSONL="$REPO_ROOT/outputs/scaffold_rex_5k/all.jsonl"
  EXPECTED_ROWS="${EXPECTED_ROWS:-0}"
fi

TRAIN_JSONL="${TRAIN_JSONL:-$DEFAULT_JSONL}"
IMAGE_FIELD="${IMAGE_FIELD:-image}"
PCA_IMAGE_FIELDS="${PCA_IMAGE_FIELDS:-pca_image,image_pca,composite_image,dino_pca_image,image}"
SUBMIT_TO_SLURM="${SUBMIT_TO_SLURM:-1}"

echo "=== Stage15 PCA Safe Submit (Local Debug First) ==="
echo "Python          : $PYTHON_BIN"
echo "TRAIN_JSONL     : $TRAIN_JSONL"
echo "USE_PCA_IMAGES  : $USE_PCA_IMAGES"
echo "EXPECTED_ROWS   : $EXPECTED_ROWS"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "ERROR: TRAIN_JSONL not found: $TRAIN_JSONL"
  echo "Create PCA JSONL first, then re-run this script."
  exit 1
fi

echo "-- Step A: Dependency import check"
"$PYTHON_BIN" - <<'PY'
import PIL, torch, transformers, peft, safetensors
print('deps_ok')
PY

echo "-- Step B: Script syntax checks"
"$PYTHON_BIN" -m py_compile \
  scripts/counting_grpo/train_stage15_sft.py \
  scripts/counting_grpo/check_stage15_pca_preflight.py
bash -n scripts/counting_grpo/launch_stage15_sft_gh200.sh
bash -n scripts/counting_grpo/submit_stage15_sft_gh200.slurm

echo "-- Step C: Dataset preflight (strict)"
"$PYTHON_BIN" scripts/counting_grpo/check_stage15_pca_preflight.py \
  --data_path "$TRAIN_JSONL" \
  --use_pca_images "$USE_PCA_IMAGES" \
  --image_field "$IMAGE_FIELD" \
  --pca_image_fields "$PCA_IMAGE_FIELDS" \
  --expected_rows "$EXPECTED_ROWS" \
  --strict 1

echo "Local debug checks passed."

if [[ "$SUBMIT_TO_SLURM" == "1" ]]; then
  echo "-- Step D: Submitting SLURM job"
  sbatch scripts/counting_grpo/submit_stage15_sft_gh200.slurm
else
  echo "SUBMIT_TO_SLURM=0, skipping submission."
fi
