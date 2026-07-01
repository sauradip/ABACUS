#!/usr/bin/env bash
# Stage 1.6b low-count scaffold SFT wrapper for the reproducible pivot.

set -euo pipefail

if [[ -n "${REPO_DIR:-}" ]]; then
  REPO_ROOT="$REPO_DIR"
elif [[ -f "scripts/counting_grpo/launch_stage15_sft_gh200.sh" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

DATA_PATH="outputs/scaffold_rex_5k_pca/sft_low_30_scaffold.jsonl"
OUTPUT_DIR="checkpoints/scaffold_rex_stage16b_low30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --stage1_checkpoint)
      STAGE1_CKPT="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export TRAIN_JSONL="$DATA_PATH"
export STAGE15_OUT_DIR="$OUTPUT_DIR"
export DATA_DIR="$(dirname "$DATA_PATH")"
export SKIP_DATA_GEN=1
export SMOKE_CHECK=0
export USE_PCA_IMAGES=1
export MAX_LENGTH="${MAX_LENGTH:-8192}"
export EXPECTED_ROWS="${EXPECTED_ROWS:-0}"
export PRECHECK_BEFORE_TRAIN="${PRECHECK_BEFORE_TRAIN:-1}"
export REQUIRE_UNIQUE_OUTPUT_DIR="${REQUIRE_UNIQUE_OUTPUT_DIR:-1}"
export ATTN_IMPL="${ATTN_IMPL:-eager}"
export BASE_MODEL="${BASE_MODEL:-OpenGVLab/InternVL2-2B}"
export STAGE1_CKPT="${STAGE1_CKPT:-$REPO_ROOT/checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"

exec bash scripts/counting_grpo/launch_stage15_sft_gh200.sh
