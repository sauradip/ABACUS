#!/bin/bash
#SBATCH --job-name=stage15_scaffold_rex
#SBATCH --output=logs/stage15_scaffold_rex_%j.log
#SBATCH --error=logs/stage15_scaffold_rex_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
# ── GH200 note ────────────────────────────────────────────────────────────────
# Override --partition and --gres to target GH200 queue on your cluster, e.g.:
#   sbatch --partition=gh200q --gres=gpu:gh200:1 submit_stage15_sft_gh200.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"
mkdir -p logs

# ── Python env ────────────────────────────────────────────────────────────────
if [[ -x ".venv311/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
fi
export PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv311/bin/python}"

# ── Torch cache (writable project path avoids compute-node tmpdir issues) ─────
_CACHE_BASE="$REPO_DIR/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_BASE/inductor"
export TORCH_COMPILE_CACHE_DIR="$_CACHE_BASE/dynamo"
export XDG_CACHE_HOME="$_CACHE_BASE/xdg"
export TMPDIR="$_CACHE_BASE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR" \
          "$XDG_CACHE_HOME" "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Dataset + checkpoint paths ────────────────────────────────────────────────
export FSC_ROOT="${FSC_ROOT:-/projects/u6fb/myprojects/FSC147_hf}"
export DATA_DIR="${DATA_DIR:-$REPO_DIR/outputs/scaffold_rex_5k_pca}"
export TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/all.jsonl}"
export STAGE1_CKPT="${STAGE1_CKPT:-$REPO_DIR/checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"
export STAGE15_OUT_DIR="${STAGE15_OUT_DIR:-$REPO_DIR/checkpoints/scaffold_rex_stage15_pca_${SLURM_JOB_ID}}"

# ── Auto-load OOM probe results if the interactive debug was run first ─────────
#   interactive_debug_stage15.sh writes oom_probe_result.env; source it here
#   so validated safe limits override the defaults below.
_OOM_ENV="$DATA_DIR/oom_probe_result.env"
if [[ -f "$_OOM_ENV" ]]; then
  echo "Sourcing OOM probe results from $_OOM_ENV"
  # shellcheck disable=SC1090
  source "$_OOM_ENV"
  # Map probe variable names → launch variable names (only if not already overridden).
  MAX_LENGTH="${MAX_LENGTH:-${SAFE_MAX_LENGTH:-16384}}"
  BATCH_SIZE="${BATCH_SIZE:-${SAFE_BATCH_SIZE:-1}}"
  GRAD_ACCUM="${GRAD_ACCUM:-${SAFE_GRAD_ACCUM:-8}}"
fi

# ── Training knobs (GH200 defaults; OOM probe above overrides if available) ───
# GH200 HBM3 target: 16k context without gradient checkpointing.
export TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export LR="${LR:-2e-5}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export VISION_SCALE="${VISION_SCALE:-1.0}"
export MIN_DYNAMIC_PATCH="${MIN_DYNAMIC_PATCH:-2}"
export MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-12}"
export FORCE_MANUAL_TILING="${FORCE_MANUAL_TILING:-1}"
export DEBUG_PATCH_SHAPES="${DEBUG_PATCH_SHAPES:-1}"
export ATTN_IMPL="${ATTN_IMPL:-eager}"
export SKIP_DATA_GEN="${SKIP_DATA_GEN:-1}"   # set to 1 to reuse preprocessed PCA data
export SMOKE_CHECK="${SMOKE_CHECK:-0}"        # already done interactively
export SMOKE_N="${SMOKE_N:-10}"
export REPORT_TO="${REPORT_TO:-none}"
export USE_PCA_IMAGES="${USE_PCA_IMAGES:-1}"
export IMAGE_FIELD="${IMAGE_FIELD:-image}"
export PCA_IMAGE_FIELDS="${PCA_IMAGE_FIELDS:-pca_image,image_pca,composite_image,dino_pca_image,image}"
export FAIL_ON_MISSING_IMAGE="${FAIL_ON_MISSING_IMAGE:-1}"
export REQUIRE_UNIQUE_OUTPUT_DIR="${REQUIRE_UNIQUE_OUTPUT_DIR:-1}"

echo "=== SCAFFOLD-Rex Stage 1.5 SFT — GH200 ==="
echo "SLURM_JOB_ID    : ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST  : ${SLURM_NODELIST:-pending}"
echo "Python          : $PYTHON_BIN"
echo "FSC root        : $FSC_ROOT"
echo "Data dir        : $DATA_DIR"
echo "Train JSONL     : $TRAIN_JSONL"
echo "Stage-1 ckpt    : $STAGE1_CKPT"
echo "Stage-1.5 out   : $STAGE15_OUT_DIR"
echo "Total samples   : $TOTAL_SAMPLES"
echo "Context length  : $MAX_LENGTH"
echo "LR              : $LR  |  Batch/accum: $BATCH_SIZE/$GRAD_ACCUM"
echo "LoRA            : rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "Vision scale    : $VISION_SCALE"
echo "Patch range     : $MIN_DYNAMIC_PATCH-$MAX_DYNAMIC_PATCH (manual tiling=$FORCE_MANUAL_TILING)"
echo "PCA mode        : $USE_PCA_IMAGES"
echo "PCA fields      : $PCA_IMAGE_FIELDS"
echo "Unique out dir  : $REQUIRE_UNIQUE_OUTPUT_DIR"
echo ""

bash scripts/counting_grpo/launch_stage15_sft_gh200.sh

echo ""
echo "=== Stage 1.5 complete. Checkpoint at: $STAGE15_OUT_DIR ==="
echo ""
echo "To continue to Stage 2.5 RankDPO, run:"
echo "  STAGE1_CKPT=$STAGE15_OUT_DIR bash scripts/counting_grpo/launch_rankdpo_stage25_gh200.sh"
