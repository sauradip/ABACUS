#!/usr/bin/env bash
# Stage 1.5 SCAFFOLD-Rex Calibration SFT — GH200 launch script.
#
# Usage:
#   bash scripts/counting_grpo/launch_stage15_sft_gh200.sh
#
# Environment overrides:
#   FSC_ROOT         — path to FSC147 dataset root (required)
#   STAGE1_CKPT      — path to Stage-1 checkpoint dir (required)
#   STAGE15_OUT_DIR  — output directory for Stage 1.5 checkpoint
#   DATA_DIR         — where SCAFFOLD-Rex JSONL + overlaid images are written
#   TRAIN_JSONL      — explicit JSONL path (defaults to $DATA_DIR/all.jsonl)
#   USE_PCA_IMAGES   — set to 1 to read PCA/composite image fields
#   PCA_IMAGE_FIELDS — comma-separated image field fallback order for PCA mode
#   PRECHECK_BEFORE_TRAIN — run strict dataset preflight before training (default 1)
#   EXPECTED_ROWS    — expected row count for preflight (default 4945 in PCA mode)
#   SKIP_DATA_GEN    — set to 1 to reuse an existing DATA_DIR
#   SMOKE_CHECK      — set to 1 to run OCR-visibility check on 10 images before full gen
#   TOTAL_SAMPLES    — total training samples in all.jsonl; default 5000
#   PYTHON_BIN       — python interpreter

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Resolve Python ──────────────────────────────────────────────────────────
if [[ -x ".venv311/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# ── Dataset location ─────────────────────────────────────────────────────────
FSC_ROOT="${FSC_ROOT:-/projects/u6fb/myprojects/FSC147_hf}"
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
  DEFAULT_DATA_DIR="$REPO_ROOT/outputs/scaffold_rex_5k_pca"
else
  DEFAULT_DATA_DIR="$REPO_ROOT/outputs/scaffold_rex_5k"
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/all.jsonl}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"

# ── Checkpoint paths ─────────────────────────────────────────────────────────
# Stage 1: the SFT checkpoint whose LM tensors are skeleton-injected into base.
STAGE1_CKPT="${STAGE1_CKPT:-$REPO_ROOT/checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"
# Stage 1.5 output.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "$USE_PCA_IMAGES" == "1" ]]; then
  DEFAULT_STAGE15_OUT="$REPO_ROOT/checkpoints/scaffold_rex_stage15_pca_${RUN_TAG}"
else
  DEFAULT_STAGE15_OUT="$REPO_ROOT/checkpoints/scaffold_rex_stage15_${RUN_TAG}"
fi
STAGE15_OUT_DIR="${STAGE15_OUT_DIR:-$DEFAULT_STAGE15_OUT}"

# ── Launch knobs ─────────────────────────────────────────────────────────────
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"
SMOKE_CHECK="${SMOKE_CHECK:-1}"
SMOKE_N="${SMOKE_N:-10}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
VISION_SCALE="${VISION_SCALE:-1.0}"
MIN_DYNAMIC_PATCH="${MIN_DYNAMIC_PATCH:-2}"
MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-12}"
FORCE_MANUAL_TILING="${FORCE_MANUAL_TILING:-1}"
DEBUG_PATCH_SHAPES="${DEBUG_PATCH_SHAPES:-1}"
BASE_MODEL="${BASE_MODEL:-OpenGVLab/InternVL2-2B}"
REPORT_TO="${REPORT_TO:-none}"
IMAGE_FIELD="${IMAGE_FIELD:-image}"
PCA_IMAGE_FIELDS="${PCA_IMAGE_FIELDS:-pca_image,image_pca,composite_image,dino_pca_image,image}"
FAIL_ON_MISSING_IMAGE="${FAIL_ON_MISSING_IMAGE:-1}"
REQUIRE_UNIQUE_OUTPUT_DIR="${REQUIRE_UNIQUE_OUTPUT_DIR:-1}"
PRECHECK_BEFORE_TRAIN="${PRECHECK_BEFORE_TRAIN:-1}"
if [[ "$USE_PCA_IMAGES" == "1" ]]; then
  EXPECTED_ROWS="${EXPECTED_ROWS:-4945}"
else
  EXPECTED_ROWS="${EXPECTED_ROWS:-0}"
fi

mkdir -p "$DATA_DIR" "$STAGE15_OUT_DIR" logs

echo "================================================================"
echo "  Stage 1.5 SCAFFOLD-Rex Calibration SFT — GH200"
echo "================================================================"
echo "  Python          : $PYTHON_BIN"
echo "  FSC root        : $FSC_ROOT"
echo "  Data dir        : $DATA_DIR"
echo "  Train JSONL     : $TRAIN_JSONL"
echo "  Stage-1 ckpt    : $STAGE1_CKPT"
echo "  Stage-1.5 out   : $STAGE15_OUT_DIR"
echo "  Total samples   : $TOTAL_SAMPLES"
echo "  Context length  : $MAX_LENGTH"
echo "  LR              : $LR"
echo "  Batch/accum     : $BATCH_SIZE / $GRAD_ACCUM"
echo "  LoRA rank/alpha : $LORA_RANK / $LORA_ALPHA"
echo "  Vision scale    : $VISION_SCALE"
echo "  Patch range     : $MIN_DYNAMIC_PATCH-$MAX_DYNAMIC_PATCH (manual tiling=$FORCE_MANUAL_TILING)"
echo "  PCA mode        : $USE_PCA_IMAGES"
echo "  Image field     : $IMAGE_FIELD"
echo "  PCA fields      : $PCA_IMAGE_FIELDS"
echo "  Strict image IO : $FAIL_ON_MISSING_IMAGE"
echo "  Unique out dir  : $REQUIRE_UNIQUE_OUTPUT_DIR"
echo "  Precheck        : $PRECHECK_BEFORE_TRAIN"
echo "  Expected rows   : $EXPECTED_ROWS"
echo "================================================================"

# ── Guard: FSC root must exist ────────────────────────────────────────────────
if [[ ! -d "$FSC_ROOT" ]]; then
  echo "ERROR: FSC_ROOT not found: $FSC_ROOT"
  echo "  Set FSC_ROOT=/path/to/FSC147_hf and re-run."
  exit 1
fi

# ── Guard: Stage-1 checkpoint must have model.safetensors ────────────────────
if [[ ! -f "$STAGE1_CKPT/model.safetensors" ]]; then
  # Auto-resolve to latest checkpoint-NNN/model.safetensors in the same directory.
  PARENT="$(dirname "$STAGE1_CKPT")"
  RESOLVED=$(find "$PARENT" -maxdepth 2 -name "model.safetensors" \
              | awk -F'/' '{for(i=1;i<=NF;i++) if($i~/^checkpoint-[0-9]+$/) print $i"/"$(i+1)}' \
              | sort -t'-' -k2 -n | tail -1)
  if [[ -n "$RESOLVED" ]]; then
    STAGE1_CKPT="$PARENT/$RESOLVED"
    STAGE1_CKPT="$(dirname "$STAGE1_CKPT")"
    echo "WARN: Auto-resolved Stage-1 checkpoint to $STAGE1_CKPT"
  else
    echo "ERROR: No model.safetensors found under $(dirname "$STAGE1_CKPT")."
    echo "  Check STAGE1_CKPT path."
    exit 1
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — OCR Visibility Smoke Check (10 images, binary-dot stats)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SMOKE_CHECK" == "1" ]]; then
  echo ""
  echo "── Step 1: OCR Visibility Smoke Check ($SMOKE_N images) ──"
  SMOKE_DIR="$DATA_DIR/smoke_check"
  "$PYTHON_BIN" scripts/counting_grpo/prepare_scaffold_rex_data.py \
    --fsc_root "$FSC_ROOT" \
    --output_dir "$SMOKE_DIR" \
    --overlay_dir "$SMOKE_DIR/images_overlay" \
    --splits "train" \
    --smoke_n "$SMOKE_N"

  echo ""
  echo "  Overlay images written to: $SMOKE_DIR/images_overlay"
  echo "  Inspect 2-3 manually to confirm dot visibility before continuing."
  echo "  (Proceeding automatically after check; set SMOKE_CHECK=0 to skip.)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Generate 5,000-sample SCAFFOLD-Rex JSONL
#
# Sample math:
#   FSC147 train = 3,659  |  val = 1,286  |  total train+val = 4,945
#   We draw train fully first, then fill from val up to TOTAL_SAMPLES.
#   Result: 4,945 samples when TOTAL_SAMPLES=5000 (practically the 5k sweet-spot).
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DATA_GEN" != "1" ]]; then
  echo ""
  echo "── Step 2: Generating $TOTAL_SAMPLES SCAFFOLD-Rex samples (train → val fill-up) ──"
  "$PYTHON_BIN" scripts/counting_grpo/prepare_scaffold_rex_data.py \
    --fsc_root "$FSC_ROOT" \
    --output_dir "$DATA_DIR" \
    --splits "train,val" \
    --total_cap "$TOTAL_SAMPLES"

  ACTUAL=$(wc -l < "$TRAIN_JSONL")
  echo "  Generated $ACTUAL records → $TRAIN_JSONL"
else
  echo ""
  echo "── Step 2: SKIP_DATA_GEN=1 — reusing $TRAIN_JSONL ──"
  if [[ ! -f "$TRAIN_JSONL" ]]; then
    echo "ERROR: $TRAIN_JSONL not found. Run with SKIP_DATA_GEN=0 first."
    exit 1
  fi
  ACTUAL=$(wc -l < "$TRAIN_JSONL")
  echo "  Found $ACTUAL records in existing JSONL."
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "ERROR: train JSONL not found: $TRAIN_JSONL"
  echo "  For PCA-SFT, preprocess data first and set TRAIN_JSONL to that file."
  exit 1
fi

if [[ "$ACTUAL" -lt 100 ]]; then
  echo "ERROR: Too few training samples ($ACTUAL < 100). Check FSC_ROOT and data generation."
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2.5 — PCA/RGB data preflight check
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$PRECHECK_BEFORE_TRAIN" == "1" ]]; then
  echo ""
  echo "── Step 2.5: Dataset preflight (paths + readability) ──"
  "$PYTHON_BIN" scripts/counting_grpo/check_stage15_pca_preflight.py \
    --data_path "$TRAIN_JSONL" \
    --use_pca_images "$USE_PCA_IMAGES" \
    --image_field "$IMAGE_FIELD" \
    --pca_image_fields "$PCA_IMAGE_FIELDS" \
    --expected_rows "$EXPECTED_ROWS" \
    --strict 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Run Stage 1.5 Calibration SFT
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 3: Stage 1.5 Calibration SFT ($ACTUAL samples, 1 epoch) ──"
echo "  Watch loss in $REPO_ROOT/logs/ — sane range: 1.5 – 3.5"

"$PYTHON_BIN" scripts/counting_grpo/train_stage15_sft.py \
  --data_path "$TRAIN_JSONL" \
  --output_dir "$STAGE15_OUT_DIR" \
  --base_model "$BASE_MODEL" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --attn_implementation "$ATTN_IMPL" \
  --model_max_length "$MAX_LENGTH" \
  --learning_rate "$LR" \
  --num_train_epochs 1.0 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --vision_scale "$VISION_SCALE" \
  --min_dynamic_patch "$MIN_DYNAMIC_PATCH" \
  --max_dynamic_patch "$MAX_DYNAMIC_PATCH" \
  --force_manual_tiling "$FORCE_MANUAL_TILING" \
  --debug_patch_shapes "$DEBUG_PATCH_SHAPES" \
  --image_field "$IMAGE_FIELD" \
  --pca_image_fields "$PCA_IMAGE_FIELDS" \
  --use_pca_images "$USE_PCA_IMAGES" \
  --fail_on_missing_image "$FAIL_ON_MISSING_IMAGE" \
  --require_unique_output_dir "$REQUIRE_UNIQUE_OUTPUT_DIR" \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 3 \
  --report_to "$REPORT_TO"

echo ""
echo "================================================================"
echo "  Stage 1.5 complete → $STAGE15_OUT_DIR"
echo "  Next: run submit_rankdpo_stage25.slurm with"
echo "    STAGE1_CKPT=$STAGE15_OUT_DIR"
echo "    DATA_JSONL=outputs/rankdpo_rex/preference_pairs.jsonl"
echo "================================================================"
