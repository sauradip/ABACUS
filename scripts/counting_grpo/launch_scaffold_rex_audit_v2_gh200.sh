#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv311/bin/python" ]]; then
  PYTHON_BIN=".venv311/bin/python"
else
  PYTHON_BIN="python3"
fi

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/scaffold_rex_stage15_4342903}"
SCAFFOLD_JSONL="${SCAFFOLD_JSONL:-$REPO_ROOT/outputs/scaffold_rex_5k/all.jsonl}"
AUDIT_JSON="${AUDIT_JSON:-$MODEL_PATH/scaffold_rex_audit_v2.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
PRINT_SAMPLES="${PRINT_SAMPLES:-3}"
START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MIN_DYNAMIC_PATCH="${MIN_DYNAMIC_PATCH:-2}"
MAX_DYNAMIC_PATCH="${MAX_DYNAMIC_PATCH:-12}"
FORCE_MANUAL_TILING="${FORCE_MANUAL_TILING:-1}"
FORCE_SCAFFOLD_OVERLAY="${FORCE_SCAFFOLD_OVERLAY:-1}"
DOT_RADIUS="${DOT_RADIUS:-4}"
DEBUG_PIXEL_SHAPE="${DEBUG_PIXEL_SHAPE:-1}"

mkdir -p "$(dirname "$AUDIT_JSON")" logs

echo "=== Scaffold-Rex Self-Audit V2 ==="
echo "Python        : $PYTHON_BIN"
echo "Model path    : $MODEL_PATH"
echo "Scaffold jsonl: $SCAFFOLD_JSONL"
echo "Output json   : $AUDIT_JSON"
echo "Max samples   : $MAX_SAMPLES (0 means full dataset)"
echo ""

"$PYTHON_BIN" scripts/counting_grpo/generate_scaffold_rex_audit_v2.py \
  --model_path "$MODEL_PATH" \
  --scaffold_jsonl "$SCAFFOLD_JSONL" \
  --output_json "$AUDIT_JSON" \
  --attn_implementation "$ATTN_IMPL" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --start_index "$START_INDEX" \
  --print_samples "$PRINT_SAMPLES" \
  --min_dynamic_patch "$MIN_DYNAMIC_PATCH" \
  --max_dynamic_patch "$MAX_DYNAMIC_PATCH" \
  --force_manual_tiling "$FORCE_MANUAL_TILING" \
  --force_scaffold_overlay "$FORCE_SCAFFOLD_OVERLAY" \
  --dot_radius "$DOT_RADIUS" \
  --debug_pixel_shape "$DEBUG_PIXEL_SHAPE"

echo ""
echo "Audit generation complete: $AUDIT_JSON"
