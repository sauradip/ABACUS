#!/bin/bash
#
# Monitor zero-shot audit performance at key training checkpoints.
# Run this during training to track format convergence (300, 500, 700 steps, etc).
#

REPO_DIR="${REPO_DIR:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/native_sft_stage1}"
FSC_ROOT="${FSC_ROOT:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/fsc147_v4/data/fsc147}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-OpenGVLab/InternVL2-2B}"
AUDIT_ATTN_IMPL="${AUDIT_ATTN_IMPL:-eager}"

# Checkpoints to audit
MILESTONES=(100 200 300 400 500 700 1000 1500 2000)

export HF_HUB_OFFLINE=1
export PYTHONWARNINGS="ignore::FutureWarning"

cd "$REPO_DIR"

echo "=== Zero-Shot Format Convergence Monitor ==="
echo "Output dir: $OUTPUT_DIR"
echo "FSC root: $FSC_ROOT"
echo ""

for step in "${MILESTONES[@]}"; do
  CKPT_PATH="$OUTPUT_DIR/checkpoint-$step"
  
  if [[ ! -d "$CKPT_PATH" ]]; then
    echo "[SKIP] checkpoint-$step not found (training still running?)"
    continue
  fi
  
  AUDIT_OUTPUT="$OUTPUT_DIR/audit_step_${step}.json"
  
  echo "[AUDIT] Running audit on checkpoint-$step..."
  python3 scripts/counting_grpo/zero_shot_point_audit.py \
    --model_path "$CKPT_PATH" \
    --tokenizer_path "$BASE_MODEL_PATH" \
    --fsc_root "$FSC_ROOT" \
    --num_samples 5 \
    --seed 42 \
    --torch_dtype float16 \
    --attn_implementation "$AUDIT_ATTN_IMPL" \
    --output_path "$AUDIT_OUTPUT" \
    --max_new_tokens 256
  
  # Extract key metrics from JSON
  if [[ -f "$AUDIT_OUTPUT" ]]; then
    CHAMFER=$(python3 -c "import json; d=json.load(open('$AUDIT_OUTPUT')); print(d.get('chamfer_mean', 'N/A'))" 2>/dev/null)
    BOUNDS=$(python3 -c "import json; d=json.load(open('$AUDIT_OUTPUT')); print('PASS' if d.get('bounds_pass') else 'FAIL')" 2>/dev/null)
    DIVERSITY=$(python3 -c "import json; d=json.load(open('$AUDIT_OUTPUT')); print('PASS' if d.get('diversity_pass') else 'FAIL')" 2>/dev/null)
    
    echo "  ✓ Step $step: bounds=$BOUNDS  diversity=$DIVERSITY  chamfer=$CHAMFER"
    echo "  Output: $AUDIT_OUTPUT"
  fi
  echo ""
done

echo "=== Format Convergence Monitor Complete ==="
echo ""
echo "To track format emergence, check if scaffold tags appear in audit outputs:"
echo "  grep -l 'scaffold' $OUTPUT_DIR/audit_step_*.json | head -3"
