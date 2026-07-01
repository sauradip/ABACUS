#!/usr/bin/env bash
# Interactive debug + OOM-probe workflow for Stage 1.5 SCAFFOLD-Rex.
#
# Run this INSIDE an interactive allocation:
#   salloc --gres=gpu:1 --partition=workq --time=01:00:00 \
#          --cpus-per-task=8 --mem=80G
#   bash scripts/counting_grpo/interactive_debug_stage15.sh
#
# The script walks through four escalating phases:
#   Phase 0 – Pre-flight (imports, checkpoint, FSC image check)
#   Phase 1 – 2-step smoke (max_steps=2, short context=2048)
#   Phase 2 – OOM probe   (walks context ladder up to 16384)
#   Phase 3 – Source result → print final sbatch command

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Python ─────────────────────────────────────────────────────────────────
if [[ -x ".venv311/bin/python" ]]; then
  PYTHON_BIN=".venv311/bin/python"
else
  PYTHON_BIN="python3"
fi

# ── Paths ──────────────────────────────────────────────────────────────────
FSC_ROOT="${FSC_ROOT:-/projects/u6fb/myprojects/FSC147_hf}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/outputs/scaffold_rex_5k}"
TRAIN_JSONL="$DATA_DIR/all.jsonl"
STAGE1_CKPT="${STAGE1_CKPT:-$REPO_ROOT/checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"
SMOKE_OUT="${SMOKE_OUT:-$REPO_ROOT/checkpoints/stage15_interactive_smoke}"
BASE_MODEL="${BASE_MODEL:-OpenGVLab/InternVL2-2B}"
VISION_SCALE="${VISION_SCALE:-0.1}"

# ── Minimal smoke parameters ────────────────────────────────────────────────
SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-2048}"
SMOKE_BATCH="${SMOKE_BATCH:-1}"
SMOKE_GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"

LOG_FILE="$REPO_ROOT/logs/stage15_interactive_debug.log"
mkdir -p "$(dirname "$LOG_FILE")" "$SMOKE_OUT" "$DATA_DIR"

header() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════"; }
pass()   { echo "  ✔  $*"; }
fail()   { echo "  ✘  $*"; exit 1; }
info()   { echo "  ▶  $*"; }

# Tee everything to log
exec > >(tee -a "$LOG_FILE") 2>&1

header "Phase 0 — Pre-flight checks"

# GPU present?
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader &>/dev/null; then
  fail "No GPU visible. Run inside an salloc allocation (see top of this script)."
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pass "GPU detected"

# Python + torch
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"
pass "torch CUDA OK"

# Stage-1 checkpoint
if [[ ! -f "$STAGE1_CKPT/model.safetensors" ]]; then
  fail "Stage-1 checkpoint missing: $STAGE1_CKPT/model.safetensors"
fi
pass "Stage-1 checkpoint: $STAGE1_CKPT"

# FSC images
FSC_IMG_COUNT=$(ls "$FSC_ROOT/images_384_VarV2/" 2>/dev/null | wc -l)
if [[ "$FSC_IMG_COUNT" -lt 100 ]]; then
  fail "Too few FSC images ($FSC_IMG_COUNT) in $FSC_ROOT/images_384_VarV2"
fi
pass "FSC images: $FSC_IMG_COUNT"

# Generate a tiny 10-sample scaffold batch if all.jsonl doesn't exist yet
if [[ ! -f "$TRAIN_JSONL" ]]; then
  info "Generating 10-sample smoke JSONL for debug (SKIP_DATA_GEN=1 to use existing) …"
  "$PYTHON_BIN" scripts/counting_grpo/prepare_scaffold_rex_data.py \
    --fsc_root "$FSC_ROOT" \
    --output_dir "$DATA_DIR" \
    --splits "train" \
    --total_cap 10 \
    --smoke_n 10
  pass "Smoke JSONL written: $TRAIN_JSONL"
else
  N=$(wc -l < "$TRAIN_JSONL")
  pass "JSONL exists: $TRAIN_JSONL ($N records)"
fi

# ═══════════════════════════════════════════════════════════════════════════
header "Phase 1 — 2-step smoke (max_steps=2, context=$SMOKE_MAX_LENGTH)"
# ═══════════════════════════════════════════════════════════════════════════
info "Running 2 optimizer steps to confirm finite loss + no crash …"

"$PYTHON_BIN" scripts/counting_grpo/train_stage15_sft.py \
  --data_path "$TRAIN_JSONL" \
  --output_dir "$SMOKE_OUT" \
  --base_model "$BASE_MODEL" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --attn_implementation eager \
  --model_max_length "$SMOKE_MAX_LENGTH" \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --per_device_train_batch_size "$SMOKE_BATCH" \
  --gradient_accumulation_steps "$SMOKE_GRAD_ACCUM" \
  --lora_rank 64 \
  --lora_alpha 128 \
  --vision_scale "$VISION_SCALE" \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --max_steps 2 2>&1 | tee -a "$LOG_FILE"

# Validate loss was finite
LAST_LOSS=$(grep -oP "loss.*?(\d+\.\d+)" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP "[\d.]+$" || echo "unknown")
info "Last logged loss: $LAST_LOSS"
if echo "$LAST_LOSS" | python3 -c "import sys,math; v=float(sys.stdin.read().strip()); sys.exit(0 if math.isfinite(v) and 0.0 < v < 15.0 else 1)" 2>/dev/null; then
  pass "Loss is finite and in plausible range"
elif [[ "$LAST_LOSS" == "unknown" ]]; then
  info "Could not parse loss from log; check $LOG_FILE manually"
else
  fail "Loss is outside sane range (got $LAST_LOSS). Check vision scale / injection."
fi

# ═══════════════════════════════════════════════════════════════════════════
header "Phase 2 — OOM Probe (walks context from 2k to 16k)"
# ═══════════════════════════════════════════════════════════════════════════
info "This will deliberately trigger OOM at the top end — that is expected."
info "The probe will report the last SAFE configuration."

"$PYTHON_BIN" scripts/counting_grpo/probe_oom_stage15.py \
  --base_model "$BASE_MODEL" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --sample_jsonl "$TRAIN_JSONL" \
  --vision_scale "$VISION_SCALE" \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_effective_batch 16 2>&1 | tee -a "$LOG_FILE"

# ═══════════════════════════════════════════════════════════════════════════
header "Phase 3 — Load OOM result + print sbatch command"
# ═══════════════════════════════════════════════════════════════════════════
OOM_RESULT="$DATA_DIR/oom_probe_result.env"
if [[ -f "$OOM_RESULT" ]]; then
  source "$OOM_RESULT"
  pass "OOM result loaded from $OOM_RESULT"
  info "Safe context : $SAFE_MAX_LENGTH"
  info "Safe batch   : $SAFE_BATCH_SIZE"
  info "Safe grad acc: $SAFE_GRAD_ACCUM"
else
  info "OOM result file not found; using conservative defaults."
  SAFE_MAX_LENGTH=8192
  SAFE_BATCH_SIZE=1
  SAFE_GRAD_ACCUM=16
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  VALIDATED sbatch command — copy-paste to submit:               ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║"
printf "║  MAX_LENGTH=%s \\\\\n" "$SAFE_MAX_LENGTH"
printf "║  BATCH_SIZE=%s \\\\\n" "$SAFE_BATCH_SIZE"
printf "║  GRAD_ACCUM=%s \\\\\n" "$SAFE_GRAD_ACCUM"
printf "║  STAGE1_CKPT=%s \\\\\n" "$STAGE1_CKPT"
printf "║  FSC_ROOT=%s \\\\\n" "$FSC_ROOT"
echo "║  sbatch scripts/counting_grpo/submit_stage15_sft_gh200.slurm"
echo "║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Full debug log: $LOG_FILE"
