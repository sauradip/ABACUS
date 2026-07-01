#!/bin/bash
#SBATCH --job-name=hf_multi_img_count_sft
#SBATCH --output=logs/hf_multi_img_count_sft_%j.log
#SBATCH --error=logs/hf_multi_img_count_sft_%j.err
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=120G

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"
mkdir -p logs

# Clean-slate defaults. Override UNILIP_BASE_MODEL with the local vanilla
# UniLIP-3B checkpoint or the canonical HF id after it is available.
export UNILIP_BASE_MODEL="${UNILIP_BASE_MODEL:-$REPO_DIR/UniLIP/UniLIP-3B}"
export SFT_DATA_PATH="${SFT_DATA_PATH:-$REPO_DIR/outputs/adaptive_hf_multi_image_count_sft_fsc147_train/train_messages.jsonl}"
export SFT_EVAL_DATA_PATH="${SFT_EVAL_DATA_PATH:-}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$REPO_DIR/checkpoints/hf_multi_image_count_sft_fsc147_train}"

export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export ALLOW_ATTN_FALLBACK="${ALLOW_ATTN_FALLBACK:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-${SLURM_GPUS_ON_NODE:-4}}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
export METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
export GREATER_IS_BETTER="${GREATER_IS_BETTER:-false}"
export PYTHONDONTWRITEBYTECODE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

export CONTAINER_PATH="${CONTAINER_PATH:-$REPO_DIR/.cache/images/nvcr_pytorch_24.08-py3_arm64_sandbox}"
export CONTAINER_PYTHONUSERBASE="${CONTAINER_PYTHONUSERBASE:-$REPO_DIR/.cache/container_py310}"
export USE_APPTAINER="${USE_APPTAINER:-auto}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/.cache/triton}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.cache/tmp}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

# Safety: submitting this file should perform only the dry run unless the caller
# explicitly flips both switches.
export DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"
export CONFIRM_FULL_RUN="${CONFIRM_FULL_RUN:-0}"

case "${UNILIP_BASE_MODEL,,}" in
  *scaffold_rex*|*stage15*|*stage1.5*|*stage16*|*rankdpo*|*grpo*|*native_sft*|*internvl*)
    echo "ERROR: refusing legacy/contaminated model path: $UNILIP_BASE_MODEL" >&2
    exit 2
    ;;
esac
case "${SFT_OUTPUT_DIR,,}" in
  *grpo*)
    echo "ERROR: refusing to write SFT output into a GRPO-marked path: $SFT_OUTPUT_DIR" >&2
    exit 2
    ;;
esac

echo "=== HF Multi-Image Count SFT ==="
echo "repo          : $REPO_DIR"
echo "model         : $UNILIP_BASE_MODEL"
echo "data          : $SFT_DATA_PATH"
echo "eval data     : ${SFT_EVAL_DATA_PATH:-<none>}"
echo "output        : $SFT_OUTPUT_DIR"
echo "attn          : $ATTN_IMPL (fallback=$ALLOW_ATTN_FALLBACK)"
echo "processes     : $NUM_PROCESSES"
echo "batch/accum   : $PER_DEVICE_TRAIN_BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS"
echo "epochs        : $NUM_TRAIN_EPOCHS"
echo "resume        : ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "save/eval     : $SAVE_STRATEGY / $EVAL_STRATEGY"
echo "best model    : $LOAD_BEST_MODEL_AT_END ($METRIC_FOR_BEST_MODEL, greater=$GREATER_IS_BETTER)"
echo "dry_run_only  : $DRY_RUN_ONLY"
echo "confirm_full  : $CONFIRM_FULL_RUN"

if [[ ! -f "$SFT_DATA_PATH" ]]; then
  echo "ERROR: missing SFT data: $SFT_DATA_PATH" >&2
  echo "Run scripts/counting_grpo/build_hf_multi_image_count_sft.py first." >&2
  exit 2
fi
if [[ -n "$SFT_EVAL_DATA_PATH" && ! -f "$SFT_EVAL_DATA_PATH" ]]; then
  echo "ERROR: missing SFT eval data: $SFT_EVAL_DATA_PATH" >&2
  exit 2
fi

if [[ -x "$REPO_DIR/.venv311/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv311/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

host_has_flash_attn() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn") is not None else 1)
PY
}

if [[ "$USE_APPTAINER" == "auto" ]]; then
  if [[ "$ATTN_IMPL" == "flash_attention_2" ]] && ! host_has_flash_attn && [[ -e "$CONTAINER_PATH" ]]; then
    USE_APPTAINER=1
  else
    USE_APPTAINER=0
  fi
fi

echo "python        : $PYTHON_BIN"
echo "apptainer     : $USE_APPTAINER"
if [[ "$USE_APPTAINER" == "1" ]]; then
  echo "container     : $CONTAINER_PATH"
  echo "container py  : $CONTAINER_PYTHONUSERBASE"
fi

run_python() {
  if [[ "$USE_APPTAINER" == "1" ]]; then
    if [[ ! -e "$CONTAINER_PATH" ]]; then
      echo "ERROR: requested Apptainer runtime but container is missing: $CONTAINER_PATH" >&2
      exit 2
    fi
    apptainer exec --nv \
      --bind /projects:/projects \
      --bind /lus:/lus \
      --bind /home:/home \
      --env PYTHONUSERBASE="$CONTAINER_PYTHONUSERBASE" \
      --env PATH="$CONTAINER_PYTHONUSERBASE/bin:$PATH" \
      --env HF_HOME="$HF_HOME" \
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      --env TMPDIR="$TMPDIR" \
      --env PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" \
      --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="$PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" \
      "$CONTAINER_PATH" \
      bash -lc 'cd "$0"; export PATH="$PYTHONUSERBASE/bin:$PATH"; export PYTHONPATH="$PWD:$PWD/UniLIP:$PYTHONUSERBASE/lib/python3.10/site-packages:${PYTHONPATH:-}"; exec python3 "$@"' \
      "$REPO_DIR" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

if [[ "$ATTN_IMPL" == "flash_attention_2" ]]; then
  run_python - <<'PY'
import importlib.util
if importlib.util.find_spec("flash_attn") is None:
    raise SystemExit("ERROR: flash_attn missing in selected runtime")
import flash_attn
print(f"flash_attn    : {getattr(flash_attn, '__version__', 'unknown')}")
PY
fi

EVAL_ARGS=()
if [[ -n "$SFT_EVAL_DATA_PATH" ]]; then
  EVAL_ARGS=(--eval_data_path "$SFT_EVAL_DATA_PATH")
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
  run_python scripts/counting_grpo/train_hf_multi_image_count_sft.py \
    --model_name_or_path "$UNILIP_BASE_MODEL" \
    --data_path "$SFT_DATA_PATH" \
    "${EVAL_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --validate_json_only
  echo "Dry-run-only mode complete. Set DRY_RUN_ONLY=0 CONFIRM_FULL_RUN=1 for training."
  exit 0
fi

if [[ "$CONFIRM_FULL_RUN" != "1" ]]; then
  echo "ERROR: refusing full training without CONFIRM_FULL_RUN=1" >&2
  exit 2
fi

if [[ ! -d "$UNILIP_BASE_MODEL" && "$UNILIP_BASE_MODEL" != kanashi6/UniLIP-* ]]; then
  echo "ERROR: local vanilla UniLIP model path does not exist: $UNILIP_BASE_MODEL" >&2
  exit 2
fi

run_python -m accelerate.commands.launch --num_processes "$NUM_PROCESSES" \
  scripts/counting_grpo/train_hf_multi_image_count_sft.py \
  --model_name_or_path "$UNILIP_BASE_MODEL" \
  --data_path "$SFT_DATA_PATH" \
  "${EVAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --output_dir "$SFT_OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --save_strategy "$SAVE_STRATEGY" \
  --eval_strategy "$EVAL_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END" \
  --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
  --greater_is_better "$GREATER_IS_BETTER" \
  --bf16 true \
  --attn_implementation "$ATTN_IMPL" \
  --allow_attn_fallback "$ALLOW_ATTN_FALLBACK"
