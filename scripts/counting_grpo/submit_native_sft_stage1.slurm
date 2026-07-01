#!/bin/bash
#SBATCH --job-name=native_sft_r64
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount/logs/native_sft_%j.log
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount/logs/native_sft_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx_a6000_risk

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-$REPO_DIR/.cache/native_sft_stage1_r64}"
HF_CACHE_DIR="${HF_HOME:-$LOCAL_CACHE_ROOT/hf_home}"
TRITON_DIR="${TRITON_CACHE_DIR:-$LOCAL_CACHE_ROOT/triton}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LOCAL_CACHE_ROOT/pip}"
CONTAINER_PATH="${CONTAINER_PATH:-}"
APPTAINER_CACHE_DIR="${APPTAINER_CACHE_DIR:-$LOCAL_CACHE_ROOT/apptainer_cache}"
APPTAINER_TMP_DIR="${APPTAINER_TMP_DIR:-$LOCAL_CACHE_ROOT/apptainer_tmp}"

export PIP_CACHE_DIR
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export APPTAINERENV_SSL_CERT_FILE=""

PYTHON_BIN="${PYTHON_BIN:-python3}"
MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-OpenGVLab/InternVL2-2B}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_DIR/outputs/fsc147_scaffold_full/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/native_sft_stage1_r64_lr2e4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
AUDIT_ATTN_IMPL="${AUDIT_ATTN_IMPL:-eager}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-6144}"
RUN_AUDIT_AFTER_100="${RUN_AUDIT_AFTER_100:-1}"
AUDIT_NUM_SAMPLES="${AUDIT_NUM_SAMPLES:-5}"
AUDIT_SEED="${AUDIT_SEED:-42}"
AUDIT_OUTPUT_PATH="${AUDIT_OUTPUT_PATH:-$OUTPUT_DIR/zero_shot_point_audit_step100.json}"
FSC_ROOT="${FSC_ROOT:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/fsc147_v4/data/fsc147}"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
  LOGGING_STEPS="${LOGGING_STEPS:-1}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
else
  MAX_STEPS="${MAX_STEPS:--1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
  LOGGING_STEPS="${LOGGING_STEPS:-10}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
  SAVE_STEPS="${SAVE_STEPS:-100}"
fi

mkdir -p "$OUTPUT_DIR" "$TRITON_DIR" "$PIP_CACHE_DIR" "$HF_CACHE_DIR" "$REPO_DIR/logs"
mkdir -p "$APPTAINER_CACHE_DIR" "$APPTAINER_TMP_DIR"

echo "=== NATIVE SFT STAGE 1 ==="
echo "Repo:             $REPO_DIR"
echo "Base model:       $BASE_MODEL_PATH"
echo "Train data:       $TRAIN_DATA"
echo "Output dir:       $OUTPUT_DIR"
echo "Model max length: $MODEL_MAX_LENGTH"
echo "Attention impl:   $ATTN_IMPL"
echo "Audit attn impl:  $AUDIT_ATTN_IMPL"
echo "Smoke test:       $SMOKE_TEST"
echo "Run audit @100:   $RUN_AUDIT_AFTER_100"
echo "FSC root:         $FSC_ROOT"
echo "Master port:      $MASTER_PORT"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: Missing training data at $TRAIN_DATA"
  exit 1
fi

cd "$REPO_DIR"

run_in_runtime() {
  if [[ -n "$CONTAINER_PATH" ]]; then
    export APPTAINER_CACHEDIR="$APPTAINER_CACHE_DIR"
    export APPTAINER_TMPDIR="$APPTAINER_TMP_DIR"
    apptainer exec --nv \
      --bind /projects:/projects \
      --bind /home:/home \
      --bind /lus:/lus \
      --env HF_HOME="$HF_HOME" \
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
      "$CONTAINER_PATH" \
      "$@"
  else
    "$@"
  fi
}

run_stage1() {
  local max_steps="$1"
  local epochs="$2"
  local save_strategy="$3"
  local save_steps="$4"

  run_in_runtime "$PYTHON_BIN" -m torch.distributed.run --master_port "$MASTER_PORT" --nproc_per_node=2 scripts/counting_grpo/train_native_sft.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --attn_implementation "$AUDIT_ATTN_IMPL" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --learning_rate 2e-4 \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "$save_strategy" \
    --save_steps "$save_steps" \
    --save_total_limit 2 \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --max_steps "$max_steps" \
    --bf16 True \
    --fp16 False \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to none \
    --remove_unused_columns False
}

run_point_audit() {
  local model_path_for_audit="$1"
  run_in_runtime "$PYTHON_BIN" scripts/counting_grpo/zero_shot_point_audit.py \
    --model_path "$model_path_for_audit" \
    --tokenizer_path "$BASE_MODEL_PATH" \
    --fsc_root "$FSC_ROOT" \
    --num_samples "$AUDIT_NUM_SAMPLES" \
    --seed "$AUDIT_SEED" \
    --torch_dtype float16 \
    --attn_implementation "$ATTN_IMPL" \
    --output_path "$AUDIT_OUTPUT_PATH" \
    --fail_on_bounds \
    --fail_on_diversity
}

if [[ "$SMOKE_TEST" == "1" ]]; then
  echo "=== Running smoke test (1 step) ==="
  run_stage1 "$MAX_STEPS" "$NUM_TRAIN_EPOCHS" "$SAVE_STRATEGY" "$SAVE_STEPS"
else
  echo "=== Running full Stage 1 training (5 epochs, ~3700 steps) ==="
  run_stage1 -1 5 steps 100
  
  echo ""
  echo "=== Training Complete. Running audit on final checkpoint ==="
  FINAL_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$FINAL_CKPT" && -d "$FINAL_CKPT" ]]; then
    echo "Auditing final checkpoint: $FINAL_CKPT"
    FINAL_AUDIT_PATH="$OUTPUT_DIR/zero_shot_point_audit_final.json"
    run_point_audit "$FINAL_CKPT"
  else
    echo "WARNING: No final checkpoint found"
  fi
fi

echo "=== Native SFT Stage 1 launch complete ==="
