#!/bin/bash
#SBATCH --job-name=scaffold_sft_stage1
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount/logs/scaffold_sft_%j.log
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount/logs/scaffold_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx_a6000_risk

set -euo pipefail

REPO_DIR="/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount"
DEFAULT_JANUS_DIR="/projects/u6bl/myprojects/Janus"
if [[ -d "$DEFAULT_JANUS_DIR" ]]; then
  JANUS_DIR="${JANUS_DIR:-$DEFAULT_JANUS_DIR}"
else
  JANUS_DIR="${JANUS_DIR:-$REPO_DIR/.cache/janus_stub}"
fi

LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-$REPO_DIR/.cache/scaffold_sft_stage1}"
LIB_DIR="${LIB_DIR:-$JANUS_DIR/python_libs}"
HF_CACHE_DIR="${HF_HOME:-$LOCAL_CACHE_ROOT/hf_home}"
TRITON_DIR="${TRITON_CACHE_DIR:-$LOCAL_CACHE_ROOT/triton}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LOCAL_CACHE_ROOT/pip}"
CONTAINER_PATH="${CONTAINER_PATH:-$JANUS_DIR/pytorch_24.08.sif}"

export PYTHONUSERBASE="/mnt/fast/nobackup/users/am04485/.local"
export PIP_CACHE_DIR
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$REPO_DIR/UniLIP/python_libs/lib/python3.10/site-packages:$REPO_DIR/UniLIP:$LIB_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONPATH="$REPO_DIR/.cache/crowdvlm_target:$PYTHONPATH"
export PATH="$LIB_DIR/bin:$PATH"
export PYTHONNOUSERSITE=0

BASE_MODEL_PATH="${BASE_MODEL_PATH:-OpenGVLab/InternVL2-2B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$BASE_MODEL_PATH}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_DIR/outputs/fsc147_scaffold_full/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/scaffold_sft_stage1}"
EXPECTED_MD5="${EXPECTED_MD5:-}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export ATTN_IMPL
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-6144}"
RUN_AUDIT_AFTER_100="${RUN_AUDIT_AFTER_100:-1}"
AUDIT_NUM_SAMPLES="${AUDIT_NUM_SAMPLES:-5}"
AUDIT_SEED="${AUDIT_SEED:-42}"
AUDIT_OUTPUT_PATH="${AUDIT_OUTPUT_PATH:-$OUTPUT_DIR/zero_shot_point_audit_step100.json}"
FSC_ROOT="${FSC_ROOT:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/fsc147_v4/data/fsc147}"

FLASH_ENV_PREFIX="${FLASH_ENV_PREFIX:-/mnt/fast/nobackup/scratch4weeks/am04485/.conda_envs/uniflash310}"
ENV_SITE_PACKAGES="$FLASH_ENV_PREFIX/lib/python3.10/site-packages"
if [[ -d "$ENV_SITE_PACKAGES" ]]; then
  export PYTHONPATH="$ENV_SITE_PACKAGES:$PYTHONPATH"
fi

if [[ -x "$FLASH_ENV_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$FLASH_ENV_PREFIX/bin/python"
  export PATH="$FLASH_ENV_PREFIX/bin:$PATH"
else
  PYTHON_BIN="python3"
fi

# Smoke defaults: one optimizer step and immediate logging.
SMOKE_TEST="${SMOKE_TEST:-1}"
if [[ "$SMOKE_TEST" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
  LOGGING_STEPS="${LOGGING_STEPS:-1}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
else
  MAX_STEPS="${MAX_STEPS:--1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
  LOGGING_STEPS="${LOGGING_STEPS:-1}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
fi

# Stage-1 production defaults for audit pivot.
if [[ "$SMOKE_TEST" != "1" && "${FORCE_100_STEP_PIVOT:-1}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-100}"
fi

mkdir -p "$OUTPUT_DIR" "$TRITON_DIR" "$PIP_CACHE_DIR" "$HF_CACHE_DIR" "$REPO_DIR/logs" "$LOCAL_CACHE_ROOT"

echo "=== SCAFFOLD-R1 STAGE 1 ==="
echo "Repo:             $REPO_DIR"
echo "Base model:       $BASE_MODEL_PATH"
echo "Tokenizer path:   $TOKENIZER_PATH"
echo "Train data:       $TRAIN_DATA"
echo "Output dir:       $OUTPUT_DIR"
echo "Model max length: $MODEL_MAX_LENGTH"
echo "Attention impl:   $ATTN_IMPL"
echo "Smoke test:       $SMOKE_TEST"
echo "Run audit @100:   $RUN_AUDIT_AFTER_100"
echo "FSC root:         $FSC_ROOT"
echo ""

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: Missing training data at $TRAIN_DATA"
  exit 1
fi

BASE_IS_LOCAL=0
if [[ -d "$BASE_MODEL_PATH" || -f "$BASE_MODEL_PATH/config.json" ]]; then
  BASE_IS_LOCAL=1
fi

if [[ "$BASE_IS_LOCAL" == "1" ]]; then
  if [[ ! -f "$BASE_MODEL_PATH/config.json" ]]; then
    echo "ERROR: Missing $BASE_MODEL_PATH/config.json"
    echo "Provide a valid local base model directory or an HF model id."
    exit 1
  fi
  ACTUAL_MD5=$(md5sum "$BASE_MODEL_PATH/config.json" | awk '{print $1}')
  echo "Base config MD5:  $ACTUAL_MD5"
  if [[ -n "$EXPECTED_MD5" && "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
    echo "ERROR: Base model MD5 mismatch! Found $ACTUAL_MD5, expected $EXPECTED_MD5."
    echo "Ensure you are using the RAW pretrained InternVL2-2B, not a UniLIP checkpoint."
    exit 1
  fi
else
  echo "Base model is treated as Hugging Face id: $BASE_MODEL_PATH"
  echo "Running trust_remote_code accessibility check..."
  if [[ -f "$CONTAINER_PATH" ]]; then
    apptainer exec --nv \
      --bind /projects:/projects \
      --bind /mnt/fast:/mnt/fast \
      --env PYTHONPATH="$PYTHONPATH" \
      --env PYTHONUSERBASE="$LIB_DIR" \
      --env HF_HOME="$HF_CACHE_DIR" \
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      "$CONTAINER_PATH" \
      python3 - <<PY
from transformers import AutoConfig
AutoConfig.from_pretrained("$BASE_MODEL_PATH", trust_remote_code=True)
print("HF model config is accessible.")
PY
  else
    "$PYTHON_BIN" - <<PY
from transformers import AutoConfig
AutoConfig.from_pretrained("$BASE_MODEL_PATH", trust_remote_code=True)
print("HF model config is accessible.")
PY
  fi
fi

NUM_EXAMPLES=$(wc -l < "$TRAIN_DATA")
echo "Training examples: $NUM_EXAMPLES"
echo ""

if [[ "$ATTN_IMPL" == "flash_attention_2" ]]; then
  echo "Checking flash-attn availability on host Python..."
  if [[ -f "$CONTAINER_PATH" ]]; then
    apptainer exec --nv \
      --bind /projects:/projects \
      --bind /mnt/fast:/mnt/fast \
      --env PYTHONPATH="$PYTHONPATH" \
      --env PYTHONUSERBASE="$LIB_DIR" \
      --env HF_HOME="$HF_CACHE_DIR" \
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      "$CONTAINER_PATH" \
      python3 - <<'PY'
import importlib.util
import sys
spec = importlib.util.find_spec("flash_attn")
if spec is None:
    print("flash-attn NOT found")
    sys.exit(1)
print("flash-attn found")
PY
  elif "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
spec = importlib.util.find_spec("flash_attn")
if spec is None:
    print("flash-attn NOT found")
    sys.exit(1)
print("flash-attn found")
PY
  then
    echo "Flash-attn check passed."
  else
    echo "ERROR: flash-attn not found in selected Python runtime."
    echo "Cannot continue with attn_implementation=$ATTN_IMPL."
    exit 1
  fi
else
  echo "Skipping flash-attn preflight because attn_implementation=$ATTN_IMPL"
fi

run_stage1() {
  local max_steps="$1"
  local epochs="$2"

  TRAIN_CMD=(
    "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=2 scripts/counting_grpo/train_scaffold_sft.py
    --model_name_or_path "$BASE_MODEL_PATH"
    --mllm_hf_path "$TOKENIZER_PATH"
    --data_path "$TRAIN_DATA"
    --output_dir "$OUTPUT_DIR"
    --base_model_md5 "$EXPECTED_MD5"
    --lora_rank 16
    --lora_alpha 32
    --learning_rate 5e-6
    --num_train_epochs "$epochs"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 8
    --logging_steps "$LOGGING_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps 200
    --save_total_limit 2
    --model_max_length "$MODEL_MAX_LENGTH"
    --max_steps "$max_steps"
    --attn_implementation "$ATTN_IMPL"
    --bf16 True
    --fp16 False
    --max_grad_norm 1.0
    --warmup_ratio 0.05
    --gradient_checkpointing True
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --report_to none
    --remove_unused_columns False
  )

  if [[ -f "$CONTAINER_PATH" ]]; then
    apptainer exec --nv \
      --bind /projects:/projects \
      --bind /mnt/fast:/mnt/fast \
      --env PYTHONPATH="$PYTHONPATH" \
      --env PYTHONUSERBASE="$LIB_DIR" \
      --env HF_HOME="$HF_CACHE_DIR" \
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      "$CONTAINER_PATH" \
      bash -lc "set -e; cd '$REPO_DIR'; ${TRAIN_CMD[*]}"
  else
    echo "WARNING: Container not found at $CONTAINER_PATH"
    echo "Falling back to host Python execution."
    cd "$REPO_DIR"
    "${TRAIN_CMD[@]}"
  fi
}

run_point_audit() {
  local model_path_for_audit="$1"
  echo "Running zero-shot point audit using model path: $model_path_for_audit"
  python scripts/counting_grpo/zero_shot_point_audit.py \
    --model_path "$model_path_for_audit" \
    --tokenizer_path "$TOKENIZER_PATH" \
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
  run_stage1 "$MAX_STEPS" "$NUM_TRAIN_EPOCHS"
else
  if [[ "$RUN_AUDIT_AFTER_100" == "1" ]]; then
    echo "Launching Stage 1 pivot run to 100 steps..."
    run_stage1 "$MAX_STEPS" "$NUM_TRAIN_EPOCHS"

    # Prefer most recent checkpoint under OUTPUT_DIR, else fall back to base.
    LAST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "$LAST_CKPT" && -d "$LAST_CKPT" ]]; then
      AUDIT_MODEL_PATH="$LAST_CKPT"
    else
      AUDIT_MODEL_PATH="$BASE_MODEL_PATH"
    fi

    run_point_audit "$AUDIT_MODEL_PATH"

    echo "Audit passed. Resuming Stage 1 to full completion..."
    MAX_STEPS=-1
    NUM_TRAIN_EPOCHS=5
    SAVE_STRATEGY=steps
    run_stage1 "$MAX_STEPS" "$NUM_TRAIN_EPOCHS"
  else
    run_stage1 "$MAX_STEPS" "$NUM_TRAIN_EPOCHS"
  fi
fi

echo "=== Scaffold-R1 Stage 1 launch complete ==="