#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_JSONL="${DATA_JSONL:-outputs/rankdpo/preference_pairs.jsonl}"
OUT_DIR="${OUT_DIR:-checkpoints/rankdpo_stage25_gh200}"
SCAFFOLD_JSONL="${SCAFFOLD_JSONL:-outputs/fsc147_scaffold_full/all.jsonl}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140}"
VERIFY_MAX_NEW_TOKENS="${VERIFY_MAX_NEW_TOKENS:-64}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
SKIP_MERGE_VERIFY="${SKIP_MERGE_VERIFY:-0}"
INIT_MODE="${INIT_MODE:-manual_inject}"
VISION_SCALE="${VISION_SCALE:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

# Prefer project venv when available to avoid host-python dependency drift.
if [[ -x .venv311/bin/python ]]; then
  PYTHON_BIN=".venv311/bin/python"
fi

if [[ -z "${AUDIT_JSON:-}" ]]; then
  if [[ -f checkpoints/native_sft_stage1_r64_lr2e4/zero_shot_audit_final.json ]]; then
    AUDIT_JSON="checkpoints/native_sft_stage1_r64_lr2e4/zero_shot_audit_final.json"
  else
    AUDIT_JSON="checkpoints/native_sft_stage1_r64_lr2e4/zero_shot_point_audit_step100.json"
  fi
fi

if [[ "$SKIP_PREPARE" == "1" ]]; then
  if [[ ! -f "$DATA_JSONL" ]]; then
    echo "[Stage2.5] ERROR: SKIP_PREPARE=1 but DATA_JSONL is missing: ${DATA_JSONL}"
    exit 1
  fi
  echo "[Stage2.5] SKIP_PREPARE=1 -> using existing RankDPO dataset: ${DATA_JSONL}"
else
  echo "[Stage2.5] Preparing RankDPO dataset -> ${DATA_JSONL}"
  "$PYTHON_BIN" scripts/counting_grpo/prepare_rankdpo_data.py \
    --scaffold_jsonl "$SCAFFOLD_JSONL" \
    --audit_json "$AUDIT_JSON" \
    --output_jsonl "$DATA_JSONL" \
    --allow_synthetic_overcount_fallback \
    --emit_rank_triplets
fi

echo "[Stage2.5] Launching RankDPO training on GH200 defaults"
TRAIN_ARGS=(
  --data_path "$DATA_JSONL"
  --output_dir "$OUT_DIR"
  --stage1_checkpoint "$STAGE1_CKPT"
  --init_mode "$INIT_MODE"
  --vision_scale "$VISION_SCALE"
  --max_length "$MAX_LENGTH"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --beta 0.1
  --learning_rate 5e-7
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --loss_type sigmoid
  --verify_max_new_tokens "$VERIFY_MAX_NEW_TOKENS"
)

if [[ "$VERIFY_ONLY" == "1" ]]; then
  TRAIN_ARGS+=(--verify_only)
fi
if [[ "$SKIP_MERGE_VERIFY" == "1" ]]; then
  TRAIN_ARGS+=(--skip_merge_verify)
fi

"$PYTHON_BIN" scripts/counting_grpo/train_rankdpo_stage25.py \
  "${TRAIN_ARGS[@]}"

echo "[Stage2.5] Done"
