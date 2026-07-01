#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
FSC147_ROOT="${FSC147_ROOT:-/projects/u6fb/myprojects/FSC147_hf}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/hf_multi_image_count_sft_fsc147_train}"
SPLITS="${SPLITS:-val test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
ALLOW_ATTN_FALLBACK="${ALLOW_ATTN_FALLBACK:-0}"

CONTAINER_PATH="${CONTAINER_PATH:-$REPO_DIR/.cache/images/nvcr_pytorch_24.08-py3_arm64_sandbox}"
CONTAINER_PYTHONUSERBASE="${CONTAINER_PYTHONUSERBASE:-$REPO_DIR/.cache/container_py310}"

cd "$REPO_DIR"
mkdir -p logs "$REPO_DIR/.cache/huggingface" "$REPO_DIR/.cache/torch" "$REPO_DIR/.cache/tmp"

run_in_container() {
  apptainer exec --nv \
    --bind /projects:/projects \
    --bind /lus:/lus \
    --bind /home:/home \
    --env PYTHONUSERBASE="$CONTAINER_PYTHONUSERBASE" \
    --env PATH="$CONTAINER_PYTHONUSERBASE/bin:$PATH" \
    --env HF_HOME="$REPO_DIR/.cache/huggingface" \
    --env TORCH_HOME="$REPO_DIR/.cache/torch" \
    --env TMPDIR="$REPO_DIR/.cache/tmp" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    "$CONTAINER_PATH" \
    bash -lc 'cd "$0"; export PATH="$PYTHONUSERBASE/bin:$PATH"; export PYTHONPATH="$PWD/scripts/counting_grpo:$PWD:$PWD/UniLIP:$PYTHONUSERBASE/lib/python3.10/site-packages:${PYTHONPATH:-}"; exec "$@"' \
    "$REPO_DIR" "$@"
}

echo "=== HF multi-image count eval ==="
echo "repo       : $REPO_DIR"
echo "model      : $MODEL_PATH"
echo "fsc147     : $FSC147_ROOT"
echo "splits     : $SPLITS"
echo "max_new    : $MAX_NEW_TOKENS"
echo "batch_size : $BATCH_SIZE"

for split in $SPLITS; do
  prompt_dir="$REPO_DIR/outputs/scaffold_prompt_fsc147_${split}_adaptive"
  prompt_jsonl="$prompt_dir/${split}_scaffold_input_only.jsonl"
  message_dir="$REPO_DIR/outputs/adaptive_hf_multi_image_count_sft_fsc147_${split}"
  message_jsonl="$message_dir/${split}_messages.jsonl"
  pred_jsonl="$message_dir/${split}_predictions.jsonl"

  echo "=== Build adaptive scaffold prompts: $split ==="
  run_in_container python3 scripts/counting_grpo/build_fsc147_train_adaptive_scaffold_prompts.py \
    --fsc147_root "$FSC147_ROOT" \
    --split "$split" \
    --output_dir "$prompt_dir" \
    --output_jsonl "$prompt_jsonl" \
    --batch_size "$BATCH_SIZE" \
    --device cuda

  echo "=== Build HF multi-image eval messages: $split ==="
  run_in_container python3 scripts/counting_grpo/build_hf_multi_image_count_sft.py \
    --prompt_jsonl "$prompt_jsonl" \
    --master_jsonl outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl \
    --fsc147_annotations "$FSC147_ROOT/annotation_FSC147_384.json" \
    --output_jsonl "$message_jsonl" \
    --missing_gt_policy error

  echo "=== Generate and audit: $split ==="
  run_in_container python3 scripts/counting_grpo/eval_hf_multi_image_count_sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$message_jsonl" \
    --output_jsonl "$pred_jsonl" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --attn_implementation "$ATTN_IMPL" \
    --allow_attn_fallback "$ALLOW_ATTN_FALLBACK"
done

echo "HF multi-image count eval complete."
