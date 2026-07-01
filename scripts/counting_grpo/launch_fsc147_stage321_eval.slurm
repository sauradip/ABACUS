#!/usr/bin/env bash
#SBATCH --job-name=fsc147_stage321_eval
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/fsc147_stage321_eval_%j.log
#SBATCH --error=logs/fsc147_stage321_eval_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
cd "$REPO_DIR"

CONTAINER_PATH="${CONTAINER_PATH:-$REPO_DIR/.cache/images/nvcr_pytorch_24.08-py3_arm64_sandbox}"
CONTAINER_PYTHONUSERBASE="${CONTAINER_PYTHONUSERBASE:-$REPO_DIR/.cache/container_py310}"
MODEL_PATH="${MODEL_PATH:-checkpoints/stage321_grpo_count_only}"
FSC147_ROOT="${FSC147_ROOT:-/projects/u6fb/myprojects/FSC147_hf}"
OUT_DIR="${OUT_DIR:-outputs/fsc147_stage321_eval}"
SPLITS="${SPLITS:-val test}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

mkdir -p logs "$OUT_DIR" \
    "$REPO_DIR/.cache/tmp" \
    "$REPO_DIR/.cache/huggingface" \
    "$REPO_DIR/.cache/container_py310"

run_in_container() {
    apptainer exec --nv \
        --bind /projects:/projects \
        --bind /lus:/lus \
        --bind /home:/home \
        --env PYTHONUSERBASE="$CONTAINER_PYTHONUSERBASE" \
        --env HF_HOME="$REPO_DIR/.cache/huggingface" \
        --env TMPDIR="$REPO_DIR/.cache/tmp" \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONPYCACHEPREFIX="$REPO_DIR/.cache/tmp/fsc147_eval_pycache_${SLURM_JOB_ID:-manual}" \
        "$CONTAINER_PATH" \
        bash -lc 'set -euo pipefail; export PATH="$PYTHONUSERBASE/bin:$PATH"; exec "$@"' _ "$@"
}

echo "=== FSC147 Stage 3.2.1 Eval ==="
echo "Repo       : $REPO_DIR"
echo "Model      : $MODEL_PATH"
echo "FSC147 root: $FSC147_ROOT"
echo "Out dir    : $OUT_DIR"
echo "Splits     : $SPLITS"
echo "Shards     : $NUM_SHARDS"
echo "Attention  : $ATTN_IMPL"
echo "Container  : $CONTAINER_PATH"

python3 scripts/counting_grpo/build_fsc147_count_only_eval.py \
    --fsc147_root "$FSC147_ROOT" \
    --output_dir "$OUT_DIR" \
    --splits $SPLITS

eval_split() {
    local split="$1"
    local dataset="$OUT_DIR/fsc147_${split}_count_only.jsonl"
    local prefix="$OUT_DIR/fsc147_${split}_stage321_predictions"
    local total
    total="$(wc -l < "$dataset")"
    local shard_size=$(( (total + NUM_SHARDS - 1) / NUM_SHARDS ))

    echo "=== Split $split: total=$total shard_size=$shard_size ==="
    rm -f "${prefix}".shard*.jsonl "${prefix}.jsonl" "${prefix}.metrics.json"

    local pids=()
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        local start=$(( shard * shard_size ))
        local count="$shard_size"
        if (( start >= total )); then
            continue
        fi
        if (( start + count > total )); then
            count=$(( total - start ))
        fi

        local gpu=$(( shard % 4 ))
        local shard_out="${prefix}.shard${shard}.jsonl"
        echo "Launching split=$split shard=$shard gpu=$gpu start=$start count=$count -> $shard_out"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            run_in_container python3 scripts/counting_grpo/generate_count_only_audit.py \
                --model_path "$MODEL_PATH" \
                --dataset_jsonl "$dataset" \
                --output_jsonl "$shard_out" \
                --start_index "$start" \
                --max_samples "$count" \
                --attn_implementation "$ATTN_IMPL" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --device cuda
        ) &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
        local shard_out="${prefix}.shard${shard}.jsonl"
        if [[ -s "$shard_out" ]]; then
            cat "$shard_out" >> "${prefix}.jsonl"
        fi
    done

    local produced
    produced="$(wc -l < "${prefix}.jsonl")"
    echo "Split $split produced $produced / $total predictions"
    if [[ "$produced" != "$total" ]]; then
        echo "ERROR: split $split produced incomplete predictions"
        exit 1
    fi

    python3 scripts/counting_grpo/audit_count_only_mae.py \
        --predictions_jsonl "${prefix}.jsonl" \
        --prediction_field completion | tee "${prefix}.metrics.json"
}

for split in $SPLITS; do
    eval_split "$split"
done

echo "FSC147 Stage 3.2.1 eval complete."
