#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Post-training: extract a PEFT adapter from each checkpoint-XXX/ in the run
# directory, then run FSC-147 val MAE eval on each one.
#
# Usage:
#   bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \
#        /data/amondal/unicount_runs/lora_counting_sft_variantB_effbatch16_10ep_<STAMP>
#
# Writes one JSON per checkpoint:
#   outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg_effbatch16_step<N>.json
#
# Eval = CARC (T=100, max_depth=3) on FSC val (1,286 images), 8×GPU.
# Wallclock per checkpoint: ~3-5 min. 10 checkpoints ≈ 30-50 min total.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
source unicount/bin/activate

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <RUN_DIR>" >&2
    echo "  RUN_DIR must contain checkpoint-*/ subdirs and a final adapter/ dir" >&2
    exit 1
fi

RUN_DIR="$1"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
    exit 1
fi

# Reference adapter for adapter_config.json + key-set check.
# Prefer the run's own final adapter; fall back to the known-good Variant B run.
REF_ADAPTER="$RUN_DIR/adapter"
if [[ ! -f "$REF_ADAPTER/adapter_config.json" ]]; then
    REF_ADAPTER="/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter"
    echo "[INFO] Using fallback reference adapter: $REF_ADAPTER"
fi

VAL_JSON="outputs/experiment_lora_counting_sft/val/val_counting.json"
OUT_BASE="outputs/experiment_lora_counting_sft/eval"
TAG="${2:-effbatch16}"
mkdir -p "$OUT_BASE" logs

CKPTS=( $(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n) )
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoint-*/ dirs found in $RUN_DIR" >&2
    exit 1
fi

echo "============================================================"
echo " Extract + Eval all checkpoints in:"
echo "   $RUN_DIR"
echo " Reference adapter: $REF_ADAPTER"
echo " Found ${#CKPTS[@]} checkpoints:"
for c in "${CKPTS[@]}"; do echo "   $(basename $c)"; done
echo "============================================================"

# ── Phase 1: extract adapters ────────────────────────────────────────────
for CKPT in "${CKPTS[@]}"; do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    ADAPTER_OUT="$RUN_DIR/adapter_step${STEP}"

    if [[ -f "$ADAPTER_OUT/adapter_model.safetensors" ]]; then
        echo "[skip] $ADAPTER_OUT already exists"
        continue
    fi

    echo "── Extracting step $STEP → $ADAPTER_OUT"
    python scripts/experiment_lora_counting_sft/extract_adapter_from_checkpoint.py \
        --checkpoint_dir    "$CKPT" \
        --reference_adapter "$REF_ADAPTER" \
        --out_dir           "$ADAPTER_OUT"
done

# ── Phase 2: eval each adapter on FSC val ────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2: FSC val eval per epoch checkpoint"
echo "============================================================"

for CKPT in "${CKPTS[@]}"; do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    ADAPTER_DIR="$RUN_DIR/adapter_step${STEP}"
    OUT_JSON="$OUT_BASE/val_recursive_T100_d3_avg_${TAG}_step${STEP}.json"
    LOG="logs/${TAG}_eval_val_step${STEP}_$(date +%Y%m%d_%H%M%S).log"

    if [[ -f "$OUT_JSON" ]]; then
        echo "[skip] $OUT_JSON already exists"
        continue
    fi

    echo ""
    echo "── Eval step $STEP ── log: $LOG"

    accelerate launch \
        --num_processes=8 \
        --mixed_precision=no \
        scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
            --T              100 \
            --max_depth      3 \
            --checkpoint_dir "$ADAPTER_DIR" \
            --val_json       "$VAL_JSON" \
            --out_json       "$OUT_JSON" \
        > "$LOG" 2>&1

    # Print headline MAE from log
    echo "── step $STEP result:"
    grep -E "MAE\s+:|RMSE" "$LOG" | head -3 || true
done

echo ""
echo "============================================================"
echo " Per-epoch val MAE summary"
echo "============================================================"
python3 - <<EOF
import json, glob, os, re
files = sorted(glob.glob("$OUT_BASE/val_recursive_T100_d3_avg_${TAG}_step*.json"),
               key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
print(f"{'step':>6}  {'epoch':>6}  {'val_MAE':>9}  {'val_RMSE':>9}")
print("-"*38)
best = (None, float("inf"))
for f in files:
    step = int(re.search(r"step(\d+)", f).group(1))
    epoch = step / 229.0
    d = json.load(open(f))
    mae  = d.get("MAE")  or d.get("mae")
    rmse = d.get("RMSE") or d.get("rmse")
    print(f"{step:>6}  {epoch:>6.2f}  {mae:>9.3f}  {rmse:>9.3f}")
    if mae < best[1]:
        best = (step, mae)
print("-"*38)
print(f"BEST: step {best[0]} (epoch ~{best[0]/229.0:.2f})  val MAE = {best[1]:.3f}")
print(f"  → adapter: $RUN_DIR/adapter_step{best[0]}")
EOF
