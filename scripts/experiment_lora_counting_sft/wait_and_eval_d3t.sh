#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wait for the active D3T training to finish, then run per-epoch eval on
# both the cold and warm runs, with separate TAGs so results don't collide.
# Designed to be launched with nohup before disconnecting.
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: keep going even if one ckpt eval fails

cd "$(dirname "$0")/../.."

COLD=/data/amondal/unicount_runs/lora_counting_sft_d3t_cold_20260501_104341
WARM=/data/amondal/unicount_runs/lora_counting_sft_d3t_warm_20260501_112126

mkdir -p logs
LOG="logs/d3t_eval_queue_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1

echo "============================================================"
echo "D3T post-training eval queue   $(date)"
echo "  COLD = $COLD"
echo "  WARM = $WARM"
echo "  log  = $LOG"
echo "============================================================"

# Wait for any running train_lora_counting_sft process to exit.
echo "[wait] polling for trainer to finish..."
while pgrep -f train_lora_counting_sft > /dev/null; do
    sleep 60
done
echo "[wait] no trainer processes detected at $(date) — proceeding."

# Make sure GPUs released memory before launching eval (safety: 30s grace).
sleep 30
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -

# ── Eval COLD ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "COLD eval starting at $(date)"
echo "============================================================"
bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \
     "$COLD" d3t_cold

# ── Eval WARM ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "WARM eval starting at $(date)"
echo "============================================================"
bash scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh \
     "$WARM" d3t_warm

# ── Combined summary ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "COMBINED D3T summary at $(date)"
echo "============================================================"
python3 - <<'PY'
import json, glob, re
EVAL_DIR = "outputs/experiment_lora_counting_sft/eval"
BASELINE_VARIANT_B_VAL_MAE = 18.94

for tag in ("d3t_cold", "d3t_warm"):
    files = sorted(glob.glob(f"{EVAL_DIR}/val_recursive_T100_d3_avg_{tag}_step*.json"),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if not files:
        print(f"\n[{tag}] no eval results found.")
        continue
    print(f"\n=== {tag.upper()} ===")
    print(f"{'step':>6}  {'epoch':>6}  {'val_MAE':>9}  {'val_RMSE':>9}")
    print("-" * 38)
    best = (None, float("inf"), None)
    for f in files:
        step = int(re.search(r"step(\d+)", f).group(1))
        epoch = step / 258.0   # 4123 / eff_batch 16
        d = json.load(open(f))
        mae  = d.get("MAE")  or d.get("mae")
        rmse = d.get("RMSE") or d.get("rmse")
        print(f"{step:>6}  {epoch:>6.2f}  {mae:>9.3f}  {rmse:>9.3f}")
        if mae < best[1]:
            best = (step, mae, rmse)
    print("-" * 38)
    delta = best[1] - BASELINE_VARIANT_B_VAL_MAE
    sign = "+" if delta >= 0 else ""
    print(f"BEST: step {best[0]} (epoch ~{best[0]/258.0:.2f})  "
          f"val MAE={best[1]:.3f}  RMSE={best[2]:.3f}   "
          f"vs baseline 18.94 = {sign}{delta:.3f}")

print("\nDecision gate:")
print("  best < 18.0      → run test + cross-dataset on best ckpt")
print("  18.0–18.5        → marginal; run test + decide")
print("  18.5–18.94       → very marginal (test only if cold-start best)")
print("  > 18.94          → REVERT, D3T didn't help under full supervision")
PY

echo ""
echo "============================================================"
echo "D3T eval queue COMPLETE at $(date)"
echo "============================================================"
