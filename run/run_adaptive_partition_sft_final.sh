#!/usr/bin/env bash
set -euo pipefail

cd /data/amondal/UniCount
source /data/amondal/UniCount/unicount/bin/activate

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/data/amondal/unicount_runs/adaptive_partition_sft_final/logs_${RUN_TS}"
mkdir -p "${LOG_DIR}"

TRAIN_LOG="${LOG_DIR}/train.log"
VRAM_LOG="${LOG_DIR}/vram.csv"

echo "Logging to: ${TRAIN_LOG}"
echo "VRAM log:   ${VRAM_LOG}"

cleanup() {
  if [[ -n "${SMI_PID:-}" ]]; then
    kill "${SMI_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 10 > "${VRAM_LOG}" &
SMI_PID=$!

stdbuf -oL -eL accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  scripts/experiment_partitioned_counting/train_adaptive_partition_sft.py \
  --model_name_or_path /data/amondal/unicount_runs/partitioned_double_scaffold_sft_v1/checkpoint-1000 \
  --output_dir /data/amondal/unicount_runs/adaptive_partition_sft_final \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3.0 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 500 \
  --save_strategy "steps" \
  2>&1 | tee -a "${TRAIN_LOG}"

python3 - "${TRAIN_LOG}" "${VRAM_LOG}" <<'PY'
import ast
import sys

train_log, vram_log = sys.argv[1], sys.argv[2]

losses = []
lrs = []

with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
            except Exception:
                continue
            if "loss" in d:
                try:
                    losses.append(float(d["loss"]))
                except Exception:
                    pass
            if "learning_rate" in d:
                try:
                    lrs.append(float(d["learning_rate"]))
                except Exception:
                    pass

first_100_losses = losses[:10]  # logging_steps=10 => first 10 logs ~= first 100 steps
if first_100_losses:
    avg = sum(first_100_losses) / len(first_100_losses)
    print(f"AVG_LOSS_FIRST_100_STEPS={avg:.6f}")
else:
    print("AVG_LOSS_FIRST_100_STEPS=NA (no loss logs found)")

if lrs:
    warmup_started = any(x > 0 for x in lrs)
    print(f"WARMUP_INITIATED={'YES' if warmup_started else 'NO'}")
    print(f"FIRST_LR={lrs[0]:.10f}")
    print(f"MAX_LR_SEEN={max(lrs):.10f}")
else:
    print("WARMUP_INITIATED=NA (no learning_rate logs found)")

peak_pct = 0.0
peak_row = None
with open(vram_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used = float(parts[2])
            total = float(parts[3])
            pct = 100.0 * used / total if total > 0 else 0.0
            if pct > peak_pct:
                peak_pct = pct
                peak_row = line.strip()
        except Exception:
            continue

print(f"PEAK_VRAM_PERCENT={peak_pct:.2f}")
if peak_row:
    print(f"PEAK_VRAM_ROW={peak_row}")
if peak_pct > 95.0:
    print("ALERT: VRAM exceeded 95% threshold.")
else:
    print("VRAM_OK: Peak stayed at or below 95%.")
PY

echo
echo "If eval_loss at step 500 jumps significantly over baseline, stop immediately."
echo "Training log: ${TRAIN_LOG}"
echo "VRAM log:     ${VRAM_LOG}"
