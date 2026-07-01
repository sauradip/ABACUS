#!/usr/bin/env bash
# Phase 3 Visual Jigsaw SFT launcher
# Uses DeepSpeed ZeRO-3 via Accelerate plugin for VRAM-safe 512-token visual sequence training.
set -euo pipefail

cd "$(dirname "$0")"
source /data/amondal/UniCount/unicount/bin/activate

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/data/amondal/unicount_runs/jigsaw_sft_base_init/logs_${RUN_TS}"
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
  --deepspeed_config_file /data/amondal/UniCount/configs/deepspeed_zero3.json \
  /data/amondal/UniCount/scripts/experiment_jigsaw/train_jigsaw_sft.py \
  --model_name_or_path /data/amondal/model_cache/UniLIP-3B \
  --data_path /data/amondal/UniCount/outputs/experiment_jigsaw/train/train_jigsaw.jsonl \
  --output_dir /data/amondal/unicount_runs/jigsaw_sft_base_init \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1024 \
  --num_train_epochs 3.0 \
  --logging_steps 10 \
  --save_steps 500 \
  2>&1 | tee -a "${TRAIN_LOG}"

# ---- Post-run telemetry summary ----
python3 - "${TRAIN_LOG}" "${VRAM_LOG}" <<'PY'
import ast, sys

train_log, vram_log = sys.argv[1], sys.argv[2]

losses, lrs = [], []
with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
            except Exception:
                continue
            if "loss" in d:
                losses.append(float(d["loss"]))
            if "learning_rate" in d:
                lrs.append(float(d["learning_rate"]))

first5 = losses[:5]
if first5:
    print(f"FIRST_5_STEPS_LOSSES={first5}")
    print(f"AVG_LOSS_FIRST_5={sum(first5)/len(first5):.6f}")
else:
    print("FIRST_5_STEPS_LOSSES=NA")

if lrs:
    print(f"FIRST_LR={lrs[0]:.10f}")
    print(f"WARMUP_OK={'YES' if any(x > 0 for x in lrs) else 'NO'}")

peak_pct, peak_row = 0.0, None
with open(vram_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used, total = float(parts[2]), float(parts[3])
            pct = 100.0 * used / total if total > 0 else 0.0
            if pct > peak_pct:
                peak_pct, peak_row = pct, line.strip()
        except Exception:
            continue

print(f"PEAK_VRAM_PERCENT={peak_pct:.2f}")
if peak_row:
    print(f"PEAK_VRAM_ROW={peak_row}")
if peak_pct > 70.0:
    print("ALERT: VRAM exceeded 70% — ZeRO-3 offload may not be working correctly.")
else:
    print("VRAM_OK: Peak within expected ZeRO-3 range (<=70% per GPU).")
PY

echo
echo "Training log: ${TRAIN_LOG}"
echo "VRAM log:     ${VRAM_LOG}"
