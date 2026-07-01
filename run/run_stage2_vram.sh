#!/usr/bin/env bash
set -e
cd /data/amondal/UniCount
source unicount/bin/activate

S2_DIR=/data/amondal/unicount_runs/jigsaw_vram_capture
mkdir -p "$S2_DIR"

# Poll VRAM every 1 s
nvidia-smi --query-gpu=index,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 > "$S2_DIR/vram.csv" &
SMI_PID=$!
echo "SMI PID=$SMI_PID  output=$S2_DIR/vram.csv"

accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  --deepspeed_config_file configs/deepspeed_zero3.json \
  scripts/experiment_jigsaw/train_jigsaw_sft.py \
  --model_name_or_path /data/amondal/model_cache/UniLIP-3B \
  --data_path outputs/experiment_jigsaw/train/train_jigsaw.jsonl \
  --output_dir "$S2_DIR" \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1024 \
  --max_steps 5 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  2>&1 | grep -E "^\{|loss|Error|Traceback"

kill "$SMI_PID" 2>/dev/null || true

echo "=== VRAM PEAK ==="
python3 - "$S2_DIR/vram.csv" << 'PY'
import sys
peaks = {}
with open(sys.argv[1]) as f:
    for line in f:
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 3: continue
        try:
            g, used, total = int(parts[0]), float(parts[1]), float(parts[2])
            peaks[g] = max(peaks.get(g, 0), used)
        except:
            pass
print("Per-GPU peak VRAM (MiB):")
for g in sorted(peaks):
    print(f"  GPU {g}: {peaks[g]:.0f} / 81920 MiB  ({100*peaks[g]/81920:.1f}%)")
if peaks:
    pk = max(peaks.values())
    thresh = 55 * 1024
    status = "PASS ✓" if pk < thresh else "FAIL ✗  OVER 55GB"
    print(f"Overall peak: {pk:.0f} MiB  ->  VRAM CHECK {status}")
PY
echo "STAGE2_DONE"
