#!/usr/bin/env bash
# run_jigsaw_final.sh — Phase 3 Visual Jigsaw SFT, 3-epoch full run
#
# Base model  : /data/amondal/model_cache/UniLIP-3B  (fresh LoRA, NO prior checkpoint)
# Global batch: 8 GPUs × batch_size=4 × grad_accum=2 = 64 samples/step
# Steps/epoch : ceil(3659 / 64) = 58
# Total steps : 58 × 3 = 174  (~8.7 min)
# Checkpoints : saved every epoch (steps 58, 116, 174) via --save_steps 58
#               No eval set → no automatic best-ckpt selection;
#               lowest-loss epoch identified post-hoc from loss curve below.

set -euo pipefail

cd /data/amondal/UniCount
source unicount/bin/activate

RUN_ID="jigsaw_sft_final_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/data/amondal/unicount_runs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Phase 3 Jigsaw SFT — Final Run ==="
echo "Base model : /data/amondal/model_cache/UniLIP-3B  (no prior checkpoint)"
echo "Run ID     : ${RUN_ID}"
echo "Output     : ${OUTPUT_DIR}"
echo "Started    : $(date)"
echo ""

accelerate launch \
  --num_processes=8 \
  --mixed_precision=bf16 \
  --deepspeed_config_file configs/deepspeed_zero3.json \
  scripts/experiment_jigsaw/train_jigsaw_sft.py \
  --model_name_or_path /data/amondal/model_cache/UniLIP-3B \
  --data_path /data/amondal/UniCount/outputs/experiment_jigsaw/train/train_jigsaw.jsonl \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 1024 \
  --num_train_epochs 3.0 \
  --logging_steps 5 \
  --save_steps 58 \
  --save_total_limit 3 \
  --allow_attn_fallback 1 \
  --report_to none \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Training complete: $(date) ==="

# ── Loss & grad_norm curve ────────────────────────────────────────────────────
echo ""
echo "=== Loss + grad_norm curve ==="
grep -E "^\{.*\"loss\"" "${OUTPUT_DIR}/train.log" | python3 -c "
import sys, ast
rows = []
for line in sys.stdin:
    try:
        d = ast.literal_eval(line.strip())
        if 'loss' in d:
            rows.append(d)
    except:
        pass
for d in rows:
    ep   = float(d.get('epoch', 0))
    loss = float(d['loss'])
    gnorm = float(d.get('grad_norm', -1))
    flag = '  *** HIGH NORM' if gnorm > 2.0 else ''
    print(f'  epoch={ep:.4f}  loss={loss:.4f}  grad_norm={gnorm:.3f}{flag}')
print()

# Step-50 audit
step50 = [d for d in rows if abs(float(d.get('epoch',0)) - 50/58) < 0.05]
if step50:
    gn = float(step50[-1].get('grad_norm', -1))
    status = 'PASS ✓' if gn < 2.0 else 'WARN — coordinate regression fighting linguistic priors'
    print(f'Step-50 grad_norm audit: {gn:.3f}  ->  {status}')
"

# ── JSON schema check at step ~100 ───────────────────────────────────────────
echo ""
echo "=== JSON schema check (step ~100 completion sample) ==="
python3 -c "
import re, ast, json

with open('${OUTPUT_DIR}/train.log') as f:
    text = f.read()

# Find any logged completion after epoch ~100/58 ≈ 1.72
completions = re.findall(r'completion[\"\':\s]+(\{[^\n]+\})', text, re.IGNORECASE)
if completions:
    for c in completions[:3]:
        try:
            obj = json.loads(c)
            has_bbox  = isinstance(obj.get('bbox'), list) and len(obj['bbox']) == 4
            has_count = 'count' in obj
            whitespace = c != c.strip() or '  ' in c
            print(f'  Sample  : {c}')
            print(f'  bbox OK : {has_bbox}')
            print(f'  count OK: {has_count}')
            print(f'  minified: {not whitespace}')
        except:
            print(f'  (Could not parse: {c[:80]})')
else:
    print('  (No inline completions logged — inspect train.log manually at step ~100)')
" 2>/dev/null || true

# ── TTC estimate from first-100-step throughput ───────────────────────────────
echo ""
echo "=== TTC estimate ==="
python3 -c "
import ast

rows = []
with open('${OUTPUT_DIR}/train.log') as f:
    for line in f:
        s = line.strip()
        if s.startswith('{') and 'train_runtime' in s:
            try: rows.append(ast.literal_eval(s))
            except: pass

if rows:
    d = rows[-1]
    sps  = float(d.get('train_steps_per_second', 0))
    total_steps = 174
    ttc_s = total_steps / sps if sps > 0 else 0
    print(f'  Steps/sec   : {sps:.3f}')
    print(f'  Total steps : {total_steps}')
    print(f'  TTC         : {ttc_s:.0f} s  ({ttc_s/60:.1f} min)')
else:
    print('  (Training still running or train_runtime not yet logged)')
" 2>/dev/null || true

# ── Best checkpoint by loss ───────────────────────────────────────────────────
echo ""
echo "=== Epoch checkpoints ==="
for ckpt in "${OUTPUT_DIR}"/checkpoint-*/; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    echo "  $ckpt  (step ${step})"
done
echo ""
echo "To select best: compare epoch losses above and use the checkpoint at"
echo "the corresponding step (step 58=epoch1, 116=epoch2, 174=epoch3)."
echo ""
echo "Next step: Global Consistency Audit"
echo "  Verify: sum(local_counts over 4-patch tiling) ≈ global_count (≤5% variance)"
