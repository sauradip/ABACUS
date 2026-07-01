# Native SFT Stage 1 — Full Training Runbook

## 🚀 Ready to Launch

All infrastructure is set. The native SFT trainer and audit pipeline are production-ready. Here's how to unleash the full run:

---

## **Step 1: Submit the Full Training Job**

```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm
```

### What happens:
- **Training**: 5 full epochs (~3,700 steps total)
- **Checkpoints**: Saved every 100 steps (36 total checkpoints)
- **Duration**: ~8-10 hours on 2x A6000 GPUs
- **Time limit**: 12 hours (sufficient)

### Expected output:
```
=== Running full Stage 1 training (5 epochs, ~3700 steps) ===
...
=== Training Complete. Running audit on final checkpoint ===
```

---

## **Step 2: Monitor Format Convergence (Optional but Recommended)**

While training runs, track when the model learns the `<|scaffold|>` syntax by running audits at milestone checkpoints:

```bash
# In a separate terminal, once training has progressed past step 100:
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
bash scripts/counting_grpo/monitor_checkpoint_audits.sh
```

This script will audit checkpoints at: **100, 200, 300, 400, 500, 700, 1000, 1500, 2000**

### What to watch for:
1. **Syntax Lock (~step 300-500)**: Model starts emitting `<|scaffold|>` and `<|count|>` tokens instead of prose
2. **Format Stabilization (~step 700-1000)**: All outputs match the format
3. **Accuracy Improvement (~step 1500+)**: Chamfer distances begin decreasing

Example output:
```
[AUDIT] Running audit on checkpoint-300...
  ✓ Step 300: bounds=FAIL  diversity=FAIL  chamfer=800.5
  
[AUDIT] Running audit on checkpoint-500...
  ✓ Step 500: bounds=FAIL  diversity=FAIL  chamfer=650.2

[AUDIT] Running audit on checkpoint-1000...
  ✓ Step 1000: bounds=FAIL  diversity=PASS  chamfer=450.1
```

---

## **Step 3: Check Training Logs in Real-Time**

```bash
# Monitor training loss:
tail -f /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount/logs/native_sft_*.log

# Or check the job queue:
squeue --me
```

---

## **Step 4: Inspect Format Emergence**

To quickly see if scaffold format is emerging:

```bash
# After running monitor_checkpoint_audits.sh, check which checkpoints have scaffold tokens:
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
for f in checkpoints/native_sft_stage1/audit_step_*.json; do
  if grep -q '<|scaffold|>' "$f"; then
    STEP=$(basename "$f" | sed 's/[^0-9]//g')
    echo "✓ Step $STEP: Contains scaffold format"
  fi
done
```

---

## **Step 5: Post-Training Analysis**

Once training completes, the final audit will run automatically. To inspect it:

```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
cat checkpoints/native_sft_stage1/zero_shot_point_audit_final.json | python3 -m json.tool | head -100
```

---

## **Key Milestones & Diagnostics**

| Step Range | Expected Behavior | Diagnostic |
|------------|-------------------|-----------|
| 0-100 | Generic prose with coordinates | No `<|scaffold|>` tokens |
| 100-300 | Format emerging sporadically | Mixing prose and scaffold |
| 300-500 | Syntax lock-in | Consistent `<|scaffold|>` tags |
| 500-1000 | Format stabilized | All outputs match format |
| 1000+ | Accuracy refinement | Chamfer distance improving |

### **If at Step 1000+ model is STILL writing essays:**
1. Check if system prompt in inference matches training system prompt ✓ (Already verified)
2. Verify chat template alignment ✓ (Already verified)
3. Consider if LoRA rank r=16 is sufficient (can increase to r=32 for next run)

---

## **Quick Commands Cheat Sheet**

```bash
# Check job status
squeue --me

# View full training output
tail -f logs/native_sft_*.log

# List all saved checkpoints
ls -la checkpoints/native_sft_stage1/checkpoint-*/

# Run audit on a specific checkpoint
python3 scripts/counting_grpo/zero_shot_point_audit.py \
  --model_path checkpoints/native_sft_stage1/checkpoint-500 \
  --tokenizer_path OpenGVLab/InternVL2-2B \
  --fsc_root /mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/fsc147_v4/data/fsc147 \
  --num_samples 5 \
  --seed 42 \
  --torch_dtype float16 \
  --attn_implementation eager \
  --output_path /tmp/audit_step_500.json

# Extract Chamfer score from audit
python3 -c "import json; d=json.load(open('checkpoints/native_sft_stage1/audit_step_500.json')); print(f\"Chamfer: {d['chamfer_mean']}\")"

# Count scaffold occurrences in output
grep -o '<|scaffold|>' checkpoints/native_sft_stage1/audit_step_500.json | wc -l
```

---

## **Success Criteria**

✅ **Training completes without crashes**
✅ **Loss decreases monotonically** (expected from step 100 onwards)
✅ **By step 500**: Model outputs strict `<|scaffold|> [...] <|count|>` format
✅ **By step 1000**: Chamfer distance < 500 (accuracy meaningful)
✅ **Final**: Ready for Stage 2 (GRPO refinement)

---

## **Next Steps After Full Training**

1. Save the best checkpoint (based on format + Chamfer)
2. Run Stage 2: GRPO fine-tuning on true rewards
3. Deploy to production counting pipeline

---

**You are cleared to launch. 🚀**
