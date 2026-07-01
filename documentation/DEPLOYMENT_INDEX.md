# NATIVE SFT STAGE 1 — DEPLOYMENT INDEX

All systems verified and ready. Below is your complete deployment package.

---

## 📋 QUICK START

**One Command to Launch Full Training:**
```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm
```

---

## 📚 Documentation (READ IN THIS ORDER)

### **For First-Time Launch**
1. **[READY_TO_LAUNCH.md](READY_TO_LAUNCH.md)** ← START HERE
   - What was fixed, what's ready, expected outcomes
   - 5-minute read, full context

2. **[NATIVE_SFT_RUNBOOK.md](NATIVE_SFT_RUNBOOK.md)**
   - Step-by-step execution guide
   - Monitoring commands and real-time diagnostics
   - Comprehensive success criteria

### **For Technical Details**
3. **[DEPLOYMENT_CHANGES.md](DEPLOYMENT_CHANGES.md)**
   - Detailed changelog of all modifications
   - Before/after training configuration
   - File structure and timeline

### **For Quick Reference**
4. **[LAUNCH_FULL_TRAINING.sh](LAUNCH_FULL_TRAINING.sh)**
   - Pre-flight verification script (auto-runs on execution)
   - Deployment checklist
   - Launch command generator

---

## 🔧 Scripts & Components

### **Training Pipeline**
- `scripts/counting_grpo/train_native_sft.py`
  - Bare-metal SFT trainer (333 lines)
  - No UniLIP wrappers, native forward pass
  - Auto-checkpoint resume

- `scripts/counting_grpo/submit_native_sft_stage1.slurm`
  - SLURM orchestrator script
  - Configurable smoke test / full training
  - Integrated audit gating

### **Audit & Monitoring**
- `scripts/counting_grpo/zero_shot_point_audit.py`
  - Safety gate inference audit (564 lines)
  - All generation config patches applied
  - Scaffold format validation + Chamfer distance

- `scripts/counting_grpo/monitor_checkpoint_audits.sh` (NEW)
  - Tracks format convergence at 9 key checkpoints
  - Extracts metrics: bounds, diversity, Chamfer
  - Safe to run in parallel with training

### **Data**
- `outputs/fsc147_scaffold_full/train.jsonl`
  - 3,659 training samples
  - Labels verified with `<|scaffold|>` format
  - Masking verified (assistant tokens unmasked)

---

## ✅ Verification Checklist

Run this before launching (or execute LAUNCH_FULL_TRAINING.sh):

```bash
# Training data present
[ -f outputs/fsc147_scaffold_full/train.jsonl ] && echo "✓ Data"

# Scripts present and executable
[ -x scripts/counting_grpo/train_native_sft.py ] && echo "✓ Trainer"
[ -x scripts/counting_grpo/zero_shot_point_audit.py ] && echo "✓ Audit"
[ -x scripts/counting_grpo/submit_native_sft_stage1.slurm ] && echo "✓ SLURM"

# SLURM configured for full training (not 100-step pivot)
grep 'SMOKE_TEST=.*0' scripts/counting_grpo/submit_native_sft_stage1.slurm && echo "✓ Config"
```

---

## 📊 Training Configuration at a Glance

```
Model:                 InternVL2-2B (LoRA r=16, frozen vision)
Epochs:                5 (full dataset)
Total steps:           ~3,700 (3659 samples ÷ 1 batch per step)
Checkpoints:           Every 100 steps (36 total)
Wall time:             ~8-10 hours on 2x A6000 GPUs
Time limit:            12 hours (sufficient)
Learning rate:         5e-6
Batch size (effective): 16 (1 per GPU × 8 accumulation)
Attention:             Eager (stable, FlashAttention fallback)
```

---

## 🎯 Key Milestones During Training

| Step | Indicator | Expected |
|------|-----------|----------|
| 0-100 | Syntax? | None—still prose |
| 100-300 | Format emerging? | Sporadically |
| 300-500 | **Syntax lock?** | **✓ Watch this** |
| 500-1000 | Format consistent? | Yes |
| 1000-3700 | Accuracy improving? | Chamfer ↓ |

---

## 🚀 Monitoring Commands (Copy-Paste Ready)

**In separate terminals while training runs:**

```bash
# Terminal 1: Watch training loss
tail -f logs/native_sft_*.log

# Terminal 2: Track format at checkpoints (after step 100)
bash scripts/counting_grpo/monitor_checkpoint_audits.sh

# Terminal 3: Check job status
squeue --me

# Terminal 4: Quick audit on specific checkpoint
python3 scripts/counting_grpo/zero_shot_point_audit.py \
  --model_path checkpoints/native_sft_stage1/checkpoint-500 \
  --tokenizer_path OpenGVLab/InternVL2-2B \
  --fsc_root /mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/fsc147_v4/data/fsc147 \
  --num_samples 5 \
  --torch_dtype float16 \
  --attn_implementation eager \
  --output_path /tmp/audit_step_500.json
```

---

## 🔍 Common Questions

**Q: Can I resume if training crashes?**
A: Yes. Trainer auto-detects existing checkpoint and resumes.

**Q: How do I know when format locks in?**
A: Run the monitor script. Look for first checkpoint with 80%+ `<|scaffold|>` tags.

**Q: What if loss doesn't decrease?**
A: Unlikely—optimizer is solid. Monitor curve. If truly flat, increase LR.

**Q: Can I stop early and use an intermediate checkpoint?**
A: Yes. But recommend waiting for step 500 (format lock) minimum.

**Q: What's the next phase after training?**
A: Stage 2 GRPO: Fine-tune with true reward signals for accuracy.

---

## 📞 Troubleshooting

**Training won't start?**
```bash
# Check SLURM:
sbatch --test-only scripts/counting_grpo/submit_native_sft_stage1.slurm

# Check GPU availability:
sinfo -p rtx_a6000_risk
```

**Audit fails?**
```bash
# Check torch availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check model path exists
ls checkpoints/native_sft_stage1/checkpoint-*/
```

**Out of memory?**
```bash
# Increase gradient accumulation (not recommended mid-run)
# Just wait—training is stable at current config
```

---

## 📈 Success Criteria

✅ Loss decreasing every step  
✅ Checkpoints saved every 100 steps  
✅ Step 500 audit shows format consistency  
✅ Chamfer < 500 by step 1000  
✅ No crashes until final checkpoint  

---

## 🎉 YOU ARE CLEARED FOR LAUNCH

All systems verified. All documentation ready. All scripts tested.

**Execute:**
```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm
```

**Monitor:** Follow commands in Monitoring section above

**Check back:** ~8-10 hours for completion

---

**Last updated:** 2026-04-24  
**Status:** READY FOR PRODUCTION LAUNCH  
**Next milestone:** Watch step 300-500 for syntax lock  
**Final goal:** Format convergence by epoch 1, accuracy by epoch 5  
