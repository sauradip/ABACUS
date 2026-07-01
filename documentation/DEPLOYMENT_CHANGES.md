# Full Training Deployment — Changes Summary

## 🎯 Objective
Transition from 100-step pivot testing to full 5-epoch production training with automatic format convergence monitoring.

## ✅ Changes Made

### 1. **SLURM Script Refactored** (`submit_native_sft_stage1.slurm`)
**What changed**: Removed the 100-step MAX_STEPS pivot logic
- **Before**: `FORCE_100_STEP_PIVOT=1` → restricted to 100 steps
- **After**: Direct full training with `MAX_STEPS=-1` (unlimited)
- **Saves**: Checkpoints every 100 steps (36 total across 5 epochs)
- **Audit**: Runs once on final checkpoint only (inline with main job)

### 2. **Checkpoint Monitor Script Created** (`monitor_checkpoint_audits.sh`)
**Purpose**: Track format convergence during training
- Audits 9 key milestones: 100, 200, 300, 400, 500, 700, 1000, 1500, 2000
- Extracts Chamfer and format metrics
- **Can run in parallel** with training (safe to call while job is running)
- Provides format emergence diagnostics

### 3. **Training Configuration**
```
Max epochs:              5 (full dataset, ~740 samples/epoch = ~3700 steps)
Batch size:              1 per GPU × 8 gradient accumulation = 16 effective
Learning rate:           5e-6 (LoRA only, base model frozen)
Attention implementation: eager (stable on A6000)
Save interval:           every 100 steps
Save limit:              keep last 2 checkpoints (economize disk)
```

### 4. **Inference Patches (Already Applied)**
✓ Generation config: `max_length=None` + explicit `max_new_tokens=256`
✓ Image context token ID: Dynamically resolved and assigned
✓ Full 256-token generation now working

## 📊 Training Timeline

| Phase | Steps | Duration | Key Event |
|-------|-------|----------|-----------|
| Early | 0-100 | ~30 min | Prose + coordinates (no format) |
| **Syntax Lock** | 100-500 | ~2 hours | Format emerges, stabilizes |
| **Refinement** | 500-1500 | ~3 hours | Accuracy improves |
| **Late** | 1500-3700 | ~3 hours | Final convergence |
| **Total** | 3700 | ~8-10 hours | Production ready |

## 🚀 How to Launch

```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm
```

Expected output:
```
Submitted batch job XXXXXXX
```

## 📈 How to Monitor

**Terminal 1** — Watch training logs:
```bash
tail -f logs/native_sft_*.log
```

**Terminal 2** — Track format convergence (after step 100 is saved):
```bash
bash scripts/counting_grpo/monitor_checkpoint_audits.sh
```

**Terminal 3** — Check job status:
```bash
squeue --me
```

## 🎯 Success Indicators

✅ Loss decreasing smoothly
✅ Step 300 audit shows scaffold format emerging
✅ Step 500 audit shows consistent format
✅ Step 1000 audit shows Chamfer < 500
✅ Final checkpoint ready for Stage 2 GRPO

## 🛑 Watchdog: If Something Goes Wrong

**If training fails before step 500:**
- Check logs for CUDA/OOM errors
- Verify checkpoint auto-resume: add `--resume_from_checkpoint $OUTPUT_DIR`

**If step 500+ still shows prose (no scaffold):**
- This would indicate chat template mismatch (already verified—unlikely)
- Log the audit output and investigate token sequences

**If Chamfer >> 1000 at step 1000:**
- Normal—coordinates are being hallucinated early
- Monitor trajectory: should decrease after step 500

---

## 📁 File Structure After Training

```
checkpoints/native_sft_stage1/
├── checkpoint-100/          (32 more checkpoints through checkpoint-3700)
├── checkpoint-200/
├── ...
├── checkpoint-3700/         ← FINAL (automatically loaded by monitor)
├── trainer_state.json       (training metadata, loss curves)
├── training_args.bin        (frozen for reproducibility)
├── zero_shot_point_audit_final.json  (final audit report)
└── audit_step_*.json        (from monitor_checkpoint_audits.sh)
```

---

## 🎓 Learning from This Run

After training completes:
1. Plot loss curve from `trainer_state.json`
2. Track Chamfer distance across checkpoints
3. Identify exact step where format locks in
4. Use that checkpoint as baseline for GRPO Stage 2

---

**Ready. All systems go. Launch on your signal. 🚀**
