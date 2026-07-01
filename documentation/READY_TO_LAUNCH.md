# 🚀 NATIVE SFT STAGE 1 — DEPLOYMENT COMPLETE

## Executive Summary

The native SFT training pipeline is **production-ready** and fully debugged. All infrastructure components have been verified and optimized for full 5-epoch training (~3,700 steps over ~8-10 hours).

---

## 🎯 What Was Fixed

### **Inference Generation Issues (NOW RESOLVED)**

1. **Hidden `max_length=20` constraint**
   - Problem: Generation config had a hard limit of 20 tokens total
   - Impact: With 31-token input prompt, max_new_tokens became negative → empty strings
   - **Fix**: Override to `max_length=None` with explicit `max_new_tokens=256`

2. **Missing `img_context_token_id`**
   - Problem: InternVL2's custom generate() asserted this attribute must be set
   - Impact: AssertionError during inference
   - **Fix**: Dynamically resolve from tokenizer and assign before generation

### **Result**
✅ Model now generates full 256-token outputs (no truncation)
✅ Current output: "To determine the number of apples..." (structured prose, not empty)
✅ Next phase: Model will learn to emit `<|scaffold|>` format as training progresses

---

## 📊 Training Plan

### **Phase 1: Syntax Learning (Steps 0-500)**
- Model learns to emit `<|scaffold|>` and `<|count|>` tokens
- Coordinates may be inaccurate but format locks in
- Expected: By step 300-500, all outputs match target format

### **Phase 2: Accuracy Refinement (Steps 500-3700)**
- Model learns to place coordinates correctly
- Chamfer distance decreases as model understands spatial relationships
- Expected: By step 1000, Chamfer < 500; by final epoch, further improvement

### **Convergence Indicators**
| Milestone | Indicator | Action |
|-----------|-----------|--------|
| Step 100 | Any scaffold tokens? | Run audit |
| Step 300 | Consistent format? | Monitor emerging |
| Step 500 | 80%+ format match? | Tracking on schedule |
| Step 1000 | Chamfer < 500? | Accuracy phase working |
| Final | All metrics stable? | Ready for GRPO Stage 2 |

---

## 🚀 Launch Instructions

### **Quick Start (Copy-Paste Ready)**

```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount

# Launch full training (5 epochs, ~3700 steps)
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm

# You should see:
# Submitted batch job XXXXXXX
```

### **Monitoring (in separate terminals)**

```bash
# Terminal 1: Watch training loss
tail -f logs/native_sft_*.log

# Terminal 2: Track format convergence (run after step 100 is saved)
bash scripts/counting_grpo/monitor_checkpoint_audits.sh

# Terminal 3: Check job status
squeue --me
```

---

## 📁 What's Ready

### **Core Training Pipeline**
✅ `train_native_sft.py` — Bare-metal SFT trainer (no UniLIP wrappers)
✅ `zero_shot_point_audit.py` — Safety gate audit (all patches applied)
✅ `submit_native_sft_stage1.slurm` — Full training orchestrator (100-step pivot removed)

### **Monitoring & Analysis**
✅ `monitor_checkpoint_audits.sh` — Auto-audit at 9 key checkpoints
✅ `NATIVE_SFT_RUNBOOK.md` — Complete step-by-step guide
✅ `DEPLOYMENT_CHANGES.md` — Technical changelog
✅ `LAUNCH_FULL_TRAINING.sh` — Pre-flight verification script

### **Training Data**
✅ `outputs/fsc147_scaffold_full/train.jsonl` — 3,659 samples with scaffold labels
✅ Labels verified: `<|thought|>`, `<|answer|>`, `<|scaffold|> [...] <|count|>` format
✅ Masking verified: Assistant tokens unmasked (will be trained)

---

## 🎓 Architecture Summary

### **Model Stack**
- Base: InternVL2-2B (vision + language)
- LoRA: r=16, α=32, applied to language_model layers (frozen vision)
- Attention: Eager mode (stable, no FlashAttention issues)
- Chat template: InternLM2 with system prompt injection

### **Training Config**
- Learning rate: 5e-6 (LoRA only, base frozen)
- Batch: 1 per GPU × 8 accumulation = 16 effective
- Loss: Native causal LM loss (no custom compute_loss)
- Checkpoints: Every 100 steps (36 total), keep 2 recent

### **Inference Fixes Applied**
- Generation config: `max_length=None`, explicit `max_new_tokens`
- Image token ID: Auto-resolved from tokenizer
- Cache handling: Bypass DynamicCache objects (transformers v4.50 compat)
- Module cache: Local-only path to bypass HF cache exhaustion

---

## 📈 Expected Outcomes

### **Baseline (Current)**
- Step 107: Generic counting prose with hallucinated coordinates
- Format: Not yet learned
- Chamfer: ~1000 (max penalty, no valid points)

### **Target (End of Epoch 1 @ ~750 steps)**
- Format: All outputs match `<|scaffold|> [...] <|count|>` format
- Coordinates: Still semi-random but properly formatted
- Chamfer: ~400-600 (large, but improving)

### **Final (End of Training @ ~3700 steps)**
- Format: Consistent, locked in
- Coordinates: Learned placement patterns
- Chamfer: < 300 (meaningful accuracy)
- Ready for GRPO Stage 2

---

## 🛑 Safeguards & Contingencies

### **If training fails:**
1. Check logs: `tail -f logs/native_sft_*.err`
2. Restart with checkpoint resume (auto-detects if exists)
3. Increase time limit if needed (currently 12 hours, usually 8-10 used)

### **If format doesn't emerge by step 500:**
1. Verify system prompt matches (already done ✓)
2. Check chat template applied correctly (already verified ✓)
3. Inspect raw token sequences in audit output
4. Consider LoRA rank increase (r=32 for next run)

### **If accuracy doesn't improve after step 1000:**
1. Normal—early training phase focuses on format over accuracy
2. Monitor loss curve: should be monotonically decreasing
3. If loss stopped decreasing, may need learning rate adjustment

---

## 📞 What to Watch For

### **Green Flags** ✅
- Loss decreasing smoothly every step
- Checkpoints saved every 100 steps
- Step 300 audit shows `<|scaffold|>` tokens appearing
- Step 500 audit shows format consistency
- Step 1000 Chamfer < 500

### **Yellow Flags** ⚠️
- Loss plateaus before step 500 (may need higher LR)
- Format inconsistent at step 500 (check prompt alignment)
- Huge variance in Chamfer across samples (training instability)

### **Red Flags** 🛑
- CUDA OOM or crash after step 100 (unlikely with current config)
- Loss NaN (usually indicates learning rate too high)
- Still writing prose at step 1000 (chat template mismatch)

---

## 🎯 Next Steps After Training Completes

1. **Identify syntax lock point**: Which checkpoint has 80%+ scaffold format?
2. **Select baseline**: Use checkpoint with good format + reasonable Chamfer
3. **Run Stage 2**: GRPO fine-tuning with true reward signals
4. **Deploy**: Use final Stage 2 checkpoint for production

---

## 🎉 Ready Status

```
Infrastructure:     ✅ READY
Training pipeline:  ✅ READY
Audit system:       ✅ READY
Data validation:    ✅ READY
Compute allocation: ✅ READY (2x A6000, 12h time limit)
Documentation:      ✅ READY
```

---

## 🚀 FINAL COMMAND

```bash
cd /mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount
sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm
```

**Expected output**: `Submitted batch job XXXXXXX`

All systems go. Launch on your signal. 🚀

---

**Prepared**: 2026-04-24  
**Checkpoint**: native_sft_stage1  
**Expected completion**: ~8-10 hours after launch  
**Next phase**: GRPO Stage 2 refinement  
