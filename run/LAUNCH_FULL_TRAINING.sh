#!/bin/bash
# Native SFT Full Training Launch Guide
# =====================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      NATIVE SFT STAGE 1 — FULL TRAINING LAUNCH                ║"
echo "║  Ready for production: 5 epochs, ~3700 steps, format learning ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

REPO_DIR="/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount"
SCRIPT_PATH="$REPO_DIR/scripts/counting_grpo/submit_native_sft_stage1.slurm"

echo "📋 DEPLOYMENT VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check training data
if [[ -f "$REPO_DIR/outputs/fsc147_scaffold_full/train.jsonl" ]]; then
  COUNT=$(wc -l < "$REPO_DIR/outputs/fsc147_scaffold_full/train.jsonl")
  echo "✅ Training data: $COUNT samples"
else
  echo "❌ Training data not found!"
  exit 1
fi

# Check trainer script
if [[ -f "$REPO_DIR/scripts/counting_grpo/train_native_sft.py" ]]; then
  echo "✅ Trainer script: Present"
else
  echo "❌ Trainer script not found!"
  exit 1
fi

# Check audit script
if [[ -f "$REPO_DIR/scripts/counting_grpo/zero_shot_point_audit.py" ]]; then
  echo "✅ Audit script: Present"
else
  echo "❌ Audit script not found!"
  exit 1
fi

# Check SLURM script
if [[ -f "$SCRIPT_PATH" ]]; then
  SMOKE_TEST=$(grep 'SMOKE_TEST=' "$SCRIPT_PATH" | head -1 | grep -o ':-[0-9]' | grep -o '[0-9]')
  if [[ "$SMOKE_TEST" == "0" ]]; then
    echo "✅ SLURM script: Configured for FULL TRAINING (SMOKE_TEST=0)"
  else
    echo "⚠️  SLURM script: SMOKE_TEST=$SMOKE_TEST (override with SMOKE_TEST=0 sbatch ...)"
  fi
else
  echo "❌ SLURM script not found!"
  exit 1
fi

echo ""
echo "🎯 TRAINING CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Epochs:               5 (full dataset)"
echo "Approx total steps:   3,700"
echo "Batch size:           1 per GPU × 8 accumulation = 16 effective"
echo "Learning rate:        5e-6 (LoRA only)"
echo "Save frequency:       Every 100 steps (36 checkpoints)"
echo "Wall time:            ~8-10 hours on 2x A6000 GPUs"
echo "SLURM time limit:     12 hours"
echo ""
echo "Output directory:     $REPO_DIR/checkpoints/native_sft_stage1/"
echo "Logs directory:       $REPO_DIR/logs/"
echo ""

echo "🚀 LAUNCH COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option 1: Default (full training)"
echo "  cd $REPO_DIR"
echo "  sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm"
echo ""
echo "Option 2: If you need a quick smoke test first (1 step)"
echo "  cd $REPO_DIR"
echo "  SMOKE_TEST=1 sbatch scripts/counting_grpo/submit_native_sft_stage1.slurm"
echo ""

echo "📊 MONITORING DURING TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Terminal 1 - Watch training logs:"
echo "  tail -f $REPO_DIR/logs/native_sft_*.log"
echo ""
echo "Terminal 2 - Track format convergence (after step 100):"
echo "  bash $REPO_DIR/scripts/counting_grpo/monitor_checkpoint_audits.sh"
echo ""
echo "Terminal 3 - Check job status:"
echo "  squeue --me"
echo ""

echo "✅ EXPECTED MILESTONES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Steps   0-100:  Prose + coordinates (no format lock)"
echo "Steps 100-300:  Format emerging sporadically"
echo "Steps 300-500:  Syntax lock-in ← WATCH FOR THIS"
echo "Steps 500-1000: Format stabilization + accuracy start"
echo "Steps 1000-3700: Final convergence + refinement"
echo ""

echo "📖 FOR DETAILED INFORMATION, READ:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • NATIVE_SFT_RUNBOOK.md - Complete execution guide"
echo "  • DEPLOYMENT_CHANGES.md - What changed and why"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "Ready to launch. All systems go. 🚀"
echo "═══════════════════════════════════════════════════════════════"
