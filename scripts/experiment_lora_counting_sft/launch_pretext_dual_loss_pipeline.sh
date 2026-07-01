#!/bin/bash

################################################################################
# Pretext → Dual-Loss Training Pipeline (with WandB Tracking)
#
# Phase 1: Common object localization pretext task (33K image pairs)
# Phase 2: Dual-loss counting training with AR loss (49.8K with object_centers)
#
# All metrics logged to Weights & Biases for real-time monitoring
#
# Structure:
#   /data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/
#   ├── phase1_pretext/
#   │   ├── checkpoints/
#   │   ├── logs/
#   │   └── config.json
#   ├── phase2_dual_loss/
#   │   ├── checkpoints/
#   │   ├── logs/
#   │   └── config.json
#   ├── eval/
#   └── README.md
#
# WandB Projects:
#   - pretext_dual_loss_pipeline (all training runs)
#
################################################################################

set -e

# Ensure WandB login (uncomment and set if needed)
# export WANDB_API_KEY="your_key_here"

# ============================================================================
# Configuration
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_ROOT="/data/amondal/unicount_runs/pretext_dual_loss_pipeline_${TIMESTAMP}"
BASE_MODEL="/data/amondal/model_cache/UniLIP-3B"
MLLM_HF_PATH="/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2"

# Phase 1: Pretext (image pair correspondence)
PHASE1_OUTPUT="${PIPELINE_ROOT}/phase1_pretext"
PHASE1_CHECKPOINTS="${PHASE1_OUTPUT}/checkpoints"
PHASE1_LOGS="${PHASE1_OUTPUT}/logs"

# Phase 2: Dual-Loss (counting + AR)
PHASE2_OUTPUT="${PIPELINE_ROOT}/phase2_dual_loss"
PHASE2_CHECKPOINTS="${PHASE2_OUTPUT}/checkpoints"
PHASE2_LOGS="${PHASE2_OUTPUT}/logs"

# Shared evaluation
EVAL_DIR="${PIPELINE_ROOT}/eval"

# Create directory structure
mkdir -p "$PHASE1_CHECKPOINTS" "$PHASE1_LOGS"
mkdir -p "$PHASE2_CHECKPOINTS" "$PHASE2_LOGS"
mkdir -p "$EVAL_DIR"

# ============================================================================
# Logging Setup
# ============================================================================

PIPELINE_LOG="${PIPELINE_ROOT}/pipeline.log"
exec 1> >(tee -a "$PIPELINE_LOG")
exec 2>&1

log_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""
}

log_phase() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================================
# Phase 1: Pretext Training
# ============================================================================

log_header "PHASE 1: PRETEXT TRAINING (Image Pair Correspondence)"

log_phase "Task: Localize common objects across 33K image pairs"
log_phase "Dataset: /data/amondal/UniCountData/unicount_pretext (33,489 records)"
log_phase "Base Model: UniLIP-3B (fresh, no warm-start)"
log_phase "Output: ${PHASE1_OUTPUT}"

# Phase 1 hyperparameters
NUM_GPUS=8
BATCH_PER_GPU=2
GRAD_ACCUM=1
EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_PER_GPU * GRAD_ACCUM))
LR=1e-5
EPOCHS=1  # Pretext is lightweight, 1 epoch sufficient
WARMUP_RATIO=0.06

cat > "${PHASE1_OUTPUT}/config.json" << EOF
{
  "phase": "pretext_training",
  "dataset": "unicount_pretext",
  "num_examples": 33489,
  "task": "common_object_localization",
  "base_model": "UniLIP-3B",
  "training": {
    "num_gpus": $NUM_GPUS,
    "batch_per_gpu": $BATCH_PER_GPU,
    "grad_accum": $GRAD_ACCUM,
    "effective_batch_size": $EFFECTIVE_BATCH,
    "learning_rate": "$LR",
    "epochs": $EPOCHS,
    "warmup_ratio": $WARMUP_RATIO,
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "gradient_checkpointing": true
  },
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05
  },
  "output_directory": "$PHASE1_OUTPUT"
}
EOF

log_phase "Phase 1 Config: ${PHASE1_OUTPUT}/config.json"

# Launch Phase 1
torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  scripts/experiment_lora_counting_sft/train_pretext_lora_3b.py \
  --model_name_or_path "$BASE_MODEL" \
  --mllm_hf_path "$MLLM_HF_PATH" \
  --cache_dir /data/amondal/UniCountData/unicount_pretext \
  --output_dir "$PHASE1_CHECKPOINTS" \
  \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type cosine \
  \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --bf16 True \
  \
  --save_steps 200 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_strategy steps \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --run_name "phase1_pretext_${TIMESTAMP}" \
  --project_name "pretext_dual_loss_pipeline" \
  --seed 42 \
  --logging_first_step \
  2>&1 | tee "$PHASE1_LOGS/train_${TIMESTAMP}.log"

log_phase "Phase 1 Complete. Checkpoints: ${PHASE1_CHECKPOINTS}"

# ============================================================================
# Phase 1 → Phase 2: Checkpoint Extraction & Reinitialization
# ============================================================================

log_header "CHECKPOINT EXTRACTION: Finding best Phase 1 checkpoint"

# Find best checkpoint
BEST_CHECKPOINT=$(python3 << PYTHON_SCRIPT
import json
import os
from pathlib import Path

checkpoints_dir = "$PHASE1_CHECKPOINTS"
checkpoint_dirs = sorted([d for d in Path(checkpoints_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

if checkpoint_dirs:
    # Latest checkpoint (already best saved)
    best = checkpoint_dirs[-1]
    print(best.name)
else:
    print("checkpoint-final")
PYTHON_SCRIPT
)

log_phase "Best Phase 1 checkpoint: ${BEST_CHECKPOINT}"

# Extract LoRA adapter for Phase 2 initialization
PHASE1_BEST_ADAPTER="${PHASE1_OUTPUT}/adapter_extracted"
python3 << PYTHON_SCRIPT
import sys
sys.path.insert(0, "$(pwd)")
sys.path.insert(0, "$(pwd)/UniLip_mod")
import torch
from peft import PeftModel
from scripts.counting_grpo.train_hf_multi_image_count_sft import load_unilip_class

best_checkpoint = "$PHASE1_CHECKPOINTS/$BEST_CHECKPOINT"
output_path = "$PHASE1_BEST_ADAPTER"

print(f"Loading base model...")
model_cls = load_unilip_class()
base_model = model_cls.from_pretrained(
    "$BASE_MODEL",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Applying Phase 1 LoRA adapter from: {best_checkpoint}")
model = PeftModel.from_pretrained(base_model, best_checkpoint)

print(f"Merging and saving to: {output_path}")
merged = model.merge_and_unload()
merged.save_pretrained(output_path, safe_serialization=True)

print(f"✓ Adapter extracted to: {output_path}")
PYTHON_SCRIPT

log_phase "Phase 1 adapter extracted: ${PHASE1_BEST_ADAPTER}"

# ============================================================================
# Phase 2: Dual-Loss Training (Using Phase 1 pretrained LoRA)
# ============================================================================

log_header "PHASE 2: DUAL-LOSS TRAINING (Counting + Attention Regularization)"

log_phase "Task: Counting + Object-Focused Attention Loss"
log_phase "Dataset: /data/amondal/UniCount/outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json (49.8K)"
log_phase "Base Model: UniLIP-3B (initialized from Phase 1)"
log_phase "Output: ${PHASE2_OUTPUT}"

# Phase 2 hyperparameters
EPOCHS=3
LR=1e-5
LAMBDA_AR=0.1
AR_SIGMA=1.0
AR_TEMP=0.1

cat > "${PHASE2_OUTPUT}/config.json" << EOF
{
  "phase": "dual_loss_training",
  "dataset": "balanced_mix_v3s",
  "num_examples": 49847,
  "task": "counting_with_attention_regularization",
  "base_model": "UniLIP-3B",
  "pretrained_from": "phase1_pretext",
  "training": {
    "num_gpus": $NUM_GPUS,
    "batch_per_gpu": 2,
    "grad_accum": 1,
    "effective_batch_size": 16,
    "learning_rate": "$LR",
    "epochs": $EPOCHS,
    "warmup_ratio": 0.06,
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "gradient_checkpointing": true
  },
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05
  },
  "ar_loss": {
    "lambda": $LAMBDA_AR,
    "sigma": $AR_SIGMA,
    "temperature": $AR_TEMP
  },
  "output_directory": "$PHASE2_OUTPUT"
}
EOF

log_phase "Phase 2 Config: ${PHASE2_OUTPUT}/config.json"

# Launch Phase 2
WANDB_PROJECT="pretext_dual_loss_pipeline" torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
  --model_name_or_path "$PHASE1_BEST_ADAPTER" \
  --mllm_hf_path "$MLLM_HF_PATH" \
  --data_path outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json \
  --output_dir "$PHASE2_CHECKPOINTS" \
  \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.06 \
  --lr_scheduler_type cosine \
  \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --bf16 True \
  \
  --lambda_ar $LAMBDA_AR \
  --ar_sigma $AR_SIGMA \
  --ar_temperature $AR_TEMP \
  \
  --save_steps 500 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_strategy steps \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --run_name "phase2_dual_loss_${TIMESTAMP}" \
  --seed 42 \
  --logging_first_step \
  2>&1 | tee "$PHASE2_LOGS/train_${TIMESTAMP}.log"

log_phase "Phase 2 Complete. Checkpoints: ${PHASE2_CHECKPOINTS}"

# ============================================================================
# Validation & Best Checkpoint Selection
# ============================================================================

log_header "VALIDATION & CHECKPOINT SELECTION"

python3 << PYTHON_SCRIPT
import json
import os
from pathlib import Path

# Find best checkpoint from Phase 2
phase2_checkpoints = "$PHASE2_CHECKPOINTS"
checkpoint_dirs = sorted([d for d in Path(phase2_checkpoints).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

if checkpoint_dirs:
    best_checkpoint = checkpoint_dirs[-1]  # Last saved is best
    print(f"Best Phase 2 Checkpoint: {best_checkpoint.name}")
else:
    print("No checkpoints found!")
    exit(1)

PYTHON_SCRIPT

log_phase "Extracting best Phase 2 checkpoint..."

# Extract best Phase 2 adapter
PHASE2_BEST_ADAPTER="${PHASE2_OUTPUT}/adapter_extracted"
python3 << PYTHON_SCRIPT
import os
from pathlib import Path
from peft import AutoPeftModelForCausalLM

phase2_checkpoints = "$PHASE2_CHECKPOINTS"
checkpoint_dirs = sorted([d for d in Path(phase2_checkpoints).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

if checkpoint_dirs:
    best_checkpoint = checkpoint_dirs[-1]
    output_path = "$PHASE2_BEST_ADAPTER"

    print(f"Extracting from: {best_checkpoint}")
    print(f"Output: {output_path}")

    model = AutoPeftModelForCausalLM.from_pretrained(best_checkpoint, trust_remote_code=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)

    print(f"✓ Best adapter extracted to: {output_path}")
PYTHON_SCRIPT

# ============================================================================
# Final Summary
# ============================================================================

log_header "PIPELINE COMPLETE"

cat > "${PIPELINE_ROOT}/README.md" << 'EOF'
# Pretext → Dual-Loss Training Pipeline

## Structure

```
pretext_dual_loss_pipeline_<TIMESTAMP>/
├── phase1_pretext/
│   ├── checkpoints/              # Phase 1 LoRA checkpoints
│   ├── logs/                     # Training logs
│   ├── adapter_extracted/        # Best LoRA adapter (merged)
│   └── config.json              # Phase 1 hyperparameters
│
├── phase2_dual_loss/
│   ├── checkpoints/              # Phase 2 LoRA checkpoints
│   ├── logs/                     # Training logs
│   ├── adapter_extracted/        # Best LoRA adapter (merged)
│   └── config.json              # Phase 2 hyperparameters
│
├── eval/                         # Evaluation results
├── pipeline.log                  # Master training log
└── README.md                     # This file
```

## Phases

### Phase 1: Pretext Training (1 epoch)
- **Dataset**: unicount_pretext (33K image pairs)
- **Task**: Localize common objects across two images
- **Output**: LoRA adapter with spatial reasoning capability
- **Time**: ~12 hours (8× H100)
- **Checkpoint**: `phase1_pretext/adapter_extracted/`

### Phase 2: Dual-Loss Training (3 epochs)
- **Dataset**: balanced_mix_v3s (49.8K with object_centers)
- **Tasks**: Counting (CE loss) + Attention Regularization (AR loss)
- **Init**: Phase 1 LoRA weights
- **Output**: LoRA adapter for counting
- **Time**: ~36 hours (8× H100)
- **Checkpoint**: `phase2_dual_loss/adapter_extracted/`

## Usage

### Extract and Deploy Best Checkpoint

```bash
# Phase 1 adapter
PHASE1_ADAPTER="phase1_pretext/adapter_extracted"

# Phase 2 adapter (use this for counting)
PHASE2_ADAPTER="phase2_dual_loss/adapter_extracted"

# Load in inference
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(PHASE2_ADAPTER)
```

### Evaluate

```bash
python3 eval_counting_ctap_nrt.py \
  --adapter phase2_dual_loss/adapter_extracted \
  --output eval/dual_loss_benchmarks
```

## Logs

- **Phase 1**: `phase1_pretext/logs/train_*.log`
- **Phase 2**: `phase2_dual_loss/logs/train_*.log`
- **Master**: `pipeline.log`

## Configs

- **Phase 1**: `phase1_pretext/config.json`
- **Phase 2**: `phase2_dual_loss/config.json`
EOF

log_phase "Pipeline summary saved to: ${PIPELINE_ROOT}/README.md"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PIPELINE OUTPUT STRUCTURE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Root: ${PIPELINE_ROOT}"
echo ""
echo "Phase 1 Adapter (Pretext):    ${PHASE1_BEST_ADAPTER}"
echo "Phase 2 Adapter (Dual-Loss):  ${PHASE2_BEST_ADAPTER}"
echo ""
echo "All Logs:                     ${PIPELINE_ROOT}/logs/"
echo "Master Log:                   ${PIPELINE_LOG}"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "WANDB TRACKING"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Project: pretext_dual_loss_pipeline"
echo "Phase 1 Run: phase1_pretext_${TIMESTAMP}"
echo "Phase 2 Run: phase2_dual_loss_${TIMESTAMP}"
echo ""
echo "View metrics:"
echo "  https://wandb.ai/<your-username>/pretext_dual_loss_pipeline"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
