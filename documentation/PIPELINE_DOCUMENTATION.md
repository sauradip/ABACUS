# Pretext → Dual-Loss Training Pipeline Documentation

## Overview

This pipeline implements a **two-phase training strategy** for UniLIP-3B object counting:

1. **Phase 1 (Pretext)**: Common object localization across image pairs (33K samples, 1 epoch)
2. **Phase 2 (Dual-Loss)**: Counting + Attention Regularization (49.8K samples, 3 epochs)

The pretext task prepares the model's spatial reasoning capability before the main counting task, resulting in improved performance across all benchmarks.

---

## Quick Start

### Launch Full Pipeline

```bash
cd /data/amondal/UniCount
bash scripts/experiment_lora_counting_sft/launch_pretext_dual_loss_pipeline.sh
```

**Expected runtime**: ~48 hours on 8× H100 GPUs (12h Phase 1 + 36h Phase 2)

### Validate Pipeline After Training

```bash
python3 scripts/experiment_lora_counting_sft/validate_pipeline.py \
  --pipeline-root /data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>
```

---

## Pipeline Structure

### Directory Organization

```
/data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/
│
├── phase1_pretext/                    # Pretext task outputs
│   ├── checkpoints/                   # LoRA checkpoints (multiple saves)
│   │   ├── checkpoint-200/
│   │   ├── checkpoint-400/
│   │   └── checkpoint-final/
│   ├── adapter_extracted/             # Best LoRA adapter (merged & extracted)
│   ├── logs/                          # TensorBoard + training logs
│   │   └── train_<TIMESTAMP>.log
│   └── config.json                    # Phase 1 hyperparameters
│
├── phase2_dual_loss/                  # Dual-loss task outputs
│   ├── checkpoints/                   # LoRA checkpoints (multiple saves)
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── checkpoint-final/
│   ├── adapter_extracted/             # Best LoRA adapter (merged & extracted)
│   ├── logs/                          # TensorBoard + training logs
│   │   └── train_<TIMESTAMP>.log
│   └── config.json                    # Phase 2 hyperparameters
│
├── eval/                              # Evaluation results (post-pipeline)
│   └── [benchmarks will be saved here]
│
├── DEPLOYMENT.json                    # Manifest pointing to best checkpoints
├── validation_report.json             # Validation results from validate_pipeline.py
├── pipeline.log                       # Master training log
└── README.md                          # This file
```

---

## Phase Details

### Phase 1: Pretext Training (Common Object Localization)

**Purpose**: Teach the model to find correspondences between image pairs and reason about spatial locations.

**Dataset**: `unicount_pretext` (33,489 examples)
- 33K image pairs from UniCount dataset
- Each pair has common object annotations (pixel coordinates)
- Task: "Localize common instances shared by the two images"
- Self-supervised signal: common_points in both images

**Configuration**:
```json
{
  "num_gpus": 8,
  "batch_per_gpu": 2,
  "effective_batch_size": 16,
  "learning_rate": "1e-5",
  "epochs": 1,
  "warmup_ratio": 0.06,
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05
  }
}
```

**Output**:
- `phase1_pretext/adapter_extracted/`: LoRA weights (spatial reasoning learned)
- `phase1_pretext/logs/train_*.log`: Training metrics (loss, learning rate, etc.)
- `phase1_pretext/config.json`: Reproducible hyperparameters

**Expected**:
- Initial loss: ~7-8 (cross-entropy for multiple choice)
- Final loss: ~4-5 (converged)
- Training time: ~12 hours

---

### Phase 2: Dual-Loss Training (Counting + Attention Regularization)

**Purpose**: Train counting model with attention-based spatial regularization, initialized from Phase 1.

**Dataset**: `balanced_mix_v3s` (49,847 records)
- All 6 annotation sources: FSC-147, UCount Part 1/2/3, UCount Crowd, ShanghaiTech
- 100% coverage with normalized object_centers [0,1]
- Dual supervision:
  - **CE Loss**: Standard language modeling on counting question-answer pairs
  - **AR Loss**: ObjectFocusedAttentionLoss pulls attention to object centers

**Configuration**:
```json
{
  "num_gpus": 8,
  "batch_per_gpu": 2,
  "effective_batch_size": 16,
  "learning_rate": "1e-5",
  "epochs": 3,
  "warmup_ratio": 0.06,
  "lora": {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05
  },
  "ar_loss": {
    "lambda": 0.1,
    "sigma": 1.0,
    "temperature": 0.1
  }
}
```

**Output**:
- `phase2_dual_loss/adapter_extracted/`: **Best checkpoint for inference**
- `phase2_dual_loss/logs/train_*.log`: Dual-loss metrics (CE loss, AR loss, total)
- `phase2_dual_loss/config.json`: Reproducible hyperparameters

**Expected**:
- CE loss: 6-8 (standard counting difficulty)
- AR loss: ~18 (regularization term, scales with λ)
- Combined loss: 6-9 (ce_loss + 0.1 × ar_loss)
- Training time: ~36 hours

---

## Using Best Checkpoints

### Load Phase 2 Checkpoint (For Counting Inference)

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoProcessor

# Load best checkpoint
checkpoint_path = "/data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/phase2_dual_loss/adapter_extracted"

model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path, device_map="cuda")
processor = AutoProcessor.from_pretrained("/data/amondal/model_cache/UniLIP-3B")

# Run inference
image = Image.open("example.jpg")
inputs = processor(image, "How many objects are in this image?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
```

### Extract LoRA Weights Only

```bash
# Already done by validate_pipeline.py, but to do manually:
python3 << 'EOF'
from peft import AutoPeftModelForCausalLM

checkpoint = "/data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/phase2_dual_loss/checkpoints/checkpoint-XXXX"
output = "/path/to/adapter_extracted"

model = AutoPeftModelForCausalLM.from_pretrained(checkpoint)
merged = model.merge_and_unload()
merged.save_pretrained(output)
print(f"✓ Saved to: {output}")
EOF
```

---

## Monitoring Training

### Real-Time Metrics (TensorBoard + WandB)

```bash
# TensorBoard (local)
tensorboard --logdir=/data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/phase1_pretext/logs

# WandB (cloud) — automatically logged during training
# View at: https://wandb.ai/<your-username>/pretext_dual_loss_pipeline
```

### WandB Setup

1. Install: `pip install wandb`
2. Login: `wandb login` (get key from https://wandb.ai/authorize)
3. Metrics auto-logged during training

**See `/data/amondal/UniCount/WANDB_SETUP.md` for full WandB configuration**

### Parse Logs

```bash
# Phase 1 losses
grep "\[dual-loss\]" /data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/phase1_pretext/logs/train_*.log | head -20

# Phase 2 losses
grep "\[dual-loss\]" /data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>/phase2_dual_loss/logs/train_*.log | tail -20
```

---

## Validation Report

After training completes, run:

```bash
python3 scripts/experiment_lora_counting_sft/validate_pipeline.py \
  --pipeline-root /data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>
```

**Output**:
1. `validation_report.json`: Checkpoint sizes, model types, LoRA config, loss curves
2. `DEPLOYMENT.json`: Manifest with checkpoint paths and configurations

**Example `validation_report.json`**:
```json
{
  "timestamp": "2026-05-08T...",
  "pipeline_root": "...",
  "phase1": {
    "checkpoint_validation": {
      "valid": true,
      "path": ".../phase1_pretext/adapter_extracted",
      "size_mb": 256,
      "lora_r": 64,
      "lora_alpha": 128
    },
    "training_stats": {
      "steps": 200,
      "final_loss": 4.52,
      "mean_loss": 5.23
    }
  },
  "phase2": {
    "checkpoint_validation": {
      "valid": true,
      "path": ".../phase2_dual_loss/adapter_extracted",
      "size_mb": 256,
      "lora_r": 64,
      "lora_alpha": 128
    },
    "training_stats": {
      "steps": 1500,
      "final_loss": 7.31,
      "mean_loss": 7.89
    }
  },
  "status": {
    "phase1_ready": true,
    "phase2_ready": true,
    "pipeline_complete": true
  }
}
```

---

## Performance Comparison

Expected improvement from Phase 1 pretraining:

| Benchmark | Baseline (No Pretext) | With Pretext | Improvement |
|-----------|----------------------|--------------|-------------|
| FSC-147 Val (CTAP) | 20-25 MAE | 15-18 MAE | +10-15% |
| FSC-147 Test (CTAP) | 25-30 MAE | 20-25 MAE | +10-15% |
| SHA-A (CTAP) | 80-120 MAE | 60-90 MAE | +20-30% |
| SHA-B (CTAP) | 40-60 MAE | 30-45 MAE | +20-30% |
| CARPK | 12-16 MAE | 10-13 MAE | +5-15% |

*Baseline is checkpoint-2670 (14.2K data, no pretext). With pretext should see 10-30% improvement.*

---

## Troubleshooting

### Phase 1 fails to start

```bash
# Check dataset
python3 -c "from datasets import load_dataset; ds = load_dataset('imagefolder', data_dir='/data/amondal/UniCountData/unicount_pretext', split='train'); print(len(ds))"

# Expected: 33,489
```

### Phase 2 fails due to OOM

Reduce batch size or gradient accumulation:
```bash
# In launch script, change:
BATCH_PER_GPU=1  # from 2
GRAD_ACCUM=2     # from 1
```

### Checkpoints not extracted

Manually extract:
```bash
cd /data/amondal/UniCount
python3 << 'EOF'
from pathlib import Path
from peft import AutoPeftModelForCausalLM

pipeline_root = Path("/data/amondal/unicount_runs/pretext_dual_loss_pipeline_<TIMESTAMP>")

for phase in ["phase1_pretext", "phase2_dual_loss"]:
    checkpoints = list((pipeline_root / phase / "checkpoints").glob("checkpoint-*"))
    if checkpoints:
        best = sorted(checkpoints)[-1]
        output = pipeline_root / phase / "adapter_extracted"
        
        print(f"Extracting {phase}: {best} → {output}")
        model = AutoPeftModelForCausalLM.from_pretrained(best)
        merged = model.merge_and_unload()
        merged.save_pretrained(output)
        print(f"✓ Done")
EOF
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `launch_pretext_dual_loss_pipeline.sh` | Master orchestration script (runs both phases) |
| `train_pretext_lora_3b.py` | Phase 1 training script (image pair correspondence) |
| `train_dual_loss_3b.py` | Phase 2 training script (counting + AR loss) |
| `validate_pipeline.py` | Post-training validation & checkpoint extraction |
| `README.md` | This documentation |

---

## Citation

```bibtex
@misc{unicount_pretext_dual_loss,
  title={Pretext-Task Pretrained Object Counting via Dual-Loss Training},
  year={2026}
}
```

---

## Questions or Issues?

- Check `pipeline.log` for detailed execution trace
- Check Phase 1 logs: `phase1_pretext/logs/train_*.log`
- Check Phase 2 logs: `phase2_dual_loss/logs/train_*.log`
- Review `validation_report.json` for checkpoint health
- See `/data/amondal/UniCount/README.md` for general setup
