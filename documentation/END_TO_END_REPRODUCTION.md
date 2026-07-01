# End-to-End Reproduction Guide: From Data to Paper Results

**Goal**: Reproduce all paper results (counting + generation) on FSC-147, CARPK, ShanghaiTech, REC-8K, and generation benchmarks.

**Timeline**: ~72 hours (48h training + 24h evaluation/iteration)

**Prerequisites**:
- 8× NVIDIA A100 80GB GPUs (or equivalent)
- ~500 GB disk space
- Python 3.10+, PyTorch 2.5+, CUDA 12.1+

---

## Phase 0: Environment Setup (30 minutes)

### Step 0.1: Clone & Install

```bash
# Clone ABACUS repository
cd /data/amondal
git clone https://github.com/mondalanindya/ABACUS.git
cd ABACUS/code/UniCount_github_bundle

# Create virtual environment
python3.10 -m venv venv_abacus
source venv_abacus/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install additional tools
pip install transformers peft trl deepspeed pxr click webdataset pillow scipy
```

### Step 0.2: Download Base Model & Datasets

```bash
# Set up directories
export MODEL_CACHE=/data/amondal/model_cache
export DATA_ROOT=/data/amondal/datasets
mkdir -p $MODEL_CACHE $DATA_ROOT

# Download UniLIP-3B model (instructions in model_cache/README.md)
# Download InternVL2 processor
# Download FSC-147 dataset
# Download other benchmarks (CARPK, ShanghaiTech, REC-8K)

# Verify downloads
ls -lh $MODEL_CACHE/UniLIP-3B/model-*.safetensors
ls -lh $DATA_ROOT/FSC-147/Train_Test_Val_FSC_147.json
```

### Step 0.3: Verify Environment

```bash
# Test GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Test imports
python -c "from transformers import AutoModel; from peft import get_peft_model; print('All imports OK')"

# Verify dataset structure
python scripts/check_dataset_structure.py --data_root $DATA_ROOT
```

---

## Phase 1: Data Preparation (2 hours)

### Step 1.1: FSC-147 Data Preprocessing

```bash
# Extract object centers from FSC-147 annotations
python scripts/preprocessing/extract_fsc147_centers.py \
  --fsc147_root $DATA_ROOT/FSC-147 \
  --output_dir outputs/fsc147_preprocessed \
  --split train

# Generate SFT training data (counting prompts + images)
python scripts/preprocessing/build_fsc147_sft_data.py \
  --fsc147_preprocessed outputs/fsc147_preprocessed \
  --output_json outputs/fsc147_counting_train.json

# Verify data
python -c "
import json
with open('outputs/fsc147_counting_train.json') as f:
    data = json.load(f)
    print(f'Training samples: {len(data)}')
    print(f'Sample: {data[0]}')
"
```

### Step 1.2: Prepare Other Benchmarks

```bash
# CARPK (Drone-based parking lot images)
python scripts/preprocessing/build_carpk_eval_data.py \
  --carpk_root $DATA_ROOT/CARPK \
  --output_json outputs/carpk_eval.json

# ShanghaiTech crowd counting
python scripts/preprocessing/build_shanghaitech_eval_data.py \
  --shanghaitech_root $DATA_ROOT/ShanghaiTech \
  --output_json outputs/shanghaitech_eval.json

# REC-8K referring expression counting
python scripts/preprocessing/build_rec8k_eval_data.py \
  --rec8k_root $DATA_ROOT/REC-8K \
  --output_json outputs/rec8k_eval.json

# Verify all datasets
python scripts/verify_all_datasets.py --output_dir outputs/
```

### Step 1.3: Create Boundary-Aware Training Data

```bash
# Generate quadrant crops + consistency training data
python scripts/preprocessing/build_boundary_training_data.py \
  --source_json outputs/fsc147_counting_train.json \
  --output_dir outputs/boundary_training \
  --num_quadrant_samples 2000 \
  --consistency_weight 0.5

# Result: outputs/boundary_training/fsc147_boundary_aware.jsonl
```

---

## Phase 2: Model Training (48 hours)

### Step 2.1: Stage 1 - Objectness Localization

```bash
# Train Stage 1 with AR-Loss for spatial grounding
python UniLIP/unilip/train/train_stage1.py \
  --model_name_or_path $MODEL_CACHE/UniLIP-3B \
  --data_path outputs/fsc147_counting_train.json \
  --output_dir runs/stage1_objectness_lora32 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --bf16 True \
  --loss_weight_ar 1.0 \
  --save_steps 100 \
  --save_total_limit 3 \
  --deepspeed_config_file UniLIP/deepspeed_scripts/zero2.json \
  --logging_steps 10 \
  --log_level info

# Expected output:
# - runs/stage1_objectness_lora32/checkpoint-final/
# - runs/stage1_objectness_lora32/adapter_extracted/
# - Training time: ~12 hours on 8×A100
# - Final AR-loss: ~0.2-0.5
# - Counting validation MAE: ~12-15
```

### Step 2.2: Stage 2 - Boundary-Aware Dual-Loss Training

```bash
# Warm-start from Stage 1 checkpoint
export STAGE1_ADAPTER=runs/stage1_objectness_lora32/adapter_extracted

# Train Stage 2 with counting + boundary-aware GRPO preparation
python UniLIP/unilip/train/train_stage2.py \
  --model_name_or_path $MODEL_CACHE/UniLIP-3B \
  --pretrain_mm_mlp_adapter $STAGE1_ADAPTER/adapter_model.bin \
  --data_path outputs/boundary_training/fsc147_boundary_aware.jsonl \
  --output_dir runs/stage2_boundary_dual_loss \
  --num_train_epochs 3 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --bf16 True \
  --loss_weight_ar 0.5 \
  --loss_weight_counting 1.0 \
  --save_steps 100 \
  --deepspeed_config_file UniLIP/deepspeed_scripts/zero2.json

# Expected output:
# - runs/stage2_boundary_dual_loss/adapter_extracted/
# - Training time: ~18 hours on 8×A100
# - Validation MAE (FSC-147 val): ~9-11
```

### Step 2.3: Stage 3 - GRPO Training for Boundary Handling

```bash
# Prepare GRPO data with boundary scenarios
python scripts/preprocessing/build_grpo_curriculum_data.py \
  --base_data outputs/fsc147_counting_train.json \
  --output_jsonl outputs/grpo_curriculum.jsonl \
  --k_curriculum 8

# Launch GRPO training
python scripts/counting_grpo/train_grpo_v32.py \
  --model_name_or_path $MODEL_CACHE/UniLIP-3B \
  --processor_name_or_path OpenGVLab/InternVL2-2B \
  --dataset_name outputs/grpo_curriculum.jsonl \
  --reward_script scripts/counting_grpo/grpo_reward_boundary_aware.py \
  --output_dir runs/stage3_boundary_aware_grpo \
  --num_generations 8 \
  --max_prompt_length 2048 \
  --max_completion_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --num_train_epochs 2 \
  --num_iterations 2 \
  --beta 0.05 \
  --save_steps 50 \
  --bf16 True \
  --attn_implementation flash_attention_2

# Expected output:
# - runs/stage3_boundary_aware_grpo/final_adapter/
# - Training time: ~18 hours on 8×A100
# - Reward curve: 0.40 → 0.85
```

### Step 2.4: Optional - Generation Fine-tuning (Stage 3 for images)

```bash
# If generating images, fine-tune SANA with count constraints
python scripts/generation/train_sana_count_aware.py \
  --model_name_or_path /path/to/SANA-1.5B \
  --data_path outputs/generation_training.json \
  --output_dir runs/stage3_generation_sana \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 True \
  --deepspeed_config_file UniLIP/deepspeed_scripts/zero2.json

# Expected output:
# - runs/stage3_generation_sana/adapter_extracted/
```

---

## Phase 3: Evaluation & Benchmark (24 hours)

### Step 3.1: Load Best Checkpoint

```bash
# Use Stage 3 GRPO checkpoint (best overall)
export BEST_CHECKPOINT=runs/stage3_boundary_aware_grpo/final_adapter
export BEST_MODEL=$MODEL_CACHE/UniLIP-3B

# Verify checkpoint exists
ls -lh $BEST_CHECKPOINT/adapter_model.bin
```

### Step 3.2: Evaluate on FSC-147 (Gold Standard)

```bash
# FSC-147 Validation Set (1,286 images)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --val_json $DATA_ROOT/FSC-147/fsc147_val_counting.json \
  --ann_json $DATA_ROOT/FSC-147/fsc147_instances_val.json \
  --output_dir outputs/eval_fsc147_val \
  --T 100 \
  --max_depth 3 \
  --min_size 50 \
  --num_workers 8 \
  --device cuda:0-7

# Collect results
python scripts/analysis/collect_eval_results.py \
  --eval_dir outputs/eval_fsc147_val \
  --metric_type "CTAP+NRT" \
  --output_json outputs/fsc147_val_results.json

# Expected:
# ├─ MAE: 8.48 ± 0.5
# ├─ RMSE: 40.91 ± 2.3
# └─ Correct format: ✓

# FSC-147 Test Set (1,190 images)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --val_json $DATA_ROOT/FSC-147/fsc147_test_counting.json \
  --output_dir outputs/eval_fsc147_test \
  --T 100 \
  --max_depth 3

# Expected:
# ├─ MAE: 10.47
# ├─ RMSE: 73.21
# └─ Paper match: ✓
```

### Step 3.3: Cross-Dataset Evaluation

```bash
# CARPK (drone parking lot)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_general.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --dataset_name carpk \
  --data_json outputs/carpk_eval.json \
  --output_json outputs/carpk_results.json

# Expected: MAE 10.33 ± 1.2

# ShanghaiTech-A (dense crowd)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_general.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --dataset_name shanghaitech_a \
  --data_json outputs/shanghaitech_eval.json \
  --dataset_split a \
  --output_json outputs/shanghaitech_a_results.json

# Expected: MAE 93.85 ± 5.0

# ShanghaiTech-B (sparse crowd)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_general.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --dataset_name shanghaitech_b \
  --dataset_split b \
  --output_json outputs/shanghaitech_b_results.json

# Expected: MAE 16.07 ± 1.5

# REC-8K (referring expressions)
python scripts/experiment_lora_counting_sft/eval_counting_referring.py \
  --model_path $BEST_MODEL \
  --adapter_path $BEST_CHECKPOINT \
  --rec8k_json outputs/rec8k_eval.json \
  --output_json outputs/rec8k_results.json

# Expected: MAE 28.5 (referring expression counting)
```

### Step 3.4: Aggregate Results

```bash
# Compile all evaluation results into master table
python scripts/analysis/compile_paper_results.py \
  --eval_results outputs/*_results.json \
  --output_table paper_results_table.md

# Expected output file: paper_results_table.md
# Contains table matching paper's Table 1, 2, 3
```

### Step 3.5: Generation Evaluation (Optional)

```bash
# If trained generation model, evaluate on CountBench
python scripts/generation/eval_generation_benchmarks.py \
  --model_path $BEST_MODEL \
  --gen_adapter runs/stage3_generation_sana/adapter_extracted \
  --countbench_root $DATA_ROOT/CountBench \
  --output_json outputs/generation_results.json

# Expected: GenEval ↑5%, CountBench MAE ↓12%
```

---

## Phase 4: Validation & Reporting (1 hour)

### Step 4.1: Verify Results Match Paper

```bash
# Create comparison table
cat << 'EOF' > scripts/validation/expected_results.txt
Dataset,Metric,Paper,Reproduced,Tolerance
FSC-147-val,MAE,8.48,?,±1.5
FSC-147-test,MAE,10.47,?,±1.5
CARPK,MAE,10.33,?,±1.5
ShanghaiTech-A,MAE,93.85,?,±5
ShanghaiTech-B,MAE,16.07,?,±2
REC-8K,MAE,28.5,?,±3
EOF

# Run validation
python scripts/validation/compare_with_paper.py \
  --expected_file scripts/validation/expected_results.txt \
  --results_dir outputs/ \
  --tolerance_column Tolerance
```

### Step 4.2: Generate Report

```bash
# Create comprehensive markdown report
python scripts/analysis/generate_reproduction_report.py \
  --eval_results outputs/ \
  --training_logs runs/ \
  --output_report REPRODUCTION_REPORT.md

# Outputs:
# ├─ REPRODUCTION_REPORT.md (full details)
# ├─ RESULTS_SUMMARY.txt (one-page summary)
# └─ figures/
#    ├─ loss_curves.png
#    ├─ mae_comparison.png
#    └─ cross_dataset_generalization.png
```

### Step 4.3: Checklist

```bash
cat << 'EOF' > REPRODUCTION_CHECKLIST.md
# Reproduction Verification Checklist

## Data ✓
- [ ] FSC-147 preprocessed (centers extracted, 4,903 train samples)
- [ ] CARPK, ShanghaiTech, REC-8K downloaded
- [ ] All datasets verified with check_dataset_structure.py

## Training ✓
- [ ] Stage 1 completed (AR-loss converged, checkpoint saved)
- [ ] Stage 2 completed (dual-loss optimized, MAE < 12)
- [ ] Stage 3 GRPO completed (reward curve converged to 0.8+)

## Evaluation ✓
- [ ] FSC-147 val: MAE within ±1.5 of 8.48
- [ ] FSC-147 test: MAE within ±1.5 of 10.47
- [ ] CARPK: MAE within ±1.5 of 10.33
- [ ] ShanghaiTech-A: MAE within ±5 of 93.85
- [ ] ShanghaiTech-B: MAE within ±2 of 16.07
- [ ] REC-8K: MAE within ±3 of 28.5

## Documentation ✓
- [ ] REPRODUCTION_REPORT.md generated
- [ ] Loss curves plotted
- [ ] Training timeline documented
- [ ] Any deviations from paper explained

## Success Criteria ✓
- [ ] ≥5/6 benchmarks match paper within tolerance
- [ ] Computational cost < 72 hours total
- [ ] All intermediate checkpoints saved
EOF

cat REPRODUCTION_CHECKLIST.md
```

---

## Troubleshooting Guide

### Training Issues

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| OOM on 8×A100 | Batch size too large | Reduce `per_device_train_batch_size` from 8→6→4 |
| Gradient explosion | Learning rate too high | Reduce `learning_rate` from 2e-5→1e-5 |
| Model diverges | DeepSpeed misconfiguration | Check stage (0, 1, 2, 3) matches available memory |
| AR-loss is NaN | Invalid object centers | Check centers normalized to [0, 1]; debug with `print()` |
| Eval crashes | Model not on GPU | Add `--device cuda:0` to eval script |

### Evaluation Issues

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Eval is very slow | No multi-GPU parallelization | Use `--num_workers 8` + `--device cuda:0-7` |
| Results don't match | Different preprocessing | Verify image resizing (448×448), normalization params |
| Invalid JSON output | Model format changed | Check prompt template hasn't drifted |

### Common Mistakes

❌ **Using wrong dataset split**: Make sure FSC-147 val/test are separate
❌ **Forgetting warmup in GRPO**: Always use `--warmup_ratio 0.03`
❌ **Mixing different model architectures**: Ensure adapter matches base model exactly
❌ **Evaluating partial checkpoint**: Always use final adapter_extracted, not intermediate

---

## Timeline Summary

```
Phase 0 (Setup):        30 min  ██░
Phase 1 (Data):         2 hr    ████░
Phase 2 (Training):     48 hr   ████████████████░
├─ Stage 1:            12 hr
├─ Stage 2:            18 hr
├─ Stage 3:            18 hr
Phase 3 (Evaluation):   20 hr   ██████████░
└─ All benchmarks:      20 hr
Phase 4 (Report):       1 hr    ██░

Total:                  ~71.5 hours
```

---

## Output Structure

```
/data/amondal/unicount_runs/
├── stage1_objectness_lora32/
│   ├── checkpoints/
│   ├── adapter_extracted/
│   └── training_log.txt
├── stage2_boundary_dual_loss/
│   ├── checkpoints/
│   ├── adapter_extracted/
│   └── training_log.txt
├── stage3_boundary_aware_grpo/
│   ├── checkpoints/
│   ├── final_adapter/
│   └── training_log.txt
└── final_results/
    ├── fsc147_val_results.json
    ├── fsc147_test_results.json
    ├── carpk_results.json
    ├── shanghaitech_results.json
    ├── rec8k_results.json
    └── paper_results_table.md
```

---

## Success Criteria

✅ **Reproduction successful if:**
1. FSC-147 val MAE = 8.48 ± 1.5
2. FSC-147 test MAE = 10.47 ± 1.5
3. ≥4/5 other benchmarks match within tolerance
4. Training completes in < 72 hours
5. All checkpoints saved and documented

❌ **Reproduction failed if:**
- Any core benchmark differs by > 2× tolerance
- Model training crashes or doesn't converge
- Eval scripts fail to run

---

## Next Steps After Reproduction

1. **Ablation studies**: Disable AR-loss, GRPO; measure impact
2. **Hyperparameter sweep**: Vary `lora_r`, `learning_rate`, reward weights
3. **Custom datasets**: Test on your own counting task
4. **Deployment**: Use Stage 3 checkpoint for inference server

---

## Support & Questions

- Check logs first: `tail -100 runs/stage*/training_log.txt`
- Review AGENTS.md for exact hyperparameter values
- See BOUNDARY_AWARE_GRPO_EXPLAINED.md for GRPO troubleshooting
- See OBJECTNESS_MAP_TRAINING.md for AR-loss issues
