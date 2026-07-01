#!/bin/bash
# Phase 8b: Full dual-loss training on 45.4K balanced_mix_v3s records with extracted object_centers
# Improvements over Phase 8a (15.8K attn_regularizer_train):
# - 3× more training data (45.4K vs 15.8K)
# - Full balanced_mix_v3s coverage (91.1%)
# - ShanghaiTech included (3,500 high-density samples)
# - All annotation sources represented

set -e
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="phase8b_balanced_mix_45k_${TIMESTAMP}"
export OUTPUT_DIR="/data/amondal/unicount_runs/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "=========================================="
echo "Phase 8b: Full Dual-Loss Training"
echo "Dataset: balanced_mix_v3s (45,413 records)"
echo "AR Loss: λ=0.1, σ=1.0, temp=0.1"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Hardware config (8x A100 / 8x H100)
NUM_GPUS=8
BATCH_PER_GPU=2
GRAD_ACCUM=1
EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_PER_GPU * GRAD_ACCUM))

# Training hyperparameters (matching v3s + Phase 8 validation)
LR=1e-5
EPOCHS=3
WARMUP_RATIO=0.06
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
MAX_LENGTH=512

# AR loss config (validated in Phase 5 dry-run)
LAMBDA_AR=0.1
AR_SIGMA=1.0
AR_TEMP=0.1

# Launch training
torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  scripts/experiment_lora_counting_sft/train_dual_loss_3b.py \
  --base_model /data/amondal/model_cache/UniLIP-3B \
  --train_json outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json \
  --output_dir "$OUTPUT_DIR" \
  \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type cosine \
  \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --unfreezeconn \
  \
  --model_max_length $MAX_LENGTH \
  --gradient_checkpointing True \
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
  --report_to tensorboard \
  --seed 42 \
  --logging_first_step \
  2>&1 | tee "$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"

echo ""
echo "=========================================="
echo "Phase 8b Training Complete"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract best checkpoint:"
echo "   python3 extract_lora_adapter.py --checkpoint $OUTPUT_DIR/checkpoint-XXXX --output $OUTPUT_DIR/adapter_extracted"
echo "2. Run CTAP+NRT evaluation:"
echo "   python3 eval_counting_ctap_nrt.py --adapter $OUTPUT_DIR/adapter_extracted --output outputs/experiment_lora_counting_sft/eval/phase8b_benchmarks"
echo "3. Compare against Phase 8a (15.8K) baseline"
echo ""
