#!/usr/bin/env bash
set -euo pipefail

cd /home/nvidia/amondal/UniCount

BASE_MODEL="/home/nvidia/amondal/model_cache/UniLIP-3B"
PROCESSOR="OpenGVLab/InternVL3-2B-hf"

ADAPTIVE_TRAIN="/home/nvidia/amondal/UniCount/outputs/scaffold_prompt_fsc147_train_adaptive/train_scaffold_input_only.jsonl"
VAL_DATA="/home/nvidia/amondal/UniCount/outputs/scaffold_prompt_fsc147_val_adaptive/val_scaffold_input_only.jsonl"
FSC147_ANN="/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json"

TRAIN_DATA="$ADAPTIVE_TRAIN"
echo "Using adaptive train JSONL: $TRAIN_DATA"

python - <<'PY' "$TRAIN_DATA"
import json, sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    row = json.loads(next(f))
print('first_row_keys=', sorted(row.keys()))
print('has_gt_count=', 'gt_count' in row)
if 'gt_count' not in row:
    print('note: gt_count missing in row; loader will resolve GT via --fsc147_annotations')
PY

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="hybrid_anchor_v2_logmse_softplus_10ep_${STAMP}"
OUTPUT_DIR="/data/unicount_runs/${RUN_NAME}"

echo "run_name: ${RUN_NAME}"
echo "output_dir: ${OUTPUT_DIR}"

auth_msg="Starting 8-GPU hybrid training (10 epochs, CE + 0.1*log-MSE, softplus regression head)."
echo "$auth_msg"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 \
  scripts/counting_grpo/train_unilip_regression.py \
  --model_name_or_path "$BASE_MODEL" \
  --processor_name_or_path "$PROCESSOR" \
  --data_path "$TRAIN_DATA" \
  --eval_data_path "$VAL_DATA" \
  --fsc147_annotations "$FSC147_ANN" \
  --loss_type hybrid \
  --mse_variant log \
  --mse_weight 0.1 \
  --regression_output_activation softplus \
  --unfreeze_queries true \
  --lora_rank 128 \
  --lora_alpha 256 \
  --count_norm_factor 1000.0 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --bf16 true \
  --attn_implementation eager \
  --logging_steps 10 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME"

echo "Training finished. Artifacts: $OUTPUT_DIR"
