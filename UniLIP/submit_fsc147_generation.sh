#!/bin/bash
#SBATCH --job-name=unilip_fsc147_gen
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

UNILIP_DIR="/projects/u6bl/myprojects/UniLIP"
LIB_DIR="$UNILIP_DIR/python_libs"
CACHE_DIR="$UNILIP_DIR/.cache"
TRITON_DIR="$UNILIP_DIR/.triton_cache"
HF_CACHE_DIR="$UNILIP_DIR/.hf_cache"
CONTAINER_IMAGE="/projects/u6bl/myprojects/Janus/pytorch_24.08.sif"
PROMPTS_JSON="/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_filename_class_count_prompt_qwen3vl.json"
OUTPUT_DIR="$UNILIP_DIR/generated_samples/fsc147_ft"
MODEL_PATH="$UNILIP_DIR/work_dirs/1b_stage3_fsc147_t2i_v3_infer"
GUIDANCE_SCALE="3.0"
SKIP_EXISTING=1
START_INDEX=0
LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts-json) PROMPTS_JSON="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --guidance-scale) GUIDANCE_SCALE="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONPATH="$UNILIP_DIR:${PYTHONPATH:-}:$LIB_DIR/lib/python3.10/site-packages"
export PATH="$LIB_DIR/bin:$PATH"

mkdir -p "$UNILIP_DIR/logs" "$OUTPUT_DIR" "$LIB_DIR" "$CACHE_DIR" "$TRITON_DIR" "$HF_CACHE_DIR"

apptainer exec --nv \
  --bind /projects:/projects \
  --env PYTHONPATH="$PYTHONPATH" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER_IMAGE" \
  bash <<EOF
set -e
cd "$UNILIP_DIR"

FILTERED_REQUIREMENTS="\$(mktemp /tmp/unilip_requirements_XXXXXX.txt)"
trap 'rm -f "\$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(bitsandbytes|deepspeed|flash_attn|opencv_python|torch|torchvision|xformers)==' requirements.txt > "\$FILTERED_REQUIREMENTS"
python -m pip install --user --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r "\$FILTERED_REQUIREMENTS"
python -m pip uninstall -y opencv-python opencv_python >/dev/null 2>&1 || true
python -m pip install --user --no-cache-dir opencv_python_headless==4.11.0.86
python -m pip install --user --no-cache-dir -e .

# Transformers 4.52 blocks loading .bin with torch<2.6 for security reasons.
# Convert fine-tuned .bin checkpoints to safetensors so inference can load safely.
if [[ -f "$MODEL_PATH/pytorch_model.bin" && ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Converting $MODEL_PATH/pytorch_model.bin to safetensors..."
  python - "$MODEL_PATH" <<'PY'
import os
import sys
import torch
from safetensors.torch import save_file

model_dir = sys.argv[1]
bin_path = os.path.join(model_dir, "pytorch_model.bin")
safe_path = os.path.join(model_dir, "model.safetensors")

state_dict = torch.load(bin_path, map_location="cpu")
save_file(state_dict, safe_path)
print(f"Wrote safetensors checkpoint: {safe_path}")
PY
  mv "$MODEL_PATH/pytorch_model.bin" "$MODEL_PATH/pytorch_model.bin.backup"
fi

CMD=(python generate_from_prompts_fsc147.py --prompts-json "$PROMPTS_JSON" --output-dir "$OUTPUT_DIR" --model-path "$MODEL_PATH" --guidance-scale "$GUIDANCE_SCALE" --start-index "$START_INDEX")
if [[ "$LIMIT" != "0" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  CMD+=(--skip-existing)
fi
"\${CMD[@]}"
EOF
