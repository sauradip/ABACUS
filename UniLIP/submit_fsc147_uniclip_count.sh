#!/bin/bash
#SBATCH --job-name=unilip_fsc147_count
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

INPUT_JSON="/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_ft_yolocount_final.json"
OUTPUT_JSON="$INPUT_JSON"
IMAGE_DIR="/projects/u6bl/myprojects/Datasets/FSC-147/images_384_VarV2"
MODEL_PATH="$UNILIP_DIR/.resolved_models/UniLIP-1B"
MAX_NEW_TOKENS=32
START_INDEX=0
LIMIT=0
SAVE_EVERY=100
RESUME=1
STORE_RAW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-json) INPUT_JSON="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --store-raw) STORE_RAW=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Missing input json: $INPUT_JSON" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Missing image dir: $IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model path: $MODEL_PATH" >&2
  exit 1
fi

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONPATH="$UNILIP_DIR:${PYTHONPATH:-}:$LIB_DIR/lib/python3.10/site-packages"
export PATH="$LIB_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

mkdir -p "$UNILIP_DIR/logs" "$LIB_DIR" "$CACHE_DIR" "$TRITON_DIR" "$HF_CACHE_DIR"

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

CMD=(python -u extract_uniclip_counts_fsc147.py
  --input-json "$INPUT_JSON"
  --output-json "$OUTPUT_JSON"
  --model-path "$MODEL_PATH"
  --image-dir "$IMAGE_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --start-index "$START_INDEX"
  --limit "$LIMIT"
  --save-every "$SAVE_EVERY"
  --log-every 10)

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "$STORE_RAW" == "1" ]]; then
  CMD+=(--store-raw)
fi

"\${CMD[@]}"
EOF
