#!/bin/bash
#SBATCH --job-name=unilip_demo
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# --- 1. CONFIGURATION ---
UNILIP_DIR="/projects/u6bl/myprojects/UniLIP"
LIB_DIR="$UNILIP_DIR/python_libs"
CACHE_DIR="$UNILIP_DIR/.cache"
TRITON_DIR="$UNILIP_DIR/.triton_cache"
HF_CACHE_DIR="$UNILIP_DIR/.hf_cache"
CONTAINER_IMAGE="/projects/u6bl/myprojects/Janus/pytorch_24.08.sif"
MODE="generation"
MODEL_PATH="$UNILIP_DIR/UniLIP-3B"
IMAGE_PATH=""
QUESTION="Describe this image in detail."
MAX_NEW_TOKENS="128"
HF_HOST_CACHE_ROOT="/projects/u6bl/myprojects/vscode_pseudo_home/.cache/huggingface/hub"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --image-path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --question)
            QUESTION="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage:
  sbatch submit_iterative.sh [--model-path PATH]
  sbatch submit_iterative.sh --mode understanding --image-path PATH [--question TEXT] [--max-new-tokens N] [--model-path PATH]
USAGE
            exit 0
            ;;
        *)
            if [[ "$1" == --* ]]; then
                echo "Unknown option: $1" >&2
                exit 1
            fi
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

if [[ "$MODE" != "generation" && "$MODE" != "understanding" ]]; then
    echo "Unsupported mode: $MODE" >&2
    exit 1
fi

if [[ "$MODE" == "understanding" && -z "$IMAGE_PATH" ]]; then
    echo "--image-path is required when --mode understanding" >&2
    exit 1
fi

if [[ "$MODE" == "understanding" ]]; then
    if [[ ! -f "$IMAGE_PATH" && -f "$UNILIP_DIR/$IMAGE_PATH" ]]; then
        IMAGE_PATH="$UNILIP_DIR/$IMAGE_PATH"
    fi
    if [[ ! -f "$IMAGE_PATH" ]]; then
        echo "Input image not found: $IMAGE_PATH" >&2
        echo "Tip: pass an existing absolute path or a path relative to $UNILIP_DIR" >&2
        exit 1
    fi
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
CACHE_BLOB_DIR="$HF_HOST_CACHE_ROOT/models--kanashi6--$MODEL_NAME/blobs"
RESOLVED_MODEL_DIR="$UNILIP_DIR/.resolved_models/$MODEL_NAME"

# --- 2. EXPORT VARS ---
export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONPATH="$UNILIP_DIR:$PYTHONPATH:$LIB_DIR/lib/python3.10/site-packages"
export PATH="$LIB_DIR/bin:$PATH"

mkdir -p "$UNILIP_DIR/logs" "$LIB_DIR" "$CACHE_DIR" "$TRITON_DIR" "$HF_CACHE_DIR"

if [ -d "$MODEL_PATH" ] && [ ! -r "$MODEL_PATH/config.json" ] && [ -d "$CACHE_BLOB_DIR" ]; then
    mkdir -p "$RESOLVED_MODEL_DIR"
    while IFS= read -r model_file; do
        file_name="$(basename "$model_file")"
        if [ -L "$model_file" ]; then
            blob_hash="$(basename "$(readlink "$model_file")")"
            if [ -f "$CACHE_BLOB_DIR/$blob_hash" ]; then
                cp -f "$CACHE_BLOB_DIR/$blob_hash" "$RESOLVED_MODEL_DIR/$file_name"
            fi
        elif [ -f "$model_file" ]; then
            cp -f "$model_file" "$RESOLVED_MODEL_DIR/$file_name"
        fi
    done < <(find "$MODEL_PATH" -maxdepth 1 \( -type f -o -type l \) | sort)
    MODEL_PATH="$RESOLVED_MODEL_DIR"
fi

if [ ! -r "$MODEL_PATH/config.json" ]; then
    echo "Resolved model path is missing a readable config.json: $MODEL_PATH" >&2
    exit 1
fi

apptainer exec --nv \
    --bind /projects:/projects \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "$CONTAINER_IMAGE" \
    bash <<EOF
set -e
cd "$UNILIP_DIR"

echo 'Installing UniLIP dependencies from requirements.txt...'
# Use the container's preinstalled CUDA Torch stack and skip training-only or unavailable wheels.
FILTERED_REQUIREMENTS="\$(mktemp /tmp/unilip_requirements_XXXXXX.txt)"
trap 'rm -f "\$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(bitsandbytes|deepspeed|flash_attn|opencv_python|torch|torchvision|xformers)==' requirements.txt > "\$FILTERED_REQUIREMENTS"
python -m pip install --user --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r "\$FILTERED_REQUIREMENTS"

echo 'Ensuring headless OpenCV is used inside the container...'
python -m pip uninstall -y opencv-python opencv_python >/dev/null 2>&1 || true
python -m pip install --user --no-cache-dir opencv_python_headless==4.11.0.86

echo 'Installing UniLIP package...'
python -m pip install --user --no-cache-dir -e .

if [ "$MODE" = "understanding" ]; then
    echo "Starting UniLIP understanding with model path: $MODEL_PATH"
    echo "Input image: $IMAGE_PATH"
    python scripts/inference_understanding.py "$MODEL_PATH" "$IMAGE_PATH" "$QUESTION" --max-new-tokens "$MAX_NEW_TOKENS"
else
    echo "Starting UniLIP generation with model path: $MODEL_PATH"
    python scripts/inference_gen.py "$MODEL_PATH"
fi
EOF
