#!/bin/bash
#SBATCH --job-name=unilip_fsc147_t2i_ft
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail

UNILIP_DIR="/projects/u6bl/myprojects/UniLIP"
LIB_DIR="$UNILIP_DIR/python_libs"
CACHE_DIR="$UNILIP_DIR/.cache"
TRITON_DIR="$UNILIP_DIR/.triton_cache"
HF_CACHE_DIR="$UNILIP_DIR/.hf_cache"
CONTAINER_IMAGE="/projects/u6bl/myprojects/Janus/pytorch_24.08.sif"

PROMPTS_JSON="/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_filename_class_count_prompt_qwen3vl.json"
IMAGE_DIR="/projects/u6bl/myprojects/Datasets/FSC-147/images_384_VarV2"
GEN_SFT_DIR="/projects/u6bl/myprojects/Datasets/FSC-147/gen_sft_fsc147"

# Defaults to released 1B checkpoint for continued fine-tuning.
STAGE2_CKPT="$UNILIP_DIR/.resolved_models/UniLIP-1B"
OUTPUT_FOLDER="$UNILIP_DIR/work_dirs/1b_stage3_fsc147_t2i_v2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts-json) PROMPTS_JSON="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --gen-sft-dir) GEN_SFT_DIR="$2"; shift 2 ;;
    --stage2-ckpt) STAGE2_CKPT="$2"; shift 2 ;;
    --output-folder) OUTPUT_FOLDER="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$PROMPTS_JSON" ]]; then
  echo "Missing prompts json: $PROMPTS_JSON" >&2
  exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Missing image dir: $IMAGE_DIR" >&2
  exit 1
fi

if [[ ! -d "$STAGE2_CKPT" ]]; then
  echo "Missing stage2 checkpoint dir: $STAGE2_CKPT" >&2
  echo "Pass --stage2-ckpt /path/to/checkpoint-xxxx" >&2
  exit 1
fi

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
WORLD_SIZE="${SLURM_NNODES:-1}"
RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONPATH="$UNILIP_DIR:${PYTHONPATH:-}:$LIB_DIR/lib/python3.10/site-packages"
export PATH="$LIB_DIR/bin:$PATH"

mkdir -p "$UNILIP_DIR/logs" "$LIB_DIR" "$CACHE_DIR" "$TRITON_DIR" "$HF_CACHE_DIR" "$GEN_SFT_DIR"

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
# Keep deepspeed for training, skip unsupported GPU-specific wheels here.
grep -Ev '^(bitsandbytes|flash_attn|opencv_python|torch|torchvision|xformers)==' requirements.txt > "\$FILTERED_REQUIREMENTS"
python -m pip install --user --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r "\$FILTERED_REQUIREMENTS"
python -m pip uninstall -y opencv-python opencv_python >/dev/null 2>&1 || true
python -m pip install --user --no-cache-dir opencv_python_headless==4.11.0.86 webdataset
python -m pip install --user --no-cache-dir -e .

python data/convert_fsc147_t2i.py \
  --prompts-json "$PROMPTS_JSON" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$GEN_SFT_DIR" \
  --maxcount 10000

export WORLD_SIZE="$WORLD_SIZE"
export RANK="$RANK"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export GPUS_PER_NODE="$GPUS_PER_NODE"
export STAGE2_CKPT="$STAGE2_CKPT"
export GEN_IMG_FOLDER="$GEN_SFT_DIR"
export OUTPUT_FOLDER="$OUTPUT_FOLDER"

bash scripts/run_unilip_1b_stage3_fsc147_t2i.sh
EOF
