#!/bin/bash
# UniCount Setup Script (portable)
# Usage:
#   source setup.sh
#   bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_first_existing() {
    for candidate in "$@"; do
        if [ -n "$candidate" ] && [ -e "$candidate" ]; then
            printf "%s" "$candidate"
            return 0
        fi
    done
    return 1
}

# Core locations
UNILIP_DIR="${UNILIP_DIR:-$SCRIPT_DIR/UniLIP}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
INTERNVL3_HF_PATH="${INTERNVL3_HF_PATH:-}"
FSC147_PROMPTS="${FSC147_PROMPTS:-}"
FSC147_SPLITS="${FSC147_SPLITS:-}"
FSC147_ORIG_DATA="${FSC147_ORIG_DATA:-}"
UNFREEZE_CKPT="${UNFREEZE_CKPT:-$SCRIPT_DIR/checkpoints/unfreeze_connector}"

# Search a few sensible local defaults if not explicitly set.
if [ -z "$CONTAINER_IMAGE" ]; then
    CONTAINER_IMAGE="$(detect_first_existing \
        "$SCRIPT_DIR/pytorch_24.08.sif" \
        "$SCRIPT_DIR/images/pytorch_24.08.sif" \
        "/mnt/fast/nobackup/scratch4weeks/am04485/Codes/llmcount/images/nvcr_pytorch_24.07-py3.sif" \
        || true)"
fi

if [ -z "$INTERNVL3_HF_PATH" ]; then
    INTERNVL3_HF_PATH="$(detect_first_existing \
        "$HOME/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf" \
        "$SCRIPT_DIR/../Cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf" \
        || true)"
fi

if [ -z "$FSC147_PROMPTS" ]; then
    FSC147_PROMPTS="$(detect_first_existing \
        "$SCRIPT_DIR/data/fsc147_filename_class_count_prompt_qwen3vl.json" \
        "$SCRIPT_DIR/data/test_prompts.json" \
        || true)"
fi

if [ -z "$FSC147_SPLITS" ]; then
    FSC147_SPLITS="$(detect_first_existing \
        "$SCRIPT_DIR/data/Train_Test_Val_FSC_147.json" \
        || true)"
fi

if [ -z "$FSC147_ORIG_DATA" ]; then
    FSC147_ORIG_DATA="$(detect_first_existing \
        "$SCRIPT_DIR/data/000000.tar" \
        "$SCRIPT_DIR/outputs/webdataset/000000.tar" \
        || true)"
fi

# Derived cache directories
LIB_DIR="${LIB_DIR:-$SCRIPT_DIR/python_libs}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.cache}"
TRITON_DIR="${TRITON_DIR:-$SCRIPT_DIR/.triton_cache}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRIPT_DIR/.hf_cache}"

mkdir -p "$LIB_DIR" "$CACHE_DIR" "$TRITON_DIR" "$HF_CACHE_DIR"

export UNILIP_DIR
export CONTAINER_IMAGE
export INTERNVL3_HF_PATH
export FSC147_PROMPTS
export FSC147_SPLITS
export FSC147_ORIG_DATA
export UNFREEZE_CKPT
export LIB_DIR
export CACHE_DIR
export TRITON_DIR
export HF_CACHE_DIR

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_DIR"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONNOUSERSITE=1

if [ -d "$UNILIP_DIR/python_libs/lib/python3.10/site-packages" ]; then
    export PYTHONPATH="$UNILIP_DIR:$SCRIPT_DIR:$UNILIP_DIR/python_libs/lib/python3.10/site-packages:$LIB_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"
else
    export PYTHONPATH="$UNILIP_DIR:$SCRIPT_DIR:$LIB_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"
fi
export PATH="$LIB_DIR/bin:$PATH"

echo "UniCount environment configured"
echo "  REPO_DIR:          $SCRIPT_DIR"
echo "  UNILIP_DIR:        $UNILIP_DIR"
echo "  CONTAINER_IMAGE:   ${CONTAINER_IMAGE:-<none>}"
echo "  INTERNVL3_HF_PATH: ${INTERNVL3_HF_PATH:-<unset>}"
echo "  FSC147_PROMPTS:    ${FSC147_PROMPTS:-<unset>}"
echo "  FSC147_SPLITS:     ${FSC147_SPLITS:-<unset>}"
echo ""
echo "Next steps:"
echo "  1) source setup.sh"
echo "  2) bash scripts/submit_eval_countbench.sh"
echo "  3) bash scripts/submit_eval_fsc147.sh"
