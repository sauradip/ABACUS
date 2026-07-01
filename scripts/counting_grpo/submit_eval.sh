#!/bin/bash
#SBATCH --job-name=counting_grpo_eval
#SBATCH --output=/projects/u6bl/myprojects/omnicountgen/logs/counting_grpo_eval_%j.log
#SBATCH --error=/projects/u6bl/myprojects/omnicountgen/logs/counting_grpo_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

set -euo pipefail

JANUS_DIR="/projects/u6bl/myprojects/Janus"
WORKSPACE_DIR="/projects/u6bl/myprojects"
LIB_DIR="$JANUS_DIR/python_libs"
HF_CACHE_DIR="$JANUS_DIR/.hf_cache"

export PYTHONUSERBASE="$LIB_DIR"
export PIP_CACHE_DIR="$JANUS_DIR/.cache"
export TRITON_CACHE_DIR="$JANUS_DIR/.triton_cache"
export HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_SSL_CERT_FILE=""
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH="$WORKSPACE_DIR/UniLIP/python_libs/lib/python3.10/site-packages:$WORKSPACE_DIR:$WORKSPACE_DIR/VLM-R1/src/open-r1-multimodal/src:$WORKSPACE_DIR/UniLIP:$LIB_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PATH="$LIB_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1

SFT_CHECKPOINT="$WORKSPACE_DIR/UniLIP/work_dirs/1b_fsc147_understanding_sft"
GRPO_CHECKPOINT="$WORKSPACE_DIR/omnicountgen/counting_grpo/checkpoints_numiter2"
FSC_ROOT="$WORKSPACE_DIR/Datasets/FSC-147"
EVAL_SCRIPT="/projects/u6bl/myprojects/omnicountgen/counting_grpo/evaluate_counting.py"

apptainer exec --nv \
  --bind /projects:/projects \
  --env PYTHONPATH="$PYTHONPATH" \
  --env PYTHONUSERBASE="$LIB_DIR" \
  --env HF_HOME="$HF_CACHE_DIR" \
  --env TRITON_CACHE_DIR="$JANUS_DIR/.triton_cache" \
  "$JANUS_DIR/pytorch_24.08.sif" \
  bash -lc '
    echo "=== SFT BASELINE (1b_fsc147_understanding_sft) ==="
    python '"$EVAL_SCRIPT"' \
      --model_path '"$SFT_CHECKPOINT"' \
      --tokenizer_path '"$GRPO_CHECKPOINT"' \
      --fsc_root '"$FSC_ROOT"' \
      --split test

    echo ""
    echo "=== GRPO RUN 4 (checkpoints_numiter2, best checkpoint) ==="
    python '"$EVAL_SCRIPT"' \
      --model_path '"$GRPO_CHECKPOINT"' \
      --fsc_root '"$FSC_ROOT"' \
      --split test

    echo ""
    echo "=== Done ==="
  '
