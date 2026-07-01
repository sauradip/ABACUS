#!/bin/bash
#SBATCH --job-name=unilip_vis
#SBATCH --output=logs/vis_%j.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail

UNILIP_DIR="/projects/u6bl/myprojects/UniLIP"
VLM_DIR="/projects/u6bl/myprojects/VLM-Visualizer"
LIB_DIR="$UNILIP_DIR/python_libs"
CONTAINER_IMAGE="/projects/u6bl/myprojects/Janus/pytorch_24.08.sif"

export PYTHONUSERBASE="$LIB_DIR"
export PYTHONPATH="$UNILIP_DIR:$VLM_DIR:${PYTHONPATH:-}:$LIB_DIR/lib/python3.10/site-packages"
export PATH="$LIB_DIR/bin:$PATH"

apptainer exec --nv \
  --bind /projects:/projects \
  --env PYTHONPATH="$PYTHONPATH" \
  "$CONTAINER_IMAGE" \
  bash -c "cd $UNILIP_DIR && python attention_visualizer_unilip.py --model-path work_dirs/1b_fsc147_understanding_sft --output-prefix attn_vis_sft_v2"
