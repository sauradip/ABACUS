#!/bin/bash
#SBATCH --job-name=debug_tokens
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_tokens_%j.log

set -euo pipefail

CONTAINER_IMAGE="/projects/u6bl/myprojects/Janus/pytorch_24.08.sif"
WORK_DIR="/projects/u6bl/myprojects/UniLIP"

apptainer exec --nv \
  --bind /projects:/projects \
  "$CONTAINER_IMAGE" \
  bash -c "
    set -e
    cd $WORK_DIR
    export PYTHONPATH=$WORK_DIR/python_libs/lib/python3.10/site-packages:\$PYTHONPATH
    export HF_HOME=$WORK_DIR/.hf_cache
    python3 unilip/debug_tokens.py
  "
