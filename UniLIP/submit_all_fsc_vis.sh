#!/bin/bash
#SBATCH --job-name=fsc_full_vis
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fsc_full_vis_%j.log
#SBATCH --error=logs/fsc_full_vis_%j.err

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
    export TRANSFORMERS_CACHE=$WORK_DIR/.hf_cache
    export HF_HOME=$WORK_DIR/.hf_cache

    python3 -u infer_and_visualize_v2.py \
      --model-path work_dirs/1b_fsc147_understanding_sft \
      --vis-dir fsc147_all_visualizations \
      --num-samples -1
  "
