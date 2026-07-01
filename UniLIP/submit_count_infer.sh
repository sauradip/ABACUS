#!/bin/bash
#SBATCH --job-name=unilip_count_infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/unilip_count_infer_%j.log
#SBATCH --error=logs/unilip_count_infer_%j.err

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

    python3 -u infer_counting_sft.py \
      --model-path work_dirs/1b_fsc147_understanding_sft \
      --data-path /projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_test.json \
      --output-path fsc147_sft_v2_test_inference.json \
      --max-new-tokens 10
  "
