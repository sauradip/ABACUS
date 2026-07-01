#!/bin/bash
#SBATCH --job-name=stage32_apptainer_smoke
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage32_apptainer_smoke_%j.log
#SBATCH --error=logs/stage32_apptainer_smoke_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/u6fb/myprojects/UniCount}"
CONTAINER_PATH="${CONTAINER_PATH:-$REPO_DIR/.cache/images/nvcr_pytorch_24.08-py3_arm64_sandbox}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-$REPO_DIR/.cache/container_py310}"

cd "$REPO_DIR"
mkdir -p logs "$REPO_DIR/.cache/huggingface" "$REPO_DIR/.cache/triton" "$REPO_DIR/.cache/tmp"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$REPO_DIR/.cache/tmp/python_pycache_${SLURM_JOB_ID:-manual}}"

echo "Repo:      $REPO_DIR"
echo "Container: $CONTAINER_PATH"
echo "Userbase:  $PYTHONUSERBASE_DIR"

apptainer exec --nv \
  --bind /projects:/projects \
  --bind /lus:/lus \
  --bind /home:/home \
  --env PYTHONUSERBASE="$PYTHONUSERBASE_DIR" \
  --env PATH="$PYTHONUSERBASE_DIR/bin:$PATH" \
  --env HF_HOME="$REPO_DIR/.cache/huggingface" \
  --env TRITON_CACHE_DIR="$REPO_DIR/.cache/triton" \
  --env TMPDIR="$REPO_DIR/.cache/tmp" \
  --env PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" \
  --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
  "$CONTAINER_PATH" \
  bash -lc 'python3 - <<'"'"'PY'"'"'
import importlib.util
import socket

print("host", socket.gethostname())

for name in ["torch", "flash_attn", "transformers", "accelerate", "peft", "trl", "timm", "datasets"]:
    spec = importlib.util.find_spec(name)
    print(name, "FOUND" if spec else "MISSING")
    if spec is not None:
        module = __import__(name)
        print(" ", getattr(module, "__version__", "unknown"))

import torch
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("gpu count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0", torch.cuda.get_device_name(0))

import torch.distributed.run as distributed_run
print("torch.distributed.run", distributed_run.__file__)

from trl.trainer.grpo_config import GRPOConfig
from peft import PeftModel
from scripts.counting_grpo.train_grpo_v32 import _ensure_flash_attn

print("GRPOConfig", GRPOConfig.__name__)
print("PeftModel", PeftModel.__name__)
print("stage32 imports ok")
PY'
