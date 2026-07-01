#!/usr/bin/env python3
"""Wrapper to fix dtype issues in train_stage3.py by forcing model to fp32."""

import sys
import torch
from pathlib import Path

# Add UniLIP to path
sys.path.insert(0, '/data/amondal/UniCount/UniLIP')
sys.path.insert(0, '/data/amondal/UniCount')

# Monkey patch the model loading to use fp32
original_from_pretrained = None

def patch_from_pretrained():
    from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
    global original_from_pretrained
    original_from_pretrained = UniLIP_InternVLForCausalLM.from_pretrained

    @classmethod
    def from_pretrained_fp32(cls, *args, **kwargs):
        # Override torch_dtype to fp32
        kwargs['torch_dtype'] = torch.float32
        model = original_from_pretrained(*args, **kwargs)
        # Ensure entire model is fp32
        model = model.to(torch.float32)
        return model

    UniLIP_InternVLForCausalLM.from_pretrained = from_pretrained_fp32

patch_from_pretrained()

# Now run the training script
from unilip.train.train_stage3 import train

if __name__ == "__main__":
    train(attn_implementation="eager")
