import torch
import torch.nn as nn
from unilip.model.language_model.headlens import ObjectFocusedAttentionLoss
import matplotlib.pyplot as plt
import numpy as np

def demo():
    H, W = 16, 16
    bsz = 1
    num_heads = 1
    q_len = 1
    k_len = 256
    
    # Simulate a "spread out" attention map
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([x, y], dim=-1).float()
    
    # 4 spread out blobs
    apple_centers = [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
    raw_attn = torch.zeros(H, W)
    for center in apple_centers:
        cx, cy = center[0]*(W-1), center[1]*(H-1)
        dist_sq = torch.sum((coords - torch.tensor([cx, cy])) ** 2, dim=-1)
        raw_attn += torch.exp(-dist_sq / (2 * 1.5 ** 2)) # Wide spread
    
    attentions = [raw_attn.view(bsz, num_heads, q_len, k_len)]
    object_centers = [apple_centers]
    
    # 1. Standard Loss (no sharpening)
    loss_std_fn = ObjectFocusedAttentionLoss(sigma=1.0, use_sharpening=False)
    loss_std = loss_std_fn(attentions, object_centers, H, W)
    
    # 2. "Cleaned" Loss (with sharpening / low temperature)
    loss_sharp_fn = ObjectFocusedAttentionLoss(sigma=1.0, temperature=0.1, use_sharpening=True)
    loss_sharp = loss_sharp_fn(attentions, object_centers, H, W)
    
    print(f"Standard Loss (Spread Attention): {loss_std.item():.4f}")
    print(f"Sharpened 'Clean' Loss (Same Attention): {loss_sharp.item():.4f}")
    print("\nNote: The sharpened loss is much more sensitive to whether the peaks align with GT points.")

    # Visualization of what the loss "sees" after sharpening
    q_raw = raw_attn / raw_attn.sum()
    eps = 1e-8
    q_sharp = torch.pow(raw_attn + eps, 1.0 / 0.1)
    q_sharp = q_sharp / q_sharp.sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(q_raw.numpy(), cmap='jet')
    axes[0].set_title("What Model Outputs (Spread)")
    axes[0].axis('off')
    
    axes[1].imshow(q_sharp.numpy(), cmap='jet')
    axes[1].set_title("What the Loss 'Sees' (Sharpened/Cleaned)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("sharpened_attention_loss.png")
    print("Saved sharpened_attention_loss.png to show how the loss function 'cleans' the input internally.")

if __name__ == "__main__":
    demo()
