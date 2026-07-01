import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HeadLens(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        # Affine translator T(z) = Az + b
        # Translates from hidden_size to hidden_size (residual stream space)
        self.A = nn.Linear(hidden_size, hidden_size)

    def forward(self, raw_head_outputs, W_O, head_idx):
        """
        raw_head_outputs: (bsz, seq_len, num_heads, head_dim)
        W_O: weight matrix of o_proj (num_heads * head_dim, hidden_size) or similar
             or a Linear module.
        head_idx: int
        """
        bsz, seq_len, _, _ = raw_head_outputs.shape

        # Isolate head i: zero out other heads
        isolated_heads = torch.zeros_like(raw_head_outputs)
        isolated_heads[:, :, head_idx, :] = raw_head_outputs[:, :, head_idx, :]

        # Flatten back to (bsz, seq_len, num_heads * head_dim)
        isolated_flat = isolated_heads.reshape(bsz, seq_len, -1)

        # Get target dtype from W_O
        if isinstance(W_O, nn.Linear):
            target_dtype = W_O.weight.dtype
            isolated_flat = isolated_flat.to(target_dtype)
            projected = W_O(isolated_flat)
        else:
            target_dtype = W_O.dtype
            isolated_flat = isolated_flat.to(target_dtype)
            projected = F.linear(isolated_flat, W_O)

        # Translate to residual stream space, ensuring A is on correct device and dtype
        self.A = self.A.to(device=projected.device, dtype=projected.dtype)
        translated = self.A(projected)
        return translated

class AttentionFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.raw_head_outputs = {} # layer_idx -> raw_head_outputs (bsz, seq_len, num_heads, head_dim)

        self.register_hooks()

    def register_hooks(self):
        # Determine the layers based on standard HuggingFace patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            print("Warning: Could not find model layers for AttentionFeatureExtractor.")
            return

        hooks_registered = 0
        for idx, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn, "o_proj"):
                    o_proj = attn.o_proj
                    if hasattr(attn, "num_heads"):
                        num_heads = attn.num_heads
                    elif hasattr(attn, "config") and hasattr(attn.config, "num_attention_heads"):
                        num_heads = attn.config.num_attention_heads
                    else:
                        print(f"Warning: Could not determine num_heads for layer {idx}.")
                        continue
                    # Qwen uses head_dim, Llama uses head_dim or hidden_size // num_heads
                    if hasattr(attn, "head_dim"):
                        head_dim = attn.head_dim
                    elif hasattr(attn, "config") and hasattr(attn.config, "hidden_size"):
                        head_dim = getattr(attn.config, "head_dim", attn.config.hidden_size // num_heads)
                    else:
                        # Fallback for Llama
                        head_dim = attn.hidden_size // num_heads

                    def make_hook(layer_idx, n_h, h_d):
                        def hook(module, args):
                            # args[0] is the input to o_proj, which is attn_output before o_proj.
                            # Shape is (bsz, q_len, num_heads * head_dim)
                            inp = args[0]
                            bsz, seq_len, _ = inp.shape
                            raw = inp.reshape(bsz, seq_len, n_h, h_d)
                            # Handle NaN to prevent numerical issues with float16
                            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                            # We detach it so we can save it without creating memory leaks in the graph
                            self.raw_head_outputs[layer_idx] = raw.detach()
                        return hook

                    h = o_proj.register_forward_pre_hook(make_hook(idx, num_heads, head_dim))
                    self.hooks.append(h)
                    hooks_registered += 1

        print(f"AttentionFeatureExtractor: Registered {hooks_registered} hooks on {len(layers)} layers")

    def clear(self):
        self.raw_head_outputs.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

class ObjectFocusedAttentionLoss(nn.Module):
    def __init__(self, sigma=1.0, temperature=0.1, use_sharpening=True, eps=1e-8):
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature
        self.use_sharpening = use_sharpening
        self.eps = eps

    def forward(self, attentions, object_centers, H, W, img_token_indices=None):
        """
        attentions: list of attention weights from different layers.
                    Shape of each layer's attn: (bsz, num_heads, q_len, k_len)
        object_centers: list of list of (x, y) coordinates for each batch item.
                        Assuming coordinates are normalized [0, 1].
        H, W: grid dimensions for image patches (e.g., 16, 16 for 256 tokens).
        img_token_indices: tensor of shape (bsz, n_query) or similar,
                           identifying which keys are visual tokens.
        """
        device = attentions[0].device
        bsz = attentions[0].shape[0]

        # Calculate grid coordinates [0, H-1] and [0, W-1]
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float() # (H, W, 2)
        coords = coords.view(-1, 2) # (H*W, 2)

        total_loss = 0.0
        valid_batches = 0

        for b in range(bsz):
            centers = object_centers[b]
            if len(centers) == 0:
                continue

            # Create target distribution g (sum of Gaussians)
            u = torch.zeros(H*W, device=device)
            for center in centers:
                # Map normalized [0, 1] to [0, W-1] and [0, H-1]
                cx = center[0] * (W - 1)
                cy = center[1] * (H - 1)
                center_t = torch.tensor([cx, cy], device=device, dtype=torch.float32)

                dist_sq = torch.sum((coords - center_t) ** 2, dim=-1)
                u += torch.exp(-dist_sq / (2 * self.sigma ** 2))

            # Normalize to get target distribution g
            g = u / (u.sum() + self.eps) # (H*W)

            layer_loss = 0.0
            for attn in attentions:
                # Average attention across heads
                # avg_attn shape: (q_len, k_len)
                avg_attn = attn[b].mean(dim=0) # (q_len, k_len)

                # Extract attention over visual tokens
                if img_token_indices is not None:
                    # img_token_indices[b] contains the indices of visual tokens
                    visual_attn = avg_attn[:, img_token_indices[b]] # (q_len, H*W)
                else:
                    # Fallback: assume last H*W tokens are visual
                    visual_attn = avg_attn[:, -H*W:]

                # Normalize predicted distribution q (attention over visual tokens)
                # q shape: (q_len, H*W)
                if self.use_sharpening:
                    # Sharpen the distribution to make it point-like
                    # (q + eps) to avoid log(0)
                    q_sharp = torch.pow(visual_attn + self.eps, 1.0 / self.temperature)
                    q = q_sharp / (q_sharp.sum(dim=-1, keepdim=True) + self.eps)
                else:
                    q = visual_attn / (visual_attn.sum(dim=-1, keepdim=True) + self.eps)

                # KL-like penalty
                ce_loss = -torch.sum(g * torch.log(q + self.eps), dim=-1) # (q_len)
                layer_loss += ce_loss.mean()

            total_loss += layer_loss / len(attentions)
            valid_batches += 1

        if valid_batches > 0:
            return total_loss / valid_batches
        return torch.tensor(0.0, device=device, requires_grad=True)

def extract_attention_points(attn_map, H, W, threshold=0.1, n_points=None):
    """
    Clean a spread-out attention map into discrete points.
    attn_map: (H*W) or (H, W) tensor/array
    H, W: grid dimensions
    threshold: ignore values below this (after normalization)
    n_points: if set, return top N peaks
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    if len(attn_map.shape) == 1:
        attn_map = attn_map.reshape(H, W)

    # Normalize 0-1
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 1. Thresholding
    binary_mask = attn_map > threshold
    cleaned_map = attn_map * binary_mask

    # 2. Simple local maxima finding (3x3 window)
    from scipy.ndimage import maximum_filter
    data_max = maximum_filter(cleaned_map, size=3, mode='constant')
    maxima = (cleaned_map == data_max) & (cleaned_map > threshold)

    # Get coordinates of maxima
    y, x = np.where(maxima)
    values = cleaned_map[y, x]

    # Sort by intensity
    idx = np.argsort(values)[::-1]
    y, x, values = y[idx], x[idx], values[idx]

    if n_points is not None:
        y, x, values = y[:n_points], x[:n_points], values[:n_points]

    # Convert back to normalized 0-1 coordinates
    points = []
    for i in range(len(x)):
        points.append((float(x[i]) / (W - 1), float(y[i]) / (H - 1)))

    return points, cleaned_map
