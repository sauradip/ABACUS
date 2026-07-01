# Objectness Map Training & Attention Regularization (AR-Loss)

**Purpose**: Train the model to focus attention on object-relevant regions using spatial regularization via Gaussian-based attention loss.

**Key Component**: `ObjectFocusedAttentionLoss` in `UniLIP/unilip/model/language_model/headlens.py`

---

## 1. What Is Objectness?

In the ABACUS pipeline, "objectness" refers to the model's ability to concentrate its visual attention on regions containing countable objects, rather than dispersing attention broadly across the image.

**Why it matters:**
- **Better localization**: Model learns where objects are located
- **Reduced counting errors**: Prevents context confusion (e.g., background vs. foreground)
- **Improved generalization**: Spatial regularization helps on out-of-distribution datasets

**Training stage**: Integrated into **Stage 2** training (dual-loss framework)
- Loss 1: Counting loss (MSE on ground truth counts)
- Loss 2: **Attention Regularization (AR-loss)** (spatial grounding)

---

## 2. How AR-Loss Works

### 2.1 Core Algorithm

The `ObjectFocusedAttentionLoss` module enforces alignment between:
- **Target distribution** $g(\mathbf{x})$: Gaussian mixture centered at object locations
- **Predicted distribution** $q(\mathbf{x})$: Model's attention weights over image patches

**Mathematical formulation:**

$$
\text{AR-Loss} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{L} \sum_{l=1}^{L} \mathbb{E}_{p \in P} \left[ -\sum_{i=1}^{N} g_i \log(q_i + \epsilon) \right]
$$

Where:
- $B$ = batch size
- $L$ = number of layers with attention
- $P$ = spatial patch grid (e.g., 16×16 for 256 patches)
- $g_i$ = target Gaussian density at patch $i$
- $q_i$ = normalized attention weight at patch $i$
- $\epsilon$ = numerical stability constant (~1e-8)

### 2.2 Step-by-Step Process

```python
# 1. CREATE TARGET DISTRIBUTION (Gaussian Mixture)
# For each object center at (cx, cy) in normalized coordinates [0,1]:
u(x,y) = Σ exp(-((x-cx)² + (y-cy)²) / (2σ²))

# 2. NORMALIZE to probability distribution
g(x,y) = u(x,y) / Σ u(x,y)

# 3. EXTRACT MODEL ATTENTION
# Average attention across heads and attention layers
q(x,y) = Attention_weights(image_patches)
q(x,y) = q(x,y) / Σ q(x,y)  # normalize

# 4. COMPUTE LOSS
# Cross-entropy between target and predicted attention
L_AR = -Σ g(x,y) * log(q(x,y) + ε)
```

### 2.3 Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sigma` | 1.0 | Width of Gaussian (in grid units) |
| `temperature` | 0.1 | Sharpening factor for predicted distribution |
| `use_sharpening` | True | Apply power-law sharpening to attention |
| `eps` | 1e-8 | Numerical stability |

**Interpretation:**
- **Low sigma**: Narrow Gaussian (focus on precise object centers)
- **High sigma**: Wide Gaussian (encourage broad coverage over objects)
- **High temperature**: Sharper predictions (more peaked attention)
- **Low temperature**: Softer predictions (diffuse attention)

---

## 3. Integration into Stage 2 Training

### 3.1 Dual-Loss Training Framework

```python
# From train_stage2.py (pseudocode)

total_loss = counting_loss + α * ar_loss

where:
  counting_loss = MSE(pred_count, gt_count)
  ar_loss = ObjectFocusedAttentionLoss(attentions, object_centers, H=16, W=16)
  α = weight_ar_loss (typically 0.1-1.0)
```

### 3.2 Data Requirements

For AR-loss training, you need:
1. **Images**: Original training images
2. **Counts**: Ground truth object counts
3. **Object locations** (optional but recommended):
   - Bounding boxes
   - Center points
   - Segmentation masks
   - Point annotations (dot maps)

**Data sources used in paper:**
- FSC-147: Dot annotations (object centers extracted)
- Objects365: Bounding boxes → center points
- V3Det: Bounding boxes → center points

### 3.3 Extracting Object Centers from Annotations

```python
# From dot maps (FSC-147)
def centers_from_dot_map(dot_map):
    """Extract object centers from binary dot annotation."""
    y, x = np.where(dot_map > 0.5)
    centers = list(zip(x / dot_map.shape[1], y / dot_map.shape[0]))
    return centers  # normalized [0, 1]

# From bounding boxes
def centers_from_bboxes(bboxes):
    """Extract center from [x1, y1, x2, y2] format."""
    centers = []
    for x1, y1, x2, y2 in bboxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    return centers  # absolute coordinates

# Normalize centers to [0, 1]
def normalize_centers(centers, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in centers]
```

---

## 4. Training Configuration

### 4.1 Stage 2 Training Command

```bash
python UniLIP/unilip/train/train_stage2.py \
  --model_name_or_path /path/to/unilip-3b \
  --data_path /path/to/training_data.json \
  --output_dir /path/to/output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --bf16 True \
  --deepspeed_config_file zero2.json \
  --loss_weight_ar 1.0  # AR-loss weight
```

### 4.2 DeepSpeed Configuration (zero2.json)

```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 8,
  "fp16": {
    "enabled": false
  },
  "bfloat16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

---

## 5. Attention Hook Mechanism

### 5.1 How Attention is Extracted

The `AttentionFeatureExtractor` class uses PyTorch forward hooks to capture intermediate attention values:

```python
class AttentionFeatureExtractor:
    def __init__(self, model):
        """Register hooks on all o_proj (output projection) layers."""
        self.raw_head_outputs = {}  # layer_idx -> (bsz, seq_len, num_heads, head_dim)
        self.register_hooks()
    
    def register_hooks(self):
        # For each layer in the model:
        for idx, layer in enumerate(model.layers):
            attn = layer.self_attn
            o_proj = attn.o_proj
            
            # Register pre-hook to capture attention before projection
            def make_hook(layer_idx):
                def hook(module, args):
                    # args[0] is attention output (bsz, seq_len, num_heads*head_dim)
                    # Reshape to (bsz, seq_len, num_heads, head_dim)
                    self.raw_head_outputs[layer_idx] = reshape_for_per_head_analysis(args[0])
                return hook
            
            o_proj.register_forward_pre_hook(make_hook(idx))
```

### 5.2 Why o_proj Hooks?

- **o_proj** = output projection layer that combines multi-head attention
- Located at: `model.layers[i].self_attn.o_proj`
- Captures: Raw attention from all heads before final projection

### 5.3 Extracting Attention Points

```python
def extract_attention_points(attn_map, H, W, threshold=0.1, n_points=None):
    """Convert diffuse attention distribution into discrete peak locations."""
    
    # 1. Normalize to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # 2. Threshold: ignore low-confidence regions
    binary_mask = attn_map > threshold
    cleaned_map = attn_map * binary_mask
    
    # 3. Find local maxima (using 3×3 sliding window)
    local_maxima = find_peaks_3x3(cleaned_map)
    
    # 4. Sort by intensity and return top N
    top_peaks = sort_by_intensity(local_maxima)[:n_points]
    
    # 5. Convert grid coordinates back to [0, 1] normalized space
    normalized_points = [(x / (W-1), y / (H-1)) for x, y in top_peaks]
    
    return normalized_points, cleaned_map
```

---

## 6. Training Dynamics & Loss Curves

### 6.1 Expected Behavior

**Early training (steps 0-5K):**
- Counting loss: High (model hasn't learned to count yet)
- AR-loss: High (attention is scattered)
- Both losses decline together

**Mid training (steps 5K-25K):**
- Counting loss: Rapid decline
- AR-loss: Slower decline (attention gradually focuses)
- Divergence begins (different local optima)

**Late training (steps 25K-50K):**
- Both converge
- Counting loss: Plateaus around 0.5-2.0 MAE equivalent
- AR-loss: Stabilizes with minor fluctuations

### 6.2 Debugging AR-Loss Issues

**If AR-loss is NaN/Inf:**
- Check object centers are normalized to [0, 1]
- Verify no empty batches (centers with count=0)
- Reduce sigma if Gaussians don't overlap patches

**If AR-loss doesn't decrease:**
- Increase loss weight: `--loss_weight_ar 2.0 or 5.0`
- Verify attention hooks are registered (check logs)
- Check object annotations are accurate

**If counting performance degrades:**
- Reduce AR-loss weight: `--loss_weight_ar 0.1`
- Use curriculum learning: Start with AR disabled, then enable at step N
- Verify object locations align with actual objects

---

## 7. Validation & Evaluation

### 7.1 Quantitative Metrics

After training, evaluate:

```bash
# Single-pass counting (baseline)
python scripts/experiment_lora_counting_sft/eval_lora_counting_sft.py \
  --checkpoint /path/to/checkpoint \
  --dataset fsc147

# CTAP+NRT evaluation (benefits from better attention)
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
  --checkpoint /path/to/checkpoint \
  --dataset fsc147 \
  --T 100  # threshold for recursive split
```

### 7.2 Expected Improvement from AR-Loss

| Metric | Baseline (no AR) | With AR-Loss | Improvement |
|--------|-----------------|--------------|------------|
| FSC-147 MAE | 10.76 | 8.48 | **21.2%** |
| FSC-147 RMSE | 50.2 | 40.91 | **18.5%** |
| CARPK MAE | 10.98 | 10.33 | **5.9%** |
| ShanghaiTech-A MAE | 234.70 | 93.85 | **60.0%** |

### 7.3 Qualitative Analysis

To visualize attention improvement:

```python
# Extract and visualize attention for an image
from UniLIP.unilip.model.language_model.headlens import extract_attention_points

attn_map = model.attention_weights[layer_idx][0]  # First batch item
points, cleaned_map = extract_attention_points(attn_map, H=16, W=16, threshold=0.1)

# Overlay attention heatmap on image
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(image)
ax1.set_title("Original Image")
ax2.imshow(image, alpha=0.5)
ax2.imshow(cleaned_map.reshape(16, 16), cmap='hot', alpha=0.5)
ax2.set_title("Attention Heatmap")
plt.show()
```

---

## 8. Production Training Pipeline

### 8.1 End-to-End Script

```bash
#!/bin/bash
# scripts/train_with_ar_loss.sh

set -e

MODEL_PATH="/data/amondal/model_cache/UniLIP-3B"
DATA_PATH="/path/to/fsc147_training_data.json"
OUTPUT_DIR="/data/amondal/unicount_runs/stage2_with_ar_loss"

# Step 1: Prepare data (extract object centers)
python scripts/preprocessing/extract_object_centers.py \
  --input_json $DATA_PATH \
  --output_json $OUTPUT_DIR/data_with_centers.json

# Step 2: Train with AR-Loss
python UniLIP/unilip/train/train_stage2.py \
  --model_name_or_path $MODEL_PATH \
  --data_path $OUTPUT_DIR/data_with_centers.json \
  --output_dir $OUTPUT_DIR/checkpoints \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --loss_weight_ar 1.0 \
  --bf16 True \
  --deepspeed_config_file UniLIP/deepspeed_scripts/zero2.json

# Step 3: Evaluate
python scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \
  --checkpoint $OUTPUT_DIR/checkpoints/adapter_extracted \
  --dataset fsc147 \
  --output_json $OUTPUT_DIR/results.json

# Step 4: Report
echo "AR-Loss training complete. Results saved to $OUTPUT_DIR/results.json"
```

### 8.2 Monitoring Training

```bash
# Watch logs in real-time
tail -f $OUTPUT_DIR/training_log.txt | grep -E "loss|ar_loss|step"

# Collect loss metrics
python -c "
import json
with open('$OUTPUT_DIR/training_log.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if 'loss' in data:
            print(f\"Step {data['step']:6d}: loss={data['loss']:.4f}, ar_loss={data.get('ar_loss', 0):.4f}\")
"
```

---

## 9. Troubleshooting

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| AR-loss is NaN | Object centers out of bounds or invalid | Verify centers are in [0, 1]; check for empty batches |
| Counting degrades | AR-loss weight too high | Reduce `loss_weight_ar` from 1.0 to 0.1-0.5 |
| Attention still unfocused | Sigma too large | Reduce `--ar_loss_sigma` from 1.0 to 0.5 |
| Training is slow | Attention hook overhead | Disable AR-loss during final steps; use periodic updates |
| Model doesn't generalize | Overfit to specific dataset's object locations | Use diverse datasets; increase object location noise |

---

## 10. References

- **Implementation**: `UniLIP/unilip/model/language_model/headlens.py`
- **Training script**: `UniLIP/unilip/train/train_stage2.py`
- **Paper section**: "Attention Regularization for Spatial Grounding"
- **Related work**: Guided attention mechanisms, attention bottleneck networks

---

## 11. Summary

AR-Loss enables ABACUS to learn object-specific spatial attention through:
1. **Gaussian target distributions** centered at object locations
2. **Attention hook extraction** from model's multi-head self-attention
3. **KL-divergence minimization** between target and predicted attention
4. **Dual-loss training** combining counting + spatial regularization

Result: **21% improvement on FSC-147** and strong generalization to out-of-distribution benchmarks.
