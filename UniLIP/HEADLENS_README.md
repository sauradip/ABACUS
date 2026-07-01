# HeadLens Layer Documentation

## Overview

HeadLens is an attention analysis and regularization layer integrated into the UniLIP model. It provides mechanisms to:
- Extract and analyze attention patterns from specific transformer layers
- Compute attention metrics (entropy, concentration, activation)
- Regularize attention during training
- Visualize attention distributions

## Architecture

### Core Components

1. **HeadLens Layer** (`headlens.py`)
   - Extracts attention from specified layers (default: Layer 1 and Layer 11)
   - Computes weighted combinations of attention across layers
   - Calculates attention metrics (entropy, concentration, head activation)
   - Provides vision attention extraction utilities

2. **AttentionRegularizer** (`headlens.py`)
   - Wrapper class for easier attention regularization during training
   - Integrates HeadLens with model outputs

## Installation & Setup

### File Structure

```
unilip/model/
├── language_model/
│   ├── headlens.py                 # New: HeadLens implementation
│   └── unilip_internvl.py           # Modified: Added HeadLens integration
└── unilip_internvl.py               # Modified: Added HeadLens methods

scripts/
├── eval_headlens.py                 # New: Evaluation script
└── demo_attention_regularizer.py    # New: Training demo
```

### Integration with Model

The HeadLens layer is automatically initialized in `UniLIP_InternVLForCausalLM`:

```python
self.headlens = HeadLens(
    attention_layers=[1, 11],
    num_heads=32,
    hidden_size=2048,
    use_entropy_regularization=True,
    entropy_weight=0.1,
)
```

## Usage

### 1. Evaluation with HeadLens

Run evaluation to analyze attention patterns:

```bash
python scripts/eval_headlens.py \
    --model-path /data/amondal/unicount_runs/v3s_merged_base \
    --image-path /path/to/image.jpg \
    --output-dir ./headlens_results \
    --visualize
```

**Options:**
- `--model-path`: Path to UniLIP checkpoint
- `--image-path`: Single image for evaluation
- `--image-dir`: Directory of images for batch evaluation
- `--question`: Question to ask about the image
- `--attention-layers`: Which layers to analyze (default: 1 11)
- `--visualize`: Generate attention visualizations
- `--device`: GPU device to use (default: cuda:0)
- `--dtype`: Model precision (float32, float16, bfloat16)

**Output:**
- `evaluation_results.json`: Attention metrics for each image
- `*_attention.png`: Attention visualizations (if --visualize enabled)

### 2. Training with Attention Regularization

Use attention regularization during training:

```bash
python scripts/demo_attention_regularizer.py \
    --model-path /data/amondal/unicount_runs/v3s_merged_base \
    --num-steps 5 \
    --batch-size 2 \
    --entropy-weight 0.1 \
    --output-dir ./regularizer_results
```

**Options:**
- `--model-path`: Path to UniLIP checkpoint
- `--num-steps`: Number of training steps
- `--batch-size`: Batch size
- `--entropy-weight`: Weight for entropy regularization loss
- `--attention-layers`: Which layers to regularize (default: 1 11)
- `--output-dir`: Results directory

**Output:**
- `regularization_results.json`: Training metrics and head importance scores

### 3. Using HeadLens in Model Forward Pass

Enable HeadLens regularization during training:

```python
outputs = model.forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    use_headlens_regularization=True,  # Enable HeadLens
    output_attentions=True,             # Get attention outputs
)

# Access attention regularization loss
reg_loss = outputs.loss
```

### 4. Analyzing Attention Programmatically

```python
from unilip.model.language_model.headlens import HeadLens

headlens = model.get_headlens()

# Forward pass with output_attentions=True
with torch.no_grad():
    outputs = language_model(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )

# Analyze attention
attention_dict = {idx: attn for idx, attn in enumerate(outputs.attentions)}
processed_attn, metrics = headlens.forward(attention_dict, return_metrics=True)

print(f"Mean Entropy: {metrics['mean_entropy']}")
print(f"Max Attention: {metrics['max_attention']}")
print(f"Concentration: {metrics['attention_concentration']}")
```

## Attention Metrics

### Entropy
- **Definition**: Shannon entropy of attention distribution
- **Interpretation**: 
  - High entropy: Attention spread uniformly across tokens
  - Low entropy: Attention focused on few tokens
- **Formula**: `-sum(p * log(p))`

### Attention Concentration (Gini Coefficient)
- **Definition**: Measures how concentrated attention is
- **Range**: 0 (uniform) to 1 (concentrated)
- **Use**: Identifies whether model attends to specific regions

### Head Activation
- **Definition**: Maximum attention value per head
- **Use**: Identifies which heads are most active

### Head Importance Scores
- **Definition**: Learned weights for each attention head
- **Use**: Prioritizes heads that contribute most to output

## Configuration

### HeadLens Parameters

```python
HeadLens(
    attention_layers=[1, 11],           # Layers to analyze
    num_heads=32,                       # Number of attention heads
    hidden_size=2048,                   # Hidden dimension
    use_entropy_regularization=True,    # Enable entropy loss
    entropy_weight=0.1,                 # Entropy loss weight
)
```

### Adjusting Attention Layers

For different model sizes, adjust `attention_layers`:
- **Small models**: Use [1] (only first layer)
- **Medium models**: Use [1, 8, 15]
- **Large models**: Use [1, 11, 22, 31]

If Layer 11 doesn't work for your model, try:
1. Use only Layer 1: `attention_layers=[1]`
2. Use early + middle layers: `attention_layers=[1, 6]`
3. Check your model's total number of layers

## Results Interpretation

### Evaluation Results Example

```json
{
  "image_path": "/path/to/image.jpg",
  "mean_entropy": 4.2541,
  "max_attention": 0.8432,
  "attention_concentration": 0.6234,
  "layer_entropies": [4.2541, 4.1823]
}
```

**Interpretation:**
- High entropy (4.25) → Attention is distributed
- Max attention (0.84) → Some tokens receive strong focus
- Concentration (0.62) → Moderate focus on specific regions

### Training Results Example

```json
{
  "final_metrics": {
    "final_loss": 0.0234,
    "mean_entropy": 4.1512,
    "entropy_std": 0.3421
  },
  "config": {
    "attention_layers": [1, 11],
    "entropy_weight": 0.1
  }
}
```

## Advanced Usage

### Custom Attention Layer Selection

```python
# Use only early layers
headlens_early = HeadLens(attention_layers=[1, 2, 3])

# Use only middle/late layers
headlens_late = HeadLens(attention_layers=[9, 10, 11])

# Use all layers
headlens_all = HeadLens(attention_layers=list(range(1, 12)))
```

### Computing Regularization Loss Only

```python
headlens = model.get_headlens()
attention_dict = {idx: attn for idx, attn in enumerate(outputs.attentions)}

# Get regularization loss without forward pass
reg_loss = headlens.regularization_loss(attention_dict)
total_loss = task_loss + reg_loss
```

### Extracting Vision-Specific Attention

```python
# After computing attention
vision_attn = headlens.extract_vision_attention(
    attention=processed_attn,
    vision_token_start=10,    # First vision token index
    vision_token_end=266,     # Last vision token index
    reduce_heads=True,        # Average across heads
)

# Visualize vision attention heatmap
visualization = visualize_attention_heatmap(vision_attn)
```

## Troubleshooting

### Error: "Layer 11 not found"
**Solution**: Reduce number of layers or check model size
```python
headlens = HeadLens(attention_layers=[1])  # Use only layer 1
```

### High regularization loss making training unstable
**Solution**: Reduce entropy_weight
```python
HeadLens(entropy_weight=0.01)  # Reduced from 0.1
```

### No attention outputs
**Ensure**: `output_attentions=True` is passed to model
```python
outputs = model.forward(..., output_attentions=True)
```

### Out of memory during attention analysis
**Solutions**:
1. Reduce batch size
2. Reduce `num_heads` in HeadLens
3. Analyze fewer attention layers

## References

- HeadLens provides interpretability into multi-head attention mechanisms
- Entropy-based regularization encourages balanced attention distributions
- Vision token attention extraction helps understand image understanding
- Suitable for analyzing both training and inference behavior

## Example Workflows

### Workflow 1: Analyze Pre-trained Model

```bash
# 1. Run evaluation
python scripts/eval_headlens.py \
    --model-path /data/amondal/unicount_runs/v3s_merged_base \
    --image-dir ./test_images \
    --output-dir ./analysis \
    --visualize

# 2. Check results
cat analysis/evaluation_results.json | jq '.' | head -50
```

### Workflow 2: Train with Regularization

```bash
# 1. Run training with attention regularization
python scripts/demo_attention_regularizer.py \
    --model-path /data/amondal/unicount_runs/v3s_merged_base \
    --num-steps 100 \
    --entropy-weight 0.1

# 2. Monitor regularization effectiveness
tail -50 regularization_results.json
```

## Performance Notes

- HeadLens adds minimal overhead (~2-5% memory)
- Attention extraction requires `output_attentions=True`
- Regularization loss is optional and controllable via `entropy_weight`
- Visualization generation scales with number of images
