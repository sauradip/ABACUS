# HeadLens Implementation - UniLIP

## Overview
HeadLens layer has been successfully implemented for the UniLIP model based on the reference repository at https://github.com/sauradip/UniLip_mod.git

## Implementation Details

### Files Modified/Created

1. **unilip/model/language_model/headlens.py** (UPDATED)
   - `HeadLens`: Neural module with learnable Affine translator (A matrix) that translates head outputs to residual stream space
   - `AttentionFeatureExtractor`: Uses forward_pre_hooks on `o_proj` layers to capture raw attention head outputs before projection
   - `ObjectFocusedAttentionLoss`: Training loss for focusing attention on object centers using Gaussian distributions
   - `extract_attention_points`: Utility for extracting discrete attention peaks from heatmaps

2. **scripts/eval_headlens.py** (UPDATED)
   - Loads UniLIP model with attention output enabled
   - Uses hooks to extract raw head outputs from all 28 layers
   - Generates three visualizations:
     - `all_layers_attention.png`: Standard attention heatmaps (7×4 grid of all layers)
     - `all_layers_headlens.png`: HeadLens magnitude visualizations (7×4 grid of all layers)
     - `avg_layers_1_11_headlens.png`: Averaged HeadLens from layers 1 & 11 only

3. **scripts/demo_attention_regularizer.py** (READY)
   - Demonstrates ObjectFocusedAttentionLoss usage
   - Shows how to use sharpening for point-like attention distributions

4. **unilip/model/language_model/unilip_internvl.py** (UPDATED)
   - Initializes AttentionFeatureExtractor for mechanistic interpretability
   - Passes output_attentions=True through the language model

## Key Features

### Hook-Based Extraction
- Uses `register_forward_pre_hook` on the attention output projection layer (o_proj)
- Captures raw multi-head attention outputs before they're projected to the residual stream
- Shape: (batch_size, seq_len, num_heads, head_dim)

### HeadLens Processing
1. Isolates individual attention heads
2. Projects through the output projection (W_O)
3. Translates to residual stream space using learned Affine transform
4. Computes magnitude as norm of the translated output

### Layer Selection Strategy
- Extracts from ALL 28 layers for completeness
- Focuses on **Layers 1 and 11** for final aggregated analysis (configurable via `target_layers_1_11`)
- Allows fallback to Layer 1 if Layer 11 is unavailable

### Numerical Stability
- Handles NaN values gracefully with `np.nan_to_num()`
- Clips heatmaps to valid range [0, 1]
- Uses robust min/max normalization with epsilon safety

## Usage

### Running Evaluation
```bash
cd /data/amondal/UniCount/UniLIP
PYTHONPATH=/data/amondal/UniCount/UniLIP:$PYTHONPATH python scripts/eval_headlens.py \
  --model-path /data/amondal/unicount_runs/v3s_merged_base \
  --image-path apple.jpg \
  --output-dir ./headlens_results
```

### Output Files
- Three PNG visualizations showing attention patterns overlaid on the input image
- Heatmaps are normalized to [0, 1] and scaled to 448×448 for display
- Uses 'jet' colormap for attention, 'hot' colormap for HeadLens magnitudes

## Architecture Compatibility
Tested with InternVL-based language models including:
- Qwen models (with head_dim attribute)
- Llama-based models (with head_dim derived from hidden_size)
- Custom InternVL2 architecture

## Reference Implementation
Based on the official UniLip_mod repository:
- https://github.com/sauradip/UniLip_mod.git
- Follows the exact same approach for raw head extraction and HeadLens computation
