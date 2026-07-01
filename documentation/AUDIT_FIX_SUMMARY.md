# Stage 1.5 Audit Script Fix Log

**Date:** April 26, 2026
**Checkpoint:** `checkpoints/scaffold_rex_stage15_4353822` (job 4353822, loss=0.5297)
**Final result:** `pred_count=10` vs `gt_count=13` for `7.jpg` ✅

---

## Problem

The Phase 2 sanity audit (`generate_scaffold_rex_audit_v2.py`) was either crashing or producing `pred_count=0` across multiple SLURM jobs. The goal was to verify that the Stage 1.5 checkpoint could produce a non-zero prediction for `7.jpg` (GT=13, category: peppers).

---

## Bug Chain (6 layers)

### Bug 1 — Wrong checkpoint hardcoded
- **Job:** 4354629
- **Symptom:** `pred_count=0`
- **Root cause:** Script had `4342903` hardcoded instead of `4353822`
- **Fix:** Pass `MODEL_PATH` explicitly via env var in SLURM submission

---

### Bug 2 — `apply_stability_overrides()` never called
- **Job:** 4354803
- **Symptom:** `pred_count=0`
- **Root cause:** The function existed but was never invoked in the main inference path
- **Fix:** Added the function + call in `main()`

---

### Bug 3 — Call was in dead code branch
- **Job:** 4355345
- **Symptom:** `pred_count=0` (completed, 0:37)
- **Root cause:** `apply_stability_overrides()` call was indented inside the `if injected < 120: raise` error branch — it was unreachable under normal conditions
- **Fix:** Corrected indentation so the call runs unconditionally after injection

---

### Bug 4 — `_wrap_fp32_forward` only cast output, not input
- **Job:** 4355499 — FAILED (0:09)
- **Error:** `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` at attention linear layer
- **Root cause:** The wrapper cast the output back to `bfloat16` but left the input as `float32`. Downstream attention layers (which stay `bfloat16`) then received `float32` activations.
- **Fix:**
```python
def _forward_cast(x, *args, **kwargs):
    input_dtype = x.dtype if torch.is_tensor(x) else None
    if torch.is_tensor(x):
        x = x.float()          # cast input to float32
    out = original_forward(x, *args, **kwargs)
    if input_dtype is not None and torch.is_tensor(out):
        return out.to(input_dtype)   # restore original dtype on output
    return out
```

---

### Bug 5 — LM head weight cast to float32, activations are bfloat16
- **Job:** 4355693 — FAILED (0:09)
- **Error:** `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float` at `logits = self.output(hidden_states)` in `modeling_internlm2.py:1081`
- **Root cause:** `apply_stability_overrides()` was calling `output_embeddings.to(torch.float32)`, casting the LM head Linear weight to `float32`. After the fixed RMSNorm wrapper restored bfloat16 activations, `F.linear(bf16_input, fp32_weight)` crashed.
- **Fix — two changes:**
  1. Removed `output_embeddings.to(torch.float32)` from `apply_stability_overrides`
  2. Added `torch.amp.autocast("cuda", dtype=torch.bfloat16)` around `model.generate()`

```python
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output_ids = model.generate(**generation_kwargs)
```

- **Verification:** All 4 local dtype tests passed in `debug_dtype_audit.py` before submission
- **Job 4355925:** COMPLETED (0:17, exit 0) — no crash, but `pred_count=0`

---

### Bug 6 — Stage 1.5 LoRA deltas never applied
- **Jobs:** 4355925, 4356227 — COMPLETED but `pred_count=0`
- **Root cause:** The checkpoint stores 240 LoRA keys (`lora_A`, `lora_B`) as unmerged PEFT weights. The `inject_stage1_backbone_from_safetensors()` function explicitly skipped them (`if "lora_" in key: continue`). The model was running with only the Stage-0 base backbone — none of the Stage 1.5 fine-tuning was applied.
- **Fix:** Added `merge_lora_deltas_from_safetensors()` function that:
  1. Reads all `lora_A` / `lora_B` pairs from checkpoint safetensors
  2. Computes delta = `lora_B @ lora_A × (alpha / rank)` = `lora_B @ lora_A × 2.0`
  3. Adds delta in-place to the corresponding base weight (`module.weight`)
  4. Handles the `.weight` suffix (stem points to the module, not the tensor)

```python
delta = (lora_B.float() @ lora_A.float()) * scaling   # scaling = 128/64 = 2.0
# ...
param.data.add_(delta.to(device=param.device, dtype=param.dtype))
```

- **Job 4356465:** LoRA Merge applied 0 deltas — key suffix bug (`.weight` missing)
- **Fix:** Appended `.weight` to the resolved module path before lookup
- **Job 4356501:** ✅ `LoRA Merge Complete: applied 120 LoRA delta(s) (scaling=2.000)` → `pred_count=10, gt_count=13`

---

## Final State

| File | Change |
|------|--------|
| `scripts/counting_grpo/generate_scaffold_rex_audit_v2.py` | All 5 fixes applied |
| `scripts/counting_grpo/debug_dtype_audit.py` | New: local CPU dtype test (4 assertions, no PIL/transformers needed) |

### Key functions added/modified

| Function | Change |
|----------|--------|
| `_wrap_fp32_forward()` | Casts **input** to float32, restores original dtype on output |
| `apply_stability_overrides()` | Removed `output_embeddings.to(float32)` |
| `run_generate()` | Added `autocast("cuda", dtype=bfloat16)` around `model.generate()` |
| `merge_lora_deltas_from_safetensors()` | **New** — merges LoRA deltas into base weights at inference time |
| `main()` | Calls `merge_lora_deltas_from_safetensors()` after backbone injection |

---

## Submission History

| Job | State | Elapsed | pred_count | Notes |
|-----|-------|---------|-----------|-------|
| 4354629 | COMPLETED | 0:37 | 0 | Wrong checkpoint |
| 4354803 | COMPLETED | 0:37 | 0 | `apply_stability_overrides` not called |
| 4355345 | COMPLETED | 0:37 | 0 | Call in dead branch |
| 4355499 | FAILED | 0:09 | — | float32 in attention |
| 4355693 | FAILED | 0:09 | — | float32 LM head weight |
| 4355925 | COMPLETED | 0:17 | 0 | LoRA deltas not applied |
| 4356227 | COMPLETED | 0:20 | 0 | LoRA deltas not applied (vs4, scale=4.0) |
| 4356465 | COMPLETED | 0:28 | 0 | LoRA merge: 0 deltas (`.weight` suffix bug) |
| **4356501** | **COMPLETED** | **0:31** | **10** | ✅ All fixes applied |

---

## Next Steps

1. Run full audit: `MAX_SAMPLES=0` with the fixed script
2. Generate RankDPO pairs from full audit output
3. Launch Stage 2.5 DPO training

---

## Output Logs

### Failed Jobs — Key Errors

**Job 4355499** (`float != bfloat16` at attention):
```
Traceback (most recent call last):
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

**Job 4355693** (`bfloat16 != float` at LM head):
```
Traceback (most recent call last):
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float
```

**Job 4355925** (no crash, but LoRA not applied):
```
Injection Complete: Reconstructed 171 Stage-1 backbone tensors from 2 file(s) (skipped=0).
[1/1] 7.jpg pred=0 gt=13 parsed_json=True
Zero-count rows: 1/1 (100.00%)
```

**Job 4356465** (LoRA merge ran but applied 0 deltas — `.weight` suffix bug):
```
Injection Complete: Reconstructed 171 Stage-1 backbone tensors from 2 file(s) (skipped=0).
LoRA Merge Complete: applied 0 LoRA delta(s) (scaling=2.000).
[warn] No LoRA deltas found in checkpoint — running without Stage-1.5 LoRA.
[1/1] 7.jpg pred=0 gt=13 parsed_json=True
Zero-count rows: 1/1 (100.00%)
```

---

### Successful Job 4356501 — Full `.log`

```
=== Submit Scaffold Self-Audit V2 ===
SLURM_JOB_ID   : 4356501
SLURM_NODELIST : nid010295
MODEL_PATH     : /projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_4353822
SCAFFOLD_JSONL : /projects/u6fb/myprojects/UniCount/outputs/scaffold_rex_5k/all.jsonl
AUDIT_JSON     : /projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_4353822/sanity_audit_lora_v2.json
MAX_SAMPLES    : 1
Launching...
=== Scaffold-Rex Self-Audit V2 ===
Python        : .venv311/bin/python
Model path    : /projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_4353822
Scaffold jsonl: /projects/u6fb/myprojects/UniCount/outputs/scaffold_rex_5k/all.jsonl
Output json   : .../sanity_audit_lora_v2.json
Max samples   : 1 (0 means full dataset)

[warn] AutoImageProcessor load from checkpoint failed: no preprocessor_config.json
[info] Falling back AutoImageProcessor to base model: OpenGVLab/InternVL2-2B
[info] Model init path: OpenGVLab/InternVL2-2B
FlashAttention2 is not installed.
Warning: Flash attention is not available, using eager attention instead.
Injection Complete: Reconstructed 171 Stage-1 backbone tensors from 2 file(s) (skipped=0).
LoRA Merge Complete: applied 120 LoRA delta(s) (scaling=2.000).
[info] Applied stability overrides: vision_scale=1.0
[info] Vision scaling: using default model projector scale (1.0)
[debug] pixel_values.shape=(9, 3, 448, 448), num_patches=9, min_dynamic_patch=2, max_dynamic_patch=12, force_manual_tiling=True
[1/1] 7.jpg pred=10 gt=13 parsed_json=True
=== Scaffold-Rex Self-Audit Complete ===
Rows: 1
JSON parse success: 1/1 (100.00%)
Zero-count rows: 0/1 (0.00%)
Wrote: .../sanity_audit_lora_v2.json
```

---

### Successful Job 4356501 — Output JSON (`sanity_audit_lora_v2.json`)

```json
{
  "model_path": "/projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_4353822",
  "num_rows": 1,
  "json_parse_success": 1,
  "json_parse_success_rate": 100.0,
  "zero_count_rows": 0,
  "zero_count_rate": 0.0,
  "rows": [
    {
      "image": "7.jpg",
      "prompt": "<image>\nThe image is overlaid with a 6x6 dot matrix. Dots are labeled (x,y). ...\nCount the peppers and group instances by nearest anchor.",
      "prediction_text": "{\"total_count\":10, \"anchors_summary\":\"Objects identified near coordinates (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (2,1), (2,3), (2,5), (2,6).\", \"clusters\":[{\"anchor\":[1,1],\"count\":1,\"region_bbox\":[81,66,101,86]},{\"anchor\":[1,2],\"count\":1,\"region_bbox\":[137,64,157,84]},{\"anchor\":[1,3],\"count\":1,\"region_bbox\":[201,64,221,84]},{\"anchor\":[1,4],\"count\":1,\"region_bbox\":[266,64,286,84]},{\"anchor\":[1,5],\"count\":1,\"region_bbox\":[320,64,340,84]},{\"anchor\":[1,6],\"count\":1,\"region_bbox\":[384,64,404,84]},{\"anchor\":[2,1],\"count\":1,\"region_bbox\":[75,127,95,147]},{\"anchor\":[2,3],\"count\":1,\"region_bbox\":[201,127,221,147]},{\"anchor\":[2,5],\"count\":1,\"region_bbox\":[320,127,340,147]},{\"anchor\":[2,6],\"count\":1,\"region_bbox\":[384,127,404,147]}]}",
      "pred_count": 10,
      "gt_count": 13
    }
  ]
}
```
