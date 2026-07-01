# Native SFT Audit Error Summary

This document summarizes the native Stage-1 training and zero-shot audit failures that occurred after the first successful finite-loss smoke run.

## Executive Summary

- Training is stable on the native path.
- The first successful finite-loss smoke run was job `2112816`.
- Every subsequent failure happened in the audit/inference path, not in the training path.
- The current unresolved blocker is inside the remote `InternLM2` generation code used by `InternVL2-2B` under the installed Transformers stack: `prepare_inputs_for_generation()` dereferences `past_key_values[0][0].shape` when that entry is `None`.

## Run Timeline

| Job | Result | What Worked | Failure |
| --- | --- | --- | --- |
| `2112816` | `COMPLETED` | Native smoke run finished with finite loss | No audit in this smoke run |
| `2112817` | `FAILED` | 100-step pivot training completed | Audit loaded with unsupported `sdpa` attention |
| `2112818` | `FAILED` | Training completed; audit started | Audit used wrong checkpoint loading path and missed `img_context_token_id` |
| `2112834` | `FAILED` | PEFT checkpoint reconstruction worked | Patched model still lacked a working `generate()` chain |
| `2112858` | `FAILED` | `GenerationMixin` patch worked | `generation_config` was `None` on inner language model |
| `2112860` | `FAILED` | `generation_config` patch worked | Generation failed with `past_key_values[0][0] is None` |
| `2112870` | `FAILED` | Inference-mode cache patch applied | Duplicate `use_cache` keyword passed through wrapper |
| `2112871` | `FAILED` | Duplicate `use_cache` removed | Same `past_key_values[0][0] is None` bug remains |

## Stable Milestone

The first proof that the native trainer itself works is the smoke run:

- Job `2112816`
- Log shows finite, non-zero loss:

```text
{'train_runtime': 3.5922, 'train_samples_per_second': 4.454, 'train_steps_per_second': 0.278, 'train_loss': 3.152094841003418, 'epoch': 0.0}
```

Subsequent full runs also reached the 100-step pivot successfully and reported finite losses before failing in audit.

## Error Progression

### 1. Unsupported attention backend during audit load

- Job: `2112817`
- Symptom:

```text
ValueError: InternVLChatModel does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet.
```

- Root cause:
  The audit attempted to load `InternVLChatModel` with `attn_implementation="sdpa"`.

- Fix attempted:
  Added SDPA-to-eager fallback in `zero_shot_point_audit.py` and made audit attention default to eager in the launcher.

## 2. Wrong checkpoint interpretation and missing image context token

- Job: `2112818`
- Symptoms:

```text
AssertionError
```

and earlier warnings showed that many checkpoint weights were unused while many language-model weights were newly initialized.

- Root causes:
  The audit loaded a Trainer-saved PEFT-style full checkpoint as if it were a plain `AutoModel` checkpoint.
  This caused LoRA/base-layer weights to be ignored and many LM weights to be freshly initialized.
  The audit also did not set `model.img_context_token_id`, causing InternVL's generate path to assert.

- Fix attempted:
  Added checkpoint inspection, PEFT full-state reconstruction, and explicit `img_context_token_id` initialization.

## 3. PEFT generate path missing on inner language model

- Job: `2112834`
- Symptoms:

```text
AttributeError: 'LoraModel' object has no attribute 'generate'
AttributeError: 'InternLM2ForCausalLM' object has no attribute 'generate'
```

- Root cause:
  After reconstructing the checkpoint, the inner language model still did not expose a valid `generate()` path under the installed Transformers version.

- Fix attempted:
  Merged LoRA into the base model with `merge_and_unload()` and dynamically patched `GenerationMixin` back onto `InternLM2ForCausalLM`.

## 4. Missing generation config on patched model

- Job: `2112858`
- Symptom:

```text
AttributeError: 'NoneType' object has no attribute '_from_model_config'
```

- Root cause:
  After patching `GenerationMixin`, the inner LM still had no initialized `generation_config` object.

- Fix attempted:
  Explicitly created `GenerationConfig.from_model_config(model.language_model.config)` before generation.

## 5. `past_key_values` initialization bug in generation

- Jobs: `2112860`, `2112871`
- Symptom:

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

- Exact failing line inside remote model code:

```text
modeling_internlm2.py:1116
past_length = past_key_values[0][0].shape[2]
```

- Root cause:
  In the current generation stack, the inner `InternLM2` generation path receives `past_key_values` where at least one entry is `None`, but `prepare_inputs_for_generation()` assumes a fully materialized cache tuple and dereferences `.shape` unconditionally.

- Fixes attempted before the error persisted:
  Disabled gradient checkpointing on the loaded audit model.
  Forced `model.config.use_cache = True` and `model.language_model.config.use_cache = True`.
  Created `generation_config` explicitly.
  Re-ran generation through the repaired wrapper.

- Outcome:
  The error still reproduces after these patches, indicating the blocker is deeper than the audit script's wrapper-level config setup.

## 6. Duplicate `use_cache` keyword during wrapper call

- Job: `2112870`
- Symptom:

```text
TypeError: transformers.generation.utils.GenerationMixin.generate() got multiple values for keyword argument 'use_cache'
```

- Root cause:
  The audit script passed `use_cache=True` directly to `model.generate()`, but InternVL's own wrapper already forwards `use_cache=True` to the inner language model.

- Fix attempted:
  Removed the explicit `use_cache` argument from the audit call and kept cache enabled via configs only.

## Current Status

- The training side is working.
- The audit still fails in the underlying remote `InternLM2` generation implementation.
- The remaining failure is not the original launcher/trainer architecture issue; it is now a generation-runtime incompatibility in the remote model stack used for inference.

## Practical Conclusion

The debugging effort successfully proved the following:

- Native Stage-1 training reaches finite loss reliably.
- Checkpoint resume works.
- The 100-step pivot completes.
- The audit script can now load checkpoints, reconstruct PEFT weights, patch generation support, and reach the inner LM generation path.

The unresolved blocker is now a lower-level model/runtime bug:

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

inside the remote `modeling_internlm2.py` generation code when it expects populated `past_key_values` but receives `None`.

## Files Involved

- `scripts/counting_grpo/train_native_sft.py`
- `scripts/counting_grpo/submit_native_sft_stage1.slurm`
- `scripts/counting_grpo/zero_shot_point_audit.py`
- `checkpoints/native_sft_stage1/modeling_internvl_chat.py`
- `logs/native_sft_2112816.log`
- `logs/native_sft_2112817.err`
- `logs/native_sft_2112818.err`
- `logs/native_sft_2112834.err`
- `logs/native_sft_2112858.err`
- `logs/native_sft_2112860.err`
- `logs/native_sft_2112870.err`
- `logs/native_sft_2112871.err`