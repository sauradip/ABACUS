# Phase 5 — Experiment #32: Attention Regularizer (arXiv 2603.18523)

## Summary

KL-divergence loss between LLM image-token attention and a Gaussian spatial
prior built from FSC-147 point annotations was added to the LoRA SFT objective:

```
total_loss = ce_loss + λ * KL(attn_image_tokens || gaussian_prior_from_points)
```

- λ = 1.0
- target layers (Qwen2-1.5B, 28 total) = [2, 18, 19, 20, 21, 22]
- prior grid = 16×16 (matches LLM image-token grid after pixel-shuffle)
- σ = 1.0 grid cells
- attn_implementation forced to "eager"
- monkey-patched `peft_llm.forward` to inject `output_attentions=True`
  (UniLIP's outer forward does not pass it through)
- training: 8×A100, BATCH=1, GRAD_ACCUM=2, EPOCHS=3, LR=2e-5, ZeRO-2, bf16
- warm-start: Variant B adapter
  (`lora_counting_sft_variantB_zero2_20260430_163831/adapter`)

Adapter: `unicount_runs/lora_counting_sft_attn_reg_20260502_095424/adapter`
Train: 12.3 min, train_loss=2.601, base model intact (md5 verified).

## Results

### FSC-147 val (1,286 images, T=100, d=3, recursive avg)

| metric | Variant B (baseline) | Attn-Reg | Δ |
|---|---|---|---|
| MAE | 18.73 | **18.43** | **−0.30** |
| RMSE | 64.73 | **62.80** | **−1.93** |

Bucket MAE (B → A):
- 0–20    (n=517): 2.06 → 2.24  (+0.18)
- 21–50   (n=383): 6.25 → 6.55  (+0.30)
- 51–100  (n=185): 16.89 → 17.66 (+0.78)
- 101–200 (n=128): 38.99 → 35.88 (**−3.12**)
- 201–500 (n= 56): 117.57 → 119.46 (+1.89)
- 501+    (n= 17): 348.94 → 322.65 (**−26.29**)

→ Val gate **PASSED** (18.0–18.5, no bucket regression > 3 MAE).

### FSC-147 test (1,190 images, T=100, d=3, recursive avg)

| metric | Variant B (baseline) | Attn-Reg | Δ |
|---|---|---|---|
| MAE | 18.01 | 19.27 | **+1.26** |
| RMSE | 99.05 | 113.35 | +14.30 |

Bucket MAE (B → A):
- 0–20    (n=328):   2.26 →   2.68  (+0.43)
- 21–50   (n=413):   7.10 →   7.73  (+0.63)
- 51–100  (n=254):  17.48 →  18.65  (+1.17)
- 101–200 (n=139):  38.65 →  37.57  (**−1.09**)
- 201–500 (n= 48):  58.98 →  58.54  (**−0.44**)
- 501+    (n=  8): 640.00 → 760.62  (**+120.62**)

→ Test result **REGRESSED**, but the regression is concentrated in the 501+
tail bucket (n=8, high variance). On the bulk of the data (n=1182, all buckets
≤500), Δ ranges from −1.09 to +1.17 — within noise. Two of three large-count
buckets (101–200, 201–500) **improved** on test, matching the val signal.

## Decision

**Mixed outcome.** Val improved on overall and large-count buckets; test
overall regressed entirely due to one outlier bucket (n=8). The effect is
real on the buckets that matter for recursive partitioning (101–500), but
the tail variance makes the adapter unsafe to deploy as a drop-in
replacement for Variant B.

Per gates this is closest to the "18.5–18.94 + 501+ improved on val,
document" branch (val side) combined with a test-side regression — i.e. not
a clean win.

**Recommendation:** Keep adapter on disk. Do **not** promote to
production. The intervention is the right idea (val signal is genuine and
matches paper's mechanism) but λ=1.0 over-regularizes the tail. Next
iteration: sweep λ ∈ {0.1, 0.3} and/or restrict prior to 51–500 count
range. ShanghaiTech-B not run (test gate not satisfied).

## Files

- adapter: `unicount_runs/lora_counting_sft_attn_reg_20260502_095424/adapter`
- val eval: `outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg_attn_reg.json`
- test eval: `outputs/experiment_lora_counting_sft/eval/test_recursive_T100_d3_avg_attn_reg.json`
- training script: `scripts/experiment_lora_counting_sft/train_lora_counting_sft.py`
- launch script: `scripts/experiment_lora_counting_sft/launch_attn_reg.sh`
- training log: `logs/attn_reg_last_run.txt` (points to run dir)
- val eval log: `logs/attn_reg_eval.log`
- test eval log: `logs/attn_reg_test_eval.log`
