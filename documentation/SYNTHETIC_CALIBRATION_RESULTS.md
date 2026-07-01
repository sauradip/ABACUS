# Synthetic Counting Calibration — Results & Decision

## Setup
- **Spec:** Counting Circuits (arXiv 2603.18523) — train on synthetic ONLY, warm-start from Variant B (no LoRA-arch change, no cold start).
- **Data:** 8,400 synthetic dot images (448×448, uniform 7-bucket sampling, grid-hashed non-overlap; high-density buckets ≥301 use radius 3–6 + min_dist_factor=1.5).
- **Generator:** `scripts/experiment_lora_counting_sft/generate_synthetic_counting.py` → `data/synthetic_dots_train.json` (verified 87s, 0% saturated).
- **Train:** `accelerate + ds_zero2`, 8×A100, bf16, lr=2e-5, cosine, warmup 0.03, eff_batch 16; warm-start `--init_adapter_from <Variant_B_adapter>`.
- **Eval:** CARC `T=100, max_depth=3, avg_50` on FSC-147 val (1,286 imgs), 8 GPUs.

## Eval Outputs

| Run | val MAE | val RMSE |
|---|---:|---:|
| **Variant B baseline (leaf_resize)** | **18.73** | **64.73** |
| synthetic_1ep | 19.74 | 68.10 |
| synthetic_3ep_step525 | 20.55 | 72.42 |
| synthetic_3ep_step1050 | 20.71 | 72.44 |
| synthetic_3ep_step1575 | 20.87 | 74.46 |
| synthetic_3ep_final | 20.87 | 74.46 |

## Per-Bucket MAE (the only thing that matters: 201-500 and 501+)

| bucket | n | Variant B | synth_1ep | synth_3ep_step525 | synth_3ep_step1050 | synth_3ep_step1575 |
|---|---:|---:|---:|---:|---:|---:|
| 0-20    | 517 | 2.06   | 2.70   | 3.39   | 3.27   | 3.34   |
| 21-50   | 383 | 6.25   | 6.15   | 6.75   | 6.83   | 6.80   |
| 51-100  | 185 | 16.89  | 16.68  | 16.76  | 17.30  | 17.34  |
| 101-200 | 128 | 38.99  | 39.27  | 40.26  | 40.23  | 40.84  |
| 201-500 |  56 | **117.57** | 137.80 | 137.48 | 142.04 | 139.77 |
| 501+    |  17 | **348.94** | **341.29** | 360.76 | 354.12 | 367.59 |

## Decision (per spec gates)

- Best synthetic val MAE = **19.74** (1ep) > **18.94** (Variant B spec baseline = 18.73 reproduced).
- **Gate hit:** "Val MAE > 18.94 → Revert."
- 501+ improved marginally only at 1ep (−7.65 MAE), but regressed at all 3ep checkpoints (+5.18 to +18.65 MAE).
- 201-500 regressed catastrophically in every synthetic checkpoint (+19.91 to +24.47 MAE).

## Conclusion

**REVERT.** Synthetic dot calibration with the current generator + recipe does not transfer to FSC-147. The single mild win (501+ at 1ep) does not survive longer training, and the 201-500 bucket — the largest bucket of natural images that needs help — gets strictly worse. Do not run FSC-147 test or SHT-B. Keep Variant B as deployed.

### Hypotheses for failure (unverified)
- Domain gap: solid uniform dots on white backgrounds vs natural cluttered scenes.
- Recursion mismatch: synthetic samples are flat density; CARC tile-and-recurse hierarchy is never exercised on synthetic data.
- 7-way uniform bucketing with only 17 native FSC-147 images in 501+ means the held-out signal is dominated by ≤200-count regimes, where calibration helped least.

## Artifacts
- Generator: `scripts/experiment_lora_counting_sft/generate_synthetic_counting.py`
- Train data: `data/synthetic_dots_train.json` (8,400 entries)
- Train images: `/data/amondal/UniCountData/synthetic_dots/images/`
- Train logs: `logs/synthetic_{1ep,3ep}_*.log`
- Adapters:
  - 1ep: `/data/amondal/unicount_runs/lora_counting_sft_synthetic_1ep_20260501_193627/adapter/`
  - 3ep: `/data/amondal/unicount_runs/lora_counting_sft_synthetic_3ep_20260501_194418/{checkpoint-{525,1050,1575},adapter}/`
- Eval JSONs: `outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg_synthetic*.json`
- Driver: `scripts/experiment_lora_counting_sft/launch_synthetic.sh`, `run_synthetic_evals.sh`
- Adapter extractor (used to recover intermediate ckpts): `scripts/experiment_lora_counting_sft/extract_adapter_from_checkpoint.py`
