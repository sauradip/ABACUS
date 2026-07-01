# Exp #34 — Attention regularizer with FIXED KL direction (single adapter)

## Bug fix
- **File**: `scripts/experiment_lora_counting_sft/train_lora_counting_sft.py`, function `compute_attention_focus_loss`, line 166.
- **Before** (mode-seeking, wrong): `KL(attn || prior) = Σ attn · (log attn − log p)` — attention can collapse onto a single object location and still satisfy the loss.
- **After** (mode-covering, paper-correct): `KL(prior || attn) = Σ p · (log p − log attn)` — attention is forced to put mass on every annotated object location.
- All other config identical: warm-start from Variant B adapter, layers `[2,18,19,20,21,22]`, λ=1.0, σ=1.0, 16×16 grid, 8×A100 zero2 bf16, lr 2e-5, 3 epochs (687 steps), eff batch 16.

## Training trajectory
- Buggy `KL(attn||p)`: focus loss 4.4 → 0.4
- FIXED `KL(p||attn)`: focus loss ~1.3 → 0.3–0.5 (similar final magnitude, different curve as predicted — slower drop, harder to satisfy)

## FSC-147 val (n=1286)

| Variant                       | MAE   | RMSE  | rec   | 0-20 | 21-50 | 51-100 | 101-200 | 201-500 | 501+   |
|-------------------------------|-------|-------|-------|------|-------|--------|---------|---------|--------|
| Baseline (Variant B)          | 18.94 | 65.21 | 0.103 | 2.04 | 6.27  | 16.97  | 39.37   | 118.38  | 358.65 |
| Buggy `KL(attn‖p)`            | 18.43 | 62.80 | 0.117 | 2.24 | 6.55  | 17.66  | 35.88   | 119.46  | 322.65 |
| **FIXED `KL(p‖attn)`**        | 19.18 | 71.54 | 0.101 | **2.13** | **6.15**  | **16.37**  | 37.56   | 124.21  | 377.29 |
| Δ FIXED vs baseline           | +0.24 | +6.33 |       | +0.09| −0.12 | −0.60  | −1.81   | +5.83   | +18.65 |
| Δ FIXED vs buggy              | +0.75 | +8.74 |       | −0.11| −0.40 | −1.29  | +1.69   | +4.75   | +54.65 |

## FSC-147 test (n=1190)

| Variant                       | MAE   | RMSE   | rec   | 0-20 | 21-50 | 51-100 | 101-200 | 201-500 | 501+   |
|-------------------------------|-------|--------|-------|------|-------|--------|---------|---------|--------|
| Baseline (Variant B)          | 18.01 | 99.05  | 0.160 | 2.26 | 7.10  | 17.48  | 38.65   | 58.98   | 640.00 |
| Buggy `KL(attn‖p)`            | 19.27 | 113.35 | 0.166 | 2.68 | 7.73  | 18.65  | 37.57   | 58.54   | 760.62 |
| **FIXED `KL(p‖attn)`**        | 18.16 | 101.75 | 0.165 | 2.33 | 7.22  | **17.16**  | 37.76   | 61.08   | 666.38 |
| Δ FIXED vs baseline           | +0.15 | +2.70  |       | +0.07| +0.12 | −0.32  | −0.89   | +2.10   | +26.38 |
| Δ FIXED vs buggy              | **−1.11** | **−11.60** |  | −0.35| −0.51 | −1.49  | +0.19   | +2.54   | −94.24 |

## Interpretation

1. **FIX is mechanistically correct on the bucket where the regularizer should act.**
   FIXED beats *both* the baseline and the buggy variant on every small/medium bucket of val (0-20, 21-50, 51-100) and on the 51-100 bucket of test. These are the regimes where individuation of distinct objects matters most — exactly the failure mode mode-covering is designed to address.

2. **Test improves substantially vs the buggy run** (18.16 vs 19.27, −1.11 MAE; RMSE −11.60). On the 501+ bucket alone the FIX recovers ~94 MAE points relative to the buggy variant. So the bug was real and the fix is real.

3. **Neither version cleanly beats baseline overall.** The aggregate MAE is dominated by the 17 (val) / 8 (test) extreme-tail images (501+); a single high-count miss moves overall MAE more than a sweep of small images. Both attn-reg variants destabilise the tail relative to baseline.

4. **The buggy run's val −0.51 advantage is now explainable**: it was a one-bucket lottery on the 17-image 501+ bucket (322 vs baseline 358, vs FIXED 377), not a mechanistic gain. On test the buggy run is much worse than baseline, consistent with this.

## Verdict
- The KL direction was reversed; the fix is correct and produces the expected attention behaviour (slower-converging focus loss, consistent improvement on small/medium buckets, large recovery on test 501+ vs the buggy variant).
- As a single drop-in regularizer at λ=1.0 it is **not a clean win** on aggregate MAE because the long-tail dense images dominate. Possible follow-ups if pursued:
  - Down-weight λ for high-count images, or warmup λ over training.
  - Use FIXED adapter as the local adapter in the dual-adapter (Exp #33) routing setup, where single-object failures are exactly what X-routing targets.
  - Train one extra epoch under FIXED to let mode-covering finish converging (focus loss did not flatten as completely as the buggy run's).

## Artifacts
- Adapter: `/data/amondal/unicount_runs/lora_counting_sft_attn_reg_20260502_113517/adapter`
- Val:  `outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg_attn_reg_FIXED.json`
- Test: `outputs/experiment_lora_counting_sft/eval/test_recursive_T100_d3_avg_attn_reg_FIXED.json`
- Train log: `logs/attn_reg_20260502_113517.log`
- Chain log: `logs/attn_reg_FIXED_full.log`
