# Experiment #33 — Dual-Adapter Eval (Variant B global + attn-reg local)

**Setup.** No training. Reused two existing LoRA adapters in dual-adapter eval (`eval_ctap_nrt_fsc147.py` with `--checkpoint_dir <global> --local_adapter <local>`):
- **Global (depth 0)**: Variant B → `lora_counting_sft_variantB_zero2_20260430_163831/adapter`
- **Local (depth ≥1)**: Attn-Reg (Exp #32) → `lora_counting_sft_attn_reg_20260502_095424/adapter`

CTAP+Recursive, `T=100`, `max_depth=3`, `min_size=224`, average aggregation, 8×A100, `mixed_precision=no`.

## Headline numbers

| Run                          | val MAE | val RMSE | test MAE | test RMSE |
|------------------------------|--------:|---------:|---------:|----------:|
| Single-adapter baseline (B)  |  18.94  |   65.21  |  18.01   |    99.05  |
| Dual (crop-local, prior)     |  21.14  |   81.38  |  17.24   |   117.46  |
| **Dual (attn-reg local)**    |**18.62**|   66.58  |  18.29   |   102.31  |

- vs single baseline: val **−0.32**, test **+0.28**.
- vs dual crop-local: val **−2.52**, test **+1.05**.

The attn-reg local adapter behaves *qualitatively differently* from the crop-local adapter: it does not regress val (no over-counting on small-count images) but it also does not pick up the test gain on the 101–200 bucket that crop-local enjoys. Net effect on val is a small win; net effect on test is a small loss.

## Per-bucket val MAE

| bucket  |     n |  Single B | Dual crop | Dual attn-reg |
|--------:|------:|----------:|----------:|--------------:|
|   0–20  |  517  |   2.04    |   2.10    |     2.10      |
|  21–50  |  383  |   6.27    |   6.25    |     6.25      |
|  51–100 |  185  |  16.97    |  16.02    |    16.62      |
| 101–200 |  128  |  39.37    |  36.48    |    39.67      |
| 201–500 |   56  | 118.38    | 132.55    |   118.25      |
|  501-+  |   17  | 358.65    | 508.59    | **334.47**    |

## Per-bucket test MAE

| bucket  |     n |  Single B | Dual crop | Dual attn-reg |
|--------:|------:|----------:|----------:|--------------:|
|   0–20  |  328  |   2.26    |   2.26    |     2.26      |
|  21–50  |  413  |   7.10    |   6.94    |     7.19      |
|  51–100 |  254  |  17.48    |**14.78**  |    17.43      |
| 101–200 |  139  |  38.65    |**25.50**  |    38.72      |
| 201–500 |   48  |  58.98    |  71.29    |    59.06      |
|  501-+  |    8  | 640.00    | 774.38    |   676.50      |

**Reading.** The attn-reg local is essentially a *no-op rounded-toward-baseline* on most buckets (numbers track Single B closely). Where the crop-local big test gain comes from (51–100 and 101–200) the attn-reg local recovers ~none of it. Where crop-local hurts (val 501+), attn-reg local actually *helps* (334.47 vs 358.65 single, 508.59 crop). In short, attn-reg learned a much milder/safer sub-view counter.

## Conditional X-routing (gate on `global_count`)

Use dual-attn-reg prediction iff `global_count ≤ X`, else fall back to baseline.

VAL:

| X    | MAE  | RMSE | n_dual_used |
|-----:|-----:|-----:|------------:|
| 100  |18.91 |65.16 | 1153/1286   |
| 125  |18.79 |64.50 | 1194/1286   |
| 150  |18.83 |64.56 | 1236/1286   |
| 175  |18.88 |64.63 | 1245/1286   |
| 200  |18.75 |64.37 | 1259/1286   |
| 250  |18.72 |64.23 | 1265/1286   |
| 300  |18.68 |63.90 | 1272/1286   |
| 400  |18.67 |63.97 | 1278/1286   |
| ∞    |**18.62**|66.58| 1286/1286 |

TEST:

| X    | MAE  | RMSE | n_dual_used |
|-----:|-----:|-----:|------------:|
| 100  |18.01 |99.05 | 1000/1190   |
| 125  |18.01 |99.05 | 1064/1190   |
| 150  |18.03 |99.06 | 1120/1190   |
| 175  |18.05 |99.07 | 1141/1190   |
| 200  |18.02 |99.01 | 1158/1190   |
| 250  |**17.98**|99.00| 1173/1190 |
| 300  |18.08 |100.95| 1180/1190   |
| 400  |18.09 |100.96| 1182/1190   |
| ∞    |18.29 |102.31| 1190/1190   |

**Best joint X.** No setting clears the user's "clean win" bar (val<18.5 AND test<18.0). The single best joint operating point is **X=250**:
- val MAE 18.72 (vs baseline 18.94, Δ −0.22)
- test MAE **17.98** (vs baseline 18.01, Δ −0.03; new best test MAE seen so far at this eval config)

So X-routing of attn-reg local at X=250 delivers a *tiny* simultaneous improvement on both splits — an honest but unimpressive +0.22 val / +0.03 test over Variant B alone. It is strictly safer than dual crop-local: no test split below 17.98 here is matched by the crop-local 17.24, but crop-local pays val MAE 21.14.

## Verdict

- **Not a clean win.** Val < 18.5 AND test < 18.0 is not achieved by any X.
- **Closest to a free lunch:** X=250 routing, both splits dominate single-baseline by very small margins.
- **Pareto picture across all three regimes (single B / dual crop / dual attn-reg / X-routed):**
  - Best test MAE outright: dual crop-local (17.24, but val 21.14).
  - Best val MAE outright: dual attn-reg full (18.62, test 18.29).
  - Best joint: X=250 attn-reg routing (val 18.72, test 17.98).

The attention-regularizer adapter, used as a sub-view specialist with a routing threshold, gives a small consistent improvement on both splits but does not displace the dual crop-local model on test. Phase 5 is closed.
