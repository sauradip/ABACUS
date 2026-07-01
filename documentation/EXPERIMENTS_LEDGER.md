# Experiment Ledger — LoRA Counting SFT

Workspace state after cleanup (May 2, 2026). All paths are absolute or relative to
`/data/amondal/UniCount/`.

---

## Shipped result

| Item | Path |
|---|---|
| Adapter (production) | `/data/amondal/unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/adapter/` |
| Symlink | `/data/amondal/unicount_runs/SHIPPED_ADAPTER` |
| Val eval | `outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg.json` (MAE=18.94, RMSE=65.21) |
| Test eval | `outputs/experiment_lora_counting_sft/eval/test_recursive_T100_d3_avg.json` (MAE=18.01, RMSE=99.05) |
| SHT-B eval | `outputs/experiment_lora_counting_sft/eval/cross_dataset/shtb_carc_T100_d3_avg.json` |
| SHT-A eval | `outputs/experiment_lora_counting_sft/eval/cross_dataset/shta_carc_T100_d3_avg.json` |
| CARPK eval | `outputs/experiment_lora_counting_sft/eval/cross_dataset/carpk_carc_T100_d3_avg.json` |
| Checksums | `SHIPPED_CHECKSUMS.md5` (verify with `md5sum -c`) |

**Eval recipe:** CTAP+Recursive, T=100, max_depth=3, avg aggregation, min_size=224.

---

## Pending decision: min_size

- `outputs/experiment_lora_counting_sft/eval/val_recursive_T100_d3_avg_ms100.json` — val MAE=18.67 (better than ms=224)
- `outputs/experiment_lora_counting_sft/eval/test_recursive_T100_d3_avg_ms100.json` — test MAE=18.12 (slightly worse)

Sweep (val): ms={224, 150, 100, 50} → MAE = {18.94, 18.69, 18.67, 18.67}. Saturates at 100.
Test does NOT confirm: aggregate MAE +0.11, RMSE +4.5 due to 501+ bucket regression on n=8.
**Decision pending.** Default remains ms=224. See conversation log for full per-bucket breakdown.

---

## Cross-dataset ablation (Variant B adapter)

CARC + recursive (T=100, max_depth=3, avg). Default = single LoRA adapter.
Dual = global LoRA + local-attention adapter combined at inference.

| Dataset | n | Default MAE | Default RMSE | Dual MAE | Dual RMSE |
|---|---:|---:|---:|---:|---:|
| SHT-B | 316 | **25.21** | 38.70 | 25.75 | 40.84 |
| SHT-A | 182 | **119.38** | 201.00 | 128.21 | 227.81 |
| CARPK | 459 | 15.56 | 20.02 | **14.28** | 18.12 |

**Files (live):**
```
outputs/experiment_lora_counting_sft/eval/cross_dataset/
    shtb_carc_T100_d3_avg.json,  shtb_carc_T100_d3_avg_dual.json
    shta_carc_T100_d3_avg.json,  shta_carc_T100_d3_avg_dual.json
    carpk_carc_T100_d3_avg.json, carpk_carc_T100_d3_avg_dual.json
    {shtb,shta,carpk}_scheme_sweep.txt   # per-X aggregation sweeps
```

Notes:
- CARPK is the only dataset where the dual adapter wins (-1.28 MAE, -1.90 RMSE).
- SHT-A regresses substantially under dual (+8.83 MAE).
- The `*_scheme_sweep.txt` files contain per-aggregation-X bucketed MAE.

---

## Regression-head experiment — Step 0 linear probe gate (NO-GO)

Goal: replace LM-head text generation with a small MLP regression on the last
hidden state, hypothesizing the autoregressive tokenization ceiling (~450) is
the bottleneck.

Gate: extract last-layer hidden state (forward hook on
`model.get_model().language_model.model.layers[-1]`) for 500 random val
images, fit 5-fold StandardScaler+RidgeCV across three pools and two targets.

| Pool | Target | CV R² | CV MAE |
|---|---|---:|---:|
| last_prompt token | count | **+0.114 ± 0.081** | 48.62 |
| last_prompt token | log1p(count) | +0.075 ± 0.072 | 40.03 |
| last image token  | count | -0.022 ± 0.021 | 55.30 |
| last image token  | log1p(count) | -0.075 ± 0.032 | 45.37 |
| mean over image tokens | count | -0.022 ± 0.021 | 55.30 |
| mean over image tokens | log1p(count) | -0.075 ± 0.032 | 45.37 |

N=500 val items, count range [8, 1022], mean=60.3.

**Decision: NO-GO.** Best R² = +0.114 ≪ 0.3 gate threshold. The last hidden
state at any tested position does not linearly encode count. Spec implication:
**the bottleneck is upstream of the LM head — in the visual encoder, not in
text generation.** A regression MLP would not help (the in-context CARC
recursion is already extracting whatever count signal is recoverable from
this representation).

Artifact: `outputs/experiment_lora_counting_sft/eval/linear_probe_gate_n500.log`
Script: `scripts/experiment_lora_counting_sft/linear_probe_gate.py`

Steps 1–5 (full feature extraction, MLP sweep, eval, CARC integration,
cross-dataset) **not executed**: gate failed.

---

## Datasets (in place)

| Item | Path |
|---|---|
| FSC-147 SFT train | `outputs/experiment_lora_counting_sft/train/train_counting.json` |
| FSC-147 SFT val | `outputs/experiment_lora_counting_sft/val/val_counting.json` |
| FSC-147 SFT test | `outputs/experiment_lora_counting_sft/test/test_counting.json` |
| Cross-dataset eval | `data/cross_dataset/{sht_a,sht_b,carpk}/test_counting.json` |
| Clean consolidated train | `/data/amondal/UniCountData/ucount_consolidated/train_counting_clean.json` |
| Decontamination audit | `/data/amondal/UniCountData/ucount_consolidated/audit/` |
| Source FSC-147 | `/data/amondal/FSC147_hf/` |
| Source ShanghaiTech | `/data/amondal/ShanghaiTech/` |
| Source CARPK/PUCPR+ | `/data/amondal/datasets/` |
| Base model | `/data/amondal/model_cache/UniLIP-3B/` |

---

## Working scripts (in place)

```
scripts/experiment_lora_counting_sft/train_lora_counting_sft.py
scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py
scripts/experiment_lora_counting_sft/extract_and_eval_all_ckpts.sh
scripts/experiment_lora_counting_sft/extract_adapter_from_checkpoint.py
scripts/experiment_lora_counting_sft/offline_t_sweep.py
scripts/recompute_aggregation.py
scripts/scheme_sweep.py
```

---

## Archive (`archive/experiments/`)

### Adapters (`archive/experiments/adapters/`)

Per-run `.tar.gz`. **Each archive contains `adapter*/` weights + `trainer_state.json` + configs.**
The heavy DeepSpeed `checkpoint-N/` (optimizer states) and `merged/` directories were excluded
to free ~1.3 TB. Cannot resume training from these — only re-evaluate.

| Archive | Notes |
|---|---|
| `lora_counting_sft_d3t_cold_*.tar.gz` | D3T cold-start ablation |
| `lora_counting_sft_d3t_warm_*.tar.gz` | D3T warm-start ablation (no improvement vs Variant B) |
| `lora_counting_sft_combined_*.tar.gz` | FSC + crowd-crops combined data |
| `lora_counting_sft_phase1_balanced_*.tar.gz` | Bucket-balanced phase 1 |
| `lora_counting_sft_synthetic_{1,3}ep_*.tar.gz` | Synthetic dot pretrain |
| `lora_counting_sft_attn_reg_*.tar.gz` | Attention regularization variants |
| `lora_counting_sft_variantB_crco_*.tar.gz` | Variant B + CRCO ranking aux loss |
| `lora_counting_sft_variantB_effbatch16_*.tar.gz` | Variant B with effective batch 16 |
| `lora_counting_sft_variantB_zero2_20260430_16{2849,3257,3838}.tar.gz` | Variant B retries (no/empty adapter) |
| `lora_local_{adapter,dense}_*.tar.gz` | Local-attention adapter ablations |

**Shipped run NOT archived — stays in `unicount_runs/`.**

Pre-LoRA experiments (mse_*, jigsaw_*, rl_*, hybrid_*, aggressive_*, etc.) remain in
`/data/amondal/unicount_runs/` and were not part of this cleanup pass.

### Eval JSONs (`archive/experiments/evals/`)

All non-shipped eval JSONs from `outputs/experiment_lora_counting_sft/eval/`.
Includes: per-step sweeps (`*_step{N}.json`), ablation variants (`*_attn_reg`, `*_combined`,
`*_crco`, `*_dual`, `*_d3t_*`, `*_synthetic_*`, `*_leaf_*`, `*_effbatch16_*`),
and the val ms={50,150} sweep results. Includes `val_mae*.json`, `test_mae.json`,
`*_scheme_sweep.txt`, `split_distribution.txt`.

### Logs (`archive/experiments/logs/`)

All training/eval logs that were in `logs/`. The `logs/` directory was emptied
for future runs.

### Data (`archive/experiments/data/`)

| File | Notes |
|---|---|
| `fsc147_d3t.json`, `fsc147_sft_plus_d3t.json` | D3T augmentation data |
| `synthetic_dots_train.json` | Synthetic counting prompts |
| `synthetic_dots_images.tar` | Synthetic images (uncompressed tar; jpegs already compact) |
| `combined_counting_train.json`, `fsc_plus_crops_train.json` | Combined-data variants |
| `fsc147_plus_crowd_crops.json`, `fsc147_crop_augmented.json` | Crowd-crop augmentations |
| `dense_crowd_crops.json`, `jhu_crowd_train.json`, `ucf_qnrf_train.json` | Crowd-counting auxiliary data |
| `crops.tar`, `crops_dense.tar` | Crowd-crop image dirs |
| `crco_data/` | CRCO ranking dataset (`fsc147_crco_ranking.json` + audit) |

### Scripts (`archive/experiments/scripts/`)

Copies (not moves) of experiment-specific launchers and data-generators.
Original scripts remain in their working locations for reproducibility.

---

## DO NOT touch

- `unicount_runs/lora_counting_sft_variantB_zero2_20260430_163831/` — shipped checkpoint
- `unicount_runs/SHIPPED_ADAPTER` — symlink to above
- `outputs/experiment_lora_counting_sft/eval/{val,test}_recursive_T100_d3_avg.json`
- `outputs/experiment_lora_counting_sft/eval/cross_dataset/`
- `outputs/experiment_lora_counting_sft/{train,val,test}/`
- `data/cross_dataset/`
- `/data/amondal/UniCountData/ucount_consolidated/` (clean dataset + audit)
- `/data/amondal/{FSC147_hf,ShanghaiTech,datasets,model_cache}/` (sources)
- `crco/` (implementation; `crco/data` was archived but code remains)
- `archive/` (reproducibility record)
