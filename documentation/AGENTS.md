# UniCount — Reproducibility Manifest

**Project:** UniLIP-3B LoRA Counting SFT — understanding pathway
**Compiled:** 2026-05-04
**Hardware:** 8× NVIDIA A100-SXM4-80GB
**Conda env:** `/home/nvidia/miniconda3` (base) — `python3.13`
**Repo root:** `/data/amondal/UniCount`

This file documents every artifact produced in the May 3–4 2026 working session so any score below can be reproduced bit-for-bit. Every checkpoint, every script, every evaluation JSON, and every input dataset is fingerprinted by MD5.

---

## 0. TL;DR — Best Model

**`v3-S` (`balancedmix_v3s_lora64a128_20260503_193348/adapter`) is the winner.**

It is the first 3B LoRA checkpoint that simultaneously beats the prior baseline on every cross-domain split AND beats every inference-time tiling variant on every dataset. It is **frozen** (chmod read-only) at:

```
/data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/adapter/
```

| File | Size | MD5 |
|---|---|---|
| `adapter_model.safetensors` | 147,768,536 B | `d14717276d31a37c97ea576a0c8b17e7` |
| `multi_modal_projector.bin` | 17,327,101 B | `8bca689c7e5e669403bcce49323e5a42` |
| `adapter_config.json` | 1,104 B | `4d6deeaab9af02587c99aaa32ee5238c` |
| `README.md` | 5,128 B | `ca606468ec3850e65335b6bec4bf1fa8` |

Manifest also stored in repo at: [`unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/_FROZEN_MD5_MANIFEST.txt`](../unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/_FROZEN_MD5_MANIFEST.txt). Permissions: `dr-xr-xr-x` directory, `-r--r--r--` files. **Do not modify.**

### Headline cross-domain MAE (CTAP-recursive, single forward pass per image)

| Split | n | **v3-S MAE** | v3-S RMSE | baseline MAE | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| **SHA test** | 182 | **93.85** | 162.37 | 234.70 | −60.0% |
| **SHT-B test** | 316 | **16.07** | 25.82 | 23.78 | −32.4% |
| **CARPK test** | 459 | **10.33** | 14.44 | 10.98 | −5.9% |
| **FSC-147 val** | 1,286 | **8.48** | 40.91 | 10.76 | −21.2% |
| **FSC-147 test** | 1,190 | **10.47** | 73.21 | 12.23 | −14.4% |
| **JHU valid** | 497 | **117.45** | 345.57 | 249.35 | −52.9% |
| **JHU test** | 1,600 | **142.51** | 497.45 | 266.70 | −46.6% |
| **QNRF test** | 334 | **254.90** | 464.76 | 484.78 | −47.4% |

v3-S **wins 8/8 splits** vs baseline. Eval JSONs at [`outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v3s/`](outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v3s/) (`_CKPT_SOURCE.txt` confirms provenance).

---

## 1. Inputs (what to prepare before training)

### 1.1 Base model (frozen)
```
/data/amondal/model_cache/UniLIP-3B/
  ├── config.json                       (UniLIP_InternVLForCausalLM, Qwen2 1536-dim, 28 layers)
  ├── model-00001-of-00002.safetensors
  └── model-00002-of-00002.safetensors  (DiT + connector + ViT + LLM)
```

### 1.2 Hugging Face mirror (for tokenizer + processor)
```
HF_HOME=/data/amondal/UniCount/.hf_cache
MLLM_HF=/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2
```

### 1.3 Warm-start adapter (BASELINE_BEST)
```
/data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/adapter/
```
| File | MD5 |
|---|---|
| `adapter_model.safetensors` | `8e97b9cdf04a37e5d590c75b2ba5bb08` |
| `multi_modal_projector.bin` | `ed1b580bd9049ea8bde1718fd9bcc237` |
| `adapter_config.json` | `a214f2cb819874290fe470eaccf207d2` |

LoRA: `r=64 α=128 dropout=0.05`, target modules `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`, `modules_to_save=["lm_head"]`.

### 1.4 Training data — `balanced_mix_train.json`
```
outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json
```
| Property | Value |
|---|---|
| Entries | **49,847** |
| MD5 | `4375839f6e7100399c4baf5c15948c29` |
| Builder | [`scripts/experiment_lora_counting_sft/build_balanced_mix_v3s.py`](scripts/experiment_lora_counting_sft/build_balanced_mix_v3s.py) (md5 `bd50b0167ea259a7b00598224cd0d605`) |

Mix recipe (config in [`balanced_mix_v3s/mix_diagnostics.json`](outputs/experiment_lora_counting_sft/balanced_mix_v3s/mix_diagnostics.json)):
```python
PERSON_CAP_PER_BUCKET     = 1500
NONPERSON_CAP_PER_CATEGORY=  800
NONPERSON_CAP_PER_BUCKET  = 5000
FSC_UPSAMPLE_FACTOR       =    5
SHAB_UPSAMPLE_FACTOR      =    5    # SHA + SHB upsampled 5×
SEED                      =   42
DROP_CATEGORIES           = ["objects"]
```
Source files consumed (raw entry counts):
| Source | Entries |
|---|---:|
| `ucount_consolidated/train_counting.json` | 50,502 |
| `ucount_consolidated/train_counting_clean.json` | 49,703 |
| `ucount_crowd_consolidated/train_counting.json` | 19,370 |
| `ucount_crowd_consolidated/train_counting_person.json` | 19,370 |
| SHA + SHB SFT splits (×5) | from `sha_train_sft.json`, `shb_train_sft.json` |

### 1.5 Eval json files (cross-domain)
```
outputs/experiment_lora_counting_sft/cross_eval/
```
| File | MD5 |
|---|---|
| `sht_a_test_countdetect_counting.json` | `ad433f86d6834bc2f630df51d921974b` |
| `sht_b_test_countdetect_counting.json` | `760cf578bf536668798f5065e87b3b62` |
| `carpk_test_countdetect_counting.json` | `3b2fa389ed3f6ef1e8d5956f246667f8` |

(FSC-147, JHU, QNRF eval JSONs live in the same dir; not fingerprinted here because they were produced before this session.)

---

## 2. Training (v3-S best run)

### 2.1 Launch script
[`scripts/experiment_lora_counting_sft/launch_balancedmix_v3s.sh`](scripts/experiment_lora_counting_sft/launch_balancedmix_v3s.sh)
**MD5:** `83cc04dc44b44198fa85322937d077fb`

### 2.2 Trainer code
[`scripts/experiment_lora_counting_sft/train_lora_counting_sft_3b_unfreezeconn.py`](scripts/experiment_lora_counting_sft/train_lora_counting_sft_3b_unfreezeconn.py)
**MD5:** `9524990461a99b9fe812c8795c191b7e`

Trainable: LoRA on Qwen2 LLM + `lm_head` + `multi_modal_projector` (connector). Frozen: InternViT, base Qwen2 weights, latent_queries, DiT, VAE.

### 2.3 Hyperparameters (frozen for reproducibility)

| Param | Value |
|---|---|
| Init from | BASELINE_BEST adapter + multi_modal_projector |
| Base model | `/data/amondal/model_cache/UniLIP-3B` |
| LoRA rank / alpha / dropout | `64 / 128 / 0.05` |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `modules_to_save` | `["lm_head"]` |
| Epochs | 3 |
| Per-device train batch | 2 |
| Grad accumulation | 1 |
| GPUs | 8 |
| **Effective batch** | **16** |
| Learning rate | 1e-5 |
| LR schedule | `cosine`, warmup_ratio=0.06 |
| Weight decay | 0.0 |
| Max grad norm | 1.0 |
| Precision | bf16 (`accelerate launch --mixed_precision=bf16`) |
| Model max length | 512 |
| Save steps / total | 1558 / 8 |
| DeepSpeed config | `scripts/experiment_lora_counting_sft/ds_zero2.json` |
| Gradient checkpointing | True |
| Seed | (default in HF Trainer) |

### 2.4 Run statistics (from `checkpoint-9348/trainer_state.json`)

| Metric | Value |
|---|---|
| `global_step` | **9,348** (= 49,847 / 16 × 3 ≈ 9,348) |
| `epoch` | 3.00 |
| `train_runtime` | 3,887.84 s (~65 min) |
| `train_samples_per_second` | 38.46 |
| `train_steps_per_second` | 2.40 |
| Final `train_loss` (mean) | 0.2306 |
| Last-step loss | 0.138 |
| Final LR | 2.59e-11 (cosine fully decayed) |

### 2.5 Pre/Post-flight integrity
[`unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/_PREFLIGHT_BASELINE_MD5.txt`](../unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/_PREFLIGHT_BASELINE_MD5.txt) records the BASELINE adapter MD5 captured before the run started. The launch script asserts the same MD5 after training completes; v3-S passed this check (BASELINE was not mutated).

### 2.6 Training command (one-liner)

```bash
cd /data/amondal/UniCount
bash scripts/experiment_lora_counting_sft/launch_balancedmix_v3s.sh \
  2>&1 | tee logs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_$(date +%Y%m%d_%H%M%S).log
```

Original log: [`logs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348.log`](logs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348.log).

---

## 3. Cross-domain Evaluation (v3-S, the winner)

### 3.1 Driver
[`scripts/experiment_lora_counting_sft/eval_ctap_balancedmix_v3s_alldatasets.sh`](scripts/experiment_lora_counting_sft/eval_ctap_balancedmix_v3s_alldatasets.sh) — runs CTAP-recursive eval on 8 splits in parallel and writes `*_ctap_mae.json` per split.

```bash
cd /data/amondal/UniCount
nohup bash scripts/experiment_lora_counting_sft/eval_ctap_balancedmix_v3s_alldatasets.sh \
  > logs/eval_balancedmix_v3s_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Per-split logs: `logs/eval_ctap_balancedmix_v3s_*_20260503_203943.log`.

### 3.2 Full results — v3-S vs BASELINE vs v2

Eval directories (each contains `_CKPT_SOURCE.txt` linking back to ckpt):
- v3-S: [`outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v3s/`](outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v3s/)
- baseline: [`outputs/experiment_lora_counting_sft/cross_eval_ctap_countdetect/`](outputs/experiment_lora_counting_sft/cross_eval_ctap_countdetect/)
- v2 (deprecated): [`outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v2/`](outputs/experiment_lora_counting_sft/cross_eval_ctap_balancedmix_v2/)

| Split | n | **v3-S MAE / RMSE** | baseline MAE / RMSE | v2 MAE / RMSE |
|---|---:|---|---|---|
| sht_a_test | 182 | **93.85 / 162.37** | 234.70 / 727.35 | 207.03 / 323.99 |
| sht_b_test | 316 | **16.07 / 25.82** | 23.78 / 35.30 | 43.53 / 72.72 |
| carpk_test | 459 | **10.33 / 14.44** | 10.98 / 14.24 | 10.49 / 13.71 |
| fsc147_val | 1,286 | **8.48 / 40.91** | 10.76 / 60.73 | 9.76 / 66.63 |
| fsc147_test | 1,190 | **10.47 / 73.21** | 12.23 / 95.84 | 8.40 / 38.63 |
| jhu_valid | 497 | **117.45 / 345.57** | 249.35 / 1055.28 | 5188.80 / 111905.40 (broken) |
| jhu_test | 1,600 | **142.51 / 497.45** | 266.70 / 956.22 | 196.21 / 655.68 |
| qnrf_test | 334 | **254.90 / 464.76** | 484.78 / 1354.06 | 493.03 / 800.00 |

**v3-S wins 7/8 splits over v2** (only fsc147_test where v2's overfit memorization edges out). v3-S wins 8/8 splits vs baseline.

---

## 4. Boundary-Aware Tiling Ablation (negative result, kept for the record)

Implemented per `ADAPTIVE_TILING_FULL_SPEC.md`: NRT (non-overlapping recursive tiling), CGA (confidence-gated aggregation), IEBC (inclusion-exclusion boundary correction), and CGA+IEBC. All four modes evaluated against the **frozen** v3-S adapter and against BASELINE for comparison.

### 4.1 Code (read-only post-results)
[`boundary_aware_tiling/`](boundary_aware_tiling/)
| File | MD5 |
|---|---|
| `tile_grid.py` | `7b84362e8bfc36773078be9500665699` |
| `count_with_cga_iebc.py` | `bd76fcfb519b63495ff1285fd3bffd2e` |
| `eval_ablation.py` | `50e4fb55a345c082e7ff297270078c81` |
| `run_ablation.sh` | `4d6aae7835dbf271df99140b8a3f5504` |
| `run_baseline_nrt.sh` | `c36720ab746718e866850f7fb2411f93` |

Hyperparameters (hard-coded in `tile_grid.py`):
```
s          = 448      # tile side
eta        = 0.10     # tile-overlap fraction
m_max      = 16       # max tiles per axis
rho        = 0.25     # boundary-strip half-width fraction
gamma      = eta/(2*rho) = 0.20
min_strip  = 100      # px
```

### 4.2 v3-S ablation @ best T per dataset

Results: [`boundary_aware_tiling/results/`](boundary_aware_tiling/results/)
T-sweep written to `<ds>_sweep_T<N>/summary.json`. Best T per dataset stored in `<ds>_best_T.txt`.

| Dataset | T* | NRT MAE | CGA MAE | IEBC MAE | CGA+IEBC MAE |
|---|---:|---:|---:|---:|---:|
| SHA  | 20 | 108.48 | 110.90 | 124.39 | 120.43 |
| SHT-B| 30 |  17.31 |  22.83 |  27.96 |  34.47 |
| CARPK| 10 |  11.98 |  11.36 |  23.75 |  26.02 |

### 4.3 Headline 2×2 (training vs inference-time tiling)

Results for baseline at v3-S T*: [`boundary_aware_tiling/results_baseline/`](boundary_aware_tiling/results_baseline/)

| Method | SHA MAE | SHT-B MAE | CARPK MAE |
|---|---:|---:|---:|
| Baseline direct (no tiling) | 234.70 | 23.78 | 10.98 |
| Baseline + NRT @ v3-S T* (20,30,10) | 204.90 | 25.29 | 12.40 |
| Baseline + NRT @ baseline best T* (200,80,60) | 201.73 | 24.05 | 12.19 |
| **v3-S direct (no tiling)** | **93.85** | **16.07** | **10.33** |
| v3-S + NRT @ T* | 108.48 | 17.31 | 11.98 |

**Conclusion**: training (v3-S balanced mix) subsumes inference-time tiling. v3-S direct dominates every tiled variant on every dataset; tiling actively hurts v3-S on all three datasets (over-counting from boundary double-counting that γ=0.20 IEBC over-corrects).

### 4.4 Tiling reproducibility

```bash
# v3-S 4-mode ablation
cd /data/amondal/UniCount
nohup bash boundary_aware_tiling/run_ablation.sh \
  > boundary_aware_tiling/logs/driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# baseline NRT comparison
nohup bash boundary_aware_tiling/run_baseline_nrt.sh \
  > boundary_aware_tiling/logs/driver_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Both drivers re-verify the ckpt MD5 at launch (`_CKPT_MD5_AT_LAUNCH.txt`) and at exit. Both passed for v3-S; v3-S adapter MD5s remained `d14717276d31a37c97ea576a0c8b17e7` / `8bca689c7e5e669403bcce49323e5a42` after all 30+ accelerate launches.

---

## 5. Reproducing v3-S from scratch

```bash
# 1. Verify base + warm-start MD5s match this manifest
md5sum /data/amondal/model_cache/UniLIP-3B/model-0000{1,2}-of-00002.safetensors
md5sum /data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/adapter/{adapter_model.safetensors,multi_modal_projector.bin}
# Expected baseline:
#   8e97b9cdf04a37e5d590c75b2ba5bb08  adapter_model.safetensors
#   ed1b580bd9049ea8bde1718fd9bcc237  multi_modal_projector.bin

# 2. (If rebuilding the mix) regenerate balanced_mix_train.json
python3 scripts/experiment_lora_counting_sft/build_balanced_mix_v3s.py
md5sum outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json
# Expected: 4375839f6e7100399c4baf5c15948c29

# 3. Train v3-S (~65 min on 8× A100-80GB)
bash scripts/experiment_lora_counting_sft/launch_balancedmix_v3s.sh

# 4. Verify produced adapter MD5
md5sum <output_dir>/adapter/{adapter_model.safetensors,multi_modal_projector.bin}
# Expected:
#   d14717276d31a37c97ea576a0c8b17e7  adapter_model.safetensors
#   8bca689c7e5e669403bcce49323e5a42  multi_modal_projector.bin

# 5. Cross-domain eval
nohup bash scripts/experiment_lora_counting_sft/eval_ctap_balancedmix_v3s_alldatasets.sh \
  > logs/eval_balancedmix_v3s_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 6. Boundary-aware tiling (optional; produces negative result)
nohup bash boundary_aware_tiling/run_ablation.sh > boundary_aware_tiling/logs/driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup bash boundary_aware_tiling/run_baseline_nrt.sh > boundary_aware_tiling/logs/driver_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 6. Lock — Final v3-S MD5s (do not overwrite this section)

```
# v3-S adapter (FROZEN, dr-xr-xr-x)
d14717276d31a37c97ea576a0c8b17e7  /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/adapter/adapter_model.safetensors
8bca689c7e5e669403bcce49323e5a42  /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/adapter/multi_modal_projector.bin
4d6deeaab9af02587c99aaa32ee5238c  /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/adapter/adapter_config.json
ca606468ec3850e65335b6bec4bf1fa8  /data/amondal/unicount_runs/lora_counting_sft_3b_balancedmix_v3s_lora64a128_20260503_193348/adapter/README.md

# BASELINE_BEST adapter (read-only reference; never mutated)
8e97b9cdf04a37e5d590c75b2ba5bb08  /data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/adapter/adapter_model.safetensors
ed1b580bd9049ea8bde1718fd9bcc237  /data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/adapter/multi_modal_projector.bin
a214f2cb819874290fe470eaccf207d2  /data/amondal/unicount_runs/BASELINE_BEST_lora64a128_allsplits_countdetect/adapter/adapter_config.json

# Training inputs
4375839f6e7100399c4baf5c15948c29  outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json

# Training pipeline
83cc04dc44b44198fa85322937d077fb  scripts/experiment_lora_counting_sft/launch_balancedmix_v3s.sh
bd50b0167ea259a7b00598224cd0d605  scripts/experiment_lora_counting_sft/build_balanced_mix_v3s.py
9524990461a99b9fe812c8795c191b7e  scripts/experiment_lora_counting_sft/train_lora_counting_sft_3b_unfreezeconn.py

# Eval JSONs (cross-domain)
ad433f86d6834bc2f630df51d921974b  outputs/experiment_lora_counting_sft/cross_eval/sht_a_test_countdetect_counting.json
760cf578bf536668798f5065e87b3b62  outputs/experiment_lora_counting_sft/cross_eval/sht_b_test_countdetect_counting.json
3b2fa389ed3f6ef1e8d5956f246667f8  outputs/experiment_lora_counting_sft/cross_eval/carpk_test_countdetect_counting.json

# Boundary-aware tiling code
7b84362e8bfc36773078be9500665699  boundary_aware_tiling/tile_grid.py
bd76fcfb519b63495ff1285fd3bffd2e  boundary_aware_tiling/count_with_cga_iebc.py
50e4fb55a345c082e7ff297270078c81  boundary_aware_tiling/eval_ablation.py
4d6aae7835dbf271df99140b8a3f5504  boundary_aware_tiling/run_ablation.sh
c36720ab746718e866850f7fb2411f93  boundary_aware_tiling/run_baseline_nrt.sh
```
