#!/usr/bin/env python3
"""
Linear probe gate for the regression-head experiment.

Extracts last-layer hidden state at the last prompt token for ~N val
images using UniLIP-3B + Variant B LoRA (frozen), fits a sklearn
LinearRegression to predict GT count, and reports R^2 + MAE.

Decision:
  R^2 > 0.7  -> proceed with full feature extraction + MLP
  0.3 < R^2 <= 0.7 -> try last image-token / mean image-token before aborting
  R^2 <= 0.3 -> abort, the visual encoder is the bottleneck

Single-GPU, no distributed. ~5 min wall time on 1xH100 for N=100.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)
from scripts.experiment_lora_counting_sft.eval_ctap_nrt_fsc147 import (
    load_model_and_tokenizer,
    build_input_ids,
)

DEFAULT_BASE_MODEL = "/data/amondal/model_cache/UniLIP-3B"
DEFAULT_MLLM_HF = (
    "/data/amondal/UniCount/.hf_cache/hub/"
    "models--OpenGVLab--InternVL3-2B-hf/snapshots/"
    "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
)
DEFAULT_ADAPTER = (
    "/data/amondal/unicount_runs/"
    "lora_counting_sft_variantB_zero2_20260430_163831/adapter"
)
DEFAULT_VAL_JSON = "outputs/experiment_lora_counting_sft/val/val_counting.json"

S = 448  # tile size — same as eval


IMG_TOKEN_ID = 151667  # UND_IMAGE_TOKEN_IDX from unilip.constants


@torch.no_grad()
def extract_one(model, tokenizer, img_proc, device, pil, system, human,
                hidden_capture):
    """Run a single forward pass and return dict of pooled hidden states."""
    pixel_values = img_proc.preprocess(
        [pil.convert("RGB")], return_tensors="pt"
    )["pixel_values"].to(device, dtype=torch.float16)

    input_ids = build_input_ids(system, human, tokenizer)
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(ids_t)

    hidden_capture.clear()
    _ = model(
        input_ids=ids_t,
        attention_mask=attn_mask,
        pixel_values=pixel_values,
        use_cache=False,
    )
    h_seq = hidden_capture["last"]  # (1, T, D)
    seq = h_seq[0].detach().to(torch.float32)  # (T, D)

    last_pos = int(attn_mask.sum(dim=1).item() - 1)
    h_last_prompt = seq[last_pos].cpu().numpy()

    img_mask = (ids_t[0] == IMG_TOKEN_ID)
    if img_mask.any():
        img_idx = img_mask.nonzero(as_tuple=False).flatten()
        h_last_img = seq[int(img_idx[-1].item())].cpu().numpy()
        h_mean_img = seq[img_idx].mean(dim=0).cpu().numpy()
    else:
        h_last_img = h_last_prompt
        h_mean_img = h_last_prompt
    return {"last_prompt": h_last_prompt,
            "last_img": h_last_img,
            "mean_img": h_mean_img}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--val_json", default=DEFAULT_VAL_JSON)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--mllm_hf", default=DEFAULT_MLLM_HF)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    print(f"[probe] loading model + adapter (Variant B)…")
    t0 = time.time()
    model, tokenizer, img_proc = load_model_and_tokenizer(
        args.base_model, args.adapter, args.mllm_hf, device,
        local_adapter=None,
    )
    print(f"[probe] loaded in {time.time()-t0:.1f}s")

    # Register forward hook on last LLM layer
    hidden_capture = {}

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_capture["last"] = h

    last_layer = model.get_model().language_model.model.layers[-1]
    handle = last_layer.register_forward_hook(hook_fn)
    print(f"[probe] hook registered on {type(last_layer).__name__}")

    # Sample N val items
    with open(args.val_json) as fh:
        val = json.load(fh)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val), size=min(args.n, len(val)), replace=False)
    items = [val[int(i)] for i in idx]
    print(f"[probe] extracting features for {len(items)} val items…")

    feats = {"last_prompt": [], "last_img": [], "mean_img": []}
    counts = []
    t0 = time.time()
    for i, it in enumerate(items):
        try:
            pil = Image.open(it["image"]).convert("RGB")
            pil = pil.resize((S, S), Image.BICUBIC)
        except Exception as e:
            print(f"  [skip] {it['image']}: {e}")
            continue
        system = it["conversations"][0]["value"]
        human = it["conversations"][1]["value"]
        gt_str = it["conversations"][-1]["value"]
        try:
            gt = float(gt_str)
        except ValueError:
            continue
        try:
            hd = extract_one(model, tokenizer, img_proc, device, pil,
                             system, human, hidden_capture)
        except Exception as e:
            print(f"  [err] {it['image']}: {e}")
            continue
        for k, v in hd.items():
            feats[k].append(v)
        counts.append(gt)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(items)}] {(time.time()-t0)/(i+1)*1000:.0f} ms/img", flush=True)

    handle.remove()
    y = np.array(counts, dtype=np.float32)
    print(f"[probe] N={len(y)}  count range [{y.min():.0f},{y.max():.0f}]  mean={y.mean():.1f}")

    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import KFold

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    def cv(X, y_target, recover=lambda p: p):
        r2s, maes = [], []
        for tr, te in kf.split(X):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("ri", RidgeCV(alphas=alphas, cv=3))])
            pipe.fit(X[tr], y_target[tr])
            p = recover(pipe.predict(X[te]))
            r2s.append(r2_score(y[te], p))
            maes.append(mean_absolute_error(y[te], p))
        return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes)), float(np.std(maes))

    print("\n[5-fold RidgeCV with StandardScaler]")
    print(f"  {'pool':<14s} {'target':<7s} {'R^2 mean':>10s} {'R^2 std':>9s} "
          f"{'MAE':>8s} {'MAE std':>8s}")
    best_r2 = -np.inf
    best_cfg = None
    for pool in ["last_prompt", "last_img", "mean_img"]:
        X = np.stack(feats[pool])
        for tgt_name, y_t, recov in [
            ("count",  y,            lambda p: p.clip(min=0)),
            ("log1p",  np.log1p(y),  lambda p: np.expm1(p).clip(min=0)),
        ]:
            r2_m, r2_s, mae_m, mae_s = cv(X, y_t, recov)
            print(f"  {pool:<14s} {tgt_name:<7s} {r2_m:>+10.3f} {r2_s:>9.3f} "
                  f"{mae_m:>8.2f} {mae_s:>8.2f}")
            if r2_m > best_r2:
                best_r2 = r2_m
                best_cfg = (pool, tgt_name)

    print(f"\n[decision] best CV R^2 = {best_r2:+.3f}  ({best_cfg})")
    if best_r2 > 0.7:
        print("  ==> GO. Hidden state encodes count. Proceed to full extraction.")
    elif best_r2 > 0.3:
        print("  ==> WEAK SIGNAL. MLP might recover more, but risky.")
    else:
        print("  ==> NO-GO. Visual encoder likely the bottleneck, not the LM head.")


if __name__ == "__main__":
    main()
