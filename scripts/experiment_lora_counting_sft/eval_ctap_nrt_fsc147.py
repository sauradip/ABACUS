#!/usr/bin/env python3
"""
CTAP + Recursive evaluation on FSC-147 test/val split — multi-GPU.

Implements recursive density-adaptive counting:
  1. Run a global 448×448 forward pass → ĉ_g
  2. If ĉ_g ≤ T: return ĉ_g  (mode="global")
  3. Else: split into 4 non-overlapping quadrants, recurse on each,
     stop when leaf count ≤ T, max_depth reached, or region < min_size.
     Final count = sum over leaves.

Non-overlapping splits eliminate spatial double-counting entirely (objects
appear in exactly one quadrant). Boundary double-counting (objects straddling
the cut line) uses only 2 cut lines per level vs. ~12 seams in a 4×4
overlapping NRT grid — ~6× less error per recursion level.

  Counter model: LoRA Variant B (adapter grafted onto UniLIP-3B), same as
  eval_lora_counting_sft.py.

Multi-GPU strategy: data-parallel sharding via accelerate (identical to
eval_lora_counting_sft.py). Each GPU loads the full model and processes
val_data[rank::world_size]. Rank 0 gathers, aggregates, and writes the JSON.

Usage (8 GPUs):
    accelerate launch --num_processes=8 --mixed_precision=no \\
        scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py \\
        [--T 100] [--max_depth 3] [--min_size 224] \\
        [--val_json outputs/experiment_lora_counting_sft/test/test_counting.json] \\
        [--out_json outputs/experiment_lora_counting_sft/eval/test_recursive_T100.json]

Usage (single GPU / debug):
    python scripts/experiment_lora_counting_sft/eval_ctap_nrt_fsc147.py --T 100
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

# ── PEFT/transformers compat shim ────────────────────────────────────────────
# Newer peft (>=0.18) calls peft.utils.save_and_load._maybe_shard_state_dict_for_tp
# which imports ALL_PARALLEL_STYLES from transformers.integrations.tensor_parallel
# — that symbol is absent in transformers 4.52. We never use TP here, so neuter
# the call to a no-op before any PeftModel.from_pretrained() call.
try:
    import peft.utils.save_and_load as _peft_sal  # noqa: PLC0415
    _peft_sal._maybe_shard_state_dict_for_tp = lambda *a, **kw: None  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# ── Defaults ──────────────────────────────────────────────────────────────────
_RUN = (
    "/data/amondal/unicount_runs/"
    "lora_counting_sft_variantB_zero2_20260430_163831"
)
DEFAULT_BASE_MODEL   = "/data/amondal/model_cache/UniLIP-3B"
DEFAULT_MLLM_HF      = (
    "/data/amondal/UniCount/.hf_cache/hub/"
    "models--OpenGVLab--InternVL3-2B-hf/snapshots/"
    "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
)
DEFAULT_CHECKPOINT   = f"{_RUN}/adapter"
DEFAULT_VAL_JSON     = (
    "outputs/experiment_lora_counting_sft/test/test_counting.json"
)
DEFAULT_OUT_JSON     = (
    "outputs/experiment_lora_counting_sft/eval/test_recursive_T100.json"
)
DEFAULT_ANN_JSON     = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"
DEFAULT_T            = 100.0
DEFAULT_MAX_DEPTH    = 3
DEFAULT_MIN_SIZE     = 224
TILE_SIZE            = 448   # s — matches model input resolution

# ── Constants (must match training exactly) ───────────────────────────────────
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
NUM_IMG_TOKENS    = 256

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\\n' }}"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}{% endif %}"
    "{{'<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n' }}{% endif %}"
)

NUMERIC = re.compile(r"\d+")


# ═══════════════════════════════════════════════════════════════════════════════
# §4.1  tile_grid.py  (exact spec implementation — pure numpy, no torch)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tile:
    x0: int; y0: int; x1: int; y1: int   # inclusive-exclusive box
    weight: float                         # overlap-correction weight w_j


def build_grid(W: int, H: int, s: int = 448, eta: float = 0.10,
               m_max: int = 16) -> List[Tile]:
    """
    Returns a list of Tiles covering the WxH canvas with edge `s`,
    minimum overlap fraction `eta`, capped at `m_max` tiles.

    Strategy: compute (n_c, n_r) from Eq. (1); if n_c*n_r > m_max,
    increase effective eta until cap is satisfied (i.e. trade overlap
    for tile count). If the image is smaller than `s` on either axis,
    we still emit a single tile padded with reflection at read time.
    """
    if W <= s and H <= s:
        return [Tile(0, 0, W, H, weight=1.0)]

    while True:
        stride = max(1, int(math.floor(s * (1.0 - eta))))
        n_c = max(1, math.ceil(max(0, W - s) / stride) + 1)
        n_r = max(1, math.ceil(max(0, H - s) / stride) + 1)
        if n_c * n_r <= m_max or eta >= 0.49:
            break
        eta = min(0.49, eta + 0.05)  # relax overlap, retry

    # Enforce hard m_max cap: if eta relaxation didn't shrink enough
    # (e.g. very large images at eta=0.49 still exceed m_max), downsample
    # the canvas so the grid fits within budget.
    if n_c * n_r > m_max:
        scale = math.sqrt(m_max / (n_c * n_r))
        W = max(s, int(W * scale))
        H = max(s, int(H * scale))
        stride = max(1, int(math.floor(s * (1.0 - eta))))
        n_c = max(1, math.ceil(max(0, W - s) / stride) + 1)
        n_r = max(1, math.ceil(max(0, H - s) / stride) + 1)

    # Distribute tiles uniformly so right/bottom edges align with W/H
    xs = _uniform_starts(W, s, n_c)
    ys = _uniform_starts(H, s, n_r)
    tiles = [Tile(x, y, x + s, y + s, weight=1.0) for y in ys for x in xs]

    # Overlap-correction weights — Eq. (3)
    cov = np.zeros((H, W), dtype=np.int32)
    for t in tiles:
        cov[t.y0:t.y1, t.x0:t.x1] += 1
    inv = 1.0 / np.maximum(cov, 1)
    for t in tiles:
        area = (t.x1 - t.x0) * (t.y1 - t.y0)
        t.weight = float(inv[t.y0:t.y1, t.x0:t.x1].sum() / area)
    return tiles


def _uniform_starts(L: int, s: int, n: int) -> List[int]:
    """Place n tiles of size s along an axis of length L, edges aligned."""
    if n == 1:
        return [max(0, (L - s) // 2)]
    if L <= s:
        return [0] * n
    span = L - s
    return [int(round(i * span / (n - 1))) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(base_model: str, checkpoint_dir: str,
                              mllm_hf: str, device: torch.device,
                              local_adapter: Optional[str] = None,
                              connector_weights: Optional[str] = None):
    """Load UniLIP-3B + LoRA adapter(s).

    If ``local_adapter`` is provided, two adapters are loaded:
      - ``checkpoint_dir`` -> adapter name 'global' (used at CARC depth 0)
      - ``local_adapter``  -> adapter name 'local'  (used at depth >= 1)
    Otherwise, a single unnamed adapter is loaded (back-compat).

    If ``connector_weights`` is provided, the trained multi_modal_projector
    state dict (multi_modal_projector.bin saved by the unfreeze-connector
    training script) is loaded into ``model.get_model().multi_modal_projector``.
    """
    apply_transformers_compat_shims()
    model_cls = load_unilip_class()

    # Full Trainer checkpoint (has config.json + model.safetensors): load directly.
    # PEFT adapter dir (has adapter_config.json): load base then apply adapter.
    ckpt_path = Path(checkpoint_dir)
    _is_full_ckpt = (ckpt_path / "config.json").exists() and not (ckpt_path / "adapter_config.json").exists()

    if _is_full_ckpt:
        print(f"[CTAP+Rec] loading full checkpoint from {checkpoint_dir}")
        model = model_cls.from_pretrained(
            checkpoint_dir,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad = False
    else:
        model = model_cls.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad = False

        from peft import PeftModel
        llm = model.get_model().language_model
        if local_adapter:
            llm = PeftModel.from_pretrained(
                llm, checkpoint_dir, adapter_name="global", is_trainable=False
            )
            llm.load_adapter(local_adapter, adapter_name="local")
            llm.set_adapter("global")
            print(f"[dual-adapter] loaded adapters: {list(llm.peft_config.keys())}  active='{llm.active_adapter}'")
        else:
            llm = PeftModel.from_pretrained(llm, checkpoint_dir, is_trainable=False)
        model.get_model().language_model = llm

    if connector_weights:
        inner = model.get_model()
        if not hasattr(inner, "multi_modal_projector"):
            raise RuntimeError("Model has no multi_modal_projector to load into.")
        sd = torch.load(connector_weights, map_location="cpu")
        missing, unexpected = inner.multi_modal_projector.load_state_dict(sd, strict=False)
        print(f"[CTAP+Rec] connector load: missing={len(missing)} unexpected={len(unexpected)} from {connector_weights}")

    model = model.to(device).eval()

    processor     = AutoProcessor.from_pretrained(mllm_hf, trust_remote_code=True)
    tokenizer     = processor.tokenizer
    img_processor = processor.image_processor
    return model, tokenizer, img_processor


def _set_carc_adapter(model, name: str) -> None:
    """Switch the active LoRA adapter on the language-model PEFT wrapper.

    No-op if the underlying LLM is not a multi-adapter PeftModel (single-adapter
    eval, or no adapter at all).
    """
    llm = model.get_model().language_model
    if hasattr(llm, "peft_config") and len(llm.peft_config) > 1:
        if llm.active_adapter != name:
            llm.set_adapter(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt / tokenisation
# ═══════════════════════════════════════════════════════════════════════════════

def build_input_ids(system: str, human: str, tokenizer) -> list:
    img_placeholder = (
        f"{IMG_START_TOKEN}"
        f"{IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS}"
        f"{IMG_END_TOKEN}"
    )
    human_filled = human.replace("<image>", img_placeholder).strip()
    tok = copy.deepcopy(tokenizer)
    tok.chat_template = CHAT_TEMPLATE
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": human_filled},
        ],
        add_generation_prompt=True,
        return_dict=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §4.2  Single-image forward pass
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def count_one_pil(model, tokenizer, img_proc, device,
                  pil: Image.Image, system: str, human: str) -> Optional[int]:
    """
    Run the model on a single PIL image (must already be 448×448 or will be
    resized by img_proc).  Returns parsed integer count, or None on parse fail.
    """
    pixel_values = img_proc.preprocess(
        [pil.convert("RGB")], return_tensors="pt"
    )["pixel_values"].to(device, dtype=torch.float16)

    input_ids = build_input_ids(system, human, tokenizer)
    ids_t     = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(ids_t)

    out = model.generate(
        input_ids      = ids_t,
        attention_mask = attn_mask,
        pixel_values   = pixel_values,
        do_sample      = False,
        num_beams      = 1,
        max_new_tokens = 8,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    # generate() returns only new tokens (inputs_embeds path)
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    m = NUMERIC.search(text)
    return int(m.group()) if m else None


# ═══════════════════════════════════════════════════════════════════════════════
# §2.4  CTAP + NRT combined pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def count_image_ctap_nrt(model, tokenizer, img_proc, device,
                          pil: Image.Image, system: str, human: str,
                          T: float = 100.0, s: int = 448,
                          eta: float = 0.10, m_max: int = 16,
                          native_wh: Optional[tuple] = None) -> dict:
    """
    §2.4 combined pipeline:
      1. Resize image to s×s → global forward → ĉ_g
      2. if ĉ_g ≤ T: return ĉ_g
      3. else: upscale to native resolution → NRT tiles → Σ w_j · ĉ_j

    native_wh: (W_native, H_native) from annotation_FSC147_384.json.  When
    provided and larger than pil.size, the image is bicubic-upscaled to the
    native canvas before build_grid() so that tile counts match the spec §4.1
    intent ("NRT operates on the native resolution of the image").

    Returns a dict with count, mode, global_count, n_tiles, tile_counts, weights.
    """
    # ── 1. Global pass ─────────────────────────────────────────────────────
    # Always use the stored (384-short-side) image resized to s×s — this
    # matches the training distribution exactly.
    global_pil = pil.resize((s, s), Image.BICUBIC)
    _c_g_raw = count_one_pil(model, tokenizer, img_proc, device, global_pil, system, human)
    global_parse_ok = _c_g_raw is not None
    c_g = _c_g_raw if global_parse_ok else 0  # spec §3: parse failure → 0

    # ── 2. Routing ──────────────────────────────────────────────────────────
    if c_g <= T:
        return dict(count=c_g, mode="global", global_count=c_g,
                    global_parse_ok=global_parse_ok,
                    n_tiles=0, tile_counts=[], weights=[],
                    parse_failed_tiles=0)

    # ── 3. NRT pass ─────────────────────────────────────────────────────────
    # Upscale to native resolution when annotation provides it and the stored
    # image is smaller.  This ensures build_grid() sees the true aspect ratio
    # and emits the spec-correct tile count (up to m_max=16).
    if native_wh is not None:
        W_nat, H_nat = native_wh
        W_cur, H_cur = pil.size
        if W_nat > W_cur or H_nat > H_cur:
            tile_pil = pil.resize((W_nat, H_nat), Image.BICUBIC)
        else:
            tile_pil = pil
    else:
        tile_pil = pil
    W, H = tile_pil.size

    tiles = build_grid(W, H, s=s, eta=eta, m_max=m_max)
    sub_counts: list = []
    weights:    list = []
    n_failed_tiles: int = 0

    for t in tiles:
        crop = tile_pil.crop((t.x0, t.y0, t.x1, t.y1))
        if crop.size != (s, s):
            # Edge tile: bicubic-resize to model input size (spec §4.1)
            crop = crop.resize((s, s), Image.BICUBIC)
        cj = count_one_pil(model, tokenizer, img_proc, device, crop, system, human)
        if cj is None:
            # spec §3: fallback to 0 (not c_g/M) — saturated c_g is unreliable
            cj = 0
            n_failed_tiles += 1
        sub_counts.append(cj)
        weights.append(t.weight)

    c_nrt = sum(w * c for w, c in zip(weights, sub_counts))
    return dict(count=int(round(c_nrt)),
                mode="tiled",
                global_count=c_g,
                global_parse_ok=global_parse_ok,
                n_tiles=len(tiles),
                tile_counts=sub_counts,
                weights=weights,
                parse_failed_tiles=n_failed_tiles)


# ═══════════════════════════════════════════════════════════════════════════════
# §4.3  Recursive density-adaptive counting (non-overlapping quadrant splits)
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_no_upscale(pil_img: Image.Image, target: int = 448,
                    fill=(128, 128, 128)) -> Image.Image:
    """Place pil_img onto a target×target canvas without upscaling.

    If both dims >= target → standard bicubic resize to target×target.
    Else → preserve aspect ratio (downscale only if needed), center on canvas.
    """
    W, H = pil_img.size
    if min(W, H) >= target:
        return pil_img.resize((target, target), Image.BICUBIC)
    scale = min(target / W, target / H, 1.0)  # never upscale
    if scale < 1.0:
        new_w, new_h = int(W * scale), int(H * scale)
        img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    else:
        img = pil_img
        new_w, new_h = W, H
    canvas = Image.new('RGB', (target, target), fill)
    canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))
    return canvas


def _leaf_resize(pil_img: Image.Image, s: int, mode: str) -> Image.Image:
    """Resize a recursion-leaf crop using the configured mode."""
    if mode == "pad":
        return _pad_no_upscale(pil_img, s, fill=(128, 128, 128))
    if mode == "pad_white":
        return _pad_no_upscale(pil_img, s, fill=(255, 255, 255))
    # "resize" or unknown → baseline behaviour
    return pil_img.resize((s, s), Image.BICUBIC)


def count_image_recursive(model, tokenizer, img_proc, device,
                          pil: Image.Image, system: str, human: str,
                          T: float = 100.0, max_depth: int = 3,
                          current_depth: int = 0, min_size: int = 224,
                          s: int = TILE_SIZE,
                          leaf_resize_mode: str = "resize") -> dict:
    """
    Recursive density-adaptive counting with non-overlapping quadrant splits.

    At each node:
      1. Resize current region to s×s → model forward → local count c
      2. If c ≤ T, depth limit, or region < min_size → leaf, return c
      3. Otherwise: split into 4 non-overlapping quadrants, recurse, sum leaves

    Non-overlapping splits mean each object appears in exactly one quadrant.
    Boundary double-counting (objects straddling the cut line) scales with
    2 cut lines per level, vs. ~12 seams in a 4×4 overlapping NRT grid.
    """
    W, H = pil.size

    # Forward pass at this node.
    # Depth-0 always resizes to s×s (global pass).  At depth>=1, honour leaf_resize_mode.
    if current_depth == 0 or leaf_resize_mode == "resize":
        resized = pil.resize((s, s), Image.BICUBIC)
    else:
        resized = _leaf_resize(pil, s, leaf_resize_mode)
    c = count_one_pil(model, tokenizer, img_proc, device, resized, system, human)
    if c is None:
        c = 0

    # Leaf condition: sparse enough, recursion limit, or region too small
    if c <= T or current_depth >= max_depth or W < min_size or H < min_size:
        return {
            "count": int(c),
            "depth": current_depth,
            "leaves": 1,
            "global_at_node": int(c),
            "children_sum": 0,
        }

    # Split into 4 non-overlapping quadrants
    mx, my = W // 2, H // 2
    quads = [
        pil.crop((0,  0,  mx, my)),   # top-left
        pil.crop((mx, 0,  W,  my)),   # top-right
        pil.crop((0,  my, mx, H )),   # bottom-left
        pil.crop((mx, my, W,  H )),   # bottom-right
    ]

    total = 0
    total_leaves = 0
    max_leaf_depth = current_depth
    for q in quads:
        r = count_image_recursive(
            model, tokenizer, img_proc, device, q,
            system, human, T, max_depth, current_depth + 1, min_size, s,
            leaf_resize_mode=leaf_resize_mode,
        )
        total += r["count"]
        total_leaves += r["leaves"]
        max_leaf_depth = max(max_leaf_depth, r["depth"])
    # Global-local averaging (GLCE-style): combine node's global count with children sum
    return {
        "count": int(round(0.5 * (c + total))),
        "depth": max_leaf_depth,
        "leaves": total_leaves,
        "global_at_node": int(c),
        "children_sum": int(total),
    }


def count_image_ctap_recursive(model, tokenizer, img_proc, device,
                                pil: Image.Image, system: str, human: str,
                                T: float = 100.0, max_depth: int = 3,
                                min_size: int = 224, s: int = TILE_SIZE,
                                native_wh: Optional[tuple] = None,
                                leaf_resize_mode: str = "resize") -> dict:
    """
    Top-level CTAP+Recursive wrapper.

    Handles native-resolution upscaling (once, before any splitting), runs the
    depth-0 global pass for routing and logging, then delegates to
    count_image_recursive starting at depth=1 on the four quadrants.

    Returns a dict compatible with run_eval row logging:
      count, mode ("global" | "recursive"), global_count,
      global_parse_ok, depth, leaves.
    """
    # Upscale to native resolution once before any splitting
    if native_wh is not None:
        W_nat, H_nat = native_wh
        W_cur, H_cur = pil.size
        if W_nat > W_cur or H_nat > H_cur:
            pil = pil.resize((W_nat, H_nat), Image.BICUBIC)

    # Depth-0 global pass — use 'global' adapter under dual-adapter mode
    _set_carc_adapter(model, "global")
    resized = pil.resize((s, s), Image.BICUBIC)
    c_global = count_one_pil(model, tokenizer, img_proc, device, resized, system, human)
    global_parse_ok = c_global is not None
    if c_global is None:
        c_global = 0

    # Routing: if count ≤ T, return global directly (no splitting)
    if c_global <= T:
        return dict(count=int(c_global), mode="global", global_count=int(c_global),
                    global_parse_ok=global_parse_ok, depth=0, leaves=1,
                    global_at_node=int(c_global), children_sum=0)

    # Switch to 'local' adapter for ALL depth>=1 quadrant forward passes
    _set_carc_adapter(model, "local")

    # Recurse into 4 non-overlapping quadrants starting at depth=1
    W, H = pil.size
    mx, my = W // 2, H // 2
    quads = [
        pil.crop((0,  0,  mx, my)),
        pil.crop((mx, 0,  W,  my)),
        pil.crop((0,  my, mx, H )),
        pil.crop((mx, my, W,  H )),
    ]

    total = 0
    total_leaves = 0
    max_depth_reached = 0
    for q in quads:
        r = count_image_recursive(
            model, tokenizer, img_proc, device, q,
            system, human, T, max_depth, 1, min_size, s,
            leaf_resize_mode=leaf_resize_mode,
        )
        total += r["count"]
        total_leaves += r["leaves"]
        max_depth_reached = max(max_depth_reached, r["depth"])

    # Global-local averaging at the top-level recursion node
    pred = int(round(0.5 * (c_global + total)))
    return dict(count=pred, mode="recursive", global_count=int(c_global),
                global_parse_ok=global_parse_ok,
                depth=max_depth_reached, leaves=total_leaves,
                global_at_node=int(c_global), children_sum=int(total))


# ═══════════════════════════════════════════════════════════════════════════════
# Main eval loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    rank        = accelerator.process_index
    world_size  = accelerator.num_processes
    device      = accelerator.device
    is_main     = accelerator.is_main_process

    if is_main:
        print(f"[CTAP+Rec] world_size  : {world_size}")
        print(f"[CTAP+Rec] checkpoint  : {args.checkpoint_dir}")
        print(f"[CTAP+Rec] val_json    : {args.val_json}")
        print(f"[CTAP+Rec] ann_json    : {args.ann_json}")
        print(f"[CTAP+Rec] T           : {args.T}")
        print(f"[CTAP+Rec] max_depth   : {args.max_depth}")
        print(f"[CTAP+Rec] min_size    : {args.min_size}")
        print(f"[CTAP+Rec] leaf_resize : {args.leaf_resize_mode}")

    model, tokenizer, img_proc = load_model_and_tokenizer(
        args.base_model, args.checkpoint_dir, args.mllm_hf, device,
        local_adapter=args.local_adapter,
        connector_weights=args.connector_weights,
    )

    # Load native-resolution lookup from FSC-147 annotation JSON.
    # Keys are bare filenames (e.g. "1050.jpg"); values are (W_native, H_native).
    native_size_map: dict = {}
    if args.ann_json and Path(args.ann_json).exists():
        with open(args.ann_json) as fh:
            ann = json.load(fh)
        for fname, entry in ann.items():
            if "W" in entry and "H" in entry:
                native_size_map[fname] = (int(entry["W"]), int(entry["H"]))
        if is_main:
            print(f"[CTAP+Rec] native_size_map: {len(native_size_map):,} entries loaded")
    else:
        if is_main:
            print("[CTAP+Rec] WARNING: no ann_json — recursion uses stored image dimensions")

    with open(args.val_json) as fh:
        val_data = json.load(fh)

    if args.max_images and args.max_images > 0:
        val_data = val_data[:args.max_images]
        if is_main:
            print(f"[CTAP+Rec] limiting to first {args.max_images} images (smoke test)")

    local_data = val_data[rank::world_size]
    if is_main:
        print(f"[CTAP+Rec] {len(val_data):,} images → {len(local_data)} per GPU\n")

    local_rows: list = []

    for item in tqdm(local_data, desc=f"rank{rank}", disable=(not is_main)):
        convs  = {c["from"]: c["value"] for c in item["conversations"]}
        system = convs.get(
            "system",
            "You are a helpful counting assistant. Answer with only a number.",
        )
        human = convs["human"]
        gt    = int(convs["gpt"])

        try:
            pil = Image.open(item["image"]).convert("RGB")
        except Exception as exc:
            if is_main:
                print(f"[WARN] Cannot open {item['image']}: {exc}")
            local_rows.append({
                "image": item["image"], "gt": gt, "pred": 0,
                "mode": "error", "global_count": 0, "depth": 0,
                "leaves": 0, "parse_ok": False,
            })
            continue

        fname = Path(item["image"]).name
        native_wh = native_size_map.get(fname)

        r = count_image_ctap_recursive(
            model, tokenizer, img_proc, device,
            pil, system, human,
            T=args.T, max_depth=args.max_depth, min_size=args.min_size,
            s=TILE_SIZE, native_wh=native_wh,
            leaf_resize_mode=args.leaf_resize_mode,
        )

        local_rows.append({
            "image"          : item["image"],
            "gt"             : gt,
            "pred"           : r["count"],
            "mode"           : r["mode"],
            "global_count"   : r["global_count"],
            "global_parse_ok": r["global_parse_ok"],
            "depth"          : r["depth"],
            "leaves"         : r["leaves"],
            "global_at_node" : r.get("global_at_node", r["global_count"]),
            "children_sum"   : r.get("children_sum", 0),
            "parse_ok"       : True,
        })

    # ── Gather rows from all ranks to rank 0 ─────────────────────────────────
    accelerator.wait_for_everyone()
    if world_size > 1:
        gathered = gather_object(local_rows)  # rank 0 receives all; others get partial
        rows = gathered if is_main else None
    else:
        rows = local_rows

    # ── Aggregate and write (rank 0 only) ────────────────────────────────────
    if is_main:
        preds = np.array([r["pred"] for r in rows], dtype=float)
        gts   = np.array([r["gt"]   for r in rows], dtype=float)
        mae   = float(np.mean(np.abs(preds - gts)))
        rmse  = float(np.sqrt(np.mean((preds - gts) ** 2)))
        n_recursive      = sum(1 for r in rows if r["mode"] == "recursive")
        frac_recursive   = n_recursive / max(len(rows), 1)
        n_global_fail    = sum(1 for r in rows if not r.get("global_parse_ok", True))
        total_leaves_run = sum(r["leaves"] for r in rows)

        split_name = Path(args.val_json).stem.split("_")[0]

        # Bucketed MAE
        if args.buckets:
            bucket_spec = []
            for tok in args.buckets.split(","):
                lo_s, hi_s = tok.strip().split("-")
                lo = int(lo_s)
                hi = 10**9 if hi_s in ("+", "inf", "") else int(hi_s)
                bucket_spec.append((lo, hi))
        else:
            bucket_spec = [(0,20),(21,50),(51,100),(101,200),(201,500),(501,10**9)]
        err = np.abs(preds - gts)
        buckets = []
        for lo, hi in bucket_spec:
            m = (gts >= lo) & (gts <= hi)
            if m.sum() > 0:
                buckets.append({"range": f"{lo}-{hi if hi<1e6 else '+'}", "n": int(m.sum()),
                                 "MAE": float(err[m].mean())})

        result = {
            "split"               : split_name,
            "dataset"             : args.dataset_name,
            "method"              : "CTAP+Recursive",
            "aggregation"         : "avg(global, children_sum)",
            "T"                   : args.T,
            "max_depth"           : args.max_depth,
            "min_size"            : args.min_size,
            "n"                   : len(rows),
            "MAE"                 : mae,
            "RMSE"                : rmse,
            "fraction_recursive"  : frac_recursive,
            "n_recursive"         : n_recursive,
            "total_leaves_run"    : total_leaves_run,
            "n_global_parse_fail" : n_global_fail,
            "checkpoint"          : args.checkpoint_dir,
            "local_adapter"       : args.local_adapter,
            "world_size"          : world_size,
            "bucketed_mae"        : buckets,
            "rows"                : rows,
        }
        if args.compute_nae:
            result["NAE"] = float(np.mean(np.abs(preds - gts)) / max(gts.mean(), 1.0))

        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  dataset             : {args.dataset_name} {split_name}  ({len(rows)} images)")
        print(f"  method              : CTAP+Recursive  T={args.T}  max_depth={args.max_depth}  min_size={args.min_size}")
        print(f"  GPUs used           : {world_size}")
        print(f"  MAE                 : {mae:.2f}")
        print(f"  RMSE                : {rmse:.2f}")
        if args.compute_nae:
            print(f"  NAE                 : {result['NAE']:.4f}")
        print(f"  fraction recursive  : {frac_recursive:.1%}  ({n_recursive}/{len(rows)})")
        print(f"  total leaves run    : {total_leaves_run}  (avg {total_leaves_run/max(n_recursive,1):.1f}/recursive image)")
        print(f"  global parse fail   : {n_global_fail}")
        print(f"  output              : {out_path}")
        print(f"\n  Bucketed MAE:")
        for b in buckets:
            print(f"    {b['range']:>8}  n={b['n']:>4}  MAE={b['MAE']:>7.2f}")
        print(f"{sep}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model",      default=DEFAULT_BASE_MODEL)
    p.add_argument("--mllm_hf",         default=DEFAULT_MLLM_HF)
    p.add_argument("--checkpoint_dir",  default=DEFAULT_CHECKPOINT,
                   help="Path to LoRA adapter (used at CARC depth 0; or all depths "
                        "if --local_adapter is not given).")
    p.add_argument("--local_adapter",   default=None,
                   help="Optional path to a second LoRA adapter used at CARC depth >= 1 "
                        "(quadrant crops). When set, --checkpoint_dir is the 'global' "
                        "adapter and --local_adapter is the 'local' adapter.")
    p.add_argument("--connector_weights", default=None,
                   help="Optional path to multi_modal_projector.bin saved by the "
                        "unfreeze-connector training script. Required for ckpts "
                        "trained with the connector unfrozen.")
    p.add_argument("--val_json",        default=DEFAULT_VAL_JSON)
    p.add_argument("--out_json",        default=DEFAULT_OUT_JSON)
    p.add_argument("--ann_json",        default=DEFAULT_ANN_JSON,
                   help="FSC-147 annotation JSON for native image dimensions")
    p.add_argument("--T",         type=float, default=DEFAULT_T,
                   help="Recursion routing threshold (default 100)")
    p.add_argument("--max_depth", type=int,   default=DEFAULT_MAX_DEPTH,
                   help="Maximum recursion depth (default 3)")
    p.add_argument("--min_size",  type=int,   default=DEFAULT_MIN_SIZE,
                   help="Minimum region dimension in pixels before stopping recursion (default 224)")
    p.add_argument("--leaf_resize_mode", default="resize",
                   choices=["resize", "pad", "pad_white"],
                   help="How to fit recursion-leaf crops to the model input. "
                        "'resize' (default) bicubic-resizes to s×s (baseline). "
                        "'pad' / 'pad_white' avoid upscaling small crops by centering "
                        "them on a gray / white s×s canvas. Depth 0 always uses 'resize'.")
    p.add_argument("--dataset_name", default="FSC-147",
                   help="Dataset name tag for output JSON (default FSC-147)")
    p.add_argument("--buckets", default="",
                   help="Comma-separated MAE buckets, e.g. '0-20,21-50,51-100,101-200,201-500'. "
                        "Empty = FSC-147 default.")
    p.add_argument("--compute_nae", action="store_true",
                   help="Add NAE = MAE / mean(GT) to output JSON.")
    p.add_argument("--max_images", type=int, default=0,
                   help="If >0, evaluate only first N images (smoke test).")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
