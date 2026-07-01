#!/usr/bin/env python3
"""Build LLaVA-format counting JSONs for cross-domain eval (CARPK, JHU-Crowd,
UCF-QNRF, ShanghaiTech A/B).  Output schema matches FSC-147 val/test JSON
consumed by eval_lora_counting_sft.py.

Each dataset only has a 'test' (and sometimes 'valid') split available locally:
  - CARPK            : test
  - JHU-Crowd        : valid, test
  - UCF-QNRF         : Test
  - ShanghaiTech A/B : test_data
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.io import loadmat

OUT_DIR = Path("/data/amondal/UniCount/outputs/experiment_lora_counting_sft/cross_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are a helpful counting assistant. Answer with only a number."
def human(category: str) -> str:
    return f"<image>\nHow many {category} are present in this image? Answer with only a number."


def make_entry(image_path: str, category: str, count: int) -> dict:
    return {
        "image": image_path,
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human", "value": human(category)},
            {"from": "gpt", "value": str(int(count))},
        ],
    }


def write(entries, name):
    p = OUT_DIR / name
    with open(p, "w") as fh:
        json.dump(entries, fh)
    print(f"  wrote {p}  ({len(entries)} images)")


# ── CARPK (test) ───────────────────────────────────────────────────────────────
def build_carpk():
    print("[CARPK]")
    root = Path("/data/amondal/datasets/CARPK_devkit/data")
    ids = (root / "ImageSets/test.txt").read_text().strip().split()
    entries = []
    for sid in ids:
        ann = root / f"Annotations/{sid}.txt"
        img = root / f"Images/{sid}.png"
        if not img.exists():
            img = root / f"Images/{sid}.jpg"
        cnt = sum(1 for ln in ann.read_text().splitlines() if ln.strip())
        entries.append(make_entry(str(img), "cars", cnt))
    write(entries, "carpk_test_counting.json")


# ── JHU-Crowd (valid, test) ────────────────────────────────────────────────────
def build_jhu():
    print("[JHU-Crowd]")
    root = Path("/data/amondal/JHU-Crowd")
    for split_dir, out in [("valid", "jhu_valid_counting.json"), ("test", "jhu_test_counting.json")]:
        labels = root / split_dir / "labels"
        images = root / split_dir / "images"
        entries = []
        for lbl in sorted(labels.glob("*.txt")):
            img = images / (lbl.stem + ".jpg")
            if not img.exists():
                continue
            cnt = sum(1 for ln in lbl.read_text().splitlines() if ln.strip())
            entries.append(make_entry(str(img), "people", cnt))
        write(entries, out)


# ── UCF-QNRF (Test) ────────────────────────────────────────────────────────────
def build_qnrf():
    print("[UCF-QNRF]")
    root = Path("/data/amondal/UCF-QNRF_ECCV18/Test")
    entries = []
    for img in sorted(root.glob("img_*.jpg")):
        ann = root / f"{img.stem}_ann.mat"
        if not ann.exists():
            continue
        m = loadmat(str(ann))
        cnt = int(m["annPoints"].shape[0])
        entries.append(make_entry(str(img), "people", cnt))
    write(entries, "qnrf_test_counting.json")


# ── ShanghaiTech part A / B (test_data) ────────────────────────────────────────
def build_sht():
    for part in ["A", "B"]:
        print(f"[ShanghaiTech part_{part}]")
        root = Path(f"/data/amondal/ShanghaiTech/part_{part}/test_data")
        images = root / "images"
        gt = root / "ground-truth"
        entries = []
        for img in sorted(images.glob("IMG_*.jpg"), key=lambda p: int(p.stem.split("_")[1])):
            ann = gt / f"GT_{img.stem}.mat"
            if not ann.exists():
                continue
            m = loadmat(str(ann))
            info = m["image_info"][0][0]
            cnt = int(info["location"][0][0].shape[0])
            entries.append(make_entry(str(img), "people", cnt))
        write(entries, f"sht_{part.lower()}_test_counting.json")


if __name__ == "__main__":
    build_carpk()
    build_jhu()
    build_qnrf()
    build_sht()
