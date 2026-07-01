"""
Build v3-S count-balanced training mix.

Differences vs v2:
  - FSC_UPSAMPLE_FACTOR: 3 -> 5
  - Adds SHA train (300 imgs) and SHB train (400 imgs) at 5x upsample each
  - Same person/non-person caps as v2 (PERSON_CAP=1500, NONPERSON_CAT=800,
    NONPERSON_BUCKET=5000)

Outputs (under outputs/experiment_lora_counting_sft/balanced_mix_v3s/):
  - sha_train_sft.json   (SHA-train converted to SFT format, 300 entries)
  - shb_train_sft.json   (SHB-train converted to SFT format, 400 entries)
  - balanced_mix_train.json  (the SFT data; ~50K entries)
  - mix_diagnostics.json
  - mix_count_distribution.csv
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------- config
CORPUS_PATHS = [
    "/data/amondal/UniCountData/ucount_consolidated/train_counting.json",
    "/data/amondal/UniCountData/ucount_consolidated/train_counting_clean.json",
    "/data/amondal/UniCountData/ucount_crowd_consolidated/train_counting.json",
    "/data/amondal/UniCountData/ucount_crowd_consolidated/train_counting_person.json",
]
FSC_PATH = "/data/amondal/UniCount/outputs/experiment_lora_counting_sft/all/all_counting_countdetect.json"

SHA_ROOT = "/data/amondal/ShanghaiTech/part_A"
SHB_ROOT = "/data/amondal/ShanghaiTech/part_B"

OUT_DIR = Path("/data/amondal/UniCount/outputs/experiment_lora_counting_sft/balanced_mix_v3s")

PERSON_CAP_PER_BUCKET     = 1500
NONPERSON_CAP_PER_CATEGORY = 800
NONPERSON_CAP_PER_BUCKET   = 5000
FSC_UPSAMPLE_FACTOR        = 5    # was 3 in v2
SHAB_UPSAMPLE_FACTOR       = 5    # NEW
SEED                       = 42

PERSON_SYNS = {
    "person", "people", "persons", "boys", "girls", "kids", "students",
    "men", "women", "man", "boy", "girl", "woman", "crowd", "crowds",
    "pedestrians", "pedestrian", "workers", "soldiers", "runners",
    "marathoners", "child", "children", "adult", "adults",
}
DROP_CATEGORIES = {"objects"}

SYS_PROMPT = "You are a helpful counting assistant. Answer with only a number."
HUMAN_TMPL = "<image>\nCount and detect all the {cat} in the image. Answer with only a number."

PAT_HOWMANY = re.compile(r"How many (.+?) are present", re.I)
PAT_COUNTDET = re.compile(r"Count and detect all the (.+?) in the image", re.I)


# ---------------------------------------------------------------- helpers
def bucket(c: int) -> str:
    if c <= 20:    return "0-20"
    if c <= 50:    return "21-50"
    if c <= 100:   return "51-100"
    if c <= 200:   return "101-200"
    if c <= 500:   return "201-500"
    return "501+"


def parse_corpus_entry(e: dict):
    h = e["conversations"][1]["value"]
    m = PAT_HOWMANY.search(h) or PAT_COUNTDET.search(h)
    if not m:
        return None
    cat = m.group(1).strip().lower()
    try:
        c = int(e["conversations"][2]["value"])
    except (ValueError, KeyError, TypeError):
        return None
    return e["image"], cat, c


def is_person_cat(cat: str) -> bool:
    if cat in PERSON_SYNS:
        return True
    toks = re.split(r"[\s\-]", cat)
    return any(t in PERSON_SYNS for t in toks)


def make_entry(image: str, category: str, count: int) -> dict:
    return {
        "image": image,
        "conversations": [
            {"from": "system", "value": SYS_PROMPT},
            {"from": "human",  "value": HUMAN_TMPL.format(cat=category)},
            {"from": "gpt",    "value": str(count)},
        ],
    }


# ---------------------------------------------------------------- SHA/B builder
def build_shab_entries(data_root: str, split: str = "train") -> list:
    gt_dir  = os.path.join(data_root, f"{split}_data", "ground-truth")
    img_dir = os.path.join(data_root, f"{split}_data", "images")
    entries = []
    for mat_path in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
        mat   = sio.loadmat(mat_path)
        try:
            dots = mat["image_info"][0][0][0][0][0]
            count = int(len(dots))
        except Exception as exc:
            print(f"    WARN: parse fail {mat_path}: {exc}")
            continue
        img_id   = os.path.basename(mat_path).replace("GT_", "").replace(".mat", "")
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        entries.append(make_entry(os.path.abspath(img_path), "people", count))
    return entries


# ---------------------------------------------------------------- corpus load
def load_corpus():
    person_by_image = {}
    nonperson_by_key = {}
    raw_person_cat_freq = Counter()
    seen_per_file = Counter()
    for p in CORPUS_PATHS:
        d = json.load(open(p))
        for e in d:
            parsed = parse_corpus_entry(e)
            if parsed is None:
                continue
            img, cat, n = parsed
            seen_per_file[p] += 1
            if is_person_cat(cat):
                raw_person_cat_freq[cat] += 1
                if img not in person_by_image:
                    person_by_image[img] = n
            else:
                if cat in DROP_CATEGORIES:
                    continue
                key = (img, cat)
                if key not in nonperson_by_key:
                    nonperson_by_key[key] = n
    return person_by_image, nonperson_by_key, raw_person_cat_freq, seen_per_file


# ---------------------------------------------------------------- balance
def balance_person(person_by_image, rng):
    buckets = defaultdict(list)
    for img, n in person_by_image.items():
        buckets[bucket(n)].append((img, n))
    out, before, after = [], {}, {}
    for b, items in buckets.items():
        before[b] = len(items)
        chosen = rng.sample(items, min(len(items), PERSON_CAP_PER_BUCKET))
        after[b] = len(chosen)
        for img, n in chosen:
            out.append(make_entry(img, "people", n))
    return out, before, after


def balance_nonperson(nonperson_by_key, rng):
    by_cat = defaultdict(list)
    for (img, cat), n in nonperson_by_key.items():
        by_cat[cat].append((img, n))
    cat_capped, before, after_cat, dropped_singletons = [], {}, {}, 0
    for cat, items in by_cat.items():
        before[cat] = len(items)
        if len(items) < 2:
            dropped_singletons += 1
            continue
        chosen = rng.sample(items, min(len(items), NONPERSON_CAP_PER_CATEGORY))
        after_cat[cat] = len(chosen)
        for img, n in chosen:
            cat_capped.append((cat, img, n))
    by_bucket = defaultdict(list)
    for cat, img, n in cat_capped:
        by_bucket[bucket(n)].append((cat, img, n))
    bucket_before = {b: len(v) for b, v in by_bucket.items()}
    bucket_after = {}
    out = []
    for b, items in by_bucket.items():
        chosen = rng.sample(items, min(len(items), NONPERSON_CAP_PER_BUCKET))
        bucket_after[b] = len(chosen)
        for cat, img, n in chosen:
            out.append(make_entry(img, cat, n))
    return out, before, after_cat, bucket_before, bucket_after, dropped_singletons


# ---------------------------------------------------------------- diag
def diag_count_dist(entries, label):
    counts = []
    for e in entries:
        try:
            counts.append(int(e["conversations"][2]["value"]))
        except Exception:
            pass
    arr = np.array(counts) if counts else np.array([0])
    return {
        "label": label,
        "n_entries": len(entries),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "buckets": {b: int(sum(1 for c in counts if bucket(c) == b))
                    for b in ["0-20","21-50","51-100","101-200","201-500","501+"]},
    }


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading UniCount corpus ...")
    person_by_image, nonperson_by_key, raw_p_freq, seen = load_corpus()
    for p, n in seen.items():
        print(f"      {n:6d} {p}")
    print(f"    unique person images: {len(person_by_image)}")
    print(f"    unique non-person (image,cat) pairs: {len(nonperson_by_key)}")

    print("[2] Count-balancing person ...")
    person_entries, p_before, p_after = balance_person(person_by_image, rng)
    for b in ["0-20","21-50","51-100","101-200","201-500","501+"]:
        print(f"      {b:>9s}: {p_before.get(b,0):6d} -> {p_after.get(b,0):6d}")

    print("[3] Capping non-person categories + count buckets ...")
    nonp_entries, np_before, np_after, npb_before, npb_after, dropped_singletons = \
        balance_nonperson(nonperson_by_key, rng)
    for b in ["0-20","21-50","51-100","101-200","201-500","501+"]:
        print(f"      {b:>9s}: {npb_before.get(b,0):6d} -> {npb_after.get(b,0):6d}")
    print(f"    dropped singleton categories: {dropped_singletons}")
    print(f"    non-person entries: {len(nonp_entries)} across {len(np_after)} categories")

    print(f"[4] Loading FSC-147 replay from {FSC_PATH} ...")
    fsc = json.load(open(FSC_PATH))
    fsc_x = fsc * FSC_UPSAMPLE_FACTOR
    print(f"    fsc base={len(fsc)}  x{FSC_UPSAMPLE_FACTOR} -> {len(fsc_x)}")

    print(f"[5] Building SHA/B from .mat files ...")
    sha = build_shab_entries(SHA_ROOT, "train")
    shb = build_shab_entries(SHB_ROOT, "train")
    sha_counts = [int(e["conversations"][2]["value"]) for e in sha]
    shb_counts = [int(e["conversations"][2]["value"]) for e in shb]
    print(f"    SHA: {len(sha)} entries, range {min(sha_counts)}-{max(sha_counts)}, "
          f"mean={np.mean(sha_counts):.0f}, median={np.median(sha_counts):.0f}")
    print(f"    SHB: {len(shb)} entries, range {min(shb_counts)}-{max(shb_counts)}, "
          f"mean={np.mean(shb_counts):.0f}, median={np.median(shb_counts):.0f}")
    json.dump(sha, open(OUT_DIR / "sha_train_sft.json", "w"), indent=2)
    json.dump(shb, open(OUT_DIR / "shb_train_sft.json", "w"), indent=2)

    sha_x = sha * SHAB_UPSAMPLE_FACTOR
    shb_x = shb * SHAB_UPSAMPLE_FACTOR
    print(f"    SHA x{SHAB_UPSAMPLE_FACTOR} = {len(sha_x)}; SHB x{SHAB_UPSAMPLE_FACTOR} = {len(shb_x)}")

    mixed = person_entries + nonp_entries + fsc_x + sha_x + shb_x
    rng.shuffle(mixed)
    total = len(mixed)
    print(f"[6] Final mix: {total} entries")
    for label, lst in [("person",person_entries),("nonperson",nonp_entries),
                        ("fsc x"+str(FSC_UPSAMPLE_FACTOR),fsc_x),
                        ("sha x"+str(SHAB_UPSAMPLE_FACTOR),sha_x),
                        ("shb x"+str(SHAB_UPSAMPLE_FACTOR),shb_x)]:
        print(f"      {label:>10s}:  {len(lst):6d}  ({len(lst)/total:.1%})")

    out_json = OUT_DIR / "balanced_mix_train.json"
    print(f"[7] Writing {out_json} ...")
    with open(out_json, "w") as f:
        json.dump(mixed, f)
    print(f"    bytes: {os.path.getsize(out_json):,}")

    diag = {
        "config": {
            "PERSON_CAP_PER_BUCKET": PERSON_CAP_PER_BUCKET,
            "NONPERSON_CAP_PER_CATEGORY": NONPERSON_CAP_PER_CATEGORY,
            "NONPERSON_CAP_PER_BUCKET": NONPERSON_CAP_PER_BUCKET,
            "FSC_UPSAMPLE_FACTOR": FSC_UPSAMPLE_FACTOR,
            "SHAB_UPSAMPLE_FACTOR": SHAB_UPSAMPLE_FACTOR,
            "SEED": SEED,
            "DROP_CATEGORIES": sorted(DROP_CATEGORIES),
        },
        "raw": {
            "per_file_entries": dict(seen),
            "unique_person_images": len(person_by_image),
            "unique_nonperson_pairs": len(nonperson_by_key),
            "raw_person_category_freq_top": dict(raw_p_freq.most_common(20)),
            "sha_train_n": len(sha), "shb_train_n": len(shb),
        },
        "person_buckets": {"before": p_before, "after": p_after},
        "nonperson_categories": {
            "n_categories_before": len(np_before),
            "n_categories_after":  len(np_after),
            "dropped_singletons":  dropped_singletons,
            "buckets_before_cap":  npb_before,
            "buckets_after_cap":   npb_after,
        },
        "fsc_replay": {"base": len(fsc), "factor": FSC_UPSAMPLE_FACTOR, "after": len(fsc_x)},
        "shab_replay": {"sha_base": len(sha), "shb_base": len(shb),
                        "factor": SHAB_UPSAMPLE_FACTOR,
                        "sha_after": len(sha_x), "shb_after": len(shb_x)},
        "final_mix_total": total,
        "count_distribution": {
            "person":    diag_count_dist(person_entries, "person"),
            "nonperson": diag_count_dist(nonp_entries,   "nonperson"),
            "fsc_x":     diag_count_dist(fsc_x,          "fsc_x"),
            "sha_x":     diag_count_dist(sha_x,          "sha_x"),
            "shb_x":     diag_count_dist(shb_x,          "shb_x"),
            "person_all_with_shab": diag_count_dist(person_entries + sha_x + shb_x, "person_all_with_shab"),
            "all":       diag_count_dist(mixed,          "all"),
        },
    }
    with open(OUT_DIR / "mix_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"    wrote {OUT_DIR / 'mix_diagnostics.json'}")

    cat_buckets = defaultdict(lambda: defaultdict(int))
    cat_total = defaultdict(int)
    for e in mixed:
        m = PAT_COUNTDET.search(e["conversations"][1]["value"])
        if not m: continue
        c = m.group(1).strip().lower()
        n = int(e["conversations"][2]["value"])
        cat_buckets[c][bucket(n)] += 1
        cat_total[c] += 1
    csv_path = OUT_DIR / "mix_count_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        bks = ["0-20","21-50","51-100","101-200","201-500","501+"]
        w.writerow(["category", "n_entries"] + bks)
        for c, tot in sorted(cat_total.items(), key=lambda kv: -kv[1]):
            w.writerow([c, tot] + [cat_buckets[c][b] for b in bks])
    print(f"    wrote {csv_path}")

    print("\n[8] Final count distribution summary:")
    cd = diag["count_distribution"]
    for k in ["person","nonperson","fsc_x","sha_x","shb_x","person_all_with_shab","all"]:
        d = cd[k]
        print(f"    {k:>22s}: n={d['n_entries']:6d}  mean={d['mean']:7.2f}  "
              f"med={d['median']:5.0f}  p99={d['p99']:6.0f}  max={d['max']:5d}  buckets={d['buckets']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
