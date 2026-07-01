"""
Build a count-balanced training mix for the next SFT run.

Inputs:
  - /data/amondal/UniCountData/ucount_consolidated/train_counting.json
  - /data/amondal/UniCountData/ucount_consolidated/train_counting_clean.json
  - /data/amondal/UniCountData/ucount_crowd_consolidated/train_counting.json
  - /data/amondal/UniCountData/ucount_crowd_consolidated/train_counting_person.json
  - /data/amondal/UniCount/outputs/experiment_lora_counting_sft/all/all_counting_countdetect.json
    (FSC-147 train, 6135 entries, 147 unique categories, "Count and detect ..." prompt)

Outputs (under outputs/experiment_lora_counting_sft/balanced_mix/):
  - balanced_mix_train.json          (the SFT data)
  - mix_diagnostics.json             (counts per source/bucket/category)
  - mix_count_distribution.csv       (per-category count buckets)

Design (per the user's specification):
  1. Canonicalize all person-synonym categories to "people" so the
     prompt matches eval format ("Count and detect all the people ...").
  2. Dedupe person entries by image path (multiple person/people aliases
     pointing at the same image are collapsed).
  3. Count-balance person bucket sizes (cap at PERSON_CAP_PER_BUCKET each
     to flatten the count prior).
  4. Keep all non-person entries except "objects" (too vague) and any
     category appearing in only 1 image. Cap any single non-person
     category at NONPERSON_CAP_PER_CATEGORY to prevent dominance
     (e.g. wheats=2699).
  5. Rewrite all corpus prompts from "How many X are present in this
     image?" to "Count and detect all the X in the image." for prompt
     consistency with FSC-147 training data and eval prompts.
  6. 3x upsample FSC-147 entries to prevent dilution.

This script is purely data preparation and emits diagnostics; it does
not start training.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- config
CORPUS_PATHS = [
    "/data/amondal/UniCountData/ucount_consolidated/train_counting.json",
    "/data/amondal/UniCountData/ucount_consolidated/train_counting_clean.json",
    "/data/amondal/UniCountData/ucount_crowd_consolidated/train_counting.json",
    "/data/amondal/UniCountData/ucount_crowd_consolidated/train_counting_person.json",
]
FSC_PATH = "/data/amondal/UniCount/outputs/experiment_lora_counting_sft/all/all_counting_countdetect.json"
OUT_DIR = Path("/data/amondal/UniCount/outputs/experiment_lora_counting_sft/balanced_mix_v2")

PERSON_CAP_PER_BUCKET = 1500
NONPERSON_CAP_PER_CATEGORY = 800
NONPERSON_CAP_PER_BUCKET = 5000   # NEW: trim low-count non-person dominance
FSC_UPSAMPLE_FACTOR = 3
SEED = 42

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
    """Return (image, raw_category, count) or None."""
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


# ---------------------------------------------------------------- load
def load_corpus():
    """Returns dedup'd person and non-person pools."""
    person_by_image = {}        # image -> count (canonical "people")
    nonperson_by_key = {}       # (image, cat) -> count
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
                # canonicalize: dedupe by image alone, keep first count
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
def balance_person(person_by_image: dict, rng: random.Random):
    """Cap each count bucket at PERSON_CAP_PER_BUCKET."""
    buckets = defaultdict(list)
    for img, n in person_by_image.items():
        buckets[bucket(n)].append((img, n))
    out, before, after = [], {}, {}
    for b, items in buckets.items():
        before[b] = len(items)
        if len(items) > PERSON_CAP_PER_BUCKET:
            chosen = rng.sample(items, PERSON_CAP_PER_BUCKET)
        else:
            chosen = items
        after[b] = len(chosen)
        for img, n in chosen:
            out.append(make_entry(img, "people", n))
    return out, before, after


def balance_nonperson(nonperson_by_key: dict, rng: random.Random):
    """Cap each category at NONPERSON_CAP_PER_CATEGORY; drop singletons;
    then count-bucket-cap at NONPERSON_CAP_PER_BUCKET to trim the 0-20
    dominance."""
    by_cat = defaultdict(list)
    for (img, cat), n in nonperson_by_key.items():
        by_cat[cat].append((img, n))
    cat_capped, before, after_cat, dropped_singletons = [], {}, {}, 0
    for cat, items in by_cat.items():
        before[cat] = len(items)
        if len(items) < 2:
            dropped_singletons += 1
            continue
        if len(items) > NONPERSON_CAP_PER_CATEGORY:
            chosen = rng.sample(items, NONPERSON_CAP_PER_CATEGORY)
        else:
            chosen = items
        after_cat[cat] = len(chosen)
        for img, n in chosen:
            cat_capped.append((cat, img, n))

    # Count-bucket cap on the per-category-capped pool
    by_bucket = defaultdict(list)
    for cat, img, n in cat_capped:
        by_bucket[bucket(n)].append((cat, img, n))
    bucket_before = {b: len(v) for b, v in by_bucket.items()}
    bucket_after = {}
    out = []
    for b, items in by_bucket.items():
        if len(items) > NONPERSON_CAP_PER_BUCKET:
            chosen = rng.sample(items, NONPERSON_CAP_PER_BUCKET)
        else:
            chosen = items
        bucket_after[b] = len(chosen)
        for cat, img, n in chosen:
            out.append(make_entry(img, cat, n))
    return out, before, after_cat, bucket_before, bucket_after, dropped_singletons


# ---------------------------------------------------------------- diagnostics
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

    print("[1] Loading corpus ...")
    person_by_image, nonperson_by_key, raw_p_freq, seen = load_corpus()
    print(f"    raw entries per file:")
    for p, n in seen.items():
        print(f"      {n:6d} {p}")
    print(f"    unique person images: {len(person_by_image)}")
    print(f"    unique non-person (image,cat) pairs: {len(nonperson_by_key)}")

    print("[2] Count-balancing person ...")
    person_entries, p_before, p_after = balance_person(person_by_image, rng)
    print(f"    person buckets (before -> after, cap={PERSON_CAP_PER_BUCKET}):")
    for b in ["0-20","21-50","51-100","101-200","201-500","501+"]:
        print(f"      {b:>9s}: {p_before.get(b,0):6d} -> {p_after.get(b,0):6d}")

    print("[3] Capping non-person categories + count buckets ...")
    nonp_entries, np_before, np_after, npb_before, npb_after, dropped_singletons = \
        balance_nonperson(nonperson_by_key, rng)
    print(f"    non-person categories (before -> after, cap={NONPERSON_CAP_PER_CATEGORY}):")
    for cat, n in sorted(np_before.items(), key=lambda kv: -kv[1])[:15]:
        print(f"      {n:6d} -> {np_after.get(cat,0):6d}  {cat}")
    print(f"    non-person count buckets (after per-cat cap, then bucket cap={NONPERSON_CAP_PER_BUCKET}):")
    for b in ["0-20","21-50","51-100","101-200","201-500","501+"]:
        print(f"      {b:>9s}: {npb_before.get(b,0):6d} -> {npb_after.get(b,0):6d}")
    print(f"    dropped singleton categories: {dropped_singletons}")
    print(f"    non-person entries kept: {len(nonp_entries)} across "
          f"{len(np_after)} categories")

    print(f"[4] Loading FSC-147 replay from {FSC_PATH} ...")
    fsc = json.load(open(FSC_PATH))
    fsc_x = fsc * FSC_UPSAMPLE_FACTOR
    print(f"    fsc base={len(fsc)}  x{FSC_UPSAMPLE_FACTOR} -> {len(fsc_x)}")

    mixed = person_entries + nonp_entries + fsc_x
    rng.shuffle(mixed)
    print(f"[5] Final mix: {len(mixed)} entries")
    print(f"      person:     {len(person_entries):6d}  ({len(person_entries)/len(mixed):.1%})")
    print(f"      nonperson:  {len(nonp_entries):6d}  ({len(nonp_entries)/len(mixed):.1%})")
    print(f"      fsc x{FSC_UPSAMPLE_FACTOR}:  {len(fsc_x):6d}  ({len(fsc_x)/len(mixed):.1%})")

    out_json = OUT_DIR / "balanced_mix_train.json"
    print(f"[6] Writing {out_json} ...")
    with open(out_json, "w") as f:
        json.dump(mixed, f)
    print(f"    bytes: {os.path.getsize(out_json):,}")

    # Diagnostics
    diag = {
        "config": {
            "PERSON_CAP_PER_BUCKET": PERSON_CAP_PER_BUCKET,
            "NONPERSON_CAP_PER_CATEGORY": NONPERSON_CAP_PER_CATEGORY,
            "NONPERSON_CAP_PER_BUCKET": NONPERSON_CAP_PER_BUCKET,
            "FSC_UPSAMPLE_FACTOR": FSC_UPSAMPLE_FACTOR,
            "SEED": SEED,
            "DROP_CATEGORIES": sorted(DROP_CATEGORIES),
        },
        "raw": {
            "per_file_entries": {p: n for p, n in seen.items()},
            "unique_person_images": len(person_by_image),
            "unique_nonperson_pairs": len(nonperson_by_key),
            "raw_person_category_freq_top": dict(raw_p_freq.most_common(20)),
        },
        "person_buckets": {"before": p_before, "after": p_after},
        "nonperson_categories": {
            "n_categories_before": len(np_before),
            "n_categories_after":  len(np_after),
            "dropped_singletons":  dropped_singletons,
            "top_caps":            {cat: {"before": np_before[cat],
                                          "after":  np_after.get(cat, 0)}
                                    for cat in sorted(np_before,
                                                      key=lambda c: -np_before[c])[:20]},
            "buckets_before_cap":  npb_before,
            "buckets_after_cap":   npb_after,
        },
        "fsc_replay": {"base": len(fsc), "factor": FSC_UPSAMPLE_FACTOR,
                       "after": len(fsc_x)},
        "final_mix_total": len(mixed),
        "count_distribution": {
            "person":    diag_count_dist(person_entries,  "person"),
            "nonperson": diag_count_dist(nonp_entries,    "nonperson"),
            "fsc_x":     diag_count_dist(fsc_x,           "fsc_x"),
            "all":       diag_count_dist(mixed,           "all"),
        },
    }
    with open(OUT_DIR / "mix_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"    wrote {OUT_DIR / 'mix_diagnostics.json'}")

    # Per-category count-bucket csv
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

    # Print sanity head
    cd = diag["count_distribution"]
    print("\n[7] Final count distribution summary:")
    for k in ["person", "nonperson", "fsc_x", "all"]:
        d = cd[k]
        print(f"    {k:>9s}: n={d['n_entries']:6d}  mean={d['mean']:7.2f}  "
              f"med={d['median']:5.0f}  p99={d['p99']:6.0f}  max={d['max']:5d}  "
              f"buckets={d['buckets']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
