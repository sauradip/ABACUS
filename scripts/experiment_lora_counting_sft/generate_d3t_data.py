#!/usr/bin/env python3
"""
Generate D³T (Divide-and-Discern Dialogue Tuning) training samples
from FSC-147 train images with GT > 100.

Reference: WS-COC (ICLR 2026)
  https://github.com/viscom-tongji/WS-COC/blob/main/prepare_instruct_data.py
  Lines 72-117 (type16: iterative bounding)

Adaptations for our pipeline (verified against existing repo):
  * Image placeholder = literal "<image>\n" (matches train_counting.json).
  * Subsequent turns must NOT contain "<image>" (preprocess_multimodal substring-replaces).
  * System prompt = "You are a helpful counting assistant. Answer with only a number."
  * Final GPT response = bare integer GT count (matches inference parser & bucket sampler's
    _extract_gt_count, which reads the LAST gpt turn).
  * Multi-turn loss masking is already correct in train_lora_counting_sft.py:
    preprocess_internvl iterates ALL turns; every gpt turn (Yes/No + final integer)
    contributes to the loss.
"""
import json
import re
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
TRAIN_JSON   = "outputs/experiment_lora_counting_sft/train/train_counting.json"
CLASSES_FILE = "/data/amondal/FSC147_hf/ImageClasses_FSC147.txt"
OUTPUT_JSON  = "data/fsc147_d3t.json"

GT_THRESHOLD     = 100      # Generate D3T only for images with GT > this
TARGET_PRECISION = 0.2      # Stop when (upper-lower)/GT < 0.2 (matches WS-COC)
INITIAL_LOWER    = 1
INITIAL_UPPER    = 2000
MAX_ROUNDS       = 10       # Matches WS-COC line 112

SYSTEM_PROMPT = "You are a helpful counting assistant. Answer with only a number."
NUM_RE        = re.compile(r"\d+")


def load_categories(path: str) -> dict:
    """Map image filename (e.g. '935.jpg') -> category name."""
    cats = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                cats[parts[0]] = parts[1].strip()
    return cats


def gt_from_item(item: dict) -> int | None:
    for conv in item["conversations"]:
        if conv["from"] == "gpt":
            m = NUM_RE.search(conv["value"])
            return int(m.group(0)) if m else None
    return None


def build_d3t_conversation(image_path: str, category: str, gt: int) -> list:
    """Multi-round binary search dialogue ending in exact-count question."""
    conv = [{"from": "system", "value": SYSTEM_PROMPT}]

    lower, upper = INITIAL_LOWER, INITIAL_UPPER
    round_num = 1
    while (upper - lower) / max(gt, 1) > TARGET_PRECISION:
        if round_num > MAX_ROUNDS:
            break
        mid = (lower + upper) // 2
        if round_num == 1:
            # First human turn MUST contain the literal "<image>" placeholder
            # — preprocess_multimodal substring-replaces it with the UniLIP
            # IMG_CONTEXT block. Subsequent turns MUST NOT contain "<image>".
            q = (f"<image>\nLooking at this image, are there more than "
                 f"{mid} {category} in total? Please answer Yes or No.")
        else:
            q = (f"Are there more than {mid} {category} in total? "
                 f"Please answer Yes or No.")
        if gt > mid:
            ans = "Yes"
            lower = mid + 1
        else:
            ans = "No"
            upper = mid
        conv.append({"from": "human", "value": q})
        conv.append({"from": "gpt",   "value": ans})
        round_num += 1

    # Final round: exact count within narrowed range.
    final_q = (f"The count is between {lower} and {upper}. "
               f"How many {category} are there exactly? "
               f"Answer with only a number.")
    conv.append({"from": "human", "value": final_q})
    conv.append({"from": "gpt",   "value": str(gt)})
    return conv


def main() -> None:
    train = json.load(open(TRAIN_JSON))
    cats  = load_categories(CLASSES_FILE)

    print(f"Loaded {len(train):,} train samples and {len(cats):,} category mappings")

    out, skipped_no_gt, skipped_no_cat, n_under = [], 0, 0, 0
    for item in train:
        gt = gt_from_item(item)
        if gt is None:
            skipped_no_gt += 1
            continue
        if gt <= GT_THRESHOLD:
            n_under += 1
            continue
        fname = Path(item["image"]).name
        cat = cats.get(fname)
        if cat is None:
            skipped_no_cat += 1
            continue
        conv = build_d3t_conversation(item["image"], cat, gt)
        out.append({"image": item["image"], "conversations": conv})

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nSkipped (no GT parse) : {skipped_no_gt}")
    print(f"Skipped (no category) : {skipped_no_cat}")
    print(f"Below GT threshold    : {n_under}")
    print(f"Generated D3T samples : {len(out)}")
    print(f"  written → {OUTPUT_JSON}")

    # Distribution + dialogue length stats
    by_bucket = {(101, 200): 0, (201, 500): 0, (501, 10**9): 0}
    rounds = []
    for item in out:
        # last gpt turn carries the exact-count answer
        gt = int(item["conversations"][-1]["value"])
        for (lo, hi) in by_bucket:
            if lo <= gt <= hi:
                by_bucket[(lo, hi)] += 1
                break
        # rounds = number of human turns
        rounds.append(sum(1 for c in item["conversations"] if c["from"] == "human"))

    print("\nGT bucket distribution of D3T samples:")
    for (lo, hi), n in by_bucket.items():
        hi_s = "+" if hi >= 10**6 else hi
        print(f"  {lo:>4}-{hi_s:>4}: {n}")
    if rounds:
        print(f"\nAvg rounds/sample  : {sum(rounds)/len(rounds):.2f}  "
              f"(min={min(rounds)}, max={max(rounds)})")

    # Sanity-check 3 random samples
    import random
    random.seed(0)
    print("\n=== 3 random sample previews ===")
    for s in random.sample(out, min(3, len(out))):
        gt = int(s["conversations"][-1]["value"])
        print(f"\n--- {Path(s['image']).name}  GT={gt}  rounds={sum(1 for c in s['conversations'] if c['from']=='human')}")
        for c in s["conversations"]:
            v = c["value"].replace("\n", "  ")
            print(f"  [{c['from']:>5}] {v[:140]}")


if __name__ == "__main__":
    main()
