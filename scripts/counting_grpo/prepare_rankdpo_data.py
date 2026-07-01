#!/usr/bin/env python3
"""
Prepare Stage 1.6/2.5/3.x counting datasets.

This script supports two generation modes:
1) tally_sft: supervised Stage 1.6 rows with deterministic Thought+tally traces.
2) rankdpo: preference pairs for Stage 2.5/3.x DPO-style training.

In rankdpo mode, this script supports two preference styles:
1) Standard DPO pairs (GT chosen vs audit repetitive rejected).
2) RankDPO triplets collapsed into pairwise preferences:
     - rank1 > rank2 (ground truth beats close overcount)
     - rank2 > rank3 (close overcount beats babbling overcount)

Primary output format (JSONL):
{
    "prompt": "<image>\\nHow many apples ...",
    "chosen": "...",
    "rejected": "...",
    "image": "/abs/path/to/image.jpg",
    "id": "2271.jpg",
    "pair_type": "audit_raw_repetition" | "rank1_gt_vs_rank2_close" | ...
}

Stage 3.0 extension (optional):
- Emit recursive RexOmni-style pairs with thought + clusters-first JSON.
- Compute weighted composite rewards (format + consistency + accuracy).
- Keep only pairs where chosen reward beats rejected reward by a margin.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


THOUGHT_TAG = "<|thought|>"
SCAFFOLD_TAG = "<|scaffold|>"
COUNT_TAG = "<|count|>"


@dataclass
class AuditPrediction:
    image_id: str
    prompt: str
    pred_points: List[List[int]]
    pred_count: int
    gt_count: Optional[int]
    prediction_text: str


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_count(text: str) -> Optional[int]:
    match = re.search(r"<\|count\|>\s*([-+]?\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    loose = re.search(r"\b(\d+)\b", text)
    if loose:
        try:
            return int(loose.group(1))
        except ValueError:
            return None
    return None


def parse_points(text: str) -> List[List[int]]:
    # Primary path: parse JSON-like scaffold block.
    block = re.search(r"<\|(scaffold|code)\|>\s*(\[\[.*?\]\])", text, re.DOTALL)
    if block:
        candidate = block.group(2)
        try:
            parsed = json.loads(candidate)
            out = []
            for pair in parsed:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    out.append([int(round(float(pair[0]))), int(round(float(pair[1])))])
            if out:
                return out
        except Exception:
            pass

    # Fallback path: parse coordinate pairs from free text.
    out = []
    for x_raw, y_raw in re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text):
        out.append([int(round(float(x_raw))), int(round(float(y_raw)))])
    return out


def normalize_scaffold_text(points: List[List[int]], count: int, category: str) -> str:
    thought = (
        f"The image shows {category}. "
        "I will count from top-left to bottom-right and place one coordinate on each instance before giving the final total."
    )
    scaffold = json.dumps(points, separators=(",", ":"))
    return f"{THOUGHT_TAG}\n{thought}\n<|answer|>\n{SCAFFOLD_TAG} {scaffold}\n{COUNT_TAG} {count}"


def extract_category_from_prompt(prompt: str) -> str:
    match = re.search(r"How many\s+(.+?)\s+are\s+in\s+this\s+image\??", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "objects"


def load_audit_predictions(audit_path: str) -> Dict[str, AuditPrediction]:
    with open(audit_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("rows", [])
    by_image: Dict[str, AuditPrediction] = {}
    for row in rows:
        image_id = str(row.get("image", "")).strip()
        if not image_id:
            continue

        pred_points = row.get("pred_points")
        if not isinstance(pred_points, list) or not pred_points:
            pred_points = parse_points(str(row.get("prediction_text", "")))
        normalized_points = []
        for pair in pred_points:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    normalized_points.append([int(round(float(pair[0]))), int(round(float(pair[1])))])
                except Exception:
                    continue

        pred_count = row.get("pred_count")
        if not isinstance(pred_count, int):
            parsed_count = parse_count(str(row.get("prediction_text", "")))
            pred_count = parsed_count if parsed_count is not None else len(normalized_points)

        pred = AuditPrediction(
            image_id=image_id,
            prompt=str(row.get("prompt", "")),
            pred_points=normalized_points,
            pred_count=int(pred_count),
            gt_count=row.get("gt_count"),
            prediction_text=str(row.get("prediction_text", "")),
        )

        # Register multiple key forms to handle basename/path/stem mismatches.
        p = Path(image_id)
        aliases = {
            image_id,
            p.name,
            p.stem,
            f"{p.stem}.jpg",
            f"{p.stem}.png",
        }
        for key in aliases:
            if key and key not in by_image:
                by_image[key] = pred

    return by_image


def robust_parse_json(text: str) -> Optional[dict]:
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        if '"total_count"' in text:
            match = re.search(r"(\{.*\})", "{" + text, re.DOTALL)
        if not match:
            return None

    candidate = match.group(1)
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        try:
            payload = json.loads(candidate + "]}")
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None


def repair_json_structure(raw_text: str) -> Optional[dict]:
    total_match = re.search(r'"total_count"\s*:\s*(\d+)', raw_text)
    total_val = int(total_match.group(1)) if total_match else 0

    # Capture anchor/count tuples even from partially malformed cluster dictionaries.
    pair_matches = re.findall(
        r'"anchor"\s*:\s*\[(\d+)\s*,\s*(\d+)\][^{}]*?"count"\s*:\s*(\d+)',
        raw_text,
        flags=re.DOTALL,
    )

    clusters: List[dict] = []
    points: List[List[int]] = []
    for x_raw, y_raw, c_raw in pair_matches:
        x_val = int(x_raw)
        y_val = int(y_raw)
        c_val = max(1, int(c_raw))
        clusters.append({"anchor": [x_val, y_val], "count": c_val})
        for _ in range(c_val):
            points.append([x_val, y_val])

    if total_val <= 0 and points:
        total_val = len(points)

    if total_val <= 0 and not clusters:
        return None

    return {
        "total_count": total_val,
        "anchors_summary": "",
        "clusters": clusters,
    }


def load_structural_negatives(audit_path: str) -> Dict[str, Dict[str, str]]:
    with open(audit_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    out: Dict[str, Dict[str, str]] = {}
    for row in payload.get("rows", []):
        image_id = str(row.get("image", "")).strip()
        raw_text = str(row.get("prediction_text", ""))
        if not image_id or not raw_text:
            continue

        # Structural negatives are malformed JSON with no prose drift.
        if robust_parse_json(raw_text) is not None:
            continue
        if not raw_text.lstrip().startswith("{"):
            continue

        repaired = repair_json_structure(raw_text)
        if repaired is None:
            continue

        repaired_text = json.dumps(repaired, ensure_ascii=True, separators=(",", ":"))
        p = Path(image_id)
        aliases = {
            image_id,
            p.name,
            p.stem,
            f"{p.stem}.jpg",
            f"{p.stem}.png",
        }
        for key in aliases:
            if key and key not in out:
                out[key] = {"raw": raw_text, "repaired": repaired_text}

    return out


def infer_regime(gt_count: int) -> str:
    if gt_count < 30:
        return "A"
    if gt_count < 100:
        return "B"
    if gt_count < 200:
        return "C"
    return "X"


def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def point_to_anchor(point: List[float]) -> Optional[List[int]]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x_val = float(point[0])
        y_val = float(point[1])
    except Exception:
        return None

    # If already expressed in Rex 1..6 anchor coordinates, keep as-is.
    if 1.0 <= x_val <= 6.0 and 1.0 <= y_val <= 6.0:
        return [_clamp(int(round(x_val)), 1, 6), _clamp(int(round(y_val)), 1, 6)]

    # normalized_points_1000 convention: both coordinates in [0, 1000].
    row = _clamp(int(x_val / (1000.0 / 6.0)) + 1, 1, 6)
    col = _clamp(int(y_val / (1000.0 / 6.0)) + 1, 1, 6)
    return [row, col]


def build_anchor_clusters_from_points(points: List[List[float]]) -> List[dict]:
    counts: Dict[tuple[int, int], int] = {}
    for p in points:
        anchor = point_to_anchor(p)
        if anchor is None:
            continue
        key = (anchor[0], anchor[1])
        counts[key] = counts.get(key, 0) + 1

    clusters: List[dict] = []
    for row_col in sorted(counts.keys()):
        row, col = row_col
        cnt = int(counts[row_col])
        px = int(round((col / 7.0) * 448.0))
        py = int(round((row / 7.0) * 448.0))
        x1 = _clamp(px - 20, 0, 447)
        y1 = _clamp(py - 20, 0, 447)
        x2 = _clamp(px + 20, 0, 447)
        y2 = _clamp(py + 20, 0, 447)
        clusters.append(
            {
                "anchor": [row, col],
                "count": cnt,
                "region_bbox": [x1, y1, x2, y2],
            }
        )
    return clusters


def reconcile_cluster_total(clusters: List[dict], target_total: int) -> List[dict]:
    if not clusters or target_total <= 0:
        return clusters

    out: List[dict] = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        try:
            cc["count"] = max(1, int(cc.get("count", 1)))
        except Exception:
            cc["count"] = 1
        out.append(cc)

    if not out:
        return clusters

    cur_total = sum(int(c.get("count", 0)) for c in out)
    if cur_total == target_total:
        return out

    # Keep anchor coverage stable by adjusting only counts.
    if cur_total < target_total:
        delta = target_total - cur_total
        out[0]["count"] = int(out[0].get("count", 1)) + delta
        return out

    # cur_total > target_total: reduce counts but keep each cluster >= 1.
    delta = cur_total - target_total
    idx = len(out) - 1
    while delta > 0 and idx >= 0:
        c = out[idx]
        cnt = int(c.get("count", 1))
        reducible = max(0, cnt - 1)
        take = min(reducible, delta)
        if take > 0:
            c["count"] = cnt - take
            delta -= take
        idx -= 1

    # If still above target due to hard lower bound, collapse to one anchor.
    if delta > 0:
        out = [dict(out[0])]
        out[0]["count"] = target_total
    return out


def extract_total_count_from_payload(payload: Optional[dict]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    total = payload.get("total_count")
    if isinstance(total, int):
        return int(total)
    return None


def sum_cluster_counts(payload: Optional[dict]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return None
    total = 0
    seen = False
    for item in clusters:
        if isinstance(item, dict) and isinstance(item.get("count"), int):
            total += int(item["count"])
            seen = True
    return total if seen else None


def parse_count_from_text_or_payload(text: str, payload: Optional[dict]) -> Optional[int]:
    parsed_payload_count = extract_total_count_from_payload(payload)
    if parsed_payload_count is not None:
        return parsed_payload_count
    return parse_count(text)


def build_rex_json_answer(
    clusters: List[dict],
    total_count: int,
    clusters_first_json: bool,
) -> str:
    anchors = [f"({c['anchor'][0]},{c['anchor'][1]})" for c in clusters if isinstance(c, dict) and isinstance(c.get("anchor"), list)]
    anchors_summary = "Objects identified near coordinates " + ", ".join(anchors) + "." if anchors else "Objects identified near coordinates."

    if clusters_first_json:
        payload = {
            "clusters": clusters,
            "anchors_summary": anchors_summary,
            "total_count": int(total_count),
        }
    else:
        payload = {
            "total_count": int(total_count),
            "anchors_summary": anchors_summary,
            "clusters": clusters,
        }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def build_recursive_chosen_text(
    category: str,
    clusters: List[dict],
    total_count: int,
    add_thought_block: bool,
    clusters_first_json: bool,
    tally_augmented_thought: bool,
    max_tally_steps: int,
) -> str:
    json_answer = build_rex_json_answer(clusters=clusters, total_count=total_count, clusters_first_json=clusters_first_json)
    if not add_thought_block:
        return json_answer

    if tally_augmented_thought:
        capped_clusters = clusters[: max(1, int(max_tally_steps))]
        subtotal = 0
        steps = []
        for idx, c in enumerate(capped_clusters, start=1):
            anchor = c.get("anchor", [0, 0]) if isinstance(c, dict) else [0, 0]
            cnt = int(c.get("count", 0)) if isinstance(c, dict) else 0
            if cnt <= 0:
                continue
            subtotal += cnt
            steps.append(
                f"Step {idx}: anchor ({anchor[0]},{anchor[1]}) contributes {cnt}; subtotal={subtotal}."
            )

        if len(clusters) > len(capped_clusters):
            remaining = 0
            for c in clusters[len(capped_clusters) :]:
                if isinstance(c, dict) and isinstance(c.get("count"), int):
                    remaining += int(c["count"])
            if remaining > 0:
                subtotal += remaining
                steps.append(f"Final step: remaining anchors contribute {remaining}; subtotal={subtotal}.")

        if not steps:
            steps.append("Step 1: no reliable anchors found; fallback subtotal=0.")

        thought = (
            f"I will recursively tally {category} by anchor clusters and verify arithmetic before the final JSON. "
            + " ".join(steps)
            + f" Verified total_count={total_count}."
        )
    else:
        thought = (
            f"I will count {category} in three steps: (1) localize likely foreground regions, "
            "(2) aggregate per 6x6 anchor cluster, (3) verify that cluster counts sum to total_count."
        )

    return f"{THOUGHT_TAG}\n{thought}\n<|answer|>\n{json_answer}"


def build_strict_tally_thought(
    clusters: List[dict],
    category: str,
) -> tuple[str, int]:
    lines: List[str] = ["Thought:"]
    subtotal = 0

    for c in clusters:
        if not isinstance(c, dict):
            continue
        anchor = c.get("anchor")
        cnt = c.get("count")
        if not (isinstance(anchor, list) and len(anchor) == 2 and isinstance(cnt, int)):
            continue
        if int(cnt) <= 0:
            continue
        subtotal += int(cnt)
        ax = int(anchor[0])
        ay = int(anchor[1])
        noun = category if category else "objects"
        lines.append(f"Anchor ({ax},{ay}) contains {int(cnt)} {noun}. Subtotal: {subtotal}.")

    lines.append(f"Final Tally: {subtotal}.")
    return "\n".join(lines), subtotal


def build_tally_sft_response(
    category: str,
    clusters: List[dict],
    total_count: int,
) -> str:
    thought, subtotal = build_strict_tally_thought(clusters=clusters, category=category)
    if subtotal != int(total_count):
        raise ValueError(
            f"Tally subtotal mismatch: subtotal={subtotal} total_count={int(total_count)}"
        )
    json_answer = build_rex_json_answer(clusters=clusters, total_count=int(total_count), clusters_first_json=True)
    return f"{thought}\n{json_answer}"


def has_valid_json_format(text: str) -> int:
    t = str(text).lstrip()
    if t.startswith("{"):
        return 1
    if "<|answer|>" in t and "{" in t:
        return 1
    return 0


def composite_reward(
    text: str,
    gt_count: int,
    gt_anchor_set: Optional[set[tuple[int, int]]],
    w_format: float,
    w_consistency: float,
    w_accuracy: float,
    w_anchor_coverage: float,
    regime_c_boost: float,
    regime_c_accuracy_weight_mult: float,
    strict_anchor_penalty: float,
) -> Dict[str, float]:
    payload = robust_parse_json(str(text))
    pred_count = parse_count_from_text_or_payload(str(text), payload)

    r_format = float(has_valid_json_format(str(text)) and payload is not None)

    sum_clusters = sum_cluster_counts(payload)
    payload_total = extract_total_count_from_payload(payload)
    r_consistency = float(payload_total is not None and sum_clusters is not None and payload_total == sum_clusters)

    if pred_count is None or gt_count <= 0:
        r_accuracy = 0.0
    else:
        r_accuracy = max(0.0, 1.0 - (abs(float(gt_count) - float(pred_count)) / float(max(1, gt_count))))
        if infer_regime(gt_count) == "C":
            r_accuracy = min(1.0, r_accuracy * regime_c_boost)

    pred_anchor_set: set[tuple[int, int]] = set()
    clusters = payload.get("clusters") if isinstance(payload, dict) else None
    if isinstance(clusters, list):
        for item in clusters:
            anchor = item.get("anchor") if isinstance(item, dict) else None
            if isinstance(anchor, list) and len(anchor) == 2:
                try:
                    pred_anchor_set.add((int(anchor[0]), int(anchor[1])))
                except Exception:
                    continue

    if gt_anchor_set:
        overlap = len(gt_anchor_set.intersection(pred_anchor_set))
        r_anchor_coverage = float(overlap) / float(max(1, len(gt_anchor_set)))
        missing = len(gt_anchor_set) - overlap
    else:
        r_anchor_coverage = 1.0
        missing = 0

    effective_w_accuracy = float(w_accuracy)
    if infer_regime(gt_count) == "C":
        effective_w_accuracy *= float(regime_c_accuracy_weight_mult)

    total = (
        (w_format * r_format)
        + (w_consistency * r_consistency)
        + (effective_w_accuracy * r_accuracy)
        + (w_anchor_coverage * r_anchor_coverage)
    )
    if gt_count > 0 and missing > 0 and strict_anchor_penalty > 0.0:
        total -= float(strict_anchor_penalty)

    return {
        "r_format": r_format,
        "r_consistency": r_consistency,
        "r_accuracy": r_accuracy,
        "r_anchor_coverage": r_anchor_coverage,
        "missing_anchors": float(missing),
        "r_total": total,
    }


def build_record_aliases(image_id: str, image_path: str) -> List[str]:
    aliases: List[str] = []

    for raw in [image_id, image_path]:
        if not raw:
            continue
        p = Path(str(raw))
        aliases.extend([str(raw), p.name, p.stem, f"{p.stem}.jpg", f"{p.stem}.png"])

    # Keep order while dropping empties/duplicates.
    dedup = []
    seen = set()
    for item in aliases:
        if not item or item in seen:
            continue
        dedup.append(item)
        seen.add(item)
    return dedup


def build_synthetic_overcount(points: List[List[int]], factor: float = 1.6) -> List[List[int]]:
    if not points:
        return []

    out = [list(p) for p in points]
    target = max(len(points) + 1, int(round(len(points) * factor)))

    idx = 0
    while len(out) < target:
        x_coord, y_coord = out[idx % len(points)]
        # Deterministic micro-jitter to avoid exact duplicates.
        jitter_x = ((idx * 7) % 9) - 4
        jitter_y = ((idx * 11) % 9) - 4
        new_x = max(0, min(1000, x_coord + jitter_x))
        new_y = max(0, min(1000, y_coord + jitter_y))
        out.append([new_x, new_y])
        idx += 1

    return out


def build_close_overcount(points: List[List[int]], gt_count: int, close_ratio: float) -> List[List[int]]:
    if not points:
        return []
    target = max(gt_count + 1, int(round(gt_count * (1.0 + close_ratio * 0.5))))
    factor = max(1.05, float(target) / max(1, gt_count))
    return build_synthetic_overcount(points, factor=factor)


def choose_audit_path(user_path: str) -> str:
    if user_path and os.path.exists(user_path):
        return user_path

    candidates = [
        "checkpoints/native_sft_stage1_r64_lr2e4/zero_shot_audit_final.json",
        "checkpoints/native_sft_stage1_r64_lr2e4/zero_shot_point_audit_step100.json",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return user_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["rankdpo", "tally_sft"],
        default="rankdpo",
        help="Generation mode: rankdpo (preference pairs) or tally_sft (supervised Stage 1.6).",
    )
    parser.add_argument(
        "--scaffold_jsonl",
        default="outputs/fsc147_scaffold_full/all.jsonl",
        help="Scaffold dataset JSONL with problem/solution/image/id fields.",
    )
    parser.add_argument(
        "--audit_json",
        default="",
        help="Stage-1 audit json containing overcounted predictions.",
    )
    parser.add_argument(
        "--output_jsonl",
        default="outputs/rankdpo/preference_pairs.jsonl",
        help="Destination JSONL for DPO training.",
    )
    parser.add_argument(
        "--allow_synthetic_overcount_fallback",
        action="store_true",
        help="If audit does not have an image, synthesize an overcounted rejected sample from GT points.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of scaffold examples to convert (0 means all).",
    )
    parser.add_argument(
        "--close_ratio",
        type=float,
        default=0.20,
        help="Relative threshold for rank-2 close overcount band.",
    )
    parser.add_argument(
        "--emit_rank_triplets",
        action="store_true",
        help="Emit RankDPO triplet-derived pairs in addition to standard GT-vs-audit pairs.",
    )
    parser.add_argument(
        "--emit_lq_prose",
        action="store_true",
        help="Emit extra low-quality prose negatives.",
    )
    parser.add_argument(
        "--structural_audit_json",
        default="",
        help="Optional audit JSON used to mine structurally malformed outputs as hard negatives.",
    )
    parser.add_argument(
        "--emit_structural_repair_pairs",
        action="store_true",
        help="Emit repaired-vs-raw structural pairs from structural_audit_json.",
    )
    parser.add_argument(
        "--emit_recursive_rex_pairs",
        action="store_true",
        help="Emit Stage-3.0 recursive RexOmni-style pairs using weighted composite rewards.",
    )
    parser.add_argument(
        "--clusters_first_json",
        action="store_true",
        help="When emitting recursive pairs, serialize chosen JSON with clusters before total_count.",
    )
    parser.add_argument(
        "--add_thought_block",
        action="store_true",
        help="When emitting recursive pairs, prepend a short thought block before JSON answer.",
    )
    parser.add_argument(
        "--w_format",
        type=float,
        default=0.35,
        help="Weight for format reward component.",
    )
    parser.add_argument(
        "--w_consistency",
        type=float,
        default=0.35,
        help="Weight for cluster-sum consistency reward component.",
    )
    parser.add_argument(
        "--w_accuracy",
        type=float,
        default=0.30,
        help="Weight for counting accuracy reward component.",
    )
    parser.add_argument(
        "--w_anchor_coverage",
        type=float,
        default=0.55,
        help="Weight for anchor-coverage reward component (predicted anchors vs GT anchors).",
    )
    parser.add_argument(
        "--regime_c_boost",
        type=float,
        default=1.25,
        help="Accuracy multiplier for Regime-C examples (clamped to 1.0 final accuracy).",
    )
    parser.add_argument(
        "--regime_c_accuracy_weight_mult",
        type=float,
        default=1.80,
        help="Additional multiplier applied to accuracy weight for Regime-C examples.",
    )
    parser.add_argument(
        "--strict_anchor_penalty",
        type=float,
        default=0.75,
        help="Penalty subtracted when any GT-positive anchor is missing from predicted clusters.",
    )
    parser.add_argument(
        "--tally_augmented_thought",
        action="store_true",
        help="When used with --add_thought_block, emit explicit step-by-step anchor tally in thought.",
    )
    parser.add_argument(
        "--max_tally_steps",
        type=int,
        default=24,
        help="Maximum number of explicit anchor tally steps in thought block.",
    )
    parser.add_argument(
        "--min_reward_margin",
        type=float,
        default=0.05,
        help="Minimum reward(chosen)-reward(rejected) required to emit recursive pair.",
    )
    parser.add_argument(
        "--fail_on_tally_mismatch",
        type=int,
        default=1,
        help="When mode=tally_sft, fail immediately if subtotal arithmetic mismatches total_count.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.scaffold_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    if args.mode == "tally_sft":
        out_rows: List[dict] = []
        tally_errors = 0
        skipped = 0

        for record in records:
            image_id = str(record.get("id", os.path.basename(record.get("image", ""))))
            prompt = str(record.get("problem", "")).strip()
            image_path = str(record.get("image", ""))
            category = str(record.get("category", extract_category_from_prompt(prompt))).lower()

            clusters_raw = record.get("clusters", [])
            clusters: List[dict] = []
            if isinstance(clusters_raw, list) and clusters_raw:
                for item in clusters_raw:
                    if not isinstance(item, dict):
                        continue
                    anchor = item.get("anchor")
                    count = item.get("count")
                    if not (isinstance(anchor, list) and len(anchor) == 2 and isinstance(count, int)):
                        continue
                    if int(count) <= 0:
                        continue
                    out_item = {
                        "anchor": [int(anchor[0]), int(anchor[1])],
                        "count": int(count),
                    }
                    if isinstance(item.get("region_bbox"), list) and len(item["region_bbox"]) == 4:
                        out_item["region_bbox"] = [int(v) for v in item["region_bbox"]]
                    clusters.append(out_item)

            if not clusters:
                gt_points = record.get("normalized_points_1000", [])
                if not isinstance(gt_points, list):
                    gt_points = []
                if not gt_points:
                    gt_points = parse_points(str(record.get("solution", "")))
                clusters = build_anchor_clusters_from_points(gt_points)

            if not clusters:
                skipped += 1
                continue

            clusters = reconcile_cluster_total(
                clusters,
                target_total=sum(int(c.get("count", 0)) for c in clusters if isinstance(c, dict)),
            )

            gt_count = int(record.get("ground_truth_count", 0)) if isinstance(record.get("ground_truth_count"), int) else 0
            if gt_count <= 0:
                gt_count = int(record.get("count", 0)) if isinstance(record.get("count"), int) else 0
            if gt_count <= 0:
                parsed_gt = parse_count(str(record.get("solution", "")))
                gt_count = int(parsed_gt) if parsed_gt is not None else 0
            if gt_count <= 0:
                gt_count = sum(int(c.get("count", 0)) for c in clusters if isinstance(c, dict))

            try:
                assistant_text = build_tally_sft_response(
                    category=category,
                    clusters=clusters,
                    total_count=gt_count,
                )
            except Exception:
                tally_errors += 1
                if int(args.fail_on_tally_mismatch) == 1:
                    raise
                skipped += 1
                continue

            out_record = dict(record)
            out_record["id"] = image_id
            out_record["image"] = image_path
            out_record["problem"] = prompt
            out_record["solution"] = assistant_text
            out_record["stage"] = "stage16_tally_sft"

            conv = out_record.get("conversations")
            if isinstance(conv, list) and len(conv) >= 2 and isinstance(conv[0], dict) and isinstance(conv[1], dict):
                new_conv = [dict(c) for c in conv]
                user_prompt = str(new_conv[0].get("value", prompt))
                if "Assistant: Thought:" not in user_prompt:
                    user_prompt = user_prompt.rstrip() + "\nAssistant: Thought:"
                new_conv[0]["value"] = user_prompt
                new_conv[0]["from"] = str(new_conv[0].get("from", "human"))
                new_conv[1]["value"] = assistant_text
                new_conv[1]["from"] = str(new_conv[1].get("from", "gpt"))
                out_record["conversations"] = new_conv
            else:
                out_record["conversations"] = [
                    {"from": "human", "value": (prompt.rstrip() + "\nAssistant: Thought:") if prompt else "<image>\nAssistant: Thought:"},
                    {"from": "gpt", "value": assistant_text},
                ]

            out_rows.append(out_record)

        write_jsonl(args.output_jsonl, out_rows)
        print(f"Mode: tally_sft")
        print(f"Input scaffold records: {len(records)}")
        print(f"Output supervised rows: {len(out_rows)}")
        print(f"Skipped rows: {skipped}")
        print(f"Tally mismatch errors: {tally_errors}")
        print(f"Wrote: {args.output_jsonl}")
        return

    audit_path = choose_audit_path(args.audit_json)
    audit_predictions = load_audit_predictions(audit_path) if os.path.exists(audit_path) else {}
    structural_negatives: Dict[str, Dict[str, str]] = {}
    if args.emit_structural_repair_pairs and args.structural_audit_json and os.path.exists(args.structural_audit_json):
        structural_negatives = load_structural_negatives(args.structural_audit_json)

    out_rows: List[dict] = []
    missing_audit = 0
    used_audit = 0
    used_synth = 0
    emitted_triplet_pairs = 0
    emitted_structural_pairs = 0
    emitted_recursive_pairs = 0
    dropped_recursive_pairs = 0

    unmatched_examples: List[str] = []

    for record in records:
        image_id = str(record.get("id", os.path.basename(record.get("image", ""))))
        prompt = str(record["problem"])
        chosen = str(record["solution"])
        image_path = str(record["image"])
        category = str(record.get("category", extract_category_from_prompt(prompt))).lower()

        # Rank-3 (worst): exact repetitive overcount string from the audit file.
        audit_pred = None
        for alias in build_record_aliases(image_id, image_path):
            audit_pred = audit_predictions.get(alias)
            if audit_pred is not None:
                break
        if audit_pred and audit_pred.prediction_text:
            rejected_hq = audit_pred.prediction_text
            used_audit += 1
        elif args.allow_synthetic_overcount_fallback:
            gt_points = record.get("normalized_points_1000", [])
            synth_points = build_synthetic_overcount(gt_points)
            synth_count = len(synth_points)
            rejected_hq = normalize_scaffold_text(synth_points, synth_count, category)
            used_synth += 1
            missing_audit += 1
            if len(unmatched_examples) < 8:
                unmatched_examples.append(f"id={image_id} image={os.path.basename(image_path)}")
        else:
            missing_audit += 1
            continue

        base = {
            "id": image_id,
            "prompt": prompt,
            "chosen": chosen,
            "image": image_path,
        }

        # Standard DPO pair: GT (chosen) vs exact raw repetitive audit text (rejected).
        out_rows.append({**base, "rejected": rejected_hq, "pair_type": "audit_raw_repetition"})

        # Optional low-quality prose negative.
        if args.emit_lq_prose:
            rejected_lq = (
                f"The image shows {category} in a natural scene. "
                "There appear to be multiple instances spread across the frame. "
                "I cannot provide exact grounded coordinates in the required format."
            )
            out_rows.append({**base, "rejected": rejected_lq, "pair_type": "lq_prose"})

        # Optional RankDPO triplet converted into pairwise preferences.
        if args.emit_rank_triplets:
            gt_count = parse_count(chosen)
            if gt_count is None:
                gt_count = int(record.get("count", 0))

            rank3_text = rejected_hq
            rank2_text: Optional[str] = None

            # Prefer rank-2 from audit if available and close enough.
            if audit_pred and isinstance(audit_pred.pred_count, int) and gt_count > 0:
                rel = abs(audit_pred.pred_count - gt_count) / float(gt_count)
                if rel <= args.close_ratio and audit_pred.pred_count >= gt_count:
                    rank2_text = audit_pred.prediction_text

            # Otherwise synthesize a close overcount as rank-2.
            if rank2_text is None:
                gt_points = record.get("normalized_points_1000", [])
                close_points = build_close_overcount(gt_points, gt_count=max(gt_count, 1), close_ratio=args.close_ratio)
                if close_points:
                    rank2_text = normalize_scaffold_text(close_points, len(close_points), category)

            if rank2_text:
                out_rows.append(
                    {
                        **base,
                        "chosen": chosen,
                        "rejected": rank2_text,
                        "pair_type": "rank1_gt_vs_rank2_close",
                    }
                )
                out_rows.append(
                    {
                        **base,
                        "chosen": rank2_text,
                        "rejected": rank3_text,
                        "pair_type": "rank2_close_vs_rank3_worst",
                    }
                )
                emitted_triplet_pairs += 2

        # Optional structural hard-negative pair: repaired valid JSON (chosen)
        # vs raw malformed JSON (rejected).
        if args.emit_structural_repair_pairs and structural_negatives:
            structural_item = None
            for alias in build_record_aliases(image_id, image_path):
                structural_item = structural_negatives.get(alias)
                if structural_item is not None:
                    break
            if structural_item is not None:
                out_rows.append(
                    {
                        **base,
                        "chosen": structural_item["repaired"],
                        "rejected": structural_item["raw"],
                        "pair_type": "structural_repair_from_raw",
                    }
                )
                emitted_structural_pairs += 1

        if args.emit_recursive_rex_pairs:
            gt_points = record.get("normalized_points_1000", [])
            if not isinstance(gt_points, list):
                gt_points = []
            if not gt_points:
                gt_points = parse_points(chosen)
            gt_count = int(record.get("count", 0))
            if gt_count <= 0:
                parsed_gt = parse_count(chosen)
                gt_count = int(parsed_gt) if parsed_gt is not None else 0
            if gt_count <= 0 and isinstance(record.get("ground_truth_count"), int):
                gt_count = int(record.get("ground_truth_count"))

            clusters = build_anchor_clusters_from_points(gt_points)
            if gt_count <= 0 and clusters:
                gt_count = sum(int(c.get("count", 0)) for c in clusters if isinstance(c, dict))

            if clusters and gt_count > 0:
                clusters = reconcile_cluster_total(clusters, gt_count)

            if clusters and gt_count > 0 and audit_pred is not None and audit_pred.prediction_text:
                gt_anchor_set = {
                    (int(c["anchor"][0]), int(c["anchor"][1]))
                    for c in clusters
                    if isinstance(c, dict)
                    and isinstance(c.get("anchor"), list)
                    and len(c["anchor"]) == 2
                    and isinstance(c.get("count"), int)
                    and int(c.get("count", 0)) > 0
                }
                chosen_recursive = build_recursive_chosen_text(
                    category=category,
                    clusters=clusters,
                    total_count=gt_count,
                    add_thought_block=bool(args.add_thought_block),
                    clusters_first_json=bool(args.clusters_first_json),
                    tally_augmented_thought=bool(args.tally_augmented_thought),
                    max_tally_steps=int(args.max_tally_steps),
                )
                rejected_recursive = str(audit_pred.prediction_text)

                rew_chosen = composite_reward(
                    chosen_recursive,
                    gt_count=gt_count,
                    gt_anchor_set=gt_anchor_set,
                    w_format=args.w_format,
                    w_consistency=args.w_consistency,
                    w_accuracy=args.w_accuracy,
                    w_anchor_coverage=args.w_anchor_coverage,
                    regime_c_boost=args.regime_c_boost,
                    regime_c_accuracy_weight_mult=args.regime_c_accuracy_weight_mult,
                    strict_anchor_penalty=args.strict_anchor_penalty,
                )
                rew_rejected = composite_reward(
                    rejected_recursive,
                    gt_count=gt_count,
                    gt_anchor_set=gt_anchor_set,
                    w_format=args.w_format,
                    w_consistency=args.w_consistency,
                    w_accuracy=args.w_accuracy,
                    w_anchor_coverage=args.w_anchor_coverage,
                    regime_c_boost=args.regime_c_boost,
                    regime_c_accuracy_weight_mult=args.regime_c_accuracy_weight_mult,
                    strict_anchor_penalty=args.strict_anchor_penalty,
                )
                margin = float(rew_chosen["r_total"] - rew_rejected["r_total"])

                if margin >= args.min_reward_margin:
                    out_rows.append(
                        {
                            **base,
                            "chosen": chosen_recursive,
                            "rejected": rejected_recursive,
                            "pair_type": "recursive_rex_weighted",
                            "reward_chosen": rew_chosen,
                            "reward_rejected": rew_rejected,
                            "reward_margin": margin,
                            "regime": infer_regime(gt_count),
                        }
                    )
                    emitted_recursive_pairs += 1
                else:
                    dropped_recursive_pairs += 1

    write_jsonl(args.output_jsonl, out_rows)

    rank0_focus = {"2191.jpg", "4071.jpg"}
    focus_hits = [r for r in out_rows if r.get("id") in rank0_focus and r.get("pair_type") == "audit_raw_repetition"]

    print(f"Input scaffold records: {len(records)}")
    print(f"Output preference rows: {len(out_rows)}")
    print(f"Used audit overcount rows: {used_audit}")
    print(f"Used synthetic overcount fallback: {used_synth}")
    print(f"Missing audit rows: {missing_audit}")
    coverage = (100.0 * used_audit / max(1, len(records)))
    print(f"Audit coverage over scaffold: {coverage:.2f}%")
    print(f"Triplet-derived pair rows: {emitted_triplet_pairs}")
    print(f"Structural repair pair rows: {emitted_structural_pairs}")
    print(f"Recursive weighted pair rows: {emitted_recursive_pairs}")
    print(f"Dropped recursive rows (low margin): {dropped_recursive_pairs}")
    print(f"Audit path used: {audit_path if audit_path else 'NONE'}")
    print(f"Focus audit pairs present (2191/4071): {len(focus_hits)}")
    if unmatched_examples:
        print("Sample unmatched scaffold rows:")
        for item in unmatched_examples:
            print(f"  - {item}")
    print(f"Wrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
