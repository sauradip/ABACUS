#!/usr/bin/env python3
"""Build a HF-style multi-image count-only SFT JSONL from adaptive scaffold prompts.

This promotes the prompt-only adaptive scaffold dry run into a supervised SFT
format while keeping the model input clean: two image entries plus text, and a
count-only assistant response. Raw FSC points, clusters, boxes, and scaffold
coordinates are not written to the output.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROMPT_JSONL = "outputs/scaffold_prompt_fsc147_train_adaptive/train_scaffold_input_only.jsonl"
DEFAULT_MASTER_JSONL = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
DEFAULT_OUTPUT_JSONL = "outputs/adaptive_hf_multi_image_count_sft_fsc147_train/train_messages.jsonl"
DEFAULT_FSC147_ANNOTATIONS = "/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def normalize_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    base = Path(raw).name
    stem = Path(base).stem
    return stem or base


def candidate_ids(row: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in ("question_id", "id", "image", "source_image"):
        value = row.get(key)
        if value:
            candidates.append(str(value))
            norm = normalize_id(value)
            if norm:
                candidates.append(norm)
    out: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_count_from_solution(row: Dict[str, Any]) -> Optional[int]:
    for key in ("gt_count", "ground_truth_count", "total_count"):
        if row.get(key) is not None:
            return int(row[key])

    solution = row.get("solution") or row.get("response") or row.get("target_response")
    if not solution:
        return None
    if isinstance(solution, dict) and solution.get("total_count") is not None:
        return int(solution["total_count"])
    if isinstance(solution, str):
        try:
            parsed = json.loads(solution)
            if isinstance(parsed, dict) and parsed.get("total_count") is not None:
                return int(parsed["total_count"])
        except Exception:
            pass
        match = re.search(r'"total_count"\s*:\s*(-?\d+)', solution)
        if match:
            return int(match.group(1))
    return None


def load_gt_mapping(master_jsonl: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for row in load_jsonl(master_jsonl):
        gt_count = parse_count_from_solution(row)
        if gt_count is None:
            continue
        for key in candidate_ids(row):
            mapping[key] = gt_count
            norm = normalize_id(key)
            if norm:
                mapping[norm] = gt_count
    return mapping


def load_fsc147_annotation_mapping(annotation_json: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not annotation_json.exists():
        raise FileNotFoundError(annotation_json)
    with annotation_json.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected FSC147 annotation dict at {annotation_json}")
    for image_name, payload in annotations.items():
        if not isinstance(payload, dict):
            continue
        count = len(payload.get("points", []))
        mapping[str(image_name)] = count
        norm = normalize_id(image_name)
        if norm:
            mapping[norm] = count
    return mapping


def history_text(row: Dict[str, Any]) -> str:
    history = row.get("history") or []
    if not history:
        return ""
    try:
        content = history[0]["content"]
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part).strip()
        if isinstance(content, str):
            return content.strip()
    except Exception:
        return ""
    return ""


def build_messages(row: Dict[str, Any], gt_count: int) -> List[Dict[str, Any]]:
    image_paths = row.get("image_paths")
    if not isinstance(image_paths, list) or len(image_paths) != 2:
        raise ValueError(f"Row {row.get('question_id')} must have exactly two image_paths")

    prompt = "\n\n".join(part for part in (history_text(row), str(row.get("question", "")).strip()) if part)
    if "<image>" in prompt:
        raise ValueError(f"Row {row.get('question_id')} text contains hardcoded <image> token")

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_paths[0])},
                {"type": "image", "url": str(image_paths[1])},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"total_count": int(gt_count)}, separators=(",", ":")),
                }
            ],
        },
    ]


def validate_output_row(row: Dict[str, Any], strict_images: bool) -> None:
    forbidden = {
        "annotations",
        "clusters",
        "gt_count",
        "ground_truth",
        "ground_truth_count",
        "points",
        "scaffold_points",
        "target_response",
        "total_count",
    }
    present = sorted(forbidden.intersection(row.keys()))
    if present:
        raise ValueError(f"Output row has forbidden top-level supervision fields: {present}")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Row {row.get('id')} must have exactly two messages")
    user_content = messages[0].get("content", [])
    image_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "image"]
    if len(image_items) != 2:
        raise ValueError(f"Row {row.get('id')} must have exactly two image content items")
    for item in image_items:
        if "url" not in item:
            raise ValueError(f"Row {row.get('id')} image item must use url")
        if strict_images and not Path(str(item["url"])).exists():
            raise FileNotFoundError(f"Missing image for row {row.get('id')}: {item['url']}")
    text_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "text"]
    if len(text_items) != 1 or "<image>" in text_items[0].get("text", ""):
        raise ValueError(f"Row {row.get('id')} must have one text item and no hardcoded image tokens")

    assistant_content = messages[1].get("content", [])
    if len(assistant_content) != 1 or assistant_content[0].get("type") != "text":
        raise ValueError(f"Row {row.get('id')} assistant response must be one text item")
    payload = json.loads(assistant_content[0]["text"])
    if sorted(payload.keys()) != ["total_count"] or not isinstance(payload["total_count"], int):
        raise ValueError(f"Row {row.get('id')} assistant text must be count-only JSON")


def build_dataset(
    prompt_jsonl: Path,
    master_jsonl: Path,
    fsc147_annotations: Optional[Path],
    output_jsonl: Path,
    limit: int,
    strict_images: bool,
    missing_gt_policy: str,
) -> Tuple[int, int]:
    if not prompt_jsonl.exists():
        raise FileNotFoundError(prompt_jsonl)

    gt_mapping: Dict[str, int] = {}
    if master_jsonl.exists():
        gt_mapping.update(load_gt_mapping(master_jsonl))
    elif fsc147_annotations is None:
        raise FileNotFoundError(master_jsonl)

    if fsc147_annotations is not None:
        # Prefer the original FSC147 annotations when explicitly requested.
        gt_mapping.update(load_fsc147_annotation_mapping(fsc147_annotations))
    if not gt_mapping:
        raise RuntimeError(f"No ground-truth counts could be loaded from {master_jsonl}")

    rows: List[Dict[str, Any]] = []
    prompt_rows = load_jsonl(prompt_jsonl)
    if limit > 0:
        prompt_rows = prompt_rows[:limit]

    missing_gt: List[str] = []
    for source in prompt_rows:
        row_id = normalize_id(source.get("question_id") or source.get("id"))
        gt_count = gt_mapping.get(row_id)
        if gt_count is None:
            missing_gt.append(row_id)
            continue
        out_row = {
            "id": row_id,
            "messages": build_messages(source, gt_count),
            "stage": "adaptive_hf_multi_image_count_sft",
        }
        validate_output_row(out_row, strict_images=strict_images)
        rows.append(out_row)

    if missing_gt:
        if missing_gt_policy == "error":
            raise KeyError(f"Missing GT for ids: {missing_gt[:20]}")
        print(f"WARNING: skipped {len(missing_gt)} rows with missing GT: {missing_gt[:20]}")

    written = write_jsonl(output_jsonl, rows)
    return len(prompt_rows), written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_jsonl", default=DEFAULT_PROMPT_JSONL)
    parser.add_argument("--master_jsonl", default=DEFAULT_MASTER_JSONL)
    parser.add_argument("--fsc147_annotations", default=None)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap for dry runs")
    parser.add_argument("--strict_images", type=int, default=1)
    parser.add_argument("--missing_gt_policy", choices=["error", "skip"], default="error")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_count, written = build_dataset(
        prompt_jsonl=Path(args.prompt_jsonl).resolve(),
        master_jsonl=Path(args.master_jsonl).resolve(),
        fsc147_annotations=Path(args.fsc147_annotations).resolve() if args.fsc147_annotations else None,
        output_jsonl=Path(args.output_jsonl).resolve(),
        limit=args.limit,
        strict_images=bool(args.strict_images),
        missing_gt_policy=args.missing_gt_policy,
    )
    print(f"source_rows={source_count}")
    print(f"written_rows={written}")
    print(f"output_jsonl={Path(args.output_jsonl).resolve()}")


if __name__ == "__main__":
    main()
