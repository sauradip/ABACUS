#!/usr/bin/env python3
"""Build a composite-image conversations SFT dataset from adaptive scaffold prompts.

This converts rows with two image paths (original + scaffolded) into a single
side-by-side composite image and merges the system guidance plus counting
question into one human turn. Ground-truth counts come from the master JSONL
when available, with a fallback to FSC147 annotations.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


DEFAULT_PROMPT_JSONL = "outputs/scaffold_prompt_fsc147_train_adaptive/train_scaffold_input_only.jsonl"
DEFAULT_MASTER_JSONL = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
DEFAULT_FSC147_ANNOTATIONS = "/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json"
DEFAULT_OUTPUT_DIR = "outputs/adaptive_sft_fsc147_train"
DEFAULT_OUTPUT_JSONL = "outputs/adaptive_sft_fsc147_train/adaptive_sft_fsc147_train_conversations.jsonl"
DEFAULT_STAGE = "stage16b_adaptive_scaffold_fsc147_sft"

SOURCE_PROMPT = (
    "I will provide you with two images of the same scene. "
    "The first image is the original scene. The second image is overlaid"
)
TARGET_PROMPT = (
    "I will provide you with two views of the same scene in one image. "
    "The left side is the original scene. The right side is overlaid"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
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
    return Path(raw).stem or Path(raw).name


def candidate_ids(row: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("id", "question_id", "image", "source_image"):
        value = row.get(key)
        if value is None:
            continue
        values.append(str(value))
        normalized = normalize_id(value)
        if normalized:
            values.append(normalized)
    out: List[str] = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def parse_count_from_solution(row: Dict[str, Any]) -> Optional[int]:
    for key in ("gt_count", "ground_truth_count", "total_count"):
        if row.get(key) is not None:
            return int(row[key])

    solution = row.get("solution") or row.get("response") or row.get("target_response")
    if isinstance(solution, dict) and solution.get("total_count") is not None:
        return int(solution["total_count"])
    if isinstance(solution, str):
        try:
            parsed = json.loads(solution)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("total_count") is not None:
            return int(parsed["total_count"])
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
    return mapping


def load_fsc147_annotation_mapping(annotation_json: Path) -> Dict[str, int]:
    payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected annotation dict at {annotation_json}")
    mapping: Dict[str, int] = {}
    for image_name, item in payload.items():
        if not isinstance(item, dict):
            continue
        count = len(item.get("points", []))
        mapping[str(image_name)] = count
        mapping[normalize_id(image_name)] = count
    return mapping


def extract_history_text(row: Dict[str, Any]) -> str:
    history = row.get("history") or []
    if not history:
        return ""
    first = history[0]
    content = first.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def adapt_prompt_text(system_prompt: str, question_text: str) -> str:
    adapted = system_prompt.replace(SOURCE_PROMPT, TARGET_PROMPT)
    return f"<image>\n{adapted}\n\n{question_text.strip()}"


def extract_category(question_text: str) -> str:
    match = re.search(r"Count the (.*?) in the scene", question_text)
    if match:
        return match.group(1).strip()
    return "objects"


def create_composite_image(original_path: Path, scaffold_path: Path, output_path: Path) -> None:
    with Image.open(original_path).convert("RGB") as original_img, Image.open(scaffold_path).convert("RGB") as scaffold_img:
        if original_img.height != scaffold_img.height:
            aspect_ratio = scaffold_img.width / scaffold_img.height
            new_width = max(1, int(original_img.height * aspect_ratio))
            scaffold_img = scaffold_img.resize((new_width, original_img.height), Image.Resampling.LANCZOS)

        composite = Image.new("RGB", (original_img.width + scaffold_img.width, original_img.height))
        composite.paste(original_img, (0, 0))
        composite.paste(scaffold_img, (original_img.width, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite.save(output_path, quality=95)


def validate_row(row: Dict[str, Any]) -> None:
    if not row.get("id"):
        raise ValueError("Composite row is missing id")
    image_path = Path(str(row.get("image", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"Missing composite image: {image_path}")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"Row {row['id']} must have two conversation turns")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise ValueError(f"Row {row['id']} must use human/gpt turns")
    human_value = str(conversations[0].get("value", ""))
    if not human_value.startswith("<image>\n"):
        raise ValueError(f"Row {row['id']} human prompt must start with <image>")
    if SOURCE_PROMPT in human_value:
        raise ValueError(f"Row {row['id']} still contains the old two-image phrasing")
    payload = json.loads(str(conversations[1].get("value", "")))
    if sorted(payload.keys()) != ["total_count"] or not isinstance(payload["total_count"], int):
        raise ValueError(f"Row {row['id']} gpt value must be strict total_count JSON")


def build_dataset(
    prompt_jsonl: Path,
    master_jsonl: Path,
    fsc147_annotations: Optional[Path],
    output_jsonl: Path,
    composite_dir: Path,
    stage_name: str,
    limit: int,
    missing_gt_policy: str,
) -> Tuple[int, int]:
    if not prompt_jsonl.exists():
        raise FileNotFoundError(prompt_jsonl)

    gt_mapping: Dict[str, int] = {}
    if master_jsonl.exists():
        gt_mapping.update(load_gt_mapping(master_jsonl))
    if fsc147_annotations is not None and fsc147_annotations.exists():
        gt_mapping.update(load_fsc147_annotation_mapping(fsc147_annotations))
    if not gt_mapping:
        raise RuntimeError("No ground-truth mapping available")

    prompt_rows = load_jsonl(prompt_jsonl)
    if limit > 0:
        prompt_rows = prompt_rows[:limit]

    out_rows: List[Dict[str, Any]] = []
    missing_gt: List[str] = []
    for index, source in enumerate(prompt_rows, start=1):
        row_id = normalize_id(source.get("question_id") or source.get("id"))
        if not row_id:
            raise ValueError(f"Prompt row at index {index} is missing question_id/id")

        gt_count = gt_mapping.get(row_id)
        if gt_count is None:
            missing_gt.append(row_id)
            continue

        image_paths = source.get("image_paths")
        if not isinstance(image_paths, list) or len(image_paths) != 2:
            raise ValueError(f"Prompt row {row_id} must have exactly two image paths")
        original_path = Path(str(image_paths[0]))
        scaffold_path = Path(str(image_paths[1]))
        if not original_path.exists() or not scaffold_path.exists():
            raise FileNotFoundError(f"Missing source images for row {row_id}")

        composite_path = composite_dir / f"composite_{row_id}.jpg"
        create_composite_image(original_path, scaffold_path, composite_path)

        question_text = str(source.get("question", "")).strip()
        system_prompt = extract_history_text(source)
        out_row = {
            "id": row_id,
            "image": str(composite_path),
            "category": extract_category(question_text),
            "conversations": [
                {
                    "from": "human",
                    "value": adapt_prompt_text(system_prompt, question_text),
                },
                {
                    "from": "gpt",
                    "value": json.dumps({"total_count": int(gt_count)}, separators=(",", ":")),
                },
            ],
            "stage": stage_name,
        }
        validate_row(out_row)
        out_rows.append(out_row)

        if index == len(prompt_rows) or index % 100 == 0:
            print(f"processed={index}/{len(prompt_rows)} written={len(out_rows)}", flush=True)

    if missing_gt:
        if missing_gt_policy == "error":
            raise KeyError(f"Missing GT for ids: {missing_gt[:20]}")
        print(f"WARNING: skipped {len(missing_gt)} rows with missing GT: {missing_gt[:20]}", flush=True)

    written = write_jsonl(output_jsonl, out_rows)
    return len(prompt_rows), written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_jsonl", default=DEFAULT_PROMPT_JSONL)
    parser.add_argument("--master_jsonl", default=DEFAULT_MASTER_JSONL)
    parser.add_argument("--fsc147_annotations", default=DEFAULT_FSC147_ANNOTATIONS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--stage_name", default=DEFAULT_STAGE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--missing_gt_policy", choices=["error", "skip"], default="error")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    prompt_path = Path(args.prompt_jsonl).resolve()
    composite_dir = output_dir / "composite_images"
    source_rows, written = build_dataset(
        prompt_jsonl=prompt_path,
        master_jsonl=Path(args.master_jsonl).resolve(),
        fsc147_annotations=Path(args.fsc147_annotations).resolve() if args.fsc147_annotations else None,
        output_jsonl=output_jsonl,
        composite_dir=composite_dir,
        stage_name=args.stage_name,
        limit=args.limit,
        missing_gt_policy=args.missing_gt_policy,
    )
    print(f"source_rows={source_rows}")
    print(f"written_rows={written}")
    print(f"output_jsonl={output_jsonl}")


if __name__ == "__main__":
    main()