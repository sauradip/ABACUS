#!/usr/bin/env python3
"""Preflight validation for Stage 1.5 SFT datasets (PCA or RGB).

Checks:
- JSONL/JSON readability
- expected row count (optional)
- image path resolution via PCA fallback fields
- file existence
- PIL readability

Exits non-zero when strict mode is enabled and any hard issue is found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


@dataclass
class CheckResult:
    total_rows: int = 0
    readable_rows: int = 0
    missing_path_rows: int = 0
    missing_file_rows: int = 0
    unreadable_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--use_pca_images", type=int, default=1)
    parser.add_argument("--image_field", default="image")
    parser.add_argument(
        "--pca_image_fields",
        default="pca_image,image_pca,composite_image,dino_pca_image,image",
    )
    parser.add_argument("--expected_rows", type=int, default=4945)
    parser.add_argument("--strict", type=int, default=1)
    parser.add_argument("--max_report", type=int, default=20)
    parser.add_argument("--check_readable", type=int, default=1)
    return parser.parse_args()


def parse_field_list(raw_fields: str) -> List[str]:
    return [field.strip() for field in raw_fields.split(",") if field.strip()]


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {path}")
            return data
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return rows


def resolve_row_image_path(
    row: Dict[str, Any],
    data_dir: str,
    image_field: str,
    use_pca_images: bool,
    pca_image_fields: List[str],
) -> Optional[str]:
    fields: List[str] = []
    if use_pca_images:
        fields.extend(pca_image_fields)
        if image_field not in fields:
            fields.append(image_field)
    else:
        fields.append(image_field)

    for field in fields:
        value = row.get(field)
        if not value:
            continue
        path = str(value)
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(data_dir, path))
    return None


def batched_preview(items: Iterable[str], max_items: int) -> List[str]:
    out: List[str] = []
    for idx, item in enumerate(items):
        if idx >= max_items:
            break
        out.append(item)
    return out


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        print(f"ERROR: data_path not found: {data_path}")
        return 2

    rows = load_json_or_jsonl(str(data_path))
    total_rows = len(rows)
    data_dir = str(data_path.parent)
    pca_fields = parse_field_list(args.pca_image_fields)
    use_pca = bool(args.use_pca_images)

    print("=== Stage15 Dataset Preflight ===")
    print(f"data_path           : {data_path}")
    print(f"total_rows          : {total_rows}")
    print(f"use_pca_images      : {use_pca}")
    print(f"image_field         : {args.image_field}")
    print(f"pca_image_fields    : {pca_fields}")

    result = CheckResult(total_rows=total_rows)
    missing_path: List[str] = []
    missing_file: List[str] = []
    unreadable: List[str] = []
    ext_counter: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        image_path = resolve_row_image_path(
            row=row,
            data_dir=data_dir,
            image_field=args.image_field,
            use_pca_images=use_pca,
            pca_image_fields=pca_fields,
        )

        if not image_path:
            result.missing_path_rows += 1
            missing_path.append(f"row={idx} id={row.get('id')}")
            continue

        ext_counter[Path(image_path).suffix.lower()] += 1

        if not os.path.exists(image_path):
            result.missing_file_rows += 1
            missing_file.append(f"row={idx} id={row.get('id')} path={image_path}")
            continue

        if bool(args.check_readable):
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                result.unreadable_rows += 1
                unreadable.append(f"row={idx} id={row.get('id')} path={image_path} err={exc}")
                continue

        result.readable_rows += 1

    if args.expected_rows > 0 and total_rows != args.expected_rows:
        print(
            f"ERROR: expected_rows mismatch: got={total_rows}, expected={args.expected_rows}"
        )
        row_mismatch = True
    else:
        row_mismatch = False

    print("--- Summary ---")
    print(f"readable_rows       : {result.readable_rows}")
    print(f"missing_path_rows   : {result.missing_path_rows}")
    print(f"missing_file_rows   : {result.missing_file_rows}")
    print(f"unreadable_rows     : {result.unreadable_rows}")
    print(f"image_extensions    : {dict(ext_counter)}")

    if missing_path:
        print("--- Missing Path Samples ---")
        for item in batched_preview(missing_path, args.max_report):
            print(item)
    if missing_file:
        print("--- Missing File Samples ---")
        for item in batched_preview(missing_file, args.max_report):
            print(item)
    if unreadable:
        print("--- Unreadable File Samples ---")
        for item in batched_preview(unreadable, args.max_report):
            print(item)

    hard_fail = (
        row_mismatch
        or result.missing_path_rows > 0
        or result.missing_file_rows > 0
        or result.unreadable_rows > 0
    )

    if bool(args.strict) and hard_fail:
        print("PRECHECK_STATUS      : FAIL")
        return 1

    print("PRECHECK_STATUS      : PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: preflight crashed: {exc}")
        raise
