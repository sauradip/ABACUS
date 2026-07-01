"""
extract_count.py
----------------
End-to-end pipeline:

  1. Load WebDataset shards via dataloader.py
  2. For each sample, parse caption with SpaCy to extract clean object categories
     (noun chunks whose root has no adjective modifier)
  3. Feed (image, categories) into Rex-Omni for open-vocabulary detection
  4. Count bounding boxes per category and format as human-readable text,
     e.g. ["3 dog", "1 bicycle", "12 person"]

Usage
-----
    # Run on first N samples (no GPU needed for SpaCy; Rex-Omni needs one)
    python extract_count.py --max_samples 10

    # Run the SpaCy extraction unit test only (CPU, no Rex-Omni)
    python extract_count.py --nlp_test
"""

import argparse
import json
import os
import re
from pathlib import PurePosixPath
from typing import Dict, List, Optional

import spacy
from PIL import Image

# Local dataloader
from dataloader import get_dataloader

# Rex-Omni (installed from the Rex-Omni repo)
try:
    from rex_omni import RexOmniWrapper, RexOmniVisualize
    _REX_AVAILABLE = True
except ImportError:
    _REX_AVAILABLE = False
    print("[WARNING] rex_omni not importable – detection will be skipped.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A  –  SpaCy NLP helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SPACY_MODEL = "en_core_web_sm"

# ── Blacklist of scene-level, meta, spatial, and abstract nouns ────────────────
# These frequently appear in image captions but do NOT refer to discrete,
# countable physical objects in the scene.  Words here are always filtered out
# regardless of how many times Rex-Omni detects them.
SCENE_WORDS: set = {
    # meta / photo vocabulary
    "image", "picture", "photo", "photograph", "scene", "view", "shot",
    "depiction", "capture",
    # spatial / compositional
    "background", "foreground", "midground", "area", "ground", "surface",
    "inside", "outside", "indoor", "outdoor", "setting", "environment",
    "space", "depth", "right", "left", "side", "end", "top", "bottom",
    "front", "rear",
    # abstract / descriptive
    "sense", "focus", "style", "type", "size", "color", "colour",
    "range", "pattern", "design", "variety", "mix", "way", "context",
    "structure", "purpose", "period", "comfort", "thrill", "adventure",
    "grace", "serenity", "purity", "enlightenment", "reverence",
    "surrender", "motion", "action", "movement", "exercise", "calm",
    "peace", "beauty", "elegance", "theme", "detail", "feature",
    "balance", "safety", "renovation", "celebration", "performance",
    "dusk", "lighting", "shadow", "inscription", "symbol",
    # relational / collective nouns that are not themselves objects
    "row", "pair", "group", "set", "series", "collection", "line",
    "number",
}


def load_nlp() -> spacy.language.Language:
    """Load (and cache) the SpaCy pipeline."""
    try:
        nlp = spacy.load(_SPACY_MODEL)
    except OSError:
        raise OSError(
            f"SpaCy model '{_SPACY_MODEL}' not found. "
            f"Install it with:\n  python -m spacy download {_SPACY_MODEL}"
        )
    return nlp


def _is_pure_noun_phrase(chunk) -> bool:
    """
    Return True if the noun chunk contains NO adjective token (ADJ POS tag).

    Strategy
    --------
    Iterate over every token in the chunk.  If any token carries the
    universal POS tag ADJ we consider the phrase "descriptive" and discard it.

    This keeps  : "lemon", "table", "cat on the roof"
    Discards    : "green lemon", "tall wooden table", "yellow flower"
    """
    return all(token.pos_ != "ADJ" for token in chunk)


def extract_categories(
    caption: str,
    nlp: spacy.language.Language,
) -> List[str]:
    """
    Parse *caption* with SpaCy and return a deduped list of clean object
    category strings suitable for Rex-Omni.

    Pipeline
    --------
    1. Run the SpaCy pipeline to get noun chunks.
    2. Discard chunks that contain any ADJ token  (step c in spec).
    3. Take the lowercased *lemma* of each chunk's syntactic ROOT noun
       – this normalises plurals (apples → apple) and keeps the canonical
         head word even for multi-word NPs like "cat on top".
    4. Drop stop-words, punctuation-only tokens, and very short strings (<2 chars).
    5. Deduplicate while preserving first-occurrence order.

    Parameters
    ----------
    caption : str
        Raw caption / alt-text from the dataset shard.
    nlp : spacy.language.Language
        Loaded SpaCy pipeline.

    Returns
    -------
    list[str]
        E.g. ["lemon", "table", "cat", "bicycle"]
    """
    doc = nlp(caption)

    seen = {}
    for chunk in doc.noun_chunks:
        # Filter 1: discard NPs with any adjective token
        if not _is_pure_noun_phrase(chunk):
            continue

        root = chunk.root

        # Filter 2: root must be a concrete noun / proper noun, not a stop-word
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        if root.is_stop:
            continue

        lemma = root.lemma_.lower().strip()

        # Filter 3: discard very short strings
        if len(lemma) < 2:
            continue

        # Filter 4: discard known scene-level / abstract words
        if lemma in SCENE_WORDS:
            continue

        # Deduplicate, preserve order
        if lemma not in seen:
            seen[lemma] = True

    return list(seen.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A2  –  ImageNet category whitelist
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CATEGORY_FILE = os.path.join(os.path.dirname(__file__), "category.txt")
COCO_CATEGORIES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_imagenet_whitelist(path: str = DEFAULT_CATEGORY_FILE):
    """
    Load the ImageNet category list from *path* (a JSON array of strings).

    Builds two lookup structures:
    - imagenet_exact  : set of lowercase full labels, e.g. {"laptop computer", …}
    - imagenet_root   : dict mapping the lowercase *last word* of each label
                        to the candidate full label strings,
                        e.g. {"computer": ["laptop computer"]}

    This lets us match SpaCy lemmas (single words) to full ImageNet labels.

    Returns
    -------
    (list[str], set[str], dict[str, list[str]])
        (full_labels, imagenet_exact, imagenet_root)
    """
    with open(path) as f:
        content = f.read().strip()
    # Support both bare JSON array "[...]" and Python list literal
    try:
        full_labels: List[str] = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: evaluate as Python literal (handles single-quoted strings)
        import ast
        full_labels = ast.literal_eval(content)

    imagenet_exact: set = {lbl.lower() for lbl in full_labels}

    # root word → candidate full labels
    imagenet_root: dict = {}
    for lbl in full_labels:
        root_word = lbl.split()[-1].lower()
        imagenet_root.setdefault(root_word, [])
        if lbl not in imagenet_root[root_word]:
            imagenet_root[root_word].append(lbl)

    print(f"[whitelist] Loaded {len(full_labels)} ImageNet categories from '{path}'")
    return full_labels, imagenet_exact, imagenet_root


def filter_by_imagenet(
    spacy_lemmas: List[str],
    imagenet_exact: set,
    imagenet_root: dict,
    caption_text: Optional[str] = None,
) -> List[str]:
    """
    Map SpaCy-extracted lemmas to full ImageNet label strings.

    Matching priority
    -----------------
    1. Exact match  : lemma itself is a full label (e.g. "cup" → "cup")
    2. Root match   : lemma matches the last word of a label, but multi-word
                      labels are only accepted when their tokens are present
                      in the caption text.

    Only matched labels are returned; unrecognised lemmas are silently dropped.
    Duplicates are deduplicated (multiple lemmas may map to the same label).

    Parameters
    ----------
    spacy_lemmas   : output of extract_categories()
    imagenet_exact : set of lowercase full labels
    imagenet_root  : dict {root_word: [full_label, ...]}
    caption_text   : raw caption text used to disambiguate multi-word labels

    Returns
    -------
    list[str]  – full ImageNet label strings suitable for Rex-Omni
    """
    caption_tokens = set(re.findall(r"[a-z0-9]+", (caption_text or "").lower()))
    caption_lower = (caption_text or "").lower()
    matched = {}
    for lemma in spacy_lemmas:
        if lemma in imagenet_exact:
            matched[lemma] = lemma           # exact match → use as-is
        elif lemma in imagenet_root:
            candidates = imagenet_root[lemma]
            phrase_matches = []
            token_matches = []
            for full in candidates:
                full_lower = full.lower()
                if " " not in full_lower:
                    token_matches.append(full)
                    continue
                label_tokens = re.findall(r"[a-z0-9]+", full_lower)
                if not label_tokens:
                    continue
                if full_lower in caption_lower:
                    phrase_matches.append(full)
                    continue
                if caption_tokens and all(tok in caption_tokens for tok in label_tokens):
                    token_matches.append(full)

            chosen = None
            if phrase_matches:
                chosen = min(phrase_matches, key=lambda item: (len(item.split()), len(item)))
            elif token_matches:
                chosen = min(token_matches, key=lambda item: (len(item.split()), len(item)))

            if chosen is not None:
                matched[chosen] = chosen
    return list(matched.values())



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B  –  Rex-Omni detection
# ═══════════════════════════════════════════════════════════════════════════════

def build_rex(
    model_path: str = "IDEA-Research/Rex-Omni",
    attn_implementation: Optional[str] = None,
) -> "RexOmniWrapper":
    """
    Initialise the Rex-Omni wrapper with generation parameters that favour
    deterministic, low-temperature output (as recommended in the template).
    """
    if not _REX_AVAILABLE:
        raise RuntimeError("rex_omni package is not installed.")

    rex = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
        attn_implementation=attn_implementation,
    )
    return rex


def make_output_filename(
    sample_key: Optional[str],
    image_ext: Optional[str],
    fallback_index: int,
) -> str:
    """Build a stable output filename from the original WebDataset key."""
    ext = (image_ext or "jpg").lower()
    if ext == "jpeg":
        ext = "jpg"

    if not sample_key:
        return f"sample_{fallback_index}.{ext}"

    key_name = PurePosixPath(str(sample_key).replace("\\", "/")).name
    if not key_name:
        return f"sample_{fallback_index}.{ext}"

    _, existing_ext = os.path.splitext(key_name)
    if existing_ext:
        return key_name
    return f"{key_name}.{ext}"


def run_detection(
    rex: "RexOmniWrapper",
    image: Image.Image,
    categories: List[str],
) -> Dict[str, List[dict]]:
    """
    Run Rex-Omni open-vocabulary detection and return the raw predictions dict.

    Parameters
    ----------
    rex        : initialised RexOmniWrapper
    image      : PIL.Image (RGB)
    categories : list of category strings (output of extract_categories)

    Returns
    -------
    dict  { category: [{"type": "box", "coords": [x0,y0,x1,y1]}, ...], ... }
    """
    if not categories:
        return {}

    results = rex.inference(images=image, task="detection", categories=categories)
    # rex.inference returns a list (one entry per image); take index 0
    raw = results[0]
    predictions: Dict[str, List[dict]] = raw.get("extracted_predictions", {})
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C  –  Count formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_counts(
    predictions: Dict[str, List[dict]],
    min_count: int = 5,
) -> List[str]:
    """
    Convert a Rex-Omni predictions dict into a list of count strings,
    keeping only *concrete, multiple-instance* detections.

    Filtering strategy
    ------------------
    1. Remove categories whose lemma is still in SCENE_WORDS (safety net in
       case any slipped through the NLP stage).
     2. Only keep categories where count >= min_count (default 5), which enforces
         a strict "objects > 4" policy per category.

    Example
    -------
    Input : {"chair": [b]*24, "row": [b]*5, "floor": [b], "image": [b]}
    Output: ["24 chair"]   ("row" has count>=2 but is in SCENE_WORDS;
                            "floor"/"image" are scene words and count==1)

    Parameters
    ----------
    predictions : dict
        { category: [{"type": "box", "coords": [x0,y0,x1,y1]}, ...], ... }
    min_count : int
        Minimum number of detections required to keep a category (default 5).

    Returns
    -------
    list[str]
    """
    # Step 1: strip scene words and zero-count categories
    candidates = [
        (category, len(boxes))
        for category, boxes in predictions.items()
        if len(boxes) > 0 and category.lower() not in SCENE_WORDS
    ]

    if not candidates:
        return []

    # Step 2: keep only entries with count >= min_count
    filtered = [(cat, cnt) for cat, cnt in candidates if cnt >= min_count]

    if not filtered:
        return []

    # Sort by count descending so the most prominent object comes first
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [f"{cnt} {cat}" for cat, cnt in filtered]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D  –  Main orchestration loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    max_samples: Optional[int] = None,
    model_path: str = "IDEA-Research/Rex-Omni",
    attn_implementation: Optional[str] = None,
    fallback_missing_caption_to_coco: bool = False,
    external_caption_map: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    save_json: Optional[str] = None,
    output_dir: str = "count_GT",
    category_file: str = DEFAULT_CATEGORY_FILE,
    min_count: int = 5,
    save_images: bool = False,
    auto_add_categories: bool = True,
):
    """
    Full end-to-end pipeline over the WebDataset shards.

    Parameters
    ----------
    max_samples   : stop after this many successfully processed samples (None = all)
    model_path    : Rex-Omni model ID or local path
    external_caption_map : optional JSON file mapping image names to captions
    data_dir      : directory containing WebDataset .tar shards
    batch_size    : DataLoader batch size (keep 1 for Rex-Omni single-image API)
    num_workers   : DataLoader workers
    save_json     : optional path to write results as JSON Lines
    output_dir    : directory to save outputs (caption.json and optional images)
    category_file : path to category.txt with ImageNet label list
    min_count     : strict minimum detections per kept category
    save_images   : save generated sample images under output_dir/img
    auto_add_categories : append unseen caption classes to category_file
    """
    print("Loading SpaCy …")
    nlp = load_nlp()

    print("Loading ImageNet whitelist …")
    full_labels, imagenet_exact, imagenet_root = load_imagenet_whitelist(category_file)
    added_categories = set()

    def persist_new_categories(path: str):
        if not added_categories:
            return
        ordered_labels = sorted(full_labels, key=lambda x: x.lower())
        with open(path, "w") as f:
            json.dump(ordered_labels, f, indent=4)
        print(f"[category] Added {len(added_categories)} new classes to {path}")

    print("Loading Rex-Omni …")
    rex = build_rex(model_path, attn_implementation=attn_implementation)

    print("Initialising DataLoader …")
    
    # ── Early Exit Filter ───────────────────────────────────────────────────
    # We pass this to the dataloader so it can parse captions and skip 
    # downloading/decoding the JPEG entirely if no categories match ImageNet.
    def caption_processor(caption: str) -> List[str]:
        spacy_lemmas = extract_categories(caption, nlp)
        categories = filter_by_imagenet(
            spacy_lemmas,
            imagenet_exact,
            imagenet_root,
            caption_text=caption,
        )
        if auto_add_categories:
            for lemma in spacy_lemmas:
                if lemma in imagenet_exact or lemma in imagenet_root:
                    continue
                categories.append(lemma)
                imagenet_exact.add(lemma)
                root_word = lemma.split()[-1]
                imagenet_root.setdefault(root_word, [])
                if lemma not in imagenet_root[root_word]:
                    imagenet_root[root_word].append(lemma)
                full_labels.append(lemma)
                added_categories.add(lemma)
        # Deduplicate while preserving order.
        seen = set()
        out = []
        for cat in categories:
            if cat in seen:
                continue
            seen.add(cat)
            out.append(cat)
        return out

    loader = get_dataloader(
        batch_size=batch_size, 
        num_workers=num_workers,
        data_dir=data_dir,
        process_caption_fn=caption_processor,
        external_caption_map=external_caption_map,
    )

    # ── Prepare output directories ──────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "img")
    if save_images:
        os.makedirs(img_dir, exist_ok=True)
    caption_mapping = {}

    output_file = open(save_json, "w") if save_json else None
    processed = 0
    verbose_sample_limit = max(int(os.environ.get("UNICOUNT_VERBOSE_SAMPLE_LIMIT", "5")), 0)
    progress_every = max(int(os.environ.get("UNICOUNT_PROGRESS_EVERY", "100")), 1)

    try:
        for batch in loader:
            images   = batch["image"]    # list[PIL.Image]
            captions = batch["caption"]  # list[str]
            sample_keys = batch.get("sample_key")
            image_exts = batch.get("image_ext")

            categories_batch = batch.get("categories")

            for i, (image, caption) in enumerate(zip(images, captions)):
                sample_index = processed
                sample_key = sample_keys[i] if sample_keys is not None else None
                image_ext = image_exts[i] if image_exts is not None else "jpg"
                # ── Step B: Retrieve pre-computed categories ────────────────
                used_coco_fallback = False
                if categories_batch is not None:
                    categories = categories_batch[i]
                else:
                    spacy_lemmas = extract_categories(caption, nlp)
                    categories = filter_by_imagenet(
                        spacy_lemmas,
                        imagenet_exact,
                        imagenet_root,
                        caption_text=caption,
                    )

                if (
                    fallback_missing_caption_to_coco
                    and not categories
                    and not caption.strip()
                ):
                    categories = COCO_CATEGORIES.copy()
                    used_coco_fallback = True

                if not categories:
                    if sample_index < verbose_sample_limit:
                        print(f"[sample {sample_index}] No categories available – skipping.")
                    continue

                log_detailed_sample = sample_index < verbose_sample_limit

                if log_detailed_sample:
                    print(f"\n[sample {sample_index}] Caption : {caption[:120]}")
                    if sample_key:
                        print(f"[sample {sample_index}] Source key : {sample_key}")
                    if used_coco_fallback:
                        print(
                            f"[sample {sample_index}] Caption missing; using COCO fallback "
                            f"({len(categories)} categories)."
                        )
                    print(f"[sample {sample_index}] Categories : {categories}")

                # ── Step C: Rex-Omni detection ──────────────────────────────
                predictions = run_detection(rex, image, categories)

                # ── Step D: Count summarisation ─────────────────────────────
                count_strings = format_counts(predictions, min_count=min_count)
                final_caption = ", ".join(count_strings) if count_strings else "0 objects"

                # ── Save Image and Map Caption ──────────────────────────────
                img_filename = make_output_filename(sample_key, image_ext, sample_index)
                if save_images:
                    image.save(os.path.join(img_dir, img_filename))
                caption_mapping[img_filename] = final_caption

                if log_detailed_sample:
                    print(f"[sample {sample_index}] Counts : {count_strings}")
                elif (sample_index + 1) % progress_every == 0:
                    source = "coco_fallback" if used_coco_fallback else "caption"
                    print(
                        f"[progress] processed={sample_index + 1} "
                        f"latest={img_filename} source={source} count_caption={final_caption!r}"
                    )

                # ── Optional persistence ────────────────────────────────────
                if output_file:
                    record = {
                        "sample_idx": sample_index,
                        "source_key": sample_key,
                        "img_file": img_filename,
                        "caption": caption,
                        "count_caption": final_caption,
                        "category_source": "coco_fallback" if used_coco_fallback else "caption",
                        "categories": categories,
                        "predictions": {
                            cat: [b["coords"] for b in boxes]
                            for cat, boxes in predictions.items()
                        },
                        "count_strings": count_strings,
                    }
                    output_file.write(json.dumps(record) + "\n")

                processed += 1
                if max_samples is not None and processed >= max_samples:
                    print(f"\nReached max_samples={max_samples}. Stopping.")
                    break
            
            if max_samples is not None and processed >= max_samples:
                break

    finally:
        if output_file:
            output_file.close()
            print(f"\nResults saved to {save_json}")

        if auto_add_categories:
            persist_new_categories(category_file)
        
        # Save caption.json
        caption_json_path = os.path.join(output_dir, "caption.json")
        with open(caption_json_path, "w") as f:
            json.dump(caption_mapping, f, indent=4)
        print(f"Final captions saved to {caption_json_path}")
        if save_images:
            print(f"Images saved to {img_dir}")
        else:
            print("Image saving disabled (--save_images to enable).")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E  –  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _nlp_unit_test():
    """Quick CPU-only test for the SpaCy extraction logic."""
    nlp = load_nlp()
    test_cases = [
        (
            "A green lemon and a tall wooden table with a cat on top",
            ["lemon", "table", "cat"],
        ),
        (
            "Two yellow flowers and a sofa near the white chair",
            ["flower", "sofa", "chair"],
        ),
        (
            "A man and a woman drinking coffee at a cafe",
            ["man", "woman", "coffee", "cafe"],
        ),
    ]
    all_pass = True
    for caption, expected in test_cases:
        got = extract_categories(caption, nlp)
        ok  = all(e in got for e in expected)
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"{status}  caption='{caption[:50]}…'")
        print(f"        expected (subset): {expected}")
        print(f"        got             : {got}\n")

    print("All tests passed." if all_pass else "Some tests FAILED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rex-Omni count extraction pipeline")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Stop after N samples (default: process all)"
    )
    parser.add_argument(
        "--model_path", type=str, default="IDEA-Research/Rex-Omni",
        help="Rex-Omni model path or HuggingFace repo ID"
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Transformers attention backend override (default: wrapper auto-select)."
    )
    parser.add_argument(
        "--fallback_missing_caption_to_coco",
        action="store_true",
        help="Use COCO categories when a sample has no caption-derived categories."
    )
    parser.add_argument(
        "--external_caption_map", type=str, default=None,
        help="Optional JSON file mapping image names to captions for raw shards."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing WebDataset .tar shards (overrides dataloader default)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="DataLoader batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes"
    )
    parser.add_argument(
        "--save_json", type=str, default=None,
        help="Optional path to save results as JSON Lines"
    )
    parser.add_argument(
        "--output_dir", type=str, default="count_GT",
        help="Directory to save images and caption.json"
    )
    parser.add_argument(
        "--min_count", type=int, default=5,
        help="Keep categories with detections >= this value (default: 5 i.e. >4)"
    )
    parser.add_argument(
        "--save_images", action="store_true",
        help="Save output images to output_dir/img (disabled by default to save storage)"
    )
    parser.add_argument(
        "--no_auto_add_categories", action="store_true",
        help="Do not append unseen caption classes to category_file"
    )
    parser.add_argument(
        "--category_file", type=str, default=DEFAULT_CATEGORY_FILE,
        help="Path to category.txt with ImageNet label list"
    )
    parser.add_argument(
        "--nlp_test", action="store_true",
        help="Run CPU-only SpaCy unit tests and exit (no Rex-Omni required)"
    )
    args = parser.parse_args()

    if args.nlp_test:
        _nlp_unit_test()
    else:
        run_pipeline(
            max_samples=args.max_samples,
            model_path=args.model_path,
            attn_implementation=args.attn_implementation,
            fallback_missing_caption_to_coco=args.fallback_missing_caption_to_coco,
            external_caption_map=args.external_caption_map,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_json=args.save_json,
            output_dir=args.output_dir,
            category_file=args.category_file,
            min_count=args.min_count,
            save_images=args.save_images,
            auto_add_categories=not args.no_auto_add_categories,
        )
