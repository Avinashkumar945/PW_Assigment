#!/usr/bin/env python3
"""
ocrQuestion.py
Reads a question image and extracts:
  1. All visible text (so the LLM knows what the question says).
  2. Where each option (A/B/C/D/E) is located on screen (to tick the right one).
  3. Free drawing regions (so annotations don't overlap existing text).

Changes vs previous version:
  - Import aligned with updated ocrUtils (enrich_ocr_data returns a plain dict
    with "index", "elements", "free_spaces" keys — no longer a class instance).
  - question_bbox x-bounds computed only from question-region detections
    (option rows excluded).
  - Option regex unified; supports A-E and all common punctuation styles.
  - enrich_ocr_data wrapped in try/except so OCR text still returns on failure.
  - Raises FileNotFoundError / ValueError instead of sys.exit() — library code
    should not call exit(); callers decide how to handle errors.
"""

import json
import os
import re
import sys

from PIL import Image
import easyocr

# ocrUtils must be on the Python path (same scripts/ directory)
from ocrUtils import enrich_ocr_data


# ── Constants ────────────────────────────────────────────────────────────────

OPTION_LETTERS       = {"A", "B", "C", "D", "E"}
CONFIDENCE_THRESHOLD = 0.3

# Matches:  (a)  (a)text  a)  a)text  a.  a.text
_OPTION_RE = re.compile(r"^\(?([a-eA-E])[).]\s*", re.IGNORECASE)


# ── Main function ─────────────────────────────────────────────────────────────

def extract_question_info(image_path: str):
    """
    Run EasyOCR on the question image and return structured data.

    Args:
        image_path: Path to the question image.

    Returns:
        full_text        : All detected text joined as one string.
        option_positions : {letter: bbox} — each bbox is a list of 4 [x, y] points.
        option_text_map  : {letter: text_content} — text after the option label.
        question_bbox    : (x_min, y_min, x_max, y_max) of the question region only,
                           excluding option rows.  None if no text detected.
        enriched_ocr     : Dict with "elements", "index", "free_spaces".

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError:        If EasyOCR returns no results at all.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    image_width, image_height = img.size
    print(f"  Image size: {image_width}x{image_height}")

    reader  = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(image_path)

    if not results:
        raise ValueError(f"EasyOCR returned no detections for: {image_path}")

    # Drop low-confidence detections
    filtered = [
        (bbox, text, conf)
        for bbox, text, conf in results
        if conf >= CONFIDENCE_THRESHOLD
    ]
    dropped = len(results) - len(filtered)
    if dropped:
        print(f"  Dropped {dropped} low-confidence detections (< {CONFIDENCE_THRESHOLD})")

    # ── Option detection ──────────────────────────────────────────────────────
    option_positions   = {}   # letter -> bbox (list of 4 [x, y] points)
    option_text_map    = {}   # letter -> text content after the label
    option_y_positions = []   # top-y of each option row
    option_indices     = set()

    for idx, (bbox, text, _) in enumerate(filtered):
        text_stripped = text.strip()
        m = _OPTION_RE.match(text_stripped)
        if not m:
            continue
        letter = m.group(1).upper()
        if letter not in OPTION_LETTERS:
            continue
        if letter in option_positions:       # keep first detection only
            continue

        option_positions[letter]  = [[int(c) for c in pt] for pt in bbox]
        option_text_map[letter]   = text_stripped[m.end():].strip()
        option_y_positions.append(min(p[1] for p in bbox))
        option_indices.add(idx)

    # ── Question bbox (question region only, excluding option rows) ───────────
    question_bbox = None
    if filtered:
        y_cutoff = (min(option_y_positions) - 5) if option_y_positions else float("inf")

        q_pts = [
            pt
            for idx, (bbox, _, _) in enumerate(filtered)
            if idx not in option_indices
            for pt in bbox
            if pt[1] < y_cutoff
        ]

        if q_pts:
            xs = [p[0] for p in q_pts]
            ys = [p[1] for p in q_pts]
            question_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    # ── Enrich OCR data ───────────────────────────────────────────────────────
    try:
        enriched_ocr = enrich_ocr_data(filtered, image_width, image_height, question_bbox)
    except Exception as exc:
        print(f"  WARNING: enrich_ocr_data failed ({exc}); enriched_ocr set to empty dict")
        enriched_ocr = {}

    full_text = " ".join(text for _, text, _ in filtered)

    print(f"  Text regions : {len(filtered)}")
    print(f"  Options found: {sorted(option_positions.keys())}")
    print(f"  Question bbox: {question_bbox}")
    print(f"  Free spaces  : {len((enriched_ocr or {}).get('free_spaces', []))}")

    return full_text, option_positions, option_text_map, question_bbox, enriched_ocr


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    try:
        text, positions, opt_texts, q_bbox, enriched = extract_question_info(image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuestion text:\n{text}")
    print(f"\nOption positions:\n{json.dumps(positions, indent=2)}")
    print(f"\nOption texts:\n{json.dumps(opt_texts, indent=2)}")
    print(f"\nQuestion bbox:\n{q_bbox}")