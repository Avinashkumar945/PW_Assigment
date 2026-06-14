#!/usr/bin/env python3
"""
ocr_question.py
Reads a question image and extracts two things:
  1. All the text (so the LLM knows what the question says)
  2. Where each option (A/B/C/D) is located on screen (so we can tick the right one)
"""

import json
import os
import re
import sys
from PIL import Image
import easyocr
from ocr_utils import enrich_ocr_data


def extract_question_info(image_path):
    """
    Run EasyOCR on the question image and return structured data.

    Returns:
        full_text       : All detected text joined as one string
        option_positions: Where each option letter sits on screen
        question_bbox   : Bounding box of the question text (above options)
        enriched_ocr    : Richer structure including free drawing regions
    """
    # Check file before loading EasyOCR — the model takes ~15 seconds to load
    # and failing after that with a missing file is frustrating to debug
    if not os.path.exists(image_path):
        print(f"  ERROR: Image not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path)
    image_width, image_height = img.size
    print(f"  Image size: {image_width}x{image_height}")

    reader  = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(image_path)

    # Drop anything EasyOCR is not confident about
    # Keeping low-confidence results adds garbage text that confuses the LLM
    CONFIDENCE_THRESHOLD = 0.3
    filtered = [(bbox, text, conf)
                for bbox, text, conf in results
                if conf >= CONFIDENCE_THRESHOLD]

    dropped = len(results) - len(filtered)
    if dropped:
        print(f"  Dropped {dropped} low-confidence detections (< {CONFIDENCE_THRESHOLD})")

    full_text = " ".join(text for _, text, _ in filtered)

    # Match options in various formats — PDFs and screenshots use different styles:
    # (a), (A), a), A), a., A.  — we want to catch all of them
    OPTION_PATTERNS = [
        r'^\(([abcd])\)',   # (a)
        r'^([abcd])\)',     # a)
        r'^([abcd])\.',     # a.
        r'^\(([abcd])\s',   # (a followed by space
    ]

    option_positions   = {}
    option_y_positions = []

    for bbox, text, _ in filtered:
        text_stripped = text.strip()
        matched_opt   = None

        for pattern in OPTION_PATTERNS:
            m = re.match(pattern, text_stripped, re.IGNORECASE)
            if m:
                matched_opt = m.group(1).upper()
                break

        # Only keep the first detection per option letter
        if matched_opt and matched_opt not in option_positions:
            option_positions[matched_opt] = [
                [int(c) for c in pt] for pt in bbox
            ]
            option_y_positions.append(min(p[1] for p in bbox))

    # Question region = everything above the topmost option
    question_bbox = None
    if filtered:
        all_x = [p[0] for bbox, _, _ in filtered for p in bbox]
        all_y = [p[1] for bbox, _, _ in filtered for p in bbox]
        q_y_max = (min(option_y_positions) - 5) if option_y_positions else max(all_y)
        question_bbox = (int(min(all_x)), int(min(all_y)),
                         int(max(all_x)), int(q_y_max))

    enriched_ocr = enrich_ocr_data(filtered, image_width, image_height, question_bbox)

    print(f"  Text regions : {len(filtered)}")
    print(f"  Options found: {list(option_positions.keys())}")
    print(f"  Question bbox: {question_bbox}")
    print(f"  Free spaces  : {len(enriched_ocr.get('free_spaces', []))}")

    return full_text, option_positions, question_bbox, enriched_ocr


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    text, positions, q_bbox, enriched = extract_question_info(image)
    print(f"\nQuestion text:\n{text}")
    print(f"\nOption positions:\n{json.dumps(positions, indent=2)}")
    print(f"\nQuestion bbox:\n{q_bbox}")