#!/usr/bin/env python3
"""
ocr_utils.py
Utility layer on top of raw EasyOCR output.

Three things happen here:
  1. Each detected text region gets classified (is it an option? a coordinate? a formula?)
  2. An index is built so we can search for text by keyword later
  3. Empty drawing regions on the image are detected — so we know where to write annotations
     without overlapping the question text
"""

import re
import numpy as np
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any


class OCRElement:
    """
    One detected text region from EasyOCR, with semantic type attached.
    Storing x1/y1/x2/y2 directly saves recalculating min/max every time we need bounds.
    """

    def __init__(self, bbox: List[List[float]], text: str, confidence: float, index: int):
        self.bbox       = [[int(c) for c in pt] for pt in bbox]
        self.text       = text.strip()
        self.confidence = confidence
        self.index      = index

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        self.x1, self.y1 = int(min(xs)), int(min(ys))
        self.x2, self.y2 = int(max(xs)), int(max(ys))
        self.width    = self.x2 - self.x1
        self.height   = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2

        self.type = self._classify()

    def _classify(self) -> str:
        """
        Classify what kind of text this region contains.
        Order matters — more specific patterns are checked first.
        """
        t = self.text.lower()

        # Option labels: (A), (B), a), b. etc.
        if re.search(r'^\s*[\(\[-]?([a-dA-D])[\)\]\.-]?', self.text) \
                or t in ["(a)", "(b)", "(c)", "(d)"]:
            return "option"

        # Coordinate pairs like (1, 2) or A(4, 6)
        if re.search(r'\(?\d+\s*,\s*\d+\)?', t) \
                or re.search(r'\b[A-Za-z]\s*\(?\d+', t):
            return "coordinate"

        # Formula hints
        if "formula" in t or "theorem" in t:
            return "formula_reference"

        # Low-confidence short text is likely a diagram label or noise
        if self.confidence < 0.25 and len(self.text) <= 4:
            return "diagram"

        return "text"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index":      self.index,
            "text":       self.text,
            "bbox":       self.bbox,
            "type":       self.type,
            "confidence": self.confidence,
            "bounds":     [self.x1, self.y1, self.x2, self.y2],
        }


class OCRIndex:
    """
    Searchable index over all detected elements.

    Search priority: exact match > substring > fuzzy.
    This means "A (1, 2)" will always beat a fuzzy match
    when the exact string exists somewhere in the OCR results.
    """

    def __init__(self, elements: List[OCRElement]):
        self.elements = elements

    def find_by_text(self, query: str, threshold: float = 0.5) -> List[OCRElement]:
        query_lower = query.lower().strip()
        matches     = []

        for elem in self.elements:
            el = elem.text.lower().strip()

            if el == query_lower:
                matches.append((elem, 1.0))
            elif query_lower in el or el in query_lower:
                matches.append((elem, 0.9))
            else:
                ratio = SequenceMatcher(None, query_lower, el).ratio()
                if ratio >= threshold:
                    matches.append((elem, ratio))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]


def find_largest_empty_rectangle(
    width: int, height: int,
    elements: List[OCRElement],
    question_bbox: Tuple[int, int, int, int]
) -> List[Dict[str, Any]]:
    """
    Find empty regions on the image where we can safely draw annotations.

    Approach: treat every OCR bounding box as an obstacle (with small padding),
    then sweep vertical strips and find gaps between obstacles in the y direction.
    Each gap is a candidate empty rectangle.

    FIX: diagram_bbox fallback is now right-side of image, not center.
    Center was wrong — most MCQ images have diagrams on the right, not center.
    """
    PAD = 8
    obstacles   = []
    diagram_bbox = None

    for elem in elements:
        ox1 = max(0,     elem.x1 - PAD)
        oy1 = max(0,     elem.y1 - PAD)
        ox2 = min(width, elem.x2 + PAD)
        oy2 = min(height,elem.y2 + PAD)
        obstacles.append((ox1, oy1, ox2, oy2))

        if elem.type == "diagram" or (elem.confidence < 0.25 and len(elem.text) <= 2):
            diagram_bbox = (elem.x1, elem.y1, elem.x2, elem.y2)

    # Sweep candidate vertical boundaries
    left_coords  = [0]  + [x2 for (x1, y1, x2, y2) in obstacles]
    right_coords = [width] + [x1 for (x1, y1, x2, y2) in obstacles]

    candidates = []
    for xa in left_coords:
        if xa < 0 or xa >= width:
            continue
        for xb in right_coords:
            if xb <= xa or xb > width:
                continue

            strip_obs  = [o for o in obstacles if o[0] < xb and o[2] > xa]
            y_intervals = sorted([(o[1], o[3]) for o in strip_obs], key=lambda i: i[0])

            curr_y = 0
            for (oy1, oy2) in y_intervals:
                if oy1 > curr_y:
                    candidates.append((xa, curr_y, xb, oy1))
                curr_y = max(curr_y, oy2)
            if height > curr_y:
                candidates.append((xa, curr_y, xb, height))

    qx1, qy1, qx2, qy2 = question_bbox if question_bbox else (0, 0, width // 2, height // 3)

    # FIX: default diagram bbox is right side, not center
    # Most PW question images have diagrams on the right half
    if not diagram_bbox:
        diagram_bbox = (width * 2 // 3, 0, width, height // 2)

    categorized = []
    seen        = set()

    for (x1, y1, x2, y2) in candidates:
        w_r  = x2 - x1
        h_r  = y2 - y1

        # Skip slivers too small to write in
        if w_r < 180 or h_r < 120:
            continue

        rect = (x1, y1, x2, y2)
        if rect in seen:
            continue
        seen.add(rect)

        # Categorise by position relative to question and diagram
        if x1 >= qx2 - 50 or (x1 >= width * 0.45 and y1 <= qy2 + 250):
            pos, priority = "right", 1
        elif y1 >= diagram_bbox[3] - 20:
            pos, priority = "bottom_diagram", 2
        elif y1 >= qy2 - 10:
            pos, priority = "bottom_question", 3
        else:
            pos, priority = "secondary", 4

        categorized.append({
            "bounds":   [x1, y1, x2, y2],
            "area":     w_r * h_r,
            "width":    w_r,
            "height":   h_r,
            "position": pos,
            "priority": priority,
        })

    # Best region = highest priority first, then largest area
    categorized.sort(key=lambda r: (r["priority"], -r["area"]))
    return categorized


def enrich_ocr_data(
    ocr_results: List[Tuple],
    image_width: int,
    image_height: int,
    question_bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """
    Wrap raw EasyOCR output into structured, searchable, enriched data.
    This is what gets passed to render_video so it knows where things are.
    """
    elements = [
        OCRElement(bbox, text, conf, idx)
        for idx, (bbox, text, conf) in enumerate(ocr_results)
    ]

    return {
        "elements":      [e.to_dict() for e in elements],
        "index":         OCRIndex(elements),
        "free_spaces":   find_largest_empty_rectangle(
                             image_width, image_height, elements, question_bbox),
        "full_text":     " ".join(e.text for e in elements),
        "element_count": len(elements),
    }