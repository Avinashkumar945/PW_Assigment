#!/usr/bin/env python3
"""
ocrUtils.py — OCR enrichment utilities for the PW Annotation System.

Provides:
  - OCRIndex       : fast substring/fuzzy lookup over OCR detections
  - enrich_ocr_data: adds free-space regions and builds an OCRIndex
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OCRElement:
    """A single OCR detection with its bounding box and text."""
    text: str
    x1:   int
    y1:   int
    x2:   int
    y2:   int
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class FreeSpace:
    """A rectangular region of the image that appears free of text."""
    bounds: Tuple[int, int, int, int]   # x1, y1, x2, y2
    area:   int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bounds
        self.area = max(0, (x2 - x1)) * max(0, (y2 - y1))


# ── OCR Index ─────────────────────────────────────────────────────────────────

class OCRIndex:
    """
    Lightweight index over a list of OCRElement objects.
    Supports fast exact-substring and simple fuzzy text lookup.
    """

    def __init__(self, elements: List[OCRElement]):
        self._elements = elements

    # ------------------------------------------------------------------
    def find_by_text(self, query: str, threshold: float = 0.4) -> List[OCRElement]:
        """
        Return elements whose text contains `query` (case-insensitive).
        If no exact match, fall back to token-overlap scoring >= threshold.

        Args:
            query:     Text to search for.
            threshold: Minimum token-overlap ratio for fuzzy matching.

        Returns:
            List of matching OCRElement objects, ordered by y-position.
        """
        if not query:
            return []

        q_lower = query.lower().strip()

        # 1. Exact substring match
        exact = [e for e in self._elements if q_lower in e.text.lower()]
        if exact:
            return sorted(exact, key=lambda e: e.y1)

        # 2. Token-overlap fallback
        q_tokens = set(re.findall(r"\w+", q_lower))
        if not q_tokens:
            return []

        scored = []
        for e in self._elements:
            e_tokens = set(re.findall(r"\w+", e.text.lower()))
            if not e_tokens:
                continue
            overlap = len(q_tokens & e_tokens) / len(q_tokens)
            if overlap >= threshold:
                scored.append((overlap, e))

        scored.sort(key=lambda x: (-x[0], x[1].y1))
        return [e for _, e in scored]

    # ------------------------------------------------------------------
    def all_elements(self) -> List[OCRElement]:
        return list(self._elements)


# ── Free-space detector ───────────────────────────────────────────────────────

def _detect_free_spaces(
    elements:      List[OCRElement],
    image_width:   int,
    image_height:  int,
    question_bbox: Optional[Tuple[int, int, int, int]],
    min_height:    int = 60,
    min_width:     int = 200,
) -> List[FreeSpace]:
    """
    Detect rectangular free-space regions where annotations can be written.

    Strategy:
      - Collect all text-occupied y-bands.
      - Look for gaps between text rows that are large enough.
      - Prefer the region BELOW the question text but ABOVE the options.
    """
    if not elements:
        # Fallback: use bottom quarter of the image
        y_start = int(image_height * 0.65)
        return [FreeSpace((20, y_start, image_width - 20, image_height - 20))]

    # Sort elements top-to-bottom
    sorted_elems = sorted(elements, key=lambda e: e.y1)

    # Build a list of occupied y-intervals (merge overlapping ones)
    occupied: List[Tuple[int, int]] = []
    for e in sorted_elems:
        if occupied and e.y1 <= occupied[-1][1] + 5:
            # Extend the last interval
            occupied[-1] = (occupied[-1][0], max(occupied[-1][1], e.y2))
        else:
            occupied.append((e.y1, e.y2))

    # Find gaps between occupied bands
    free_spaces: List[FreeSpace] = []
    prev_bottom = 0

    for top, bottom in occupied:
        gap = top - prev_bottom
        if gap >= min_height:
            x1 = 20
            x2 = image_width - 20
            if (x2 - x1) >= min_width:
                free_spaces.append(FreeSpace((x1, prev_bottom + 4, x2, top - 4)))
        prev_bottom = bottom

    # Gap after last text block
    gap = image_height - prev_bottom
    if gap >= min_height:
        x1, x2 = 20, image_width - 20
        if (x2 - x1) >= min_width:
            free_spaces.append(FreeSpace((x1, prev_bottom + 4, x2, image_height - 20)))

    if not free_spaces:
        # Ultimate fallback
        y_start = int(image_height * 0.65)
        free_spaces = [FreeSpace((20, y_start, image_width - 20, image_height - 20))]

    # Sort by area descending so the largest space comes first
    free_spaces.sort(key=lambda s: s.area, reverse=True)
    return free_spaces


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_ocr_data(
    filtered_results: list,
    image_width:      int,
    image_height:     int,
    question_bbox:    Optional[Tuple[int, int, int, int]],
) -> dict:
    """
    Convert raw EasyOCR results into enriched OCR data.

    Args:
        filtered_results: List of (bbox, text, conf) tuples from EasyOCR.
        image_width:      Width of the source image in pixels.
        image_height:     Height of the source image in pixels.
        question_bbox:    (x1, y1, x2, y2) bounding box of question region,
                          or None if unknown.

    Returns:
        dict with keys:
          "elements"    : List[OCRElement]
          "index"       : OCRIndex  (for fast text lookup)
          "free_spaces" : List[FreeSpace]  (sorted largest-first)
    """
    elements: List[OCRElement] = []

    for bbox, text, conf in filtered_results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        elements.append(OCRElement(
            text=text,
            x1=int(min(xs)),
            y1=int(min(ys)),
            x2=int(max(xs)),
            y2=int(max(ys)),
            conf=float(conf),
        ))

    index       = OCRIndex(elements)
    free_spaces = _detect_free_spaces(elements, image_width, image_height, question_bbox)

    return {
        "elements":    elements,
        "index":       index,
        "free_spaces": free_spaces,
    }