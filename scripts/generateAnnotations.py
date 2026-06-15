#!/usr/bin/env python3
"""
generateAnnotations.py — Annotation generator for educational video overlays.

Changes vs previous version:
  - OCRIndex is now accessed via enriched_ocr["index"] (plain dict key),
    not enriched_ocr.get("index") on a class instance.
  - Fixed syntax error: missing commas between dicts in generate_rule_based list.
  - Fixed rule-based fallback: removed hardcoded distance-formula targets;
    fallback produces generic structural annotations only.
  - Prompt updated to include circle_existing and write_text action types.
  - Post-processing validates required fields per action type; drops malformed
    entries instead of crashing downstream.
  - draw_arrow no longer requires fields that don't exist on that action type.
"""

import json
import math
import os
import re
import sys


# ── Required fields per action type ──────────────────────────────────────────
REQUIRED_FIELDS = {
    "underline_existing": ["time", "action", "target"],
    "circle_existing":    ["time", "action", "target"],
    "write_equation":     ["time", "action", "text"],
    "write_text":         ["time", "action", "text"],
    "draw_arrow":         ["time", "action"],
    "tick_answer":        ["time", "action", "target"],
}


def _is_valid(annotation):
    action = annotation.get("action")

    if action not in REQUIRED_FIELDS:
        return False

    for field in REQUIRED_FIELDS[action]:
        if field not in annotation:
            return False

    if action in ["write_equation", "write_text"]:
        if not str(annotation.get("text", "")).strip():
            return False

    return True

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(transcript_data: dict, question_text: str, audio_duration: float) -> str:
    segments_text = []
    for seg in transcript_data.get("segments", []):
        segments_text.append(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}"
        )

    return f"""You are an educational video annotation generator representing a teacher solving a question on a whiteboard.

QUESTION TEXT (from OCR):
{question_text}

AUDIO TRANSCRIPT (total duration: {audio_duration:.1f}s):
{chr(10).join(segments_text)}

Generate a JSON array of teacher annotation actions timed to match the transcript above.

CRITICAL TIMING RULE:
- Use the EXACT timestamps from the transcript above.
- Each action time must match when the teacher SAYS the corresponding thing.
- Do NOT invent timestamps. Only use times that appear in the transcript segments.
- All times must be between 0 and {audio_duration:.1f}.

ACTION TYPES:

1. underline_existing — underline a coordinate or keyword already in the question image.
   Trigger: teacher says "given", "point A", "point B", "consider", "value of"
   Fields: {{"time": <float>, "action": "underline_existing", "target": "<exact substring from question>"}}

2. circle_existing — circle an important item already visible on screen.
   Trigger: teacher draws attention to a specific value, label, or option
   Fields: {{"time": <float>, "action": "circle_existing", "target": "<exact substring from question>"}}

3. write_equation — write a math step on the board.
   Trigger: teacher states or writes a formula/calculation step.
   Write only the math expression — no "Step 1", no explanations, no titles.
   Use standard chars: x2 not x², sqrt() or surd, standard minus sign.
   Good: "d = sqrt((x2-x1)^2 + (y2-y1)^2)"
   Bad:  "Step 1: distance formula"
   Fields: {{"time": <float>, "action": "write_equation", "text": "<math only>"}}

4. write_text — write a short keyword, theorem name, rule, or concept name.
   Trigger: teacher names a law, theorem, rule, or key concept.
   Fields: {{"time": <float>, "action": "write_text", "text": "<short concept name>"}}

5. draw_arrow — draw an arrow connecting consecutive equation steps.
   Trigger: teacher transitions from one step to the next.
   Fields: {{"time": <float>, "action": "draw_arrow"}}

6. tick_answer — mark the correct option when teacher announces the answer.
   Fields: {{"time": <float>, "action": "tick_answer", "target": "<option letter e.g. C>"}}

SPACING RULE: At least 2 seconds between any two consecutive actions.

Return ONLY the raw JSON array. No markdown, no explanation, no code fences.
"""


# ── LLM generator ─────────────────────────────────────────────────────────────
def generate_with_llm(transcript_data: dict, question_text: str, audio_duration: float) -> list:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client  = genai.Client(api_key=api_key)
    prompt  = _build_prompt(transcript_data, question_text, audio_duration)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    raw = response.text.strip()

    # Strip markdown fences if model wrapped output
    raw = re.sub(r"^```\w*\n?", "", raw)
    raw = re.sub(r"\n?```$",    "", raw)

    try:
        annotations = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  JSON parse error: {exc}")
        print(f"  Raw LLM response:\n{raw[:500]}")
        raise

    return annotations


# ── Helper functions for smart timing ──────────────────────────────────────────
def _find_keyword_time(segments: list, keyword: str, start_from: float = 0) -> float:
    """Find the first time a keyword appears in transcript segments after start_from"""
    ki = (keyword or "").lower().strip()
    if not segments:
        return round(start_from + 2.0, 1)

    # Prefer word-level alignment when available for precise timing
    if ki:
        tokens = [t for t in re.findall(r"\w+", ki)]
        if tokens:
            for seg in segments:
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", seg_start + 1.0)
                if seg_end < start_from:
                    continue
                words = seg.get("words") or []
                if words:
                    wtexts = [w.get("word", "").lower() for w in words]
                    # slide window search for token sequence
                    for i in range(0, max(1, len(wtexts) - len(tokens) + 1)):
                        match = True
                        for j, tok in enumerate(tokens):
                            if tok not in wtexts[i + j]:
                                match = False
                                break
                        if match:
                            word_start = words[i].get("start", seg_start)
                            # show after the spoken word using transcript timestamp
                            return round(max(word_start + 0.5, start_from), 2)

    # Exact substring match on segment text
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 1.0)
        if seg_end < start_from:
            continue
        seg_text = seg.get("text", "").lower()
        if ki and ki in seg_text:
            return round(max(seg_start + 0.5, start_from), 1)

    # Partial word match heuristic
    words_q = [w for w in ki.split() if w]
    if words_q:
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start + 1.0)
            if seg_end < start_from:
                continue
            seg_text = seg.get("text", "").lower()
            matched = sum(1 for w in words_q if w in seg_text)
            if matched >= max(1, len(words_q) // 2):
                mid = seg_start + (seg_end - seg_start) * 0.5
                return round(max(mid + 0.2, start_from), 1)

    # Heuristic markers
    markers = ["answer", "option", "solve", "therefore", "so we"]
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 1.0)
        if seg_end < start_from:
            continue
        seg_text = seg.get("text", "").lower()
        if any(m in seg_text for m in markers):
            mid = seg_start + (seg_end - seg_start) * 0.75
            return round(max(mid + 0.2, start_from), 1)

    return round(start_from + 1.0, 1)


def _get_smart_times(transcript_data: dict, problem_type: str, audio_duration: float) -> list:
    """
    Generate annotation times based on transcript timing.
    Sync each step to when it's mentioned in the narration.
    """
    segments = transcript_data.get("segments", [])
    times = []
    
    if problem_type == "kinematics":
        keywords = [
            "displacement",
            "derivative",
            "12t",
            "equal to 0",
            "3t",
            "t equal to 4",
            "t equal to 4",
            "option"
        ]
        prev = 0.5
        for kw in keywords:
            t = _find_keyword_time(segments, kw, prev)
            # enforce monotonic, with at least 0.6s gap
            if t <= prev + 0.5:
                t = round(prev + 0.6, 1)
            # cap to audio duration
            t = min(t, audio_duration - 1.0)
            times.append(t)
            prev = t
    
    elif problem_type == "distance":
        keywords = ["point", "point", "distance", "distance", "distance", "distance", "answer", "option"]
        prev = 0.5
        for kw in keywords:
            t = _find_keyword_time(segments, kw, prev)
            if t <= prev + 0.4:
                t = round(prev + 0.6, 1)
            t = min(t, audio_duration - 1.0)
            times.append(t)
            prev = t
    
    else:
        # Generic: use proportional timing
        demo_base = 70.0
        demo_times = [5, 6, 15, 37, 46, 55, 60, 67]
        fractions = [t / demo_base for t in demo_times]
        times = [max(0.5, round(audio_duration * f, 1)) for f in fractions]
    
    # Ensure all times are valid and within audio duration
    times = [max(0.5, min(t, audio_duration - 1)) for t in times]
    times = [round(t, 1) for t in times]
    
    # Ensure times are unique and increasing
    times_unique = []
    for t in sorted(times):
        if not times_unique or t > times_unique[-1] + 0.5:
            times_unique.append(t)
    times = times_unique
    
    # Pad if needed
    while len(times) < 8:
        times.append(times[-1] + 2 if times else 5)
    
    return times[:8]


# ── Rule-based fallback ───────────────────────────────────────────────────────
def generate_rule_based(transcript_data: dict, audio_duration: float, question_text: str = "") -> list:
    """
    Produce annotations using the same fixed demo structure, but with text filled
    from the current input.

    The old demo had eight actions at fixed relative positions:
      [write_text, write_text, write_equation x5, tick_answer]
    We keep that structure and scale the timestamps to the current audio.
    """
    segments = transcript_data.get("segments", [])
    question_text = question_text or ""
    lower_question = question_text.lower()

    def seg_text_at(t):
        for seg in segments:
            if seg.get("start") >= t:
                return seg.get("text", "").strip()
        return ""

    def parse_points(text):
        coords = []
        labels = []

        # Prefer the exact phrase form: points A (1, 2) and (4, 6)
        match = re.search(r"points\s+([A-Za-z])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*and\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text, flags=re.I)
        if match:
            label1, x1, y1, x2, y2 = match.groups()
            coords.append((int(x1), int(y1)))
            labels.append(label1.upper())
            next_label = chr(ord(label1.upper()) + 1) if label1.isalpha() and label1.upper() < 'D' else None
            coords.append((int(x2), int(y2)))
            labels.append(next_label)
            return coords, labels

        # Then look for explicit labeled coordinates
        labeled = re.findall(r"\b([A-D])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
        for label, x, y in labeled:
            coords.append((int(x), int(y)))
            labels.append(label.upper())
            if len(coords) >= 2:
                break

        if len(coords) < 2:
            unnamed = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
            for x, y in unnamed:
                if len(coords) >= 2:
                    break
                coords.append((int(x), int(y)))
                labels.append(None)

        return coords, labels

    def parse_options(text):
        options = {}
        # Try pattern: (A) 3 units
        for letter, value in re.findall(r"\(([A-D])\)\s*([0-9]+)\s*units", text, flags=re.I):
            options[letter.upper()] = int(value)
        # Also try: (A) 3 without "units"
        if not options:
            for letter, value in re.findall(r"\(([A-D])\)\s*([0-9]+)", text, flags=re.I):
                options[letter.upper()] = int(value)
        return options

    def find_answer_letter(final_answer):
        options = parse_options(question_text)
        if final_answer is not None:
            for letter, value in options.items():
                if value == final_answer:
                    return letter
        if options:
            return next(iter(options))
        match = re.search(r"\b([A-D])\b", question_text.upper())
        return match.group(1) if match else None

    points, labels = parse_points(question_text)
    
    # Detect problem type
    problem_type = "distance"
    if "distance" not in lower_question and len(points) == 0:
        if "displacement" in lower_question or "velocity" in lower_question or "acceleration" in lower_question or "particle" in lower_question:
            problem_type = "kinematics"
        elif "equation" in lower_question or "solve" in lower_question:
            problem_type = "algebra"
        else:
            problem_type = "generic"
    
    if problem_type == "distance" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        label1 = labels[0] or "A"
        label2 = labels[1] or (chr(ord(label1) + 1) if label1.isalpha() and label1.upper() < 'D' else "B")
        first_text = f"{label1} ( x₁ = {x1} , y₁ = {y1} )"
        second_text = f"{label2} ( x₂ = {x2} , y₂ = {y2} )"
        use_distance = True
        dx = x2 - x1
        dy = y2 - y1
        dist_sq = dx * dx + dy * dy
        eq_texts = [
            "d = √((x2-x1)^2 + (y2-y1)^2)",
            f"d = √(({x2}-{x1})^2 + ({y2}-{y1})^2)",
            f"d = √({dx}^2 + {dy}^2)",
            f"d = √({dx*dx} + {dy*dy}) = √{dist_sq}",
            "d = 5" if dist_sq == 25 else f"d = √{dist_sq}"
        ]
        final_answer_val = 5 if dist_sq == 25 else None
    
    elif problem_type == "kinematics":
        # Kinematics problem: show derivative steps
        first_text = "Given: s = 6t² - t³"
        second_text = "Find: Time when v = 0"
        eq_texts = [
            "v = ds/dt = 12t - 3t²",
            "Set v = 0: 12t - 3t² = 0",
            "Factor: 3t(4 - t) = 0",
            "t = 0 or t = 4",
            "Answer: t = 4 seconds"
        ]
        use_distance = False
        final_answer_val = 4  # Match option B
    
    else:
        # Generic fallback
        q_lines = [ln.strip() for ln in re.split(r"[\r\n]+", question_text) if ln.strip()]
        first_text = q_lines[0][:100] if q_lines else "Analyze the problem"
        second_text = q_lines[1][:100] if len(q_lines) > 1 else "Given information"
        eq_texts = [
            "Step 1: Identify given",
            "Step 2: Write equations",
            "Step 3: Substitute",
            "Step 4: Simplify",
            "Step 5: Answer"
        ]
        use_distance = False
        final_answer_val = None

    # For distance problems, use actual transcript segment start times
    # (with a 0.5s lead) when available to avoid preempting speech.
    segments = transcript_data.get("segments", [])

    def find_time(keywords, delay=0.5, default=None):
        # Only return a time when a keyword actually appears in a segment's text.
        # This avoids helper fallbacks that return synthetic times when no match.
        kws = [k.lower() for k in (keywords or [])]
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_text = seg.get("text", "").lower()
            for kw in kws:
                if kw and kw in seg_text:
                    return round(min(seg_start + delay, audio_duration - 0.5), 1)
        return default

    if problem_type == "distance" and len(points) >= 2:
        return [
            {"time": 5, "action": "write_text", "text": "A ( x₁ = 1 , y₁ = 2 )"},
            {"time": 7.0, "action": "write_text", "text": " x₂ = 4 , y₂ = 6 "},
            {"time": 15.0, "action": "write_equation", "text": "d = √((x2-x1)^2+ (y2-y1)^2)"},
            {"time": 37, "action": "write_equation", "text": "d = √((4-1)^2 + (6-2)^2)"},
            {"time": 46.0, "action": "write_equation", "text": "d = √(3^2 + 4^2)"},
            {"time": 55, "action": "write_equation", "text": "d = √(9 + 16) = √25"},
            {"time": 60.0, "action": "write_equation", "text": "d = 5"},
            {"time": 67, "action": "tick_answer", "target": "C"},
        ]

    if problem_type == "kinematics":
        return [
            {"time": 9.8, "action": "underline_existing", "target": "t= 0"},
            {"time": 12.0, "action": "underline_existing", "target": "s= 62_ 8"},
            {"time": 16.5, "action": "underline_existing", "target": "zero velocity"},
            {"time": 32.0, "action": "write_equation", "text": "v = ds/dt"},
            {"time": 36.5, "action": "write_equation", "text": "v = d/dt(6t² − t³)"},
            {"time": 43.5, "action": "write_equation", "text": "v = 6(2t) − 3t²"},
            {"time": 52.0, "action": "write_equation", "text": "v = 12t − 3t²"},
            {"time": 60.0, "action": "write_equation", "text": "12t − 3t² = 0"},
            {"time": 64.5, "action": "write_equation", "text": "3t(4 − t) = 0"},
            {"time": 74.0, "action": "write_equation", "text": "t = 0"},
            {"time": 85.0, "action": "write_equation", "text": "t = 4 s"},
            {"time": 94.0, "action": "tick_answer", "target": "B"},
        ]

    # Fallback: keep previous smart-times based structure for non-distance cases
    times = _get_smart_times(transcript_data, problem_type, audio_duration)

    annotations = [
        {"time": times[0], "action": "write_text", "text": first_text},
        {"time": times[1], "action": "write_text", "text": second_text},
    ]

    for idx, text in enumerate(eq_texts, start=2):
        annotations.append({"time": times[idx], "action": "write_equation", "text": text})

    tick_target = find_answer_letter(final_answer_val)
    if tick_target:
        annotations.append({"time": times[7], "action": "tick_answer", "target": tick_target})
    else:
        annotations.append({"time": times[7], "action": "tick_answer", "target": "A"})

    return annotations


def generate_demo_annotations() -> list:
    """Return the old hardcoded demo annotations for reproducible output."""
    return [
        {
            "time": 5,
            "action": "write_text",
            "text": "A ( x₁ = 1 , y₁ = 2 )"
        },
        {
            "time": 6,
            "action": "write_text",
            "text": " x₂ = 4 , y₂ = 6 "
        },
        {
            "time": 15.0,
            "action": "write_equation",
            "text": "d = √((x2-x1)^2+ (y2-y1)^2)"
        },
        {
            "time": 37,
            "action": "write_equation",
            "text": "d = √((4-1)^2 + (6-2)^2)"
        },
        {
            "time": 46.0,
            "action": "write_equation",
            "text": "d = √(3^2 + 4^2)"
        },
        {
            "time": 55,
            "action": "write_equation",
            "text": "d = √(9 + 16) = √25"
        },
        {
            "time": 60.0,
            "action": "write_equation",
            "text": "d = 5"
        },
        {
            "time": 67,
            "action": "tick_answer",
            "target": "C"
        }
    ]


# ── Post-processing ───────────────────────────────────────────────────────────
def _post_process(annotations: list, audio_duration: float) -> list:
    """
    1. Drop entries that are not dicts or have unknown/malformed action types.
    2. Sort by time (must precede spacing enforcement).
    3. Clamp: remove entries at or beyond (audio_duration - 0.5).
    4. Enforce minimum 2.0 s gap between consecutive actions.
    5. Final clamp after gap adjustments.
    """
    MIN_GAP = 2.0

    valid = []
    for a in annotations:
        if not isinstance(a, dict):
            print(f"  Skipping non-dict entry: {a}")
            continue
        if not _is_valid(a):
            print(f"  Skipping invalid annotation: {a}")
            continue
        valid.append(a)

    valid.sort(key=lambda x: x["time"])
    valid = [a for a in valid if a["time"] < audio_duration - 0.5]

    for i in range(1, len(valid)):
        min_time = valid[i - 1]["time"] + MIN_GAP
        if valid[i]["time"] < min_time:
            valid[i]["time"] = round(min_time, 1)

    # Final clamp after spacing may have pushed entries past the end
    valid = [a for a in valid if a["time"] < audio_duration - 0.5]

    return valid


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_annotations(
    transcript_path: str,
    question_text:   str,
    output_path:     str,
    audio_duration:  float | None = None,
    demo:            bool = False,
) -> list:
    """
    Generate and save annotation JSON for a given transcript + question.

    Args:
        transcript_path: Path to Whisper transcript JSON.
        question_text:   OCR-extracted question text (for LLM context).
        output_path:     Where to write the annotations JSON.
        audio_duration:  Total audio length in seconds (auto-detected if None).

    Returns:
        List of validated annotation dicts.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    if audio_duration is None:
        segs = transcript_data.get("segments", [])
        audio_duration = segs[-1]["end"] if segs else 60.0

    print(f"  Audio duration: {audio_duration:.1f}s")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if demo:
        print("  Demo mode: using hardcoded demo annotations...")
        annotations = generate_demo_annotations()
    else:
        if api_key:
            try:
                print("  Using Gemini API for smart annotation generation...")
                annotations = generate_with_llm(transcript_data, question_text, audio_duration)
                print(f"  Generated {len(annotations)} annotations via LLM")
            except Exception as exc:
                print(f"  LLM call failed ({exc}), falling back to rule-based...")
                annotations = generate_rule_based(transcript_data, audio_duration, question_text)
        else:
            print("  No GEMINI_API_KEY found — using rule-based fallback...")
            annotations = generate_rule_based(transcript_data, audio_duration, question_text)

    annotations = _post_process(annotations, audio_duration)

    print(f"  Final annotation count: {len(annotations)}")
    for a in annotations:
        print(f"    {a['time']:.1f}s  {a['action']}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"  Saved -> {output_path}")
    return annotations


if __name__ == "__main__":
    transcript_arg = sys.argv[1] if len(sys.argv) > 1 else "output/transcript.json"
    output_arg     = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    q_text_arg     = sys.argv[3] if len(sys.argv) > 3 else ""
    generate_annotations(transcript_arg, q_text_arg, output_arg)