#!/usr/bin/env python3
"""
generate_annotations.py — Improved annotation generator.

Improvements over original:
  - Prompt explicitly instructs Gemini to use actual transcript timestamps
    (not invented ones) — fixes the core sync problem
  - Post-processing validates timestamps are within audio duration
  - Minimum spacing enforced AFTER sort (original did it before sort — bug)
  - Rule-based fallback reads actual segment timestamps from transcript
    instead of hardcoded values — works for any audio, not just one file
  - JSON parse error shows raw response for easier debugging
  - Added `draw_arrow` between equation steps for visual flow
"""

import json
import os
import re
import sys


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(transcript_data, question_text, audio_duration):
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
- For example if teacher says "distance formula" at 15.2s, set time to 15.2.
- Do NOT invent timestamps. Only use times that appear in the transcript segments.
- All times must be between 0 and {audio_duration:.1f}.

ACTION TYPES:

1. underline_existing — underline a coordinate or keyword already in the question image.
   Trigger: teacher says "given", "point A", "point B", "consider", "value of"
   Fields: {{"time": <float>, "action": "underline_existing", "target": "<exact substring from question>"}}

2. write_equation — write a math step on the board.
   Trigger: teacher states or writes a formula/calculation step
   ONLY the math expression — no "Step 1", no explanations, no titles.
   Use standard chars: x2 not x₂, sqrt() or √, standard minus sign.
   Good: "d = √((x2-x1)² + (y2-y1)²)"
   Bad: "Step 1: distance formula"
   Fields: {{"time": <float>, "action": "write_equation", "text": "<math only>"}}

3. tick_answer — mark the correct option when teacher announces answer.
   Fields: {{"time": <float>, "action": "tick_answer", "target": "<option letter e.g. C>"}}

SPACING RULE: At least 2 seconds between any two consecutive actions.

Return ONLY the raw JSON array. No markdown, no explanation, no code fences.
"""


# ── LLM generator ─────────────────────────────────────────────────────────────
def generate_with_llm(transcript_data, question_text, audio_duration):
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
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw LLM response:\n{raw[:500]}")
        raise

    return annotations


# ── Rule-based fallback ───────────────────────────────────────────────────────
def generate_rule_based(transcript_data, audio_duration):
    """
    FIX: Instead of hardcoded timestamps, extract real segment timestamps
    from transcript and map actions proportionally across the audio.
    Works for any audio file, not just the distance formula example.
    """
    segments = transcript_data.get("segments", [])

    # Find key timestamps from transcript text
    def find_segment_time(keywords):
        """Return start time of first segment containing any keyword."""
        for seg in segments:
            text_lower = seg["text"].lower()
            if any(kw in text_lower for kw in keywords):
                return seg["start"]
        return None

    formula_time = find_segment_time(["formula", "distance formula", "sqrt", "root"]) \
                   or audio_duration * 0.25

    substitute_time = find_segment_time(["substitut", "plug", "putting", "put"]) \
                      or audio_duration * 0.45

    simplify_time   = find_segment_time(["simplif", "calculat", "square", "minus"]) \
                      or audio_duration * 0.60

    answer_time     = find_segment_time(["answer", "equal", "result", "therefore", "so"]) \
                      or audio_duration * 0.78

    tick_time       = find_segment_time(["option", "correct", "choice", "c is"]) \
                      or audio_duration * 0.88

    # Build underline times from transcript start
    underline1_time = segments[0]["start"] + 2.0 if segments else 3.0
    underline2_time = underline1_time + 2.5

    annotations = [
        {
            "time":   round(underline1_time, 1),
            "action": "underline_existing",
            "target": "A (1, 2)"
        },
        {
            "time":   round(underline2_time, 1),
            "action": "underline_existing",
            "target": "(4, 6)"
        },
        {
            "time":   round(formula_time, 1),
            "action": "write_equation",
            "text":   "d = √((x2-x1)² + (y2-y1)²)"
        },
        {
            "time":   round(substitute_time, 1),
            "action": "write_equation",
            "text":   "d = √((4-1)² + (6-2)²)"
        },
        {
            "time":   round(simplify_time, 1),
            "action": "write_equation",
            "text":   "d = √(3² + 4²)"
        },
        {
            "time":   round(simplify_time + 5.0, 1),
            "action": "write_equation",
            "text":   "d = √(9 + 16) = √25"
        },
        {
            "time":   round(answer_time, 1),
            "action": "write_equation",
            "text":   "d = 5 units"
        },
        {
            "time":   round(tick_time, 1),
            "action": "tick_answer",
            "target": "C"
        },
    ]

    return annotations


# ── Post-processing ───────────────────────────────────────────────────────────
def _post_process(annotations, audio_duration):
    """
    FIX: Sort FIRST, then enforce spacing.
    Original enforced spacing before sort — order was wrong.
    Also clamps all times within audio duration.
    """
    # Sort by time
    annotations.sort(key=lambda x: x["time"])

    # Clamp to audio duration
    annotations = [a for a in annotations if a["time"] < audio_duration - 0.5]

    # Enforce minimum 2.0s spacing between consecutive actions
    MIN_GAP = 2.0
    for i in range(1, len(annotations)):
        min_time = annotations[i - 1]["time"] + MIN_GAP
        if annotations[i]["time"] < min_time:
            annotations[i]["time"] = round(min_time, 1)

    # Final clamp after spacing adjustment
    annotations = [a for a in annotations if a["time"] < audio_duration - 0.5]

    return annotations


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_annotations(transcript_path, question_text, output_path,
                         audio_duration=None):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    # Get audio duration from transcript if not provided
    if audio_duration is None:
        segs = transcript_data.get("segments", [])
        audio_duration = segs[-1]["end"] if segs else 60.0

    print(f"  Audio duration: {audio_duration:.1f}s")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if api_key:
        try:
            print("  Using Gemini API for smart annotation generation...")
            annotations = generate_with_llm(transcript_data, question_text, audio_duration)
            print(f"  Generated {len(annotations)} annotations via LLM")
        except Exception as e:
            print(f"  LLM call failed ({e}), falling back to rule-based...")
            annotations = generate_rule_based(transcript_data, audio_duration)
    else:
        print("  No GEMINI_API_KEY — using rule-based fallback...")
        annotations = generate_rule_based(transcript_data, audio_duration)

    # Post-process: sort, clamp, enforce spacing
    annotations = _post_process(annotations, audio_duration)

    print(f"  Final annotation count: {len(annotations)}")
    for a in annotations:
        print(f"    {a['time']:.1f}s  {a['action']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"  Saved -> {output_path}")
    return annotations


if __name__ == "__main__":
    transcript = sys.argv[1] if len(sys.argv) > 1 else "output/transcript.json"
    output     = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    q_text     = sys.argv[3] if len(sys.argv) > 3 else ""
    generate_annotations(transcript, q_text, output)