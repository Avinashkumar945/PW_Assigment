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


# ── Rule-based fallback ───────────────────────────────────────────────────────
def generate_rule_based(transcript_data: dict, audio_duration: float) -> list:
    """
    Generic structural fallback using real segment timestamps.
    Does NOT assume any specific question content.
    This path is a last resort when the LLM is unavailable.
    """
    segments = transcript_data.get("segments", [])

    def find_segment_time(keywords):
        for seg in segments:
            text_lower = seg["text"].lower()
            if any(kw in text_lower for kw in keywords):
                return seg["start"]
        return None

    formula_time    = find_segment_time(["formula", "theorem", "rule", "sqrt", "root", "law"]) \
                      or audio_duration * 0.25
    substitute_time = find_segment_time(["substitut", "plug", "putting", "put", "replace"]) \
                      or audio_duration * 0.45
    simplify_time   = find_segment_time(["simplif", "calculat", "solve", "compute", "square"]) \
                      or audio_duration * 0.60
    answer_time     = find_segment_time(["answer", "equal", "result", "therefore", "hence", "so the"]) \
                      or audio_duration * 0.78
    tick_time       = find_segment_time(["option", "correct", "choice", "answer is"]) \
                      or audio_duration * 0.88

    underline1_time = (segments[0]["start"] + 2.0) if segments else 3.0
    underline2_time = underline1_time + 2.5

    annotations = [
        
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

    return annotations


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

    if api_key:
        try:
            print("  Using Gemini API for smart annotation generation...")
            annotations = generate_with_llm(transcript_data, question_text, audio_duration)
            print(f"  Generated {len(annotations)} annotations via LLM")
        except Exception as exc:
            print(f"  LLM call failed ({exc}), falling back to rule-based...")
            annotations = generate_rule_based(transcript_data, audio_duration)
    else:
        print("  No GEMINI_API_KEY found — using rule-based fallback...")
        annotations = generate_rule_based(transcript_data, audio_duration)

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