#!/usr/bin/env python3
"""
PW Automated Annotation System
================================
Generates annotated educational videos from a question image + audio narration.

Usage:
    python main.py
    python main.py --image input/question.png --audio input/narration.mp3
    python main.py --skip-transcribe   # reuse existing transcript

Pipeline:
    1. Transcribe audio  (Whisper)         -> word-level timestamps
    2. OCR question image (EasyOCR)        -> question text + option positions
    3. Generate annotations (Gemini/rules) -> timestamped actions synced to audio
    4. Render video (PIL + MoviePy)        -> final annotated video with audio

Set GEMINI_API_KEY for smart LLM-based annotations (works for any question).
Without it the system falls back to keyword-based timestamp matching.
"""

import argparse
import json
import os
import sys

# ── Add scripts/ to path ─────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

# ── Imports — file names must match exactly (case-sensitive on Linux/Mac) ────
from transcribe          import transcribe_audio        # transcribe.py
from ocrQuestion         import extract_question_info   # ocrQuestion.py
from generateAnnotations import generate_annotations    # generateAnnotations.py
from renderVideo         import render_video            # renderVideo.py


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_inputs(image_path: str, audio_path: str) -> None:
    """Fail immediately with a clear message if input files are missing."""
    missing = []
    if not os.path.exists(image_path):
        missing.append(f"Image not found: {image_path}")
    if not os.path.exists(audio_path):
        missing.append(f"Audio not found: {audio_path}")
    if missing:
        for msg in missing:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


def _get_audio_duration(transcript_path: str) -> float | None:
    """
    Extract audio duration from transcript so generate_annotations
    can validate and clamp timestamps correctly.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        return float(segs[-1]["end"]) if segs else None
    except Exception:
        return None


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PW Automated Annotation System")
    parser.add_argument("--image",           default="input/question.png",
                        help="Path to question background image")
    parser.add_argument("--audio",           default="input/narration.mp3",
                        help="Path to narration audio file")
    parser.add_argument("--output",          default="output/final.mp4",
                        help="Path for the output video")
    parser.add_argument("--transcript",      default="output/transcript.json",
                        help="Path to save/load Whisper transcript JSON")
    parser.add_argument("--annotations",     default="output/annotations.json",
                        help="Path to save/load annotations JSON")
    parser.add_argument("--demo-annotations", action="store_true",
                        help="Use demo hardcoded annotations instead of LLM/rules fallback")
    parser.add_argument("--whisper-model",   default="base",
                        help="Whisper model size: tiny | base | small | medium | large")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Reuse existing transcript instead of re-transcribing")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Guard: check inputs before touching any pipeline step
    _validate_inputs(args.image, args.audio)

    # ── Step 1: Transcribe ───────────────────────────────────────────────────
    if args.skip_transcribe and os.path.exists(args.transcript):
        print("[1/4] Skipping transcription (reusing existing transcript)")
    else:
        print("[1/4] Transcribing audio...")
        transcribe_audio(args.audio, args.transcript, args.whisper_model)

    # ── Step 2: OCR ──────────────────────────────────────────────────────────
    print("[2/4] Running OCR on question image...")
    question_text, option_positions, option_text_map, question_bbox, enriched_ocr = \
        extract_question_info(args.image)
    print(f"      Question text : {question_text[:80]}...")
    print(f"      Options found : {list(option_positions.keys())}")

    # ── Step 3: Generate annotations ─────────────────────────────────────────
    print("[3/4] Generating annotations...")
    audio_duration = _get_audio_duration(args.transcript)
    generate_annotations(
        args.transcript,
        question_text,
        args.annotations,
        audio_duration=audio_duration,
        demo=args.demo_annotations,
    )

    # ── Step 4: Render video ──────────────────────────────────────────────────
    print("[4/4] Rendering video...")
    render_video(
        args.image,
        args.annotations,
        args.audio,
        args.output,
        option_positions,
        question_bbox,
        enriched_ocr,
    )

    print(f"\n✓ Done! Video saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()