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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from transcribe import transcribe_audio
from ocr_question import extract_question_info
from generate_annotations import generate_annotations
from render_video import render_video


def _validate_inputs(image_path, audio_path):
    """
    FIX: Validate input files exist before starting the pipeline.
    Original would fail deep inside a step with a confusing error.
    Now fails immediately with a clear message.
    """
    missing = []
    if not os.path.exists(image_path):
        missing.append(f"Image not found: {image_path}")
    if not os.path.exists(audio_path):
        missing.append(f"Audio not found: {audio_path}")
    if missing:
        for msg in missing:
            print(f"ERROR: {msg}")
        sys.exit(1)


def _get_audio_duration(transcript_path):
    """
    Extract audio duration from transcript so generate_annotations
    can validate and clamp timestamps correctly.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        return segs[-1]["end"] if segs else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="PW Automated Annotation System")
    parser.add_argument("--image",           default="input/question.png")
    parser.add_argument("--audio",           default="input/narration.mp3")
    parser.add_argument("--output",          default="output/final.mp4")
    parser.add_argument("--transcript",      default="output/transcript.json")
    parser.add_argument("--annotations",     default="output/annotations.json")
    parser.add_argument("--whisper-model",   default="base",
                        help="tiny / base / small / medium / large")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Reuse existing transcript instead of re-transcribing")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Validate inputs before touching any step
    _validate_inputs(args.image, args.audio)

    # ── Step 1: Transcribe ───────────────────────────────────────────────────
    if args.skip_transcribe and os.path.exists(args.transcript):
        print("[1/4] Skipping transcription (reusing existing transcript)")
    else:
        print("[1/4] Transcribing audio...")
        transcribe_audio(args.audio, args.transcript, args.whisper_model)

    # ── Step 2: OCR ──────────────────────────────────────────────────────────
    print("[2/4] Running OCR on question image...")
    question_text, option_positions, question_bbox, enriched_ocr = \
        extract_question_info(args.image)
    print(f"      Question text: {question_text[:80]}...")
    print(f"      Options found: {list(option_positions.keys())}")

    # ── Step 3: Generate annotations ─────────────────────────────────────────
    print("[3/4] Generating annotations...")
    # FIX: Pass audio_duration so annotations are validated against actual length
    audio_duration = _get_audio_duration(args.transcript)
    generate_annotations(
        args.transcript,
        question_text,
        args.annotations,
        audio_duration=audio_duration,
    )

    # ── Step 4: Render ────────────────────────────────────────────────────────
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