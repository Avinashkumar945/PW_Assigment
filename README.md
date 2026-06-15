# PW Automated Annotation System

An automated pipeline that transforms a question image and audio narration into a fully annotated educational video — complete with character-by-character text animation, synced timestamps, and visual highlights for questions and answer options.

Table of Contents
- How It Works
- Project Structure
- Setup
- Usage
- Pipeline Breakdown
  - Step 1 — Audio Transcription
  - Step 2 — OCR Question Extraction
  - Step 3 — Annotation Generation
  - Step 4 — Video Rendering
- Bonus: Rename Questions Utility
- Output Files
- Configuration
- Tech Stack

---

## How It Works

`input/question.png` + `input/narration.mp3`

```text
              input/question.png
              input/narration.mp3
                         │
                         ▼
   ┌────────────────────────────────────┐
   │  1. Transcribe Audio (Whisper)   │
   │     → word-level timestamps       │
   └──────────┬────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │  2. OCR Question Extraction       │
   │     (EasyOCR) → text + option box │
   └──────────┬────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │  3. Generate Annotations         │
   │     (Gemini API / rule fallback)  │
   │     → timestamped solution steps  │
   └──────────┬────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │  4. Render Video                 │
   │     (PIL + MoviePy)               │
   │     → animated video + audio     │
   └──────────┬────────────────────────┘
              ▼
        output/final.mp4
```

The system reads a static question image and a teacher’s audio explanation, then automatically produces a polished instructional video where solution steps appear on screen in sync with the narration.

---

## Project Structure

PW-Automated-Annotation-System/

```text
├── main.py                           # Entry point — runs the full 4-step pipeline
├── scripts/
│   ├── transcribe.py                 # Step 1: Audio → timestamped transcript (Whisper)
│   ├── ocrQuestion.py               # Step 2: Image → question text + option boxes (EasyOCR)
│   ├── generateAnnotations.py      # Step 3: Transcript → timed annotations (Gemini/rules)
│   ├── renderVideo.py              # Step 4: Compose final annotated video (PIL + MoviePy)
│   ├── rename_questions.py        # Utility: bulk-rename images from ZIP + Excel metadata
│   └── ocrUtils.py                  # OCR enrichment helpers used by ocrQuestion.py
├── input/
│   ├── question.png                 # Source question image (MCQ with options)
│   └── narration.mp3                # Teacher’s audio explanation
├── output/
│   ├── transcript.json             # Whisper output with word-level timestamps
│   ├── annotations.json            # Generated timestamped annotations
│   └── final.mp4                   # Final rendered video
├── task02brief.md                   # Writing style profile template (Task 02)
└── task03_explanation.md          # Explanation of the rename utility (Task 03)
```

---

## Setup

1) Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies (from code usage)

- `openai-whisper` (audio transcription with word-level timestamps)
- `easyocr` + `opencv-python` (OCR)
- `moviepy` (video assembly)
- `Pillow` (frame rendering)
- `google-genai` (Gemini API for smart annotation generation)
- `pandas` + `openpyxl` (rename utility)
- `numpy` (frame array operations)

2) Set up FFmpeg

Whisper/MoviePy require FFmpeg.

If ffmpeg is not on PATH, install one of these (Windows examples):
- `winget install Gyan.FFmpeg`

3) Set GEMINI API key (recommended)

Environment variable:

- `GEMINI_API_KEY` (preferred)

Without it, the system falls back to the rule-based annotation generator.

---

## Usage

### Run full pipeline

```bash
python main.py --image input/question.png --audio input/narration.mp3 --output output/final.mp4
```

### Skip re-transcription (reuse existing transcript)

```bash
python main.py --skip-transcribe
```

---

## Pipeline Breakdown

### Step 1 — Audio Transcription (scripts/transcribe.py)

- Loads Whisper model (default: `base`)
- Transcribes with `word_timestamps=True`
- Saves output JSON with segments/words

Output: `output/transcript.json`

### Step 2 — OCR Question Extraction (scripts/ocrQuestion.py)

- Runs EasyOCR on the question image
- Extracts:
  - `full_text` (question + any detected text)
  - `option_positions` (bounding boxes per option letter)
  - `option_text_map` (text after each option label)
  - `question_bbox` (question region only; options excluded)
  - `enriched_ocr` (via `scripts/ocrUtils.py`)

### Step 3 — Annotation Generation (scripts/generateAnnotations.py)

Two modes:

1) Gemini API (primary)
- Sends transcript + OCR question text
- Asks the model to output a JSON array of timed actions

2) Rule-based fallback
- Uses pattern matching to create a reasonable generic annotation sequence

Annotation actions include:
- `underline_existing`
- `circle_existing`
- `write_equation`
- `write_text`
- `draw_arrow`
- `tick_answer`

Output: `output/annotations.json`

### Step 4 — Video Rendering (scripts/renderVideo.py)

- Loads the question image as background
- Builds a frame schedule from annotations + timings
- Renders handwriting-like overlays using PIL
- Assembles MP4 with audio using MoviePy

Output: `output/final.mp4`

---

## Bonus: Rename Questions Utility

`scripts/rename_questions.py` fixes a workflow gap:

- ZIP assets contain images with random/hashed names
- Excel metadata maps those files to Q1/Q2/... and solution images
- The script renames/copies them into a clean format

### Command

```bash
python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx --output output/renamed/
```

---

## Output Files

- `output/transcript.json` — Whisper transcript JSON
- `output/annotations.json` — timestamped annotation actions
- `output/final.mp4` — rendered video with synced audio

---

## Configuration

Environment variables:

- `GEMINI_API_KEY` (Gemini annotation generation)
- `GOOGLE_API_KEY` (alternate key name)

---

## Tech Stack

- OpenAI Whisper — speech-to-text with word-level timestamps
- EasyOCR — OCR text + option detection
- Google Gemini 2.0 Flash — smart annotation generation
- Pillow (PIL) — drawing/handwriting overlay
- MoviePy — MP4 encoding + audio sync
- OpenCV — EasyOCR dependency
- Pandas + OpenPyXL — Excel metadata parsing

