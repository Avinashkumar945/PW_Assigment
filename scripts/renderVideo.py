#!/usr/bin/env python3
"""
renderVideo.py — Annotation video renderer for the PW Annotation System.

Key fixes vs previous version:
  - enriched_ocr is now a plain dict (not a class). Access free_spaces and
    index via dict keys: enriched_ocr.get("free_spaces"), enriched_ocr.get("index").
  - _build_schedule: ocr_index fetched as enriched_ocr.get("index") dict value,
    and find_by_text called correctly on the OCRIndex object inside it.
  - Easing function (ease_in_out) for natural handwriting feel.
  - Fixed wy mutation bug in _build_schedule — wy tracked as running cursor.
  - Smarter frame cache (per-action progress rounding, bounded to 500 entries).
  - Token reveal skips whitespace tokens (no wasted reveal time).
  - Radical sign scales proportionally with font size.
  - write_equation duration clamped intelligently per token count.
"""

import json
import math
import os
import random
import re
import sys

import numpy as np
from moviepy import AudioFileClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont


# ── Constants ────────────────────────────────────────────────────────────────
PEN_COLOR  = (0, 0, 0)
PEN_WIDTH  = 3
TARGET_FPS = 24


# ── Easing ───────────────────────────────────────────────────────────────────
def ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out. Makes writing look natural instead of robotic."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ── Font loader ──────────────────────────────────────────────────────────────
def _find_font(family: str = "body", size: int = 26) -> ImageFont.FreeTypeFont:
    candidates = {
        "title": [
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "body": [
            "C:/Windows/Fonts/BRADHITC.TTF",
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoepr.ttf",
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
    }
    for path in candidates.get(family, candidates["body"]):
        if os.path.exists(path):
            try:
                font_size = size + 4 if "Inkfree.ttf" in path else size
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


# ── Tokenizer ────────────────────────────────────────────────────────────────
def split_into_math_tokens(text: str) -> list:
    """
    Split equation into meaningful tokens, filtering whitespace-only tokens
    so they don't waste reveal slots.
    """
    raw = re.findall(r'[A-Za-z0-9₀-₉⁰-⁹]+|\s+|[^\w\s]', text)
    return [t for t in raw if t.strip()]


# ── Handwriting drawing primitives ───────────────────────────────────────────
def _jitter_line(draw, x1, y1, x2, y2, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw a slightly jittery line simulating handwriting."""
    dx, dy = x2 - x1, y2 - y1
    dist   = math.hypot(dx, dy)
    steps  = max(int(dist / 4), 1)
    for i in range(steps + 1):
        t  = i / steps
        px = x1 + dx * t + random.uniform(-0.5, 0.5)
        py = y1 + dy * t + random.uniform(-0.5, 0.5)
        r  = width / 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)


def draw_progressive_underline(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    p = ease_in_out(progress)
    _jitter_line(draw, x1, y1, x1 + (x2 - x1) * p, y2, width, color)


def draw_progressive_circle(draw, cx, cy, radius, progress, width=PEN_WIDTH, color=PEN_COLOR):
    p      = ease_in_out(progress)
    steps  = max(int(p * 60), 2)
    angles = np.linspace(0, 2 * math.pi * p, steps)
    for i in range(len(angles) - 1):
        x1 = cx + radius * math.cos(angles[i])
        y1 = cy + radius * math.sin(angles[i])
        x2 = cx + radius * math.cos(angles[i + 1])
        y2 = cy + radius * math.sin(angles[i + 1])
        _jitter_line(draw, x1, y1, x2, y2, width, color)


def draw_progressive_arrow(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    p  = ease_in_out(progress)
    ex = x1 + (x2 - x1) * p
    ey = y1 + (y2 - y1) * p
    _jitter_line(draw, x1, y1, ex, ey, width, color)
    if progress > 0.8:
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L > 0:
            dx /= L; dy /= L
            al, aw = 12, 6
            bx, by = ex - dx * al, ey - dy * al
            _jitter_line(draw, ex, ey, bx + dy * aw, by - dx * aw, width, color)
            _jitter_line(draw, ex, ey, bx - dy * aw, by + dx * aw, width, color)


def draw_progressive_slash(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    p = ease_in_out(progress)
    _jitter_line(draw, x1, y1, x1 + (x2 - x1) * p, y1 + (y2 - y1) * p, width, color)


# ── Text drawing with subscript/superscript support ──────────────────────────
SUPERSCRIPTS = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
}


def _sub_font(font):
    try:
        path = getattr(font, "path", None)
        if path and os.path.exists(path):
            return ImageFont.truetype(path, max(10, int(font.size * 0.65)))
    except Exception:
        pass
    return font


def draw_custom_text(draw, x, y, text, font, color):
    """Draw text with subscript/superscript mapping. Returns width drawn."""
    sf     = _sub_font(font)
    curr_x = x
    for ch in text:
        if ch == '−':
            ch = '-'
        if '₀' <= ch <= '₉':
            glyph = str(ord(ch) - 0x2080)
            draw.text((curr_x, y + int(font.size * 0.25)), glyph, fill=color, font=sf)
            curr_x += draw.textlength(glyph, font=sf)
        elif ch in SUPERSCRIPTS:
            glyph = SUPERSCRIPTS[ch]
            draw.text((curr_x, y - int(font.size * 0.15)), glyph, fill=color, font=sf)
            curr_x += draw.textlength(glyph, font=sf)
        else:
            draw.text((curr_x, y), ch, fill=color, font=font)
            curr_x += draw.textlength(ch, font=font)
    return curr_x - x


def text_width(draw, text, font):
    sf = _sub_font(font)
    w  = 0
    for ch in text:
        if ch == '−':
            ch = '-'
        if '₀' <= ch <= '₉':
            w += draw.textlength(str(ord(ch) - 0x2080), font=sf)
        elif ch in SUPERSCRIPTS:
            w += draw.textlength(SUPERSCRIPTS[ch], font=sf)
        else:
            w += draw.textlength(ch, font=font)
    return w

                                                                                                                                                                                                                                                                            
def _parse_radicand(part):
      """√ ke baad ka text parse karke (inside, rest) return karta hai."""
      if part.startswith("("):
          depth, closing = 0, -1
          for ci, ch in enumerate(part):
              if ch == "(":
                  depth += 1
              elif ch == ")":
                  depth -= 1
                  if depth == 0:
                      closing = ci
                      break
          if closing != -1:
              return part[1:closing], part[closing + 1:]
          else:
              return part[1:], ""
      else:
          m = re.match(r'^[0-9]+', part)
          if m:
              return m.group(0), part[len(m.group(0)):]
          else:
              return part, ""

# ── Radical renderer ─────────────────────────────────────────────────────────

def draw_math_with_radicals(draw, x, y, text, font, color, full_text=None):
      if "√" not in text:
          draw_custom_text(draw, x, y, text, font, color)
          return

      fs     = font.size
      tail_h = int(fs * 0.55)
      head_h = int(fs * 0.85)
      tick_w = int(fs * 0.55)

      parts     = text.split("√")
      bar_parts = (full_text.split("√")
                   if full_text and "√" in full_text
                   else parts)
      curr_x = x

      for idx, part in enumerate(parts):
          if idx == 0:
              if part:
                  curr_x += draw_custom_text(draw, curr_x, y, part, font, color)
              continue

          inside, rest = _parse_radicand(part)

          # FULL text se bar width calculate karo — yahi fix hai
          if idx < len(bar_parts):
              full_inside, _ = _parse_radicand(bar_parts[idx])
          else:
              full_inside = inside

          full_iw = text_width(draw, full_inside, font) if full_inside else 0

          # Radical sign draw karo — bar FULL width ka hoga
          r_x0, r_y0 = curr_x,                y + tail_h // 2
          r_x1, r_y1 = curr_x + tick_w // 4,  y + tail_h
          r_x2, r_y2 = curr_x + tick_w // 2,  y + tail_h
          r_x3, r_y3 = curr_x + tick_w,        y - head_h + fs
          r_x4, r_y4 = curr_x + tick_w + int(full_iw) + 4, y - head_h + fs

          draw.line(
              [(r_x0, r_y0), (r_x1, r_y1), (r_x2, r_y2),
               (r_x3, r_y3), (r_x4, r_y4)],
              fill=color, width=2, joint="round",
          )

          if inside:
              draw_custom_text(draw, curr_x + tick_w + 2, y, inside, font, color)

          curr_x += tick_w + int(full_iw) + 8
          if rest:
              curr_x += draw_custom_text(draw, curr_x, y, rest, font, color)
              
              
# ── Schedule builder ─────────────────────────────────────────────────────────
def _build_schedule(
    annotations:      list,
    total_duration:   float,
    enriched_ocr:     dict,
    option_positions: dict,
    draw_ref,
    font,
    image_size:      tuple | None = None,
    question_bbox:   tuple | None = None,
) -> list:
    """
    Compute geometry and timing for every annotation.

    FIX: enriched_ocr is now a plain dict — access via dict keys.
    FIX: wy tracked as a running cursor correctly across passes.
    FIX: write_equation duration based on token count, not just segment gap.
    """
    enriched_ocr = enriched_ocr or {}

    # --- OCR index (OCRIndex object, or None) --------------------------------
    ocr_index   = enriched_ocr.get("index")     # OCRIndex instance or None
    free_spaces = enriched_ocr.get("free_spaces", [])

    if free_spaces:
        # FreeSpace.bounds is (x1, y1, x2, y2)
        bounds = free_spaces[0].bounds if hasattr(free_spaces[0], "bounds") else free_spaces[0]
        rx1, ry1, rx2, ry2 = bounds
    else:
        rx1, ry1, rx2, ry2 = 200, 420, 1150, 630

    img_w, img_h = image_size if image_size else (rx2, ry2)
    LINE_GAP      = 62

    # Determine write area inside the free space (prefer blank area)
    padding = 10
    write_x1 = rx1 + padding
    write_y1 = ry1 + padding
    write_x2 = rx2 - padding
    write_y2 = ry2 - padding

    # Fallback centered area when no meaningful free space
    if (write_x2 - write_x1) < 80 or (write_y2 - write_y1) < 40:
        centered_x = int(img_w * 0.32)
        centered_y = int(img_h * 0.16) - 2
        wx = min(max(centered_x, rx1 + 20), max(rx2 - 220, rx1 + 20))
        wy = centered_y
    else:
        wx = write_x1
        wy = write_y1

    # Column wrap width for multiple columns inside free space
    column_width = min(220, write_x2 - write_x1)

    schedule = []

    for i, ann in enumerate(annotations):
        action = ann["action"]
        t      = ann["time"]
        entry  = {**ann, "write_start": t}

        # Next annotation time (for duration stretching)
        t_next = total_duration - 1.0
        for j in range(i + 1, len(annotations)):
            t_next = annotations[j]["time"]
            break
        segment_gap = t_next - t

        if action == "write_equation":
            tokens    = split_into_math_tokens(ann.get("text", ""))
            n_tokens  = len(tokens)
            natural   = n_tokens / 2.0
            write_dur = max(1.5, min(natural, segment_gap * 0.88))
            entry["write_pos"]      = (wx, wy)
            entry["write_duration"] = write_dur
            entry["write_end"]      = t + write_dur
            wy += LINE_GAP
            # Wrap to next column if exceeding free space
            if 'write_y2' in locals() and wy > write_y2:
                wy = write_y1
                wx = min(wx + column_width, write_x2 - 20)

        elif action == "write_text":
            tokens    = ann.get("text", "").split()
            n_tokens  = len(tokens)
            write_dur = max(1.0, min(n_tokens * 0.4, segment_gap * 0.88))
            entry["write_pos"]      = (wx, wy)
            entry["write_duration"] = write_dur
            entry["write_end"]      = t + write_dur
            wy += LINE_GAP
            if 'write_y2' in locals() and wy > write_y2:
                wy = write_y1
                wx = min(wx + column_width, write_x2 - 20)

        elif action == "underline_existing":
            target = ann.get("target", "")
            entry["underline_duration"] = 0.8
            entry["write_end"]          = t + 0.8
            if ocr_index and target:
                matches = ocr_index.find_by_text(target, threshold=0.4)
                if matches:
                    elem = matches[0]
                    entry["underline_params"] = (elem.x1, elem.y2 + 4, elem.x2, elem.y2 + 4)
                else:
                    entry["underline_params"] = (60, 117, 300, 117)
            else:
                entry["underline_params"] = (60, 117, 300, 117)

        elif action == "circle_existing":
            target = ann.get("target", "")
            entry["circle_duration"] = 0.8
            entry["write_end"]       = t + 0.8
            if ocr_index and target:
                matches = ocr_index.find_by_text(target, threshold=0.4)
                if matches:
                    elem   = matches[0]
                    cx     = int((elem.x1 + elem.x2) / 2)
                    cy     = int((elem.y1 + elem.y2) / 2)
                    radius = max(20, int(max(elem.x2 - elem.x1, elem.y2 - elem.y1) / 2) + 8)
                    entry["circle_params"] = (cx, cy, radius)
                else:
                    entry["circle_params"] = (145, 91, 30)
            else:
                entry["circle_params"] = (145, 91, 30)

        elif action == "draw_arrow":
            entry["arrow_duration"] = 0.6
            entry["write_end"]      = t + 0.6
            entry["arrow_params"]   = (wx + 60, wy - LINE_GAP - 10, wx + 60, wy - 10)

        elif action == "tick_answer":
            target     = (ann.get("target") or ann.get("option", "")).strip().upper()
            opt_letter = target.replace("OPTION", "").strip()
            entry["tick_duration"] = 0.5
            entry["write_end"]     = t + 0.5
            if opt_letter in option_positions:
                bbox = option_positions[opt_letter]
                xs   = [p[0] for p in bbox]
                ys   = [p[1] for p in bbox]
                ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)
                entry["tick_params"] = (ox1 - 4, oy2 + 4, ox1 + 42, oy1 - 4)
            else:
                entry["tick_params"] = (18, 300, 69, 240)

        else:
            # Unknown action type — add minimal entry so it doesn't crash
            entry["write_end"] = t + 0.5

        schedule.append(entry)

    return schedule

# ── Frame renderer ────────────────────────────────────────────────────────────
def _render_frame(t, background, schedule, fonts):
    font  = fonts[0]
    frame = Image.new("RGB", background.size, (255, 255, 255))
    frame.paste(background, (0, 0))
    draw  = ImageDraw.Draw(frame, "RGBA")

    _DUR_KEYS = {
        "write_equation":     "write_duration",
        "write_text":         "write_duration",
        "underline_existing": "underline_duration",
        "circle_existing":    "circle_duration",
        "draw_arrow":         "arrow_duration",
        "tick_answer":        "tick_duration",
    }

    for ann in schedule:
        if t < ann["write_start"]:
            continue

        action   = ann["action"]
        end      = ann["write_end"]
        dur_key  = _DUR_KEYS.get(action, "write_duration")
        duration = ann.get(dur_key, 1.0)
        raw_p    = 1.0 if t >= end else (t - ann["write_start"]) / max(duration, 0.01)
        progress = ease_in_out(raw_p)

        if action in ("write_equation", "write_text"):
              full_text = ann.get("text", "")
              wx, wy    = ann["write_pos"]
              tokens    = split_into_math_tokens(full_text)
              k         = max(1, int(progress * len(tokens)))
              partial   = "".join(tokens[:k])
              draw_math_with_radicals(draw, wx, wy, partial, font, PEN_COLOR,full_text=full_text)
        elif action == "underline_existing":
            params = ann.get("underline_params")
            if params:
                draw_progressive_underline(draw, *params, progress)

        elif action == "circle_existing":
            params = ann.get("circle_params")
            if params:
                draw_progressive_circle(draw, *params, progress)

        elif action == "draw_arrow":
            params = ann.get("arrow_params")
            if params:
                draw_progressive_arrow(draw, *params, progress)

        elif action == "tick_answer":
            params = ann.get("tick_params")
            if params:
                draw_progressive_slash(draw, *params, progress)

    return np.array(frame)


# ── Frame cache ───────────────────────────────────────────────────────────────
def _cache_key(t, schedule):
    """Round progress to 2 dp as cache key — reduces redundant renders."""
    key_parts = []
    for ann in schedule:
        if t < ann["write_start"]:
            key_parts.append("pending")
        elif t >= ann["write_end"]:
            key_parts.append("done")
        else:
            action  = ann["action"]
            dur_key = {
                "write_equation":     "write_duration",
                "write_text":         "write_duration",
                "underline_existing": "underline_duration",
                "circle_existing":    "circle_duration",
                "draw_arrow":         "arrow_duration",
                "tick_answer":        "tick_duration",
            }.get(action, "write_duration")
            dur = ann.get(dur_key, 1.0)
            p   = round((t - ann["write_start"]) / max(dur, 0.01), 2)
            key_parts.append(f"active_{p}")
    return tuple(key_parts)


# ── Entry point ───────────────────────────────────────────────────────────────
def render_video(
    image_path:       str,
    annotations_path: str,
    audio_path:       str,
    output_path:      str,
    option_positions: dict | None = None,
    question_bbox:    tuple | None = None,
    enriched_ocr:     dict | None = None,
):
    """
    Render the final annotated video.

    Args:
        image_path:       Background question image.
        annotations_path: JSON file of timed annotation actions.
        audio_path:       Narration audio file.
        output_path:      Where to write the output MP4.
        option_positions: {letter: bbox} from OCR (for tick_answer).
        question_bbox:    (x1, y1, x2, y2) of question region (unused directly).
        enriched_ocr:     Dict with "index" and "free_spaces" keys from ocrUtils.
    """
    option_positions = option_positions or {}
    enriched_ocr     = enriched_ocr or {}

    background = Image.open(image_path).convert("RGB")
    # Ensure dimensions are even (yuv420p requires even width/height)
    bw, bh = background.size
    ew, eh = bw + (bw % 2), bh + (bh % 2)
    if (ew, eh) != (bw, bh):
        bg2 = Image.new("RGB", (ew, eh), (255, 255, 255))
        bg2.paste(background, (0, 0))
        background = bg2

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    font  = _find_font("body", 28)
    fonts = (font,)

    audio          = AudioFileClip(audio_path)
    total_duration = audio.duration

    dummy    = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    schedule = _build_schedule(
        annotations, total_duration, enriched_ocr,
        option_positions, dummy, font,
        background.size,
        question_bbox,
    )

    frame_cache: dict = {}

    def make_frame(t):
        key = _cache_key(t, schedule)
        if key not in frame_cache:
            frame_cache[key] = _render_frame(t, background, schedule, fonts)
            if len(frame_cache) > 500:
                oldest = next(iter(frame_cache))
                del frame_cache[oldest]
        return frame_cache[key]

    print(f"  Rendering {total_duration:.1f}s at {TARGET_FPS} fps...")

    # Diagnostic: render a sample frame to verify shape/dtype
    sample = make_frame(0.1)
    try:
        print(f"  Sample frame: shape={sample.shape}, dtype={sample.dtype}, min={sample.min()}, max={sample.max()}")
    except Exception:
        pass

    video = VideoClip(make_frame, duration=total_duration).with_fps(TARGET_FPS)
    video = video.with_effects([vfx.FadeIn(0.6), vfx.FadeOut(0.6)])
    video = video.with_audio(audio)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Prefer H.264 with a yuv420p pixel format for maximum player compatibility.
    try:
        video.write_videofile(
            output_path,
            fps=TARGET_FPS,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-movflags", "+faststart",
            ],
            logger="bar",
        )
    except Exception as e:
        # Fallback to a more widely-available codec if libx264/aac is not present
        print(f"Primary encoding failed: {e}. Falling back to MPEG4.")
        video.write_videofile(
            output_path,
            fps=TARGET_FPS,
            codec="mpeg4",
            audio_codec="libmp3lame",
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
            logger="bar",
        )
    print(f"  Done → {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    ann = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    aud = sys.argv[3] if len(sys.argv) > 3 else "input/narration.mp3"
    out = sys.argv[4] if len(sys.argv) > 4 else "output/final.mp4"
    render_video(img, ann, aud, out)