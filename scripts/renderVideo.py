#!/usr/bin/env python3
"""
render_video.py — Improved annotation video renderer.

Key improvements over original:
  - Easing function (ease_in_out) for natural handwriting feel
  - Fixed wy mutation bug in _build_schedule
  - Smarter frame cache (per-action progress rounding)
  - Token reveal skips whitespace tokens (no wasted reveal time)
  - Radical sign scales proportionally with font size
  - write_equation duration clamped intelligently per token count
  - Clean separation of geometry and rendering logic
"""

import json
import os
import sys
import math
import random
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip, vfx

# ── Constants ────────────────────────────────────────────────────────────────
PEN_COLOR  = (0, 0, 0)
PEN_WIDTH  = 3
TARGET_FPS = 24

# ── Easing ───────────────────────────────────────────────────────────────────
def ease_in_out(t):
    """
    Smooth cubic ease-in-out.
    Starts slow, speeds up in the middle, slows at the end.
    Makes writing look natural instead of robotic linear reveal.
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ── Font loader ──────────────────────────────────────────────────────────────
def _find_font(family="body", size=26):
    candidates = {
        "title": [
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "body": [
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
def split_into_math_tokens(text):
    """
    Split equation into meaningful tokens.
    FIX: filters out pure-whitespace tokens so they don't
    waste reveal slots — equations reveal faster and more naturally.
    """
    raw = re.findall(r'[A-Za-z0-9₀-₉⁰-⁹]+|\s+|[^\w\s]', text)
    return [t for t in raw if t.strip()]  # skip whitespace tokens


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
    p    = ease_in_out(progress)
    ex   = x1 + (x2 - x1) * p
    ey   = y1 + (y2 - y1) * p
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
SUPERSCRIPTS = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4',
                '⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}

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
        if ch == '−': ch = '-'
        if '₀' <= ch <= '₉':
            w += draw.textlength(str(ord(ch) - 0x2080), font=sf)
        elif ch in SUPERSCRIPTS:
            w += draw.textlength(SUPERSCRIPTS[ch], font=sf)
        else:
            w += draw.textlength(ch, font=font)
    return w


# ── Radical renderer ─────────────────────────────────────────────────────────
def draw_math_with_radicals(draw, x, y, text, font, color):
    """
    Render math text, replacing √ with a properly scaled
    hand-drawn radical sign instead of a missing-glyph box.
    FIX: radical dimensions now scale with font.size.
    """
    if "√" not in text:
        draw_custom_text(draw, x, y, text, font, color)
        return

    fs     = font.size
    tail_h = int(fs * 0.55)   # height of the tail below baseline
    head_h = int(fs * 0.85)   # height of the overline above baseline
    tick_w = int(fs * 0.55)   # horizontal width of the tick part

    parts  = text.split("√")
    curr_x = x

    for idx, part in enumerate(parts):
        if idx == 0:
            if part:
                curr_x += draw_custom_text(draw, curr_x, y, part, font, color)
            continue

        # Parse inside the radical
        if part.startswith("("):
            depth, closing = 0, -1
            for ci, ch in enumerate(part):
                if ch == "(": depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0: closing = ci; break
            if closing != -1:
                inside, rest = part[1:closing], part[closing + 1:]
            else:
                inside, rest = part[1:], ""
        else:
            m = re.match(r'^[0-9]+', part)
            if m:
                inside, rest = m.group(0), part[len(m.group(0)):]
            else:
                inside, rest = part, ""

        iw = text_width(draw, inside, font) if inside else 0

        # Radical strokes (scaled)
        r_x0, r_y0 = curr_x,              y + tail_h // 2
        r_x1, r_y1 = curr_x + tick_w // 4, y + tail_h
        r_x2, r_y2 = curr_x + tick_w // 2, y + tail_h
        r_x3, r_y3 = curr_x + tick_w,       y - head_h + fs
        r_x4, r_y4 = curr_x + tick_w + int(iw) + 4, y - head_h + fs

        draw.line(
            [(r_x0, r_y0), (r_x1, r_y1), (r_x2, r_y2),
             (r_x3, r_y3), (r_x4, r_y4)],
            fill=color, width=2, joint="round"
        )

        if inside:
            draw_custom_text(draw, curr_x + tick_w + 2, y, inside, font, color)
            curr_x += tick_w + iw + 8
        if rest:
            curr_x += draw_custom_text(draw, curr_x, y, rest, font, color)


# ── Schedule builder ─────────────────────────────────────────────────────────
def _build_schedule(annotations, total_duration, enriched_ocr, option_positions, draw_ref, font):
    """
    Compute geometry and timing for every annotation.
    FIX: wy is now tracked as a running cursor correctly across both passes.
    FIX: write_equation duration based on token count, not just segment gap.
    """
    ocr_index   = (enriched_ocr or {}).get("index")
    free_spaces = (enriched_ocr or {}).get("free_spaces", [])

    if free_spaces:
        rx1, ry1, rx2, ry2 = free_spaces[0]["bounds"]
    else:
        rx1, ry1, rx2, ry2 = 200, 420, 1150, 630

    # Writing cursor starts at top-left of free space
    wx = rx1 + 20
    wy = ry1 + 30
    LINE_GAP = 62  # pixels between annotation lines

    schedule = []

    for i, ann in enumerate(annotations):
        action = ann["action"]
        t      = ann["time"]
        entry  = {**ann, "write_start": t}

        # ── Next annotation time (for duration stretching) ──
        t_next = total_duration - 1.0
        for j in range(i + 1, len(annotations)):
            t_next = annotations[j]["time"]
            break
        segment_gap = t_next - t

        if action == "write_equation":
            tokens    = split_into_math_tokens(ann.get("text", ""))
            n_tokens  = len(tokens)
            # Natural speed: ~2 tokens/sec, clamped to segment gap
            natural   = n_tokens / 2.0
            write_dur = max(1.5, min(natural, segment_gap * 0.88))
            entry["write_pos"]      = (wx, wy)
            entry["write_duration"] = write_dur
            entry["write_end"]      = t + write_dur
            wy += LINE_GAP  # advance cursor for next line

        elif action == "underline_existing":
            target = ann.get("target", "")
            entry["underline_duration"] = 0.8
            entry["write_end"] = t + 0.8
            # Fallback coords based on common patterns
            if ocr_index:
                matches = ocr_index.find_by_text(target, threshold=0.4)
                if matches:
                    elem = matches[0]
                    entry["underline_params"] = (elem.x1, elem.y2 + 4, elem.x2, elem.y2 + 4)
                else:
                    entry["underline_params"] = (60, 117, 300, 117)
            else:
                entry["underline_params"] = (60, 117, 300, 117)

        elif action == "circle_existing":
            entry["circle_duration"] = 0.8
            entry["write_end"] = t + 0.8
            entry["circle_params"] = (145, 91, 30)

        elif action == "draw_arrow":
            entry["arrow_duration"] = 0.6
            entry["write_end"] = t + 0.6
            entry["arrow_params"] = (wx + 60, wy - LINE_GAP - 10, wx + 60, wy - 10)

        elif action == "tick_answer":
            target    = (ann.get("target") or ann.get("option", "")).strip().upper()
            opt_letter = target.replace("OPTION", "").strip()
            entry["tick_duration"] = 0.5
            entry["write_end"]     = t + 0.5
            if opt_letter in option_positions:
                bbox = option_positions[opt_letter]
                xs   = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)
                entry["tick_params"] = (ox1 - 4, oy2 + 4, ox1 + 42, oy1 - 4)
            else:
                entry["tick_params"] = (18, 300, 69, 240)

        schedule.append(entry)

    return schedule


# ── Frame renderer ────────────────────────────────────────────────────────────
def _render_frame(t, background, schedule, fonts):
    font = fonts[0]
    frame = Image.new("RGB", background.size, (255, 255, 255))
    frame.paste(background, (0, 0))
    draw  = ImageDraw.Draw(frame, "RGBA")

    for ann in schedule:
        if t < ann["write_start"]:
            continue

        action   = ann["action"]
        end      = ann["write_end"]
        dur_key  = {"write_equation":    "write_duration",
                     "underline_existing":"underline_duration",
                     "circle_existing":   "circle_duration",
                     "draw_arrow":        "arrow_duration",
                     "tick_answer":       "tick_duration"}.get(action, "write_duration")
        duration = ann.get(dur_key, 1.0)
        raw_p    = 1.0 if t >= end else (t - ann["write_start"]) / max(duration, 0.01)
        progress = ease_in_out(raw_p)

        if action == "write_equation":
            text   = ann.get("text", "")
            wx, wy = ann["write_pos"]
            tokens = split_into_math_tokens(text)
            k      = max(1, int(progress * len(tokens)))
            partial = "".join(tokens[:k])
            draw_math_with_radicals(draw, wx, wy, partial, font, PEN_COLOR)

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
    """
    FIX: Instead of returning None during active drawing (causing full re-render),
    round progress to 2 decimal places as cache key — dramatically reduces
    redundant renders while still animating smoothly.
    """
    key_parts = []
    for ann in schedule:
        if t < ann["write_start"]:
            key_parts.append("pending")
        elif t >= ann["write_end"]:
            key_parts.append("done")
        else:
            dur = ann.get("write_duration") or ann.get("underline_duration") or \
                  ann.get("circle_duration") or ann.get("arrow_duration") or \
                  ann.get("tick_duration") or 1.0
            p = round((t - ann["write_start"]) / max(dur, 0.01), 2)
            key_parts.append(f"active_{p}")
    return tuple(key_parts)


# ── Entry point ───────────────────────────────────────────────────────────────
def render_video(image_path, annotations_path, audio_path, output_path,
                 option_positions=None, question_bbox=None, enriched_ocr=None):

    option_positions = option_positions or {}
    enriched_ocr     = enriched_ocr or {}

    background = Image.open(image_path).convert("RGB")

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    font  = _find_font("body", 28)
    fonts = (font,)

    audio          = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Dummy draw for text measurement in schedule builder
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    schedule = _build_schedule(annotations, total_duration, enriched_ocr,
                               option_positions, dummy, font)

    frame_cache = {}

    def make_frame(t):
        key = _cache_key(t, schedule)
        if key not in frame_cache:
            frame_cache[key] = _render_frame(t, background, schedule, fonts)
            # Keep cache bounded
            if len(frame_cache) > 500:
                oldest = next(iter(frame_cache))
                del frame_cache[oldest]
        return frame_cache[key]

    print(f"  Rendering {total_duration:.1f}s at {TARGET_FPS} fps...")

    video = VideoClip(make_frame, duration=total_duration).with_fps(TARGET_FPS)
    video = video.with_effects([vfx.FadeIn(0.6), vfx.FadeOut(0.6)])
    video = video.with_audio(audio)

    video.write_videofile(
        output_path,
        fps=TARGET_FPS,
        codec="libx264",
        audio_codec="aac",
        logger="bar",
    )
    print(f"  Done → {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    ann = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    aud = sys.argv[3] if len(sys.argv) > 3 else "input/narration.mp3"
    out = sys.argv[4] if len(sys.argv) > 4 else "output/final.mp4"
    render_video(img, ann, aud, out)