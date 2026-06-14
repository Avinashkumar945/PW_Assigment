#!/usr/bin/env python3
"""Transcribe audio to JSON with word-level timestamps using OpenAI Whisper."""

import json
import os
import shutil
import sys


def _ensure_ffmpeg():
    """
    Find FFmpeg on PATH or in common Windows install locations.
    Sets PATH so Whisper/MoviePy can find it without hardcoding a path.
    Raises EnvironmentError if FFmpeg is not found anywhere.
    """
    # Already on PATH — nothing to do
    if shutil.which("ffmpeg"):
        return

    # Common Windows locations to probe (WinGet, Chocolatey, Scoop, manual)
    candidates = [
        # WinGet (pattern: any version under the Gyan.FFmpeg package)
        os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        ),
        # Chocolatey
        r"C:\ProgramData\chocolatey\bin",
        # Scoop
        os.path.expandvars(r"%USERPROFILE%\scoop\shims"),
        # Common manual installs
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
    ]

    # For WinGet the binary sits inside a versioned sub-folder; walk one level
    winget_base = candidates[0]
    if os.path.isdir(winget_base):
        for entry in os.listdir(winget_base):
            if "ffmpeg" in entry.lower() or "Gyan" in entry:
                sub = os.path.join(winget_base, entry)
                # Might be one more level deep (e.g. ffmpeg-x.y-full_build/bin)
                for inner in os.listdir(sub) if os.path.isdir(sub) else []:
                    bin_path = os.path.join(sub, inner, "bin")
                    if os.path.isfile(os.path.join(bin_path, "ffmpeg.exe")):
                        candidates.append(bin_path)
                bin_path = os.path.join(sub, "bin")
                if os.path.isfile(os.path.join(bin_path, "ffmpeg.exe")):
                    candidates.append(bin_path)

    for path in candidates[1:]:  # skip base WinGet dir itself
        ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
        if os.path.isfile(ffmpeg_exe):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            print(f"  FFmpeg found at: {ffmpeg_exe}")
            return

    raise EnvironmentError(
        "FFmpeg not found. Install it (e.g. `winget install Gyan.FFmpeg`) "
        "and make sure it is on your PATH, then retry."
    )


def transcribe_audio(audio_path: str, output_path: str, model_size: str = "base") -> dict:
    """
    Transcribe an audio file and save the result with word-level timestamps.

    Args:
        audio_path:  Path to audio file (.mp3, .wav, .m4a, etc.)
        output_path: Path to save the transcript JSON.
        model_size:  Whisper model size — tiny | base | small | medium | large.

    Returns:
        The Whisper result dict (also written to output_path).

    Raises:
        FileNotFoundError:  If audio_path does not exist.
        EnvironmentError:   If FFmpeg cannot be located.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _ensure_ffmpeg()

    import whisper  # imported after PATH fix so it picks up ffmpeg correctly

    print(f"  FFmpeg on PATH: {shutil.which('ffmpeg')}")
    print(f"  Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"  Transcribing: {audio_path} ...")
    result = model.transcribe(audio_path, word_timestamps=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_segments = len(result.get("segments", []))
    duration   = result.get("segments", [{}])[-1].get("end", 0.0) if result.get("segments") else 0.0
    print(f"  Saved {n_segments} segments ({duration:.1f}s) -> {output_path}")
    return result


if __name__ == "__main__":
    audio_arg  = sys.argv[1] if len(sys.argv) > 1 else "input/narration.mp3"
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.json"
    model_arg  = sys.argv[3] if len(sys.argv) > 3 else "base"
    try:
        transcribe_audio(audio_arg, output_arg, model_arg)
    except (FileNotFoundError, EnvironmentError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)