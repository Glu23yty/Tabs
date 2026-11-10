"""Audio extraction from video/audio files using ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

SAMPLE_RATE = 22050


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(src: str | Path, out_wav: str | Path | None = None,
                  sample_rate: int = SAMPLE_RATE) -> Path:
    """Decode any ffmpeg-readable file (mp4, mov, mkv, mp3, wav...) to mono WAV."""
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError("ffmpeg is not installed or not on PATH")
    src = Path(src)
    if not src.exists():
        raise AudioExtractionError(f"input not found: {src}")
    if out_wav is None:
        out_wav = Path(tempfile.mkdtemp(prefix="fretscribe_")) / "audio.wav"
    out_wav = Path(out_wav)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",                 # drop video
        "-ac", "1",            # mono
        "-ar", str(sample_rate),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # normalise level
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.exists():
        raise AudioExtractionError(
            f"ffmpeg failed (does the file have an audio track?): {proc.stderr.strip()}")
    return out_wav
