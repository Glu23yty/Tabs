"""Synthesize plucked-guitar test clips (Karplus-Strong) and wrap them in an MP4."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
STANDARD = [40, 45, 50, 55, 59, 64]  # low E .. high e


def pluck(midi: int, dur: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(sr * dur)
    period = int(round(sr / freq))
    rng = np.random.default_rng(seed + midi)
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    decay = 0.996
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = decay * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    env = np.minimum(1.0, np.linspace(0, 30, n))  # tiny attack ramp
    rel = min(n, int(0.08 * sr))                    # damp the string, no click
    env[-rel:] *= np.linspace(1, 0, rel)
    return out * env


def render(events, total: float, sr: int = SR) -> np.ndarray:
    """events: list of (start_sec, dur_sec, [midi,...])"""
    y = np.zeros(int(sr * total) + sr)
    for k, (t, d, pitches) in enumerate(events):
        for j, p in enumerate(pitches):
            s = pluck(p, d, sr, seed=k * 7 + j) * 0.5
            i = int(t * sr) + j * int(0.008 * sr)  # slight strum
            y[i:i + len(s)] += s
    y /= np.abs(y).max() + 1e-9
    return (y * 0.8).astype(np.float32)


def melody_fixture(bpm: float = 100.0):
    """A riff in A minor pentatonic, eighth notes, with a couple of chords."""
    sfret = lambda string_idx, fret: STANDARD[string_idx] + fret  # noqa: E731
    beat = 60.0 / bpm
    e = beat / 2
    seq = [
        [sfret(1, 0)], [sfret(2, 2)], [sfret(3, 2)], [sfret(4, 1)],  # Am arpeggio
        [sfret(5, 0)], [sfret(4, 1)], [sfret(3, 2)], [sfret(2, 2)],
        [sfret(1, 3)], [sfret(2, 2)], [sfret(3, 0)], [sfret(4, 0)],  # C arpeggio
        [sfret(5, 0)], [sfret(4, 0)], [sfret(3, 0)], [sfret(2, 2)],
    ]
    chords = [
        [sfret(1, 0), sfret(2, 2), sfret(3, 2), sfret(4, 1), sfret(5, 0)],  # Am
        [sfret(0, 3), sfret(1, 2), sfret(2, 0), sfret(3, 0), sfret(4, 0), sfret(5, 3)],  # G
    ]
    events = [(i * e, e * 1.6, p) for i, p in enumerate(seq)]
    t = len(seq) * e
    for c in chords:
        events.append((t, beat * 2, c))
        t += beat * 2
    return events, t


def make_mp4(out: Path, bpm: float = 100.0) -> tuple[Path, list]:
    events, total = melody_fixture(bpm)
    y = render(events, total)
    wav = out.with_suffix(".wav")
    sf.write(wav, y, SR)
    dur = len(y) / SR
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=0x333333:s=320x240:d={dur}",
                    "-i", str(wav), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    wav.unlink()
    return out, events


if __name__ == "__main__":
    import sys
    p, _ = make_mp4(Path(sys.argv[1] if len(sys.argv) > 1 else "riff.mp4"))
    print(p)
