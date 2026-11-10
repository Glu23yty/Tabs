"""End-to-end: media file -> tab."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .audio import extract_audio
from .fretboard import Guitar, assign_frets
from .render import ascii_tab, musicxml, to_json
from .rhythm import Grid, estimate_tempo, quantize
from .transcribe import transcribe

log = logging.getLogger(__name__)


@dataclass
class Result:
    tab: str
    musicxml: str
    json: str
    bpm: float
    note_count: int
    backend: str


def run(media: str | Path, *, tuning: str = "standard", capo: int = 0, bpm: float | None = None,
        backend: str = "auto", beats_per_bar: int = 4, onset_threshold: float = 0.5,
        frame_threshold: float = 0.3, title: str | None = None) -> Result:
    media = Path(media)
    guitar = Guitar.from_spec(tuning, capo=capo)
    with tempfile.TemporaryDirectory(prefix="fretscribe_") as tmp:
        wav = extract_audio(media, Path(tmp) / "audio.wav")
        from .transcribe import available_backends
        used = backend if backend != "auto" else (available_backends() or ["none"])[0]
        notes = transcribe(wav, backend, onset_threshold=onset_threshold,
                           frame_threshold=frame_threshold)
        events = assign_frets(notes, guitar)
        if bpm:
            first = events[0].start if events else 0.0
        else:
            bpm, first = estimate_tempo(wav, events)
    grid = Grid(bpm=bpm, offset=first, beats_per_bar=beats_per_bar)
    slots = quantize(events, grid)
    title = title or media.stem
    return Result(
        tab=ascii_tab(slots, grid, guitar, title=title),
        musicxml=musicxml(slots, grid, guitar, title=title),
        json=to_json(slots, grid, guitar),
        bpm=bpm,
        note_count=sum(len(ev.notes) for _, ev in slots),
        backend=used,
    )
