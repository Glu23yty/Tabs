"""Tempo estimation and rhythmic quantisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import Event


@dataclass
class Grid:
    bpm: float
    offset: float            # time (s) of the first beat
    subdivisions: int = 4    # grid slots per beat (4 = sixteenth notes)
    beats_per_bar: int = 4

    @property
    def slot_seconds(self) -> float:
        return 60.0 / self.bpm / self.subdivisions

    @property
    def slots_per_bar(self) -> int:
        return self.subdivisions * self.beats_per_bar

    def slot(self, t: float) -> int:
        return max(0, int(round((t - self.offset) / self.slot_seconds)))


def estimate_tempo(wav_path, events: list[Event]) -> tuple[float, float]:
    """Return (bpm, first_beat_time)."""
    try:
        import librosa
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        bpm, beats = 0.0, []
    if not (40 <= bpm <= 240):
        bpm = _tempo_from_onsets([e.start for e in events])
    # Keep tempo in a readable range.
    while bpm > 180:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    first = events[0].start if events else 0.0
    return round(bpm, 1), first


def _tempo_from_onsets(onsets: list[float]) -> float:
    if len(onsets) < 3:
        return 120.0
    iois = np.diff(sorted(onsets))
    iois = iois[(iois > 0.08) & (iois < 2.0)]
    if not len(iois):
        return 120.0
    beat = float(np.median(iois))
    return 60.0 / beat


def quantize(events: list[Event], grid: Grid) -> list[tuple[int, Event]]:
    """Return (slot_index, event) with events snapped to the grid. Events that
    collapse onto the same slot are merged (one note per string)."""
    out: dict[int, Event] = {}
    for ev in events:
        s = grid.slot(ev.start)
        if s in out:
            used = {n.string for n in out[s].notes}
            out[s].notes.extend(n for n in ev.notes if n.string not in used)
        else:
            out[s] = Event(ev.start, list(ev.notes))
    return sorted(out.items())
