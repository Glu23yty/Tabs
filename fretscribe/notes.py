"""Shared note data types."""
from __future__ import annotations

from dataclasses import dataclass, field

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def name_to_midi(name: str) -> int:
    name = name.strip()
    letter = name[0].upper()
    rest = name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octave = int(rest)
    return NOTE_NAMES.index(letter) + acc + (octave + 1) * 12


@dataclass
class Note:
    start: float          # seconds
    end: float            # seconds
    pitch: int            # MIDI number
    velocity: float = 0.8  # 0..1
    string: int | None = None  # 1 = high e ... 6 = low E (tab convention)
    fret: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)


@dataclass
class Event:
    """A group of notes that start together (a single note or a chord)."""
    start: float
    notes: list[Note] = field(default_factory=list)

    @property
    def end(self) -> float:
        return max(n.end for n in self.notes)
