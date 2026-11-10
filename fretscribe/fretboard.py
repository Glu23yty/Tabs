"""Map MIDI notes to playable (string, fret) positions.

A pitch can usually be played in several places on the neck. We pick positions
for the whole piece at once with dynamic programming (Viterbi), minimising a cost
that models what a guitarist actually does:

* chords must fit under one hand (small fret span, one note per string)
* the fretting hand prefers to stay in one position rather than jump around
* lower positions and open strings are slightly preferred
* a string that is still ringing should not be re-used for a different note
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .notes import Event, Note, name_to_midi

TUNINGS = {
    "standard": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "drop-d": ["D2", "A2", "D3", "G3", "B3", "E4"],
    "half-step-down": ["D#2", "G#2", "C#3", "F#3", "A#3", "D#4"],
    "full-step-down": ["D2", "G2", "C3", "F3", "A3", "D4"],
    "open-g": ["D2", "G2", "D3", "G3", "B3", "D4"],
    "open-d": ["D2", "A2", "D3", "F#3", "A3", "D4"],
    "dadgad": ["D2", "A2", "D3", "G3", "A3", "D4"],
}


@dataclass
class Guitar:
    tuning: list[int]          # MIDI pitch of open strings, low -> high
    capo: int = 0
    max_fret: int = 22

    @classmethod
    def from_spec(cls, tuning: str = "standard", capo: int = 0, max_fret: int = 22) -> "Guitar":
        names = TUNINGS.get(tuning.lower())
        if names is None:
            names = [s for s in tuning.replace(",", " ").split() if s]
            if len(names) < 4:
                raise ValueError(f"unknown tuning '{tuning}'. Use one of {list(TUNINGS)} "
                                 "or 6 note names low->high, e.g. 'E2 A2 D3 G3 B3 E4'")
        return cls([name_to_midi(n) for n in names], capo, max_fret)

    @property
    def num_strings(self) -> int:
        return len(self.tuning)

    def string_number(self, idx_low_to_high: int) -> int:
        """Tab string number: 1 = highest string."""
        return self.num_strings - idx_low_to_high

    def positions(self, pitch: int) -> list[tuple[int, int]]:
        """All (string_idx_low_to_high, fret_relative_to_capo) for a pitch."""
        out = []
        for i, open_pitch in enumerate(self.tuning):
            fret = pitch - (open_pitch + self.capo)
            if 0 <= fret <= self.max_fret - self.capo:
                out.append((i, fret))
        return out

    def lowest(self) -> int:
        return min(self.tuning) + self.capo

    def highest(self) -> int:
        return max(self.tuning) + self.max_fret


def group_events(notes: list[Note], chord_window: float = 0.045) -> list[Event]:
    events: list[Event] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if events and n.start - events[-1].start <= chord_window:
            if all(m.pitch != n.pitch for m in events[-1].notes):
                events[-1].notes.append(n)
        else:
            events.append(Event(n.start, [n]))
    return events


# ---------------------------------------------------------------- cost model
# DP state = (fingering, hand position). Hand position p means the index finger
# sits at fret p and the hand covers frets p..p+3 without shifting. Open strings
# need no hand, so an all-open fingering can be played from any position.
SPAN_COST = 0.6          # per fret of stretch inside a chord
BIG_SPAN_PENALTY = 6.0   # stretches over 4 frets are rarely playable
HEIGHT_COST = 0.04       # mild preference for lower positions
OPEN_BONUS = -0.15       # open strings are easy
MOVE_COST = 0.3          # per fret of hand movement between events
SHIFT_COST = 0.5         # fixed cost of any position shift
RING_PENALTY = 1.5       # cutting off a still-ringing string
HAND_SPAN = 3
TOP_K = 16


def _static_cost(fing) -> float:
    frets = [f for _, f in fing if f > 0]
    cost = sum(HEIGHT_COST * f for f in frets) + OPEN_BONUS * (len(fing) - len(frets))
    if frets:
        span = max(frets) - min(frets)
        cost += SPAN_COST * span + (BIG_SPAN_PENALTY * (span - 4) if span > 4 else 0)
        if min(frets) > 7 and len(fing) - len(frets):
            cost += 0.4  # open string combined with a high position is awkward
    return cost


def _hand_positions(fing, max_pos: int) -> list[int]:
    frets = [f for _, f in fing if f > 0]
    if not frets:
        return list(range(1, min(max_pos, 15) + 1))
    lo, hi = min(frets), max(frets)
    first = max(1, hi - HAND_SPAN)
    return list(range(first, max(first, lo) + 1))


def _candidates(event: Event, guitar: Guitar):
    """Return [(fingering, hand_pos, static_cost)]."""
    options = [guitar.positions(n.pitch) for n in event.notes]
    fings = []
    for combo in itertools.product(*options):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fings.append((tuple(combo), _static_cost(combo)))
    fings.sort(key=lambda c: c[1])
    out = []
    for fing, c in fings[:TOP_K]:
        for p in _hand_positions(fing, guitar.max_fret - guitar.capo):
            out.append((fing, p, c))
    return out


def _transition_cost(prev_event, prev_state, event, state) -> float:
    pf, pp, _ = prev_state
    fing, p, _ = state
    gap = max(event.start - prev_event.start, 0.05)
    cost = 0.0
    if p != pp:
        # Long gaps give the player time to shift, so movement costs less.
        cost += (SHIFT_COST + MOVE_COST * abs(p - pp)) / (1.0 + max(gap - 0.5, 0))
    still_ringing = {s: f for (s, f), n in zip(pf, prev_event.notes)
                     if n.end > event.start + 0.05}
    for s, f in fing:
        if s in still_ringing and still_ringing[s] != f:
            cost += RING_PENALTY
    return cost


def assign_frets(notes: list[Note], guitar: Guitar) -> list[Event]:
    """Assign string/fret to every note. Notes outside the instrument's range are
    moved by octaves into range. Returns the grouped events."""
    lo, hi = guitar.lowest(), guitar.highest()
    for n in notes:
        while n.pitch < lo:
            n.pitch += 12
        while n.pitch > hi:
            n.pitch -= 12
    events = group_events(notes)
    # A chord can have at most one note per string; keep the loudest.
    for ev in events:
        if len(ev.notes) > guitar.num_strings:
            ev.notes = sorted(ev.notes, key=lambda n: -n.velocity)[:guitar.num_strings]
        ev.notes.sort(key=lambda n: n.pitch)

    # Drop the quietest notes from a chord until a playable fingering exists.
    cand_lists = []
    for ev in events:
        cands = _candidates(ev, guitar)
        while not cands and len(ev.notes) > 1:
            ev.notes.remove(min(ev.notes, key=lambda n: n.velocity))
            cands = _candidates(ev, guitar)
        cand_lists.append(cands)

    keep = [i for i, c in enumerate(cand_lists) if c]
    events = [events[i] for i in keep]
    cand_lists = [cand_lists[i] for i in keep]
    if not events:
        return []

    # Viterbi
    costs = [[c for *_, c in cand_lists[0]]]
    back = [[-1] * len(cand_lists[0])]
    for t in range(1, len(events)):
        row, brow = [], []
        prev_c, prev_s = costs[t - 1], cand_lists[t - 1]
        for state in cand_lists[t]:
            best, arg = float("inf"), 0
            for j, ps in enumerate(prev_s):
                c = prev_c[j] + _transition_cost(events[t - 1], ps, events[t], state)
                if c < best:
                    best, arg = c, j
            row.append(best + state[2])
            brow.append(arg)
        costs.append(row)
        back.append(brow)
    j = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    for t in range(len(events) - 1, -1, -1):
        fing = cand_lists[t][j][0]
        for (s, f), n in zip(fing, events[t].notes):
            n.string = guitar.string_number(s)
            n.fret = f
        j = back[t][j]
    return events
