"""Output formats: ASCII tab, MusicXML (with TAB staff), JSON."""
from __future__ import annotations

import json
from xml.sax.saxutils import escape

from .fretboard import Guitar
from .notes import NOTE_NAMES, Event
from .rhythm import Grid


def _string_labels(guitar: Guitar) -> list[str]:
    """Labels high -> low, e.g. ['e','B','G','D','A','E']."""
    names = [NOTE_NAMES[p % 12] for p in guitar.tuning]
    labels = []
    seen_lowest = names[0]
    for i, n in enumerate(names):
        # Lower-case the top string if it shares a name with the bottom one (e/E).
        labels.append(n.lower() if i == len(names) - 1 and n == seen_lowest else n)
    return labels[::-1]


def ascii_tab(slots: list[tuple[int, Event]], grid: Grid, guitar: Guitar, *,
              title: str = "", bars_per_line: int = 4) -> str:
    n_str = guitar.num_strings
    labels = _string_labels(guitar)
    width = max(len(l) for l in labels)
    labels = [l.ljust(width) for l in labels]
    last_slot = slots[-1][0] if slots else 0
    n_bars = last_slot // grid.slots_per_bar + 1
    by_slot = dict(slots)
    col_w = 3 if any(n.fret >= 10 for _, ev in slots for n in ev.notes) else 2

    header = []
    if title:
        header.append(title)
    tuning_str = " ".join(f"{NOTE_NAMES[p % 12]}" for p in guitar.tuning)
    header.append(f"Tempo ~{grid.bpm:g} BPM | {grid.beats_per_bar}/4 | "
                  f"Tuning: {tuning_str}" + (f" | Capo {guitar.capo}" if guitar.capo else "")
                  + f" | 1 column = 1/{grid.subdivisions * 4} note")
    lines = ["\n".join(header), ""]

    for line_start in range(0, n_bars, bars_per_line):
        rows = [f"{labels[i]}|" for i in range(n_str)]
        bar_nums = " " * (width + 1)
        for bar in range(line_start, min(line_start + bars_per_line, n_bars)):
            bar_nums += str(bar + 1).ljust(grid.slots_per_bar * col_w + 1)
            for s in range(bar * grid.slots_per_bar, (bar + 1) * grid.slots_per_bar):
                ev = by_slot.get(s)
                cells = ["-" * col_w] * n_str
                if ev:
                    for n in ev.notes:
                        cells[n.string - 1] = str(n.fret).ljust(col_w, "-")
                for i in range(n_str):
                    rows[i] += cells[i]
            for i in range(n_str):
                rows[i] += "|"
        lines.append(bar_nums.rstrip())
        lines.extend(rows)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_json(slots, grid: Grid, guitar: Guitar) -> str:
    data = {
        "bpm": grid.bpm,
        "time_signature": f"{grid.beats_per_bar}/4",
        "tuning_midi": guitar.tuning,
        "capo": guitar.capo,
        "events": [
            {
                "slot": s,
                "bar": s // grid.slots_per_bar + 1,
                "time": round(ev.start, 3),
                "notes": [
                    {"pitch": n.pitch, "name": n.name, "string": n.string, "fret": n.fret,
                     "start": round(n.start, 3), "end": round(n.end, 3),
                     "velocity": round(n.velocity, 3)}
                    for n in ev.notes
                ],
            }
            for s, ev in slots
        ],
    }
    return json.dumps(data, indent=2)


# --------------------------------------------------------------- MusicXML
_DUR_TYPES = {16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1),
              4: ("quarter", 0), 3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0)}


def _split_duration(d: int) -> list[int]:
    """Split a duration in sixteenths into notatable pieces."""
    out = []
    for v in (16, 12, 8, 6, 4, 3, 2, 1):
        while d >= v:
            out.append(v)
            d -= v
    return out


def _pitch_xml(midi: int) -> str:
    step_alter = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0), 5: ("F", 0),
                  6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0), 10: ("A", 1), 11: ("B", 0)}
    step, alter = step_alter[midi % 12]
    octave = midi // 12 - 1
    alt = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alt}<octave>{octave}</octave></pitch>"


def musicxml(slots, grid: Grid, guitar: Guitar, title: str = "Transcription") -> str:
    """MusicXML 3.1 with a single TAB staff. Opens in MuseScore, Guitar Pro, TuxGuitar."""
    assert grid.subdivisions == 4, "MusicXML export expects a 16th-note grid"
    spb = grid.slots_per_bar
    last = slots[-1][0] if slots else 0
    n_bars = last // spb + 1
    starts = [s for s, _ in slots] + [n_bars * spb]
    by_slot = dict(slots)

    tuning_xml = []
    step_names = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    for line, p in enumerate(guitar.tuning, start=1):
        alter = 1 if NOTE_NAMES[p % 12].endswith("#") else 0
        alt = f"<tuning-alter>{alter}</tuning-alter>" if alter else ""
        tuning_xml.append(
            f'<staff-tuning line="{line}"><tuning-step>{step_names[p % 12]}</tuning-step>'
            f"{alt}<tuning-octave>{p // 12 - 1}</tuning-octave></staff-tuning>")

    measures = []
    for bar in range(n_bars):
        b0, b1 = bar * spb, (bar + 1) * spb
        parts = [f'<measure number="{bar + 1}">']
        if bar == 0:
            parts.append(
                "<attributes><divisions>4</divisions><key><fifths>0</fifths></key>"
                f"<time><beats>{grid.beats_per_bar}</beats><beat-type>4</beat-type></time>"
                "<clef><sign>TAB</sign><line>5</line></clef>"
                f"<staff-details><staff-lines>{guitar.num_strings}</staff-lines>"
                + "".join(tuning_xml)
                + (f"<capo>{guitar.capo}</capo>" if guitar.capo else "")
                + "</staff-details></attributes>"
                f'<direction placement="above"><direction-type><metronome><beat-unit>quarter'
                f"</beat-unit><per-minute>{round(grid.bpm)}</per-minute></metronome>"
                f'</direction-type><sound tempo="{round(grid.bpm)}"/></direction>')
        cursor = b0
        bar_slots = [s for s in starts[:-1] if b0 <= s < b1]
        for s in bar_slots:
            if s > cursor:
                parts += _rests(s - cursor)
            nxt = next(x for x in starts if x > s)
            dur = min(nxt, b1) - s
            ev = by_slot[s]
            pieces = _split_duration(dur)
            for pi, piece in enumerate(pieces):
                typ, dots = _DUR_TYPES[piece]
                for ni, n in enumerate(sorted(ev.notes, key=lambda n: n.string, reverse=True)):
                    tie = ""
                    if len(pieces) > 1:
                        if pi > 0:
                            tie += '<tie type="stop"/>'
                        if pi < len(pieces) - 1:
                            tie += '<tie type="start"/>'
                    parts.append(
                        "<note>" + ("<chord/>" if ni else "") + _pitch_xml(n.pitch)
                        + f"<duration>{piece}</duration>{tie}<voice>1</voice><type>{typ}</type>"
                        + "<dot/>" * dots
                        + f"<notations><technical><string>{n.string}</string>"
                        f"<fret>{n.fret}</fret></technical></notations></note>")
            cursor = s + dur
        if cursor < b1:
            parts += _rests(b1 - cursor)
        parts.append("</measure>")
        measures.append("".join(parts))

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">'
        f"<work><work-title>{escape(title)}</work-title></work>"
        '<part-list><score-part id="P1"><part-name>Guitar</part-name>'
        '<score-instrument id="P1-I1"><instrument-name>Acoustic Guitar</instrument-name>'
        '</score-instrument><midi-instrument id="P1-I1"><midi-program>26</midi-program>'
        "</midi-instrument></score-part></part-list>"
        '<part id="P1">' + "\n".join(measures) + "</part></score-partwise>\n"
    )


def _rests(d: int) -> list[str]:
    out = []
    for piece in _split_duration(d):
        typ, dots = _DUR_TYPES[piece]
        out.append(f"<note><rest/><duration>{piece}</duration><voice>1</voice>"
                   f"<type>{typ}</type>" + "<dot/>" * dots + "</note>")
    return out
