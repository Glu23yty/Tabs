import sys
from pathlib import Path

import mir_eval
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import STANDARD, make_mp4  # noqa: E402

from fretscribe.fretboard import Guitar, assign_frets  # noqa: E402
from fretscribe.notes import Note, name_to_midi  # noqa: E402
from fretscribe.pipeline import run  # noqa: E402


def test_name_to_midi():
    assert name_to_midi("E2") == 40
    assert name_to_midi("A4") == 69
    assert name_to_midi("C#3") == 49


def test_open_chord_fingering():
    g = Guitar.from_spec("standard")
    # Open E major: E2 B2 E3 G#3 B3 E4
    pitches = [40, 47, 52, 56, 59, 64]
    notes = [Note(0.0, 1.0, p) for p in pitches]
    assign_frets(notes, g)
    shape = {n.string: n.fret for n in notes}
    assert shape == {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}


def test_melody_stays_in_position():
    g = Guitar.from_spec("standard")
    # A minor pentatonic run at the 5th position
    pitches = [57, 60, 62, 64, 67, 69, 72]
    notes = [Note(i * 0.25, i * 0.25 + 0.2, p) for i, p in enumerate(pitches)]
    assign_frets(notes, g)
    frets = [n.fret for n in notes if n.fret > 0]  # open strings need no hand
    assert max(frets) - min(frets) <= 4, frets


def test_capo_and_drop_d():
    g = Guitar.from_spec("drop-d", capo=2)
    n = Note(0, 1, 40)  # E2 = 2 semitones above low D, i.e. fret 0 with capo 2
    assign_frets([n], g)
    assert (n.string, n.fret) == (6, 0)


@pytest.fixture(scope="module")
def riff(tmp_path_factory):
    out = tmp_path_factory.mktemp("clip") / "riff.mp4"
    return make_mp4(out, bpm=100)


def test_end_to_end_mp4(riff):
    path, truth = riff
    res = run(path, bpm=100)
    assert "e|" in res.tab and "E|" in res.tab
    assert "<sign>TAB</sign>" in res.musicxml

    import json
    data = json.loads(res.json)
    est = [(n["start"], n["pitch"]) for ev in data["events"] for n in ev["notes"]]
    ref = [(t, p) for t, d, ps in truth for p in ps]
    ref_iv = np.array([[t, t + 0.2] for t, _ in ref])
    est_iv = np.array([[t, t + 0.2] for t, _ in est])
    ref_hz = 440 * 2 ** ((np.array([p for _, p in ref]) - 69) / 12)
    est_hz = 440 * 2 ** ((np.array([p for _, p in est]) - 69) / 12)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz, onset_tolerance=0.06, offset_ratio=None)
    print(f"note precision={p:.2f} recall={r:.2f} F1={f:.2f}")
    assert f > 0.8


def test_web_upload(riff):
    from io import BytesIO
    from fretscribe.web import create_app
    path, _ = riff
    client = create_app().test_client()
    assert client.get("/").status_code == 200
    r = client.post("/transcribe", data={"file": (BytesIO(path.read_bytes()), "riff.mp4"),
                                         "tuning": "standard", "capo": "0", "bpm": "100",
                                         "sensitivity": "0.5"},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.data[:500]
    html = r.data.decode()
    assert "e|" in html and "/download/" in html
    job = html.split("/download/")[1].split("/")[0]
    d = client.get(f"/download/{job}/musicxml?name=riff")
    assert d.status_code == 200 and b"score-partwise" in d.data


def test_rejects_file_without_audio(tmp_path):
    import subprocess
    from fretscribe.audio import AudioExtractionError
    v = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:d=1", "-c:v", "libx264", str(v)], check=True)
    with pytest.raises(AudioExtractionError):
        run(v)
