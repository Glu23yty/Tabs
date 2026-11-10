# Fretscribe

Turn a video (or audio recording) of someone playing guitar into tablature.

```
mp4 ──ffmpeg──▶ mono wav ──Basic Pitch──▶ notes ──cleanup──▶ fret/string DP ──▶ tab + MusicXML + JSON
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Audio extraction | `audio.py` | ffmpeg decodes any video/audio, mono 22.05 kHz, loudness-normalised |
| Transcription | `transcribe.py` | Spotify's **Basic Pitch** polyphonic model (run via ONNX, no TensorFlow). Falls back to librosa **pYIN** (monophonic) |
| Guitar cleanup | `transcribe.py` | drops quiet noise, removes overtone "ghost" notes (except inside strums, where octaves are real), merges notes the model re-triggered mid-sustain unless the audio shows a real pluck |
| Fingering | `fretboard.py` | Viterbi search over (fingering, hand position) states. Costs: chord stretch, fret height, hand shifts (cheaper after long gaps), cutting off ringing strings. Open strings are free from any position |
| Rhythm | `rhythm.py` | tempo from librosa beat tracking (or `--bpm`), notes snapped to a 16th-note grid |
| Output | `render.py` | ASCII tab, MusicXML with a TAB staff (opens in MuseScore, Guitar Pro, TuxGuitar), JSON of every note |

Supports standard, drop D, half/full step down, open G, open D, DADGAD or any custom tuning (`--tuning "D2 A2 D3 G3 B3 E4"`), plus capo.

## Install

Needs Python 3.10+ and ffmpeg.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install --no-deps basic-pitch==0.4.0   # its metadata pins TensorFlow; we only use its ONNX model
```

Or with Docker: `docker build -t fretscribe . && docker run -p 8000:8000 fretscribe`

## Use

```bash
fretscribe my_video.mp4                 # prints tab, writes my_video.tab.txt / .musicxml / .json
fretscribe clip.mp4 --tuning drop-d --capo 2 --bpm 92 --sensitivity 0.6
fretscribe serve --port 8000            # web UI at http://localhost:8000
```

`--sensitivity` (0–1) trades missed notes against false notes. Raise it for quiet fingerpicking, lower it for distorted or noisy recordings.

## Tests

```bash
pytest -q
```

The end-to-end test synthesises a plucked-string riff with known notes, wraps it in an MP4, runs the full pipeline and scores it with `mir_eval` (currently F1 ≈ 0.82; the arpeggio is transcribed exactly, the misses are inside chords).

## Known limits

* Works best on one clean guitar. Full band mixes need source separation first (e.g. Demucs) — not included yet.
* Strummed chords lose a note or two; chord-shape recognition would improve them.
* Bends, slides, hammer-ons and palm mutes are not detected yet.
* Fingerings are playable but not always the ones the player used; the video itself (fretboard hand tracking) could resolve this later.
