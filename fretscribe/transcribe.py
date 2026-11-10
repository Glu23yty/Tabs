"""Audio -> notes transcription.

Two backends:
  * ``basic-pitch`` (Spotify's polyphonic note transcription model, run via ONNX).
    Handles chords and single notes. Used by default when installed.
  * ``pyin`` (librosa probabilistic YIN). Monophonic only, but has no model
    dependency. Used as a fallback, or when ``backend="pyin"`` is requested.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .notes import Note

log = logging.getLogger(__name__)

# Guitar fundamental range: low E2 (~82 Hz) dropped to ~D2 for drop tunings,
# up to roughly the 24th fret of the high E string.
MIN_FREQ = 70.0
MAX_FREQ = 1400.0


def available_backends() -> list[str]:
    backends = []
    try:
        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.ERROR)
        try:
            import basic_pitch  # noqa: F401
        finally:
            root.setLevel(level)
        backends.append("basic-pitch")
    except Exception:  # pragma: no cover - depends on environment
        pass
    try:
        import librosa  # noqa: F401
        backends.append("pyin")
    except Exception:  # pragma: no cover
        pass
    return backends


def transcribe(wav_path: str | Path, backend: str = "auto", *,
               onset_threshold: float = 0.5, frame_threshold: float = 0.3,
               min_note_ms: float = 60.0) -> list[Note]:
    backends = available_backends()
    if backend == "auto":
        backend = backends[0] if backends else "none"
    if backend not in backends:
        raise RuntimeError(
            f"transcription backend '{backend}' not available (installed: {backends or 'none'})")
    if backend == "basic-pitch":
        notes = _transcribe_basic_pitch(wav_path, onset_threshold, frame_threshold, min_note_ms)
    else:
        notes = _transcribe_pyin(wav_path, min_note_ms)
    notes = clean_notes(notes, audio_onsets=detect_onsets(wav_path))
    log.info("transcribed %d notes with %s", len(notes), backend)
    return notes


def _transcribe_basic_pitch(wav_path, onset_threshold, frame_threshold, min_note_ms) -> list[Note]:
    import contextlib
    import io

    # basic-pitch logs warnings about optional backends (TF, CoreML) and prints
    # progress to stdout; keep our CLI output clean.
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
            _, _, note_events = _predict(predict, ICASSP_2022_MODEL_PATH, wav_path, onset_threshold,
                                         frame_threshold, min_note_ms)
    finally:
        root.setLevel(level)
    return [Note(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in note_events]


def _predict(predict, model_path, wav_path, onset_threshold, frame_threshold, min_note_ms):
    return predict(
        str(wav_path),
        model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        minimum_frequency=MIN_FREQ,
        maximum_frequency=MAX_FREQ,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )


def _transcribe_pyin(wav_path, min_note_ms) -> list[Note]:
    import librosa

    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    hop = 256
    f0, voiced, prob = librosa.pyin(y, fmin=MIN_FREQ, fmax=MAX_FREQ, sr=sr,
                                    frame_length=2048, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    onset_frames = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms = rms / (rms.max() + 1e-9)

    midi = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1).astype(int)
    notes: list[Note] = []
    cur, start_i = -1, 0

    def close(i_end):
        if cur >= 0:
            s, e = times[start_i], times[min(i_end, len(times) - 1)]
            if (e - s) * 1000 >= min_note_ms:
                notes.append(Note(float(s), float(e), int(cur),
                                  float(np.clip(rms[start_i:i_end + 1].mean() * 1.5, 0.1, 1.0))))

    for i, m in enumerate(midi):
        # A new note starts on a pitch change, or on a detected re-attack of the same pitch.
        if m != cur or (i in onset_frames and m >= 0 and i - start_i > 3):
            close(i)
            cur, start_i = m, i
    close(len(midi) - 1)
    return notes


CHORD_WINDOW = 0.06


def detect_onsets(wav_path) -> np.ndarray | None:
    """Pluck onsets from the audio signal itself (spectral flux)."""
    try:
        import librosa
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    except Exception:  # pragma: no cover
        return None


def clean_notes(notes: list[Note], *, min_velocity: float = 0.12,
                relative_floor: float = 0.5, audio_onsets=None) -> list[Note]:
    """Remove common transcription artifacts for guitar audio.

    * very quiet notes (noise / sympathetic resonance), judged both absolutely and
      relative to the median loudness of the clip
    * overtone ghosts: a weaker note an octave, octave+fifth or two octaves above a
      stronger note struck at the same time. Skipped inside strummed chords, where
      octave doublings are real (e.g. G2 + G3 in an open G chord)
    * re-trigger splits: a ringing note that the model restarts mid-sustain. If the
      restart does not line up with a real pluck in the audio (``audio_onsets``),
      or it coincides with a different string being plucked, it is merged back
    """
    notes = [n for n in notes if n.end > n.start]
    if not notes:
        return []
    med = float(np.median([n.velocity for n in notes]))
    floor = max(min_velocity, relative_floor * med)
    notes = sorted((n for n in notes if n.velocity >= floor), key=lambda n: (n.start, n.pitch))

    def co_onsets(n):
        return [m for m in notes if m is not n and abs(m.start - n.start) < CHORD_WINDOW]

    # Ghost overtones
    keep = []
    for n in notes:
        others = co_onsets(n)
        ghost = False
        if len(others) < 3:  # not a strum
            for m in others:
                if n.pitch - m.pitch in (12, 19, 24) and n.velocity < 0.6 * m.velocity:
                    ghost = True
                    break
        if not ghost:
            keep.append(n)
    notes = keep

    # Re-trigger splits
    by_pitch: dict[int, list[Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    continuation = set()
    for seq in by_pitch.values():
        for a, b in zip(seq, seq[1:]):
            if 0 <= b.start - a.end < 0.03:
                continuation.add(id(b))
    merged: list[Note] = []
    last_of_pitch: dict[int, Note] = {}
    for n in notes:
        if id(n) in continuation:
            fresh = [m for m in co_onsets(n) if id(m) not in continuation]
            no_pluck = audio_onsets is not None and (
                len(audio_onsets) == 0 or np.min(np.abs(audio_onsets - n.start)) > 0.05)
            prev = last_of_pitch.get(n.pitch)
            if (fresh or no_pluck) and prev is not None:
                prev.end = max(prev.end, n.end)
                continue
        merged.append(n)
        last_of_pitch[n.pitch] = n
    return merged
