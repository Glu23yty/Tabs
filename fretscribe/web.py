"""Minimal web UI: upload a video, get tab back."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

from .fretboard import TUNINGS
from .pipeline import run

ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


def create_app(max_mb: int = 300) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024
    store = Path(tempfile.mkdtemp(prefix="fretscribe_web_"))

    @app.get("/")
    def index():
        return render_template("index.html", tunings=list(TUNINGS))

    @app.post("/transcribe")
    def transcribe_view():
        f = request.files.get("file")
        if not f or not f.filename:
            return render_template("index.html", tunings=list(TUNINGS),
                                   error="Choose a video or audio file first."), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED:
            return render_template("index.html", tunings=list(TUNINGS),
                                   error=f"Unsupported file type {ext}."), 400
        job = uuid.uuid4().hex[:12]
        job_dir = store / job
        job_dir.mkdir()
        src = job_dir / f"input{ext}"
        f.save(src)
        sens = float(request.form.get("sensitivity", 0.5))
        bpm = request.form.get("bpm") or None
        try:
            res = run(src, tuning=request.form.get("tuning", "standard"),
                      capo=int(request.form.get("capo") or 0),
                      bpm=float(bpm) if bpm else None,
                      onset_threshold=0.75 - 0.5 * sens, frame_threshold=0.45 - 0.3 * sens,
                      title=Path(f.filename).stem)
        except Exception as e:  # show the error to the user instead of a 500 page
            return render_template("index.html", tunings=list(TUNINGS),
                                   error=f"Transcription failed: {e}"), 500
        finally:
            src.unlink(missing_ok=True)
        (job_dir / "tab.txt").write_text(res.tab)
        (job_dir / "score.musicxml").write_text(res.musicxml)
        (job_dir / "notes.json").write_text(res.json)
        return render_template("result.html", res=res, job=job, name=Path(f.filename).stem)

    @app.get("/download/<job>/<kind>")
    def download(job, kind):
        files = {"tab": "tab.txt", "musicxml": "score.musicxml", "json": "notes.json"}
        if kind not in files or not job.isalnum():
            abort(404)
        path = store / job / files[kind]
        if not path.exists():
            abort(404)
        return send_file(path, as_attachment=True,
                         download_name=f"{request.args.get('name', 'tab')}.{path.suffix[1:]}")

    return app
