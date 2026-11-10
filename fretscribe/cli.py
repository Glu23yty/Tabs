"""Command line interface: ``fretscribe input.mp4``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .fretboard import TUNINGS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fretscribe",
                                description="Turn a guitar video or recording into tablature.")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("transcribe", help="transcribe a file (default command)")
    t.add_argument("input", help="video or audio file (mp4, mov, mkv, mp3, wav, ...)")
    t.add_argument("-o", "--out-dir", default=None,
                   help="where to write .tab.txt / .musicxml / .json (default: next to input)")
    t.add_argument("--tuning", default="standard",
                   help=f"{', '.join(TUNINGS)} or note names low->high, e.g. 'D2 A2 D3 G3 B3 E4'")
    t.add_argument("--capo", type=int, default=0)
    t.add_argument("--bpm", type=float, default=None, help="override detected tempo")
    t.add_argument("--beats-per-bar", type=int, default=4)
    t.add_argument("--backend", default="auto", choices=["auto", "basic-pitch", "pyin"])
    t.add_argument("--sensitivity", type=float, default=0.5,
                   help="0..1, higher detects more (quieter) notes; default 0.5")
    t.add_argument("-q", "--quiet", action="store_true")

    w = sub.add_parser("serve", help="start the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("transcribe", "serve", "-h", "--help"):
        argv.insert(0, "transcribe")
    args = p.parse_args(argv)
    if args.cmd is None:
        p.print_help()
        return 1

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if args.cmd == "serve":
        from .web import create_app
        create_app().run(host=args.host, port=args.port)
        return 0

    from .pipeline import run
    sens = min(max(args.sensitivity, 0.0), 1.0)
    res = run(args.input, tuning=args.tuning, capo=args.capo, bpm=args.bpm,
              backend=args.backend, beats_per_bar=args.beats_per_bar,
              onset_threshold=0.75 - 0.5 * sens, frame_threshold=0.45 - 0.3 * sens)
    src = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / src.stem
    Path(f"{base}.tab.txt").write_text(res.tab)
    Path(f"{base}.musicxml").write_text(res.musicxml)
    Path(f"{base}.json").write_text(res.json)
    if not args.quiet:
        print(res.tab)
        print(f"{res.note_count} notes, ~{res.bpm:g} BPM, backend={res.backend}", file=sys.stderr)
        print(f"wrote {base}.tab.txt, {base}.musicxml, {base}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
