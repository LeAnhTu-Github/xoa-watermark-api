#!/usr/bin/env python3
"""Finalize one image: move original+cleaned(+mask) to done/ or review/ and
update state.json.

Usage:
    python finalize.py --original <img> --cleaned <img> --mask <mask.png> \
        --verdict done|review --state <state.json> --input-dir <dir> \
        [--score 85.0] [--method telea] [--ms 1234.5] [--attempts 1] \
        [--reasons '["..."]']

done/ and review/ are created next to state.json. state.json entry for the
original image is updated with status, score, method, timing and reasons.
Prints ONLY JSON to stdout: {"verdict": ..., "state": "<path>"}.
Logs go to stderr. Exit 0 on success.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import log  # noqa: E402


def unique_dest(d: Path) -> Path:
    if not d.exists():
        return d
    stem, suffix = d.stem, d.suffix
    i = 1
    while True:
        c = d.with_name(f"{stem}_{i}{suffix}")
        if not c.exists():
            return c
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Finalize one image")
    ap.add_argument("--original", required=True)
    ap.add_argument("--cleaned", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--verdict", required=True, choices=["done", "review"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--score", type=float, default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--ms", type=float, default=None)
    ap.add_argument("--attempts", type=int, default=0)
    ap.add_argument("--reasons", default="[]")
    args = ap.parse_args()

    state_path = Path(args.state)
    target_dir = state_path.parent / args.verdict
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = {}
    for label, src in (("original", args.original),
                       ("cleaned", args.cleaned),
                       ("mask", args.mask)):
        s = Path(src)
        if s.is_file():
            dest = unique_dest(target_dir / s.name)
            shutil.move(str(s), str(dest))
            moved[label] = str(dest.resolve())
        else:
            log(f"warning: {label} file missing, skipping: {s}")

    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"could not parse state file: {e}")
    key = str(Path(args.original).resolve())
    try:
        reasons = json.loads(args.reasons)
    except Exception:
        reasons = [args.reasons]
    entry = state.get(key, {})
    entry.update({"status": args.verdict, "attempts": args.attempts,
                  "score": args.score, "method": args.method,
                  "ms": args.ms, "reasons": reasons,
                  "output": moved.get("cleaned"),
                  "name": Path(args.original).name})
    state[key] = entry
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    log(f"finalized {Path(args.original).name} -> {args.verdict}")
    print(json.dumps({"verdict": args.verdict, "state": str(state_path.resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
