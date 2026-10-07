#!/usr/bin/env python3
"""Scan an input directory for images that still need processing.

Usage:
    python scan.py --input <dir> --state <state.json>

Prints ONLY JSON to stdout:
    {"pending": [{"path": "<absolute>", "attempts": 0}], "count": N}

Files already marked done/review in state.json are skipped. New files are
registered as pending. Detailed logs go to stderr. Exit 0 on success.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import log  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"could not parse state file {path}: {e} - starting fresh")
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan input dir for pending images")
    ap.add_argument("--input", required=True, help="input directory with images")
    ap.add_argument("--state", required=True, help="path to state.json")
    args = ap.parse_args()

    in_dir = Path(args.input)
    state_path = Path(args.state)
    if not in_dir.is_dir():
        print(json.dumps({"ok": False, "error": f"input dir not found: {in_dir}"}))
        return 1

    state = load_state(state_path)
    pending = []
    for p in sorted(in_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        key = str(p.resolve())
        entry = state.get(key, {})
        if entry.get("status") in ("done", "review"):
            continue
        attempts = int(entry.get("attempts", 0))
        pending.append({"path": key, "attempts": attempts})
        state.setdefault(key, {"status": "pending", "attempts": attempts,
                               "name": p.name})

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    log(f"scanned {in_dir}: {len(pending)} pending")
    print(json.dumps({"pending": pending, "count": len(pending)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
