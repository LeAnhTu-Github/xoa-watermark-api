#!/usr/bin/env python3
"""Quality-check a cleaned image against its mask.

Usage:
    python qc.py --original <img> --cleaned <img> --mask <mask.png>

Prints ONLY JSON to stdout:
    {"pass": bool, "score": 0-100,
     "checks": {"residual": 0-100, "halo": 0-100, "blur": 0-100},
     "reasons": [...]}

Metrics (see lib.qc_metrics):
  residual - leftover dark/colored watermark pixels inside the mask core
             (eroded, so edge effects don't count); must be ~0.
  halo     - mean gradient in a ring around the mask vs far background;
             catches visible rims from over-dilation.
  blur     - Laplacian variance inside the repaired region vs its ring;
             catches over-smoothed smudges.
pass when score >= 70. Logs go to stderr. Exit 0 on success.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import log, qc_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="QC a cleaned image")
    ap.add_argument("--original", required=True)
    ap.add_argument("--cleaned", required=True)
    ap.add_argument("--mask", required=True, help="binary mask PNG")
    args = ap.parse_args()

    cleaned = cv2.imread(args.cleaned, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if cleaned is None:
        print(json.dumps({"ok": False,
                          "error": f"could not read cleaned: {args.cleaned}"}))
        return 1
    if mask is None:
        print(json.dumps({"ok": False,
                          "error": f"could not read mask: {args.mask}"}))
        return 1
    if mask.shape[:2] != cleaned.shape[:2]:
        mask = cv2.resize(mask, (cleaned.shape[1], cleaned.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    res = qc_metrics(cleaned, mask)
    log(f"qc {Path(args.cleaned).name}: score={res['score']} "
        f"pass={res['pass']} checks={res['checks']}")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
