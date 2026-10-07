#!/usr/bin/env python3
"""Process one image: detect watermark/logo, rebuild the region, composite.

LaMa-ONNX adapter (attempt 2 escalation):
    This script looks for the Big-LaMa model at (first hit wins):
      1. $LAMA_MODEL env var
      2. <scripts dir>/models/big-lama.onnx
      3. ./models/big-lama.onnx
    How to obtain it (done ONCE, not needed for normal operation):
      - Download the community ONNX export of Big-LaMa (search
        "big-lama onnx") and place it at scripts/models/big-lama.onnx, and
      - pip install onnxruntime   (it is OPTIONAL - see requirements.txt)
    Expected model I/O: image input float32 [0,1] RGB, shape (1,3,H,W);
    mask input float32 0/1 (1 = fill), shape (1,1,H,W); output (1,3,H,W)
    in [0,1]. The adapter pads H/W to multiples of 8 and unpads afterwards.
    Do NOT quantize to fp16 - CPU inference stays in fp32 for quality.
    When the file (or onnxruntime) is missing the adapter is silently
    skipped and the pipeline falls back to OpenCV Telea - nothing breaks.

Usage:
    python process.py --input <img> --output <img> --attempt <n> [--method auto]

Prints ONLY JSON to stdout:
    {"ok": true, "output": "<absolute>", "mask": "<absolute mask png>",
     "method": "<see below>",
     "mask_area_px": N, "stages_ms": {"detect":..,"mask":..,"inpaint":..,"composite":..}}

method is one of: telea | lama | transplant | interp | none, or "+"-joined
combinations (e.g. "interp+telea") when several strategies were used.
"interp" = vertical-interpolation rebuild of text on the product;
"none" = no watermark detected (output is a bit-identical copy).

Escalation by attempt:
    0 = classical detect + single-pass Telea
    1 = wider dilation + double-pass Telea
    2 = LaMa-ONNX for Telea regions if the model file exists, else Telea

The binary mask is saved to <output>.mask.png. When the mask bbox exceeds
400px the work is done on a crop (bbox + ~160px margin) and stitched back.
Feathering (Gaussian ~5px) is applied ONLY at the final composite onto the
original, so unmasked pixels stay bit-identical. Logs go to stderr.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import find_lama_model, log, run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Debrand a single image")
    ap.add_argument("--input", required=True, help="input image path")
    ap.add_argument("--output", required=True, help="output image path")
    ap.add_argument("--attempt", type=int, default=0,
                    help="escalation level 0..2")
    ap.add_argument("--method", default="auto", choices=["auto", "telea"],
                    help="auto = transplant for solid icons; telea = force Telea")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        print(json.dumps({"ok": False,
                          "error": f"input not found: {in_path}"}))
        return 1

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"ok": False,
                          "error": f"could not decode image: {in_path}"}))
        return 1

    lama_model = find_lama_model() if args.attempt >= 2 else None
    if args.attempt >= 2:
        log(f"LaMa model: {lama_model if lama_model else 'not found - Telea fallback'}")

    t0 = time.perf_counter()
    res = run_pipeline(img, attempt=args.attempt, method=args.method,
                       lama_model=lama_model)
    total_ms = (time.perf_counter() - t0) * 1000

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), res["result"],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(out_path), res["result"])
    mask_path = Path(str(out_path) + ".mask.png")
    cv2.imwrite(str(mask_path), (res["hard_mask"].astype("uint8")) * 255)

    methods = sorted(res["methods"])
    method_str = "+".join(methods) if methods else "none"
    stages = res["stages_ms"]
    stages["total"] = round(total_ms, 1)
    log(f"{in_path.name}: method={method_str} "
        f"mask_px={int(res['hard_mask'].sum())} total={total_ms:.0f}ms")
    print(json.dumps({
        "ok": True,
        "output": str(out_path.resolve()),
        "mask": str(mask_path.resolve()),
        "method": method_str,
        "mask_area_px": int(res["hard_mask"].sum()),
        "stages_ms": stages,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
