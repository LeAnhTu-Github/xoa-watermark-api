"""Shared image-debranding core for the n8n batch pipeline.

Implements the battle-tested rules from AGENTS.md ("Image debranding"):
  1. Corner watermarks on light bg: scan min-channel<230 for the FULL extent
     (never crop tight), mask = min-channel<adaptive(240) dilated 7px,
     cv2.inpaint TELEA r6. Verify zero remaining dark pixels.
  2. Engraved marks on metal: mask gray 45-125, keep connected components
     15-600px NOT touching the image border, dilate 2px, TELEA r3.
     Max 2-3 passes - a faint illegible ghost beats a blurry smudge.
  3. Mid-image colored watermarks: mask = per-pixel color distance from the
     LOCAL background (cv2.medianBlur ksize 21) > 28, components >20px,
     dilate 3px, TELEA r3, mask kept TIGHT to strokes.
     Solid-color icons: NEVER large-area inpaint - transplant real texture
     from a nearby clean donor with feathered edges instead.
     Product colors (yellow H 20-35, S>90) are always excluded from masks.
  4. Watermark overlapping the PRODUCT: split the mask by background; smooth
     product parts are rebuilt via vertical interpolation from clean areas
     above/below (median for robustness); textured parts get normal handling.

Cross-platform: pathlib only, no hardcoded OS paths. Works on Windows
Python 3.11+ with opencv-python, numpy, Pillow installed.

All logging goes to stderr; CLI wrappers print ONLY JSON to stdout.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- logging
def log(msg: str) -> None:
    print(f"[debrand] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- helpers
def dilate_mask(m: np.ndarray, px: int) -> np.ndarray:
    """Dilate a boolean mask by px pixels (elliptical kernel)."""
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def erode_mask(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(m.astype(np.uint8), k).astype(bool)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill holes inside a boolean mask (e.g. white text inside a badge)."""
    inv = (~m).astype(np.uint8)
    h, w = inv.shape
    flood = inv.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 2)
    holes = flood == 1
    return m | holes


def touches_border(m: np.ndarray) -> bool:
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def components(bin_img: np.ndarray, min_px: int, max_px: int | None = None):
    """Yield boolean masks of connected components filtered by pixel count."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img.astype(np.uint8), 8
    )
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        if max_px is not None and area > max_px:
            continue
        yield labels == i


def union(masks) -> np.ndarray | None:
    out = None
    for m in masks:
        out = m.copy() if out is None else (out | m)
    return out


def bbox_of(m: np.ndarray, pad: int = 0):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    h, w = m.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return (y0, y1, x0, x1)


# ---------------------------------------------------------------- detectors
def product_mask_yellow(img_bgr: np.ndarray) -> np.ndarray:
    """Yellow product areas (e.g. handles): H 20-35, S>90 in OpenCV HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 0] >= 20)
        & (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] > 90)
    )


def detect_corner_dark(img_bgr: np.ndarray):
    """Dark text on a light background (corner watermarks and friends).

    Scans the WHOLE image: min-channel < 230 finds the full extent first,
    then the mask uses an adaptive threshold (min(240, local_bg - 12)) so a
    slightly off-white background is not swallowed.
    """
    h, w = img_bgr.shape[:2]
    minch = img_bgr.min(axis=2).astype(np.int16)
    dark = minch < 230
    # Guard against giant components (e.g. a whole mid-gray background is
    # "dark" by the <230 rule): real corner text is small. Without this, the
    # "ring" check below samples interior holes (e.g. white text inside a
    # badge) and false-fires on the entire image.
    max_px = int(0.10 * h * w)
    found = []
    for comp in components(dark, 30, max_px):
        ring = dilate_mask(comp, 20) & ~dilate_mask(comp, 3)
        if ring.sum() == 0:
            continue
        if float(minch[ring].mean()) < 215:
            continue  # not a light background -> not this detector's job
        bg_level = float(np.median(minch[ring]))
        thresh = min(240.0, bg_level - 12.0)
        bb = bbox_of(comp, pad=12)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        local = np.zeros_like(dark, dtype=bool)
        local[y0:y1, x0:x1] = minch[y0:y1, x0:x1] < thresh
        found.append(local)
    return union(found), {"dilate": 7, "radius": 6, "kind": "corner_dark"}


def detect_engraved(img_bgr: np.ndarray):
    """Gray engraved/printed marks (e.g. on metal): gray 45-125, components
    15-600px NOT touching the image border."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binm = (gray >= 45) & (gray <= 125)
    found = [c for c in components(binm, 15, 600) if not touches_border(c)]
    return union(found), {"dilate": 2, "radius": 3, "kind": "engraved"}


def detect_colored(img_bgr: np.ndarray, product_mask: np.ndarray,
                   icon_mask: np.ndarray):
    """Colored text on varied backgrounds: per-pixel color distance from the
    LOCAL background > adaptive threshold, components > 20px.
    Product-colored components are excluded; huge components (a product
    photographed against a contrasting background) are excluded too.

    Kernel note: ksize 31 (not 21) - with thick text strokes a 21px median
    window is ~half text, which drags the "local background" toward the
    text color and punches holes in the mask."""
    h, w = img_bgr.shape[:2]
    local_bg = cv2.medianBlur(img_bgr, 31)
    diff = img_bgr.astype(np.float32) - local_bg.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2)
    # Noise-adaptive threshold: on textured backgrounds (foam, brushed
    # metal) plain >28 fires on background grain. Estimate per-channel noise
    # sigma via MAD and require > max(28, 4*sigma).
    sigma = float(np.median(np.abs(diff).mean(axis=2))) * 1.4826
    thresh = max(28.0, 4.0 * sigma)
    cand = (dist > thresh) & (~icon_mask)
    max_px = int(0.12 * h * w)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    found = []
    for comp in components(cand, 20, max_px):
        if product_mask[comp].mean() > 0.5:
            continue  # product itself, never a watermark
        # This detector is for COLORED text; gray object edges (low
        # saturation) belong to the product, not a watermark. Gray text is
        # covered by the corner_dark / engraved detectors.
        if float(sat[comp].mean()) < 25:
            continue
        found.append(comp)
    return union(found), {"dilate": 3, "radius": 3, "kind": "colored_text"}


def detect_product_text(img_bgr: np.ndarray, product_mask: np.ndarray):
    """Text sitting ON a uniform product (e.g. red stamp on a yellow box).

    Uses the GLOBAL product median color instead of a local median window,
    so thick strokes can't contaminate the background estimate (the failure
    mode of the local-median detector). Returns a boolean mask (or None).
    """
    if int(product_mask.sum()) < 100:
        return None
    # Close text-sized holes in the product mask: text strokes displace the
    # product color, so a raw "yellow fraction" test fails on thick text.
    # Closing with a 25px kernel fills strokes while keeping the product
    # location. Erode by 31px afterwards so the background side of the
    # product edge is never included.
    closed = cv2.morphologyEx(
        product_mask.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    inner = cv2.erode(
        closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    prod_median = np.median(img_bgr[product_mask].astype(np.float32), axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - prod_median, axis=2)
    cand = (dist > 40) & (inner > 0) & (~product_mask)
    found = [c for c in components(cand, 20)]
    return union(found)


def detect_icon(img_bgr: np.ndarray):
    """Solid-color icon/badge (e.g. blue H 85-135, S>50): holes filled so
    inner text is covered too. Strategy is transplant, never inpaint."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    solid = (
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] > 50)
    )
    found = [fill_holes(c) for c in components(solid, 300)]
    # Dilate 5 (not 2): the anti-aliased fringe around a solid icon has low
    # saturation and escapes the hue mask; the wider hard mask swallows it so
    # the final feathered composite never blends icon color back in.
    return union(found), {"dilate": 5, "radius": 0, "kind": "solid_icon"}


def detect_all(img_bgr: np.ndarray, attempt: int = 0,
               method: str = "auto"):
    """Run every detector. Returns (regions, product_mask).

    regions: list of dicts {mask, strategy, dilate, radius, kind}.
    strategy is 'telea' | 'transplant' | 'interp'.
    """
    h, w = img_bgr.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    product = product_mask_yellow(img_bgr)

    icon_m, icon_p = detect_icon(img_bgr)
    icon_m = empty if icon_m is None else icon_m

    regions = []
    corner_m, corner_p = detect_corner_dark(img_bgr)
    if corner_m is not None:
        regions.append({"mask": corner_m, "strategy": "telea", **corner_p})
    engr_m, engr_p = detect_engraved(img_bgr)
    if engr_m is not None:
        regions.append({"mask": engr_m, "strategy": "telea", **engr_p})
    color_m, color_p = detect_colored(img_bgr, product, icon_m)
    if color_m is not None:
        regions.append({"mask": color_m, "strategy": "telea", **color_p})

    if icon_m.any() and method == "auto":
        regions.append({"mask": icon_m, "strategy": "transplant", **icon_p})
    elif icon_m.any():
        # --method telea: fall back to careful inpaint instead of transplant
        regions.append({"mask": icon_m, "strategy": "telea",
                        "dilate": 4, "radius": 4, "kind": "solid_icon_telea"})

    # Watermark text overlapping the product -> split by background.
    # Smooth product parts are rebuilt by vertical interpolation (median),
    # textured parts keep the normal Telea path.
    # NOTE: the split is location-based (dilated product REGION), because
    # text pixels sitting on the product are not product-colored themselves.
    text_union = union([r["mask"] for r in regions
                        if r["strategy"] == "telea"])
    product_region = dilate_mask(product, 10)
    prod_text = detect_product_text(img_bgr, product)
    combined_text = union([m for m in (text_union, prod_text)
                           if m is not None])
    if combined_text is not None:
        on_product = combined_text & product_region
        if on_product.any():
            for r in regions:
                if r["strategy"] == "telea":
                    r["mask"] = r["mask"] & ~product_region
            # Wide dilate (10px): the work area must extend well beyond the
            # text so the final feathered composite (6px) blends on clean
            # background, never on text (which would make an orange halo).
            regions.append({"mask": on_product, "strategy": "interp",
                            "dilate": 10, "radius": 0, "kind": "on_product"})

    # Escalation with attempt number: wider dilation catches faint halos.
    for r in regions:
        if r["strategy"] == "telea":
            r["dilate"] = r["dilate"] + attempt * 3
    return regions, product


# ---------------------------------------------------------------- rebuild
def vertical_interp(img_bgr: np.ndarray, text_mask: np.ndarray,
                    product_region: np.ndarray) -> np.ndarray:
    """Rebuild text pixels sitting on a smooth product by vertical
    interpolation: each column is filled with the median of the clean
    product pixels above/below (robust to outliers).

    product_region is the dilated product LOCATION mask; the actual clean
    product pixels are re-derived by color (yellow) minus the text area.
    """
    work = dilate_mask(text_mask & product_region, 2)
    prod_color = product_mask_yellow(img_bgr) & ~dilate_mask(text_mask, 3)
    out = img_bgr.copy()
    prod_pixels = img_bgr[prod_color & ~work]
    global_med = (np.median(prod_pixels.astype(np.float32), axis=0)
                  if len(prod_pixels) else np.array([128, 128, 128]))
    for x in np.where(work.any(axis=0))[0]:
        rows = np.where(work[:, x])[0]
        clean = prod_color[:, x] & ~work[:, x]
        if int(clean.sum()) >= 3:
            med = np.median(img_bgr[clean, x].astype(np.float32), axis=0)
        else:
            # fallback: nearest column that has clean product pixels
            med = None
            for d in range(1, 60):
                for xx in (x - d, x + d):
                    if 0 <= xx < img_bgr.shape[1]:
                        c2 = prod_color[:, xx] & ~work[:, xx]
                        if int(c2.sum()) >= 3:
                            med = np.median(
                                img_bgr[c2, xx].astype(np.float32), axis=0)
                            break
                if med is not None:
                    break
            if med is None:
                med = global_med
        out[rows, x] = np.clip(med, 0, 255).astype(np.uint8)
    return out


def transplant(img_bgr: np.ndarray, icon_mask: np.ndarray) -> np.ndarray | None:
    """Replace a solid icon with real texture transplanted from a nearby
    clean donor patch. Does a HARD paste - the caller (feather_blend in
    run_pipeline) applies the single feathered composite, so the original
    icon color is never blended back in (past bug: double feathering).
    Returns None when no clean donor fits (caller falls back to Telea)."""
    bb = bbox_of(icon_mask)
    if bb is None:
        return img_bgr
    y0, y1, x0, x1 = bb
    bh, bw = y1 - y0, x1 - x0
    H, W = img_bgr.shape[:2]
    ring = dilate_mask(icon_mask, 25) & ~dilate_mask(icon_mask, 5)
    target_mean = (img_bgr[ring].mean(axis=0) if ring.any()
                   else img_bgr.mean(axis=(0, 1)))

    best, best_score = None, float("inf")
    for dy in (-(bh + 15), 0, bh + 15):
        for dx in (-(bw + 15), 0, bw + 15):
            if dx == 0 and dy == 0:
                continue
            ny0, nx0 = y0 + dy, x0 + dx
            if ny0 < 0 or nx0 < 0 or ny0 + bh > H or nx0 + bw > W:
                continue
            if icon_mask[ny0:ny0 + bh, nx0:nx0 + bw].any():
                continue  # donor must be clean
            donor = img_bgr[ny0:ny0 + bh, nx0:nx0 + bw]
            score = float(np.abs(donor.mean(axis=(0, 1))
                                 - target_mean).sum())
            if score < best_score:
                best, best_score = (ny0, nx0), score
    if best is None:
        return None

    donor = img_bgr[best[0]:best[0] + bh, best[1]:best[1] + bw]
    out = img_bgr.copy()
    region = icon_mask[y0:y1, x0:x1]
    target = out[y0:y1, x0:x1]
    target[region] = donor[region]
    return out


# ---------------------------------------------------------------- LaMa adapter
def find_lama_model() -> Path | None:
    """Locate models/big-lama.onnx without hardcoding OS paths.

    Search order: $LAMA_MODEL env var, <scripts dir>/models/big-lama.onnx,
    ./models/big-lama.onnx (cwd). Returns None when absent - the pipeline
    then silently falls back to Telea (see process.py header for how to
    obtain the model).
    """
    import os
    cands = []
    env = os.environ.get("LAMA_MODEL")
    if env:
        cands.append(Path(env))
    cands.append(Path(__file__).resolve().parent / "models" / "big-lama.onnx")
    cands.append(Path.cwd() / "models" / "big-lama.onnx")
    for c in cands:
        if c.is_file():
            return c
    return None


def lama_inpaint(img_bgr: np.ndarray, mask: np.ndarray,
                 model_path: Path) -> np.ndarray | None:
    """Big-LaMa ONNX inpainting on CPU. Input image is expected in [0,1],
    mask is a hard 0/1 mask (1 = region to fill). Pads to a multiple of 8.

    Returns the inpainted BGR uint8 image, or None on any failure
    (missing onnxruntime, bad model, shape mismatch) - the caller must
    fall back to Telea.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        log("onnxruntime not installed - skipping LaMa, using Telea")
        return None
    try:
        sess = ort.InferenceSession(str(model_path),
                                    providers=["CPUExecutionProvider"])
        in_names = [i.name for i in sess.get_inputs()]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = (mask > 0).astype(np.float32)
        h, w = m.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img_p = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m_p = np.pad(m, ((0, ph), (0, pw)), mode="constant")
        feeds = {}
        for n in in_names:
            nl = n.lower()
            if "mask" in nl:
                feeds[n] = m_p[None, None].astype(np.float32)
            else:  # image
                feeds[n] = img_p.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, feeds)[0]
        out = np.clip(out[0].transpose(1, 2, 0), 0, 1)
        out = out[:h, :w]
        return cv2.cvtColor((out * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)
    except Exception as e:  # never break the pipeline on LaMa
        log(f"LaMa failed ({e}) - falling back to Telea")
        return None


# ---------------------------------------------------------------- pipeline
def feather_blend(base: np.ndarray, work: np.ndarray,
                  hard_mask: np.ndarray, feather_px: float = 6.0) -> np.ndarray:
    """Composite the worked image onto the base with a feathered mask edge
    (~feather_px, in the 4-8px band). Uses the distance transform so the
    weight reaches exactly 1.0 everywhere >= feather_px inside the mask -
    a plain Gaussian blur of the mask never reaches 1.0 on thin regions and
    leaks the original watermark back in (past bug). Pixels outside the mask
    stay bit-identical to the base."""
    if not hard_mask.any():
        return base.copy()
    d = cv2.distanceTransform(hard_mask.astype(np.uint8), cv2.DIST_L2, 3)
    w = np.clip(d / float(feather_px), 0, 1).astype(np.float32)
    w = w[..., None]
    out = base.astype(np.float32) * (1 - w) + work.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


def run_pipeline(img_bgr: np.ndarray, attempt: int = 0,
                 method: str = "auto",
                 lama_model: Path | None = None) -> dict:
    """Full debrand pipeline. Returns dict with keys:
    result, hard_mask, methods (set), stages_ms, lama_used."""
    stages = {}
    base = img_bgr

    t = time.perf_counter()
    regions, product_mask = detect_all(base, attempt=attempt, method=method)
    stages["detect"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    built = []
    for r in regions:
        if r["strategy"] == "telea":
            m = dilate_mask(r["mask"], r["dilate"])
        else:
            m = dilate_mask(r["mask"], r["dilate"])
        if m.any():
            built.append({**r, "mask": m})
    hard_all = union([r["mask"] for r in built])
    if hard_all is None:
        hard_all = np.zeros(base.shape[:2], dtype=bool)
    stages["mask"] = (time.perf_counter() - t) * 1000

    # Crop-and-stitch for large masks: work on bbox + ~160px margin.
    use_crop, off = False, (0, 0)
    bb = bbox_of(hard_all)
    if bb is not None:
        y0, y1, x0, x1 = bb
        if max(y1 - y0, x1 - x0) > 400:
            H, W = base.shape[:2]
            cy0, cy1 = max(0, y0 - 160), min(H, y1 + 160)
            cx0, cx1 = max(0, x0 - 160), min(W, x1 + 160)
            use_crop, off = True, (cy0, cx0)
            crop_img = base[cy0:cy1, cx0:cx1].copy()
            crop_prod = product_mask[cy0:cy1, cx0:cx1]
            crop_regions = []
            for r in built:
                cm = r["mask"][cy0:cy1, cx0:cx1]
                if cm.any():
                    crop_regions.append({**r, "mask": cm})
        else:
            crop_img, crop_prod, crop_regions = base.copy(), product_mask, built
    else:
        crop_img, crop_prod, crop_regions = base.copy(), product_mask, built
    work = crop_img
    methods = set()
    lama_used = False

    t = time.perf_counter()
    for r in crop_regions:
        m8 = r["mask"].astype(np.uint8) * 255
        if r["strategy"] == "interp":
            prod_reg = dilate_mask(product_mask_yellow(work), 10)
            work = vertical_interp(work, r["mask"], prod_reg)
            methods.add("interp")
        elif r["strategy"] == "transplant":
            t_res = transplant(work, r["mask"])
            if t_res is None:
                work = cv2.inpaint(work, m8, 4, cv2.INPAINT_TELEA)
                methods.add("telea")
            else:
                work = t_res
                methods.add("transplant")
        else:  # telea (with attempt escalation)
            if attempt >= 2 and lama_model is not None:
                lama_out = lama_inpaint(work, r["mask"], lama_model)
                if lama_out is not None:
                    # blend LaMa result only inside this region
                    w = cv2.GaussianBlur(r["mask"].astype(np.float32),
                                         (0, 0), 3.0)[..., None]
                    work = (work.astype(np.float32) * (1 - w)
                            + lama_out.astype(np.float32) * w)
                    work = np.clip(work, 0, 255).astype(np.uint8)
                    methods.add("lama")
                    lama_used = True
                    continue
            passes = 2 if attempt >= 1 else 1
            for _ in range(passes):
                work = cv2.inpaint(work, m8, r["radius"], cv2.INPAINT_TELEA)
            methods.add("telea")
    stages["inpaint"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if use_crop:
        cy0, cx0 = off
        cy1, cx1 = cy0 + work.shape[0], cx0 + work.shape[1]
        full_work = base.copy()
        # feathered rect paste (8px) - the mask never touches the crop edge
        # (160px margin), so this only blends identical pixels there.
        rw = np.ones((work.shape[0], work.shape[1]), np.float32)
        feather_px = 8
        rw[:feather_px, :] *= np.linspace(0, 1, feather_px)[:, None]
        rw[-feather_px:, :] *= np.linspace(1, 0, feather_px)[:, None]
        rw[:, :feather_px] *= np.linspace(0, 1, feather_px)[None, :]
        rw[:, -feather_px:] *= np.linspace(1, 0, feather_px)[None, :]
        w3 = rw[..., None]
        full_work[cy0:cy1, cx0:cx1] = (
            full_work[cy0:cy1, cx0:cx1].astype(np.float32) * (1 - w3)
            + work.astype(np.float32) * w3
        ).astype(np.uint8)
    else:
        full_work = work
    result = feather_blend(base, full_work, hard_all, feather_px=6.0)
    stages["composite"] = (time.perf_counter() - t) * 1000

    return {"result": result, "hard_mask": hard_all, "methods": methods,
            "stages_ms": {k: round(v, 1) for k, v in stages.items()},
            "lama_used": lama_used}


# ---------------------------------------------------------------- QC
def qc_metrics(cleaned_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Score a cleaned image against its mask.

    Checks:
      residual - leftover dark/colored watermark pixels inside the mask
                 (eroded core, so edge effects don't count). Must be ~0.
      halo     - mean gradient in a ring around the mask vs far background;
                 catches blurry/visible rims.
      blur     - Laplacian variance inside the inpainted region vs the ring;
                 catches over-smoothed smudges.
    Returns {pass, score, checks, reasons}.
    """
    m = (mask > 127) if mask.dtype != bool else mask
    m = m.astype(bool)
    if int(m.sum()) == 0:
        return {"pass": True, "score": 100.0,
                "checks": {"residual": 100.0, "halo": 100.0, "blur": 100.0},
                "reasons": []}

    core = erode_mask(m, 3)
    if int(core.sum()) == 0:
        core = m

    # --- residual: watermark pixels that survived inside the mask core.
    # The "dark" threshold is adaptive (bg - 12) like the detector, and it is
    # NOT applied on product areas: a yellow product is "dark" by min-channel
    # (minch ~10) yet perfectly clean - the color-distance check covers
    # leftovers there instead.
    minch = cleaned_bgr.min(axis=2).astype(np.int16)
    far_bg = ~dilate_mask(m, 20)
    bg_level = (float(np.median(minch[far_bg])) if far_bg.any() else 255.0)
    dark_thresh = min(240.0, bg_level - 12.0)
    prod_area = dilate_mask(product_mask_yellow(cleaned_bgr), 5)
    local_bg = cv2.medianBlur(cleaned_bgr, 21)
    dist = np.linalg.norm(
        cleaned_bgr.astype(np.float32) - local_bg.astype(np.float32), axis=2
    )
    remnant = ((minch < dark_thresh) & (~prod_area)) | (dist > 28)
    # Compare against the NATURAL anomaly rate of the far background: on
    # textured backgrounds (foam, brushed metal) a fraction of pixels
    # naturally exceeds the thresholds. Only a significant EXCESS inside the
    # mask indicates leftover watermark.
    core_frac = float(remnant[core].mean())
    natural_frac = float(remnant[far_bg].mean()) if far_bg.any() else 0.0
    excess = max(0.0, core_frac - natural_frac * 1.5)  # 50% margin
    frac = excess
    residual = 100.0 * max(0.0, 1.0 - excess * 8.0)

    # --- halo: gradient ring around the mask vs far background
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    ring = (dilate_mask(m, 12) & ~dilate_mask(m, 4))
    far = ~dilate_mask(m, 20)
    rg = float(grad[ring].mean()) if ring.any() else 0.0
    bg = float(grad[far].mean()) if far.any() else float(grad.mean())
    # Absolute formulation: a pure ratio explodes on smooth backgrounds
    # (bg gradient ~ 0). Penalize only a clearly elevated ring gradient.
    halo = 100.0 * float(np.clip(1.0 - max(0.0, rg - bg - 4.0) / 20.0, 0, 1))
    ratio = rg / (bg + 1e-6)

    # --- blur: is the inpainted region suspiciously smoother than its ring?
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vin = float(lap[m].var())
    vout = float(lap[ring].var()) if ring.any() else float(lap.var())
    # On smooth backgrounds both variances are ~0 and their ratio is
    # meaningless noise (can explode to 1e10); there is no texture to
    # preserve, so the check trivially passes.
    if max(vin, vout) < 25.0:
        blur = 100.0
        lratio = 1.0
    else:
        lratio = vin / (vout + 1e-9)
        blur = 100.0 * float(np.exp(-abs(np.log(lratio + 1e-9)) / 1.5))

    reasons = []
    if residual < 70:
        reasons.append(f"residual watermark remnants inside mask: {frac*100:.1f}% of core pixels above background noise level")
    if halo < 70:
        reasons.append(f"visible halo/rim around the repaired area (gradient ratio {ratio:.2f})")
    if blur < 70:
        reasons.append(f"repaired area over-blurred vs surroundings (laplacian var ratio {lratio:.2f})")

    score = 0.5 * residual + 0.25 * halo + 0.25 * blur
    return {"pass": bool(score >= 70.0), "score": round(score, 1),
            "checks": {"residual": round(residual, 1),
                       "halo": round(halo, 1),
                       "blur": round(blur, 1)},
            "reasons": reasons}
