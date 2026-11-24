#!/usr/bin/env python3
"""Planar TIFF stitcher for flatbed CD booklet / digipak scans.

Default path is a constrained overlap-strip planar engine: translation
lock on the predicted overlap (phase correlation + ECC), then a rigid
(translation + small bounded rotation) ECC refine on the same strip so
straight lines stay straight across the join, light overlap gain, and a
narrow feather at the best-aligned column. No scale, shear or perspective;
rotation only corrects how the booklet sat on the glass between scans.

Legacy OpenCV ``cv.detail`` affine stitcher remains behind
``--legacy-opencv-stitcher`` for A/B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

# Defaults tuned for flatbed booklet/digipak panels (ICE planar analogue).
WORK_MEGAPIX = 0.6
SEAM_MEGAPIX = 0.1
COMPOSE_MEGAPIX = -1.0  # full resolution
# Predicted horizontal overlap as a fraction of each panel width.
OVERLAP_FRAC = 0.30
OVERLAP_FRAC_MIN = 0.15
OVERLAP_FRAC_MAX = 0.40
# Coarse strip registration downscale target (long edge).
STRIP_COARSE_LONG_EDGE = 900
STRIP_PHASE_MIN_RESP = 0.08
STRIP_ECC_ITERATIONS = 80
STRIP_ECC_EPS = 1e-5
# OpenCV BestOf2NearestMatcher (legacy path only).
CONF_THRESH = 0.5
MATCH_CONF = 0.5
MATCH_FALLBACKS = (
    (0.4, 0.45),
    (0.3, 0.3),
)
# Bumped when match/seam defaults change so a stale app Resources engine is obvious.
ENGINE_REVISION = "2026-11-24-rigid-strip"
# Re-placing the booklet between scans rotates it slightly (seen 0.08–1.1°).
# Translation alone cannot keep a line straight across the seam then, so the
# strip refine estimates a rigid motion. Larger angles mean a bad lock.
RIGID_MAX_ROT_DEG = 2.0
# Below this end-to-end displacement a rotation is not worth resampling for.
RIGID_MIN_EFFECT_PX = 0.5
RIGID_ECC_PYRAMID = ((0.25, 2.0), (0.5, 2.0), (1.0, 1.5))  # (scale, blur sigma)
RIGID_ECC_ITERATIONS = 200
RIGID_ECC_EPS = 1e-7
RIGID_STRIP_MARGIN = 64  # px kept off strip borders (rotation swings ends)
RIGID_MAX_SHIFT_PX = 40.0  # refine may not move the lock further than this
# The right panel keeps its scan geometry; earlier panels rotate onto it.
# Matches ICE's planar output on booklet pairs (A/B flips line up).
# Overlay residual: local phase+ECC shifts on a patch grid across the strip.
RESIDUAL_PATCH = 256
RESIDUAL_STEP_Y = 150
RESIDUAL_COLS = 6
# Seam column search window (fraction of overlap width, centred).
SEAM_SEARCH_FRAC = 0.6
# Frequency-split: wide LF feather hides seams/tone jumps; hard HF keeps screen.
BLEND_STRENGTH = 5
# "seam"=narrow mid-overlap feather (ICE-like); "no"=hard paste; also freq/feather/multiband.
BLEND_TYPE = "seam"
SEAM_FEATHER_PX = 28  # ICE planar joins are soft (~seam sharpness), not hard cuts
LF_SIGMA = 12.0  # illumination only; larger values smear mid-freq into a soft stripe
EXPOS_BLOCK = 64  # large-ish blocks for smooth scanner drift
# Probe the warped overlap near the future seam and nudge translation.
REFINE_PROBE_SCALE = 0.5
REFINE_MAX_PASSES = 3
REFINE_MIN_ABS_PX = 0.35  # full-res px; ignore tiny residuals
REFINE_MAX_ABS_PX = 40.0  # guard against runaway corrections
REFINE_MIN_ROT_DEG = 0.004  # legacy opencv path only
REFINE_MAX_ROT_DEG = 0.15  # legacy opencv path only
REFINE_SEAM_BAND = 70  # probe-px half-width around mid-overlap seam
REFINE_MIN_RESP = 0.15  # phaseCorrelate response floor per band
# Legacy path only: 1D dx(y)/dy(y) remap of the right panel near the seam.
# It fades out across the overlap, so it bends lines that cross the strip;
# the default path corrects the underlying rotation instead.
PIECEWISE_MAX_PASSES = 3
PIECEWISE_STRIP = 90
PIECEWISE_STEP = 30
PIECEWISE_HALF = 70
PIECEWISE_MIN_RESP = 0.08
PIECEWISE_CONVERGE_PX = 0.10
# Prefer LF blur over DoG/high-pass: the ~6px halftone screen creates
# false phaseCorrelate peaks that can "converge" on the wrong period.
PIECEWISE_LF_SIGMA = 2.5
PIECEWISE_SMOOTH_SIGMA = 30.0


class StitchError(Exception):
    pass


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if payload.get("ok"):
            print(payload.get("message") or "ok")
            if payload.get("outputPath"):
                print(payload["outputPath"])
        else:
            print(payload.get("message") or "error", file=sys.stderr)


def _default_output(paths: list[Path]) -> Path:
    first = paths[0]
    return first.with_name(f"{first.stem}-stitched.tif")


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def load_bgr(path: Path) -> tuple[np.ndarray, dict]:
    """Load TIFF/JPEG as BGR uint8 for OpenCV, plus metadata for rewrite."""
    im = Image.open(path)
    meta = {
        "icc_profile": im.info.get("icc_profile"),
        "dpi": im.info.get("dpi", (600.0, 600.0)),
        "mode": im.mode,
        "bits": 8,
    }
    # Pillow may report 16-bit via mode I;16 or via tag
    if im.mode in ("I;16", "I;16B", "I;16L"):
        meta["bits"] = 16
        arr = np.asarray(im, dtype=np.uint16)
        # Downscale to 8-bit for OpenCV stitch path while keeping relative tones.
        arr8 = (arr / 256.0).astype(np.uint8)
        if arr8.ndim == 2:
            bgr = cv.cvtColor(arr8, cv.COLOR_GRAY2BGR)
        else:
            bgr = cv.cvtColor(arr8, cv.COLOR_RGB2BGR)
        meta["source_16"] = True
        return bgr, meta

    if im.mode == "RGBA":
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    elif im.mode == "RGB":
        rgb = np.asarray(im, dtype=np.uint8)
    elif im.mode == "L":
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    else:
        # CMYK etc.
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    # Detect 16-bit via tag when mode is still RGB after convert
    try:
        bps = im.tag_v2.get(258) if hasattr(im, "tag_v2") else None
        if isinstance(bps, (tuple, list)) and bps and int(bps[0]) > 8:
            meta["bits"] = int(bps[0])
    except Exception:
        pass

    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    return bgr, meta


def save_tiff(
    path: Path,
    bgr: np.ndarray,
    *,
    icc_profile: bytes | None,
    dpi: tuple,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    im = Image.fromarray(rgb, mode="RGB")
    kwargs: dict = {
        "dpi": (float(dpi[0]), float(dpi[1])),
        "compression": "tiff_lzw",
    }
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    im.save(path, "TIFF", **kwargs)


def trim_opaque(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the opaque bounding box of the stitch mask."""
    if mask is None:
        return bgr, mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bgr, mask
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _clone_cameras(cameras: list) -> list:
    out = []
    for cam in cameras:
        c = cv.detail.CameraParams()
        c.focal = float(cam.focal)
        c.aspect = float(cam.aspect)
        c.ppx = float(cam.ppx)
        c.ppy = float(cam.ppy)
        c.R = np.asarray(cam.R, dtype=np.float32).reshape(3, 3).copy()
        out.append(c)
    return out


def _find_hard_seam_x(gray: np.ndarray) -> int:
    """Column with the strongest mean |horizontal gradient| in the center half."""
    h, w = gray.shape[:2]
    score = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean(axis=0)
    score[: w // 4] = 0
    score[3 * w // 4 :] = 0
    return int(np.argmax(score))


def measure_vertical_seam_offset(
    bgr: np.ndarray,
    seam_x: int | None = None,
) -> float | None:
    """How many pixels lower the right seam side sits vs the left.

    Prefers a strong horizontal light→dark boundary (page edge / printed bar),
    which is robust to halftone phase ambiguity. Falls back to Sobel-Y phase
    correlation along the seam. Returns None when the signal is unreliable.

    Pass ``seam_x`` from the hard vertical cut when known — max-gradient
    column search can lock onto tone edges that are *not* the paste seam.
    """
    if bgr is None or bgr.size == 0:
        return None
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 80 or w < 200:
        return None
    if seam_x is None:
        # Prefer the geometric mid (booklet cut) when it has a real boundary;
        # fall back to max-gradient. Evaluating both avoids false zeros.
        candidates = [w // 2, _find_hard_seam_x(gray)]
    else:
        candidates = [int(seam_x)]

    y0 = int(h * 0.55)

    def _edge_y(x0: int, x1: int) -> tuple[float, float] | None:
        col = gray[:, x0:x1].astype(np.float32).mean(axis=1)
        d = np.diff(col)
        region = d[y0:]
        if region.size < 8:
            return None
        i = int(np.argmin(region))
        ye = y0 + i
        drop = float(col[max(0, ye - 4)] - col[min(h - 1, ye + 4)])
        if drop < 40.0:
            return None
        y_a = max(0, ye - 40)
        y_b = min(h, ye + 40)
        strip = gray[y_a:y_b, x0:x1]
        up = cv.resize(
            strip,
            (strip.shape[1] * 4, strip.shape[0] * 4),
            interpolation=cv.INTER_CUBIC,
        )
        row = up.astype(np.float32).mean(axis=1)
        dd = np.diff(row)
        j = int(np.argmin(dd))
        if 0 < j < len(dd) - 1:
            a, b, c = float(dd[j - 1]), float(dd[j]), float(dd[j + 1])
            den = a - 2 * b + c
            delta = 0.0 if abs(den) < 1e-6 else 0.5 * (a - c) / den
        else:
            delta = 0.0
        return (y_a + (j + delta) / 4.0), drop

    def _offset_at(sx: int) -> float | None:
        sx = int(np.clip(sx, 80, w - 80))
        left = _edge_y(sx - 70, sx - 15)
        right = _edge_y(sx + 15, sx + 70)
        if left is not None and right is not None:
            return float(right[0] - left[0])
        edges = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        half = min(100, sx - 5, w - sx - 5)
        if half < 40:
            return None
        y_a, y_b = int(h * 0.45), int(h * 0.95)
        a = edges[y_a:y_b, sx - half : sx - 8].astype(np.float64)
        b = edges[y_a:y_b, sx + 8 : sx + half].astype(np.float64)
        mw = min(a.shape[1], b.shape[1])
        if a.shape[0] < 32 or mw < 16:
            return None
        a = a[:, :mw]
        b = b[:, :mw]
        win = np.hanning(a.shape[0])[:, None] * np.hanning(mw)[None, :]
        shift, resp = cv.phaseCorrelate(a * win, b * win)
        if resp < 0.25 or abs(shift[1]) > h * 0.05:
            return None
        return float(shift[1])

    best: float | None = None
    for sx in candidates:
        off = _offset_at(sx)
        if off is None:
            continue
        if best is None or abs(off) > abs(best):
            best = off
    return best


def _warp_pair_to_canvas(
    cams: list,
    images: list[np.ndarray],
    *,
    work_scale: float,
    probe_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]], int, int, int]:
    """Warp both panels into a shared probe canvas. Returns gray cans, masks, corners, seam_x, H, W."""
    aspect = probe_scale / work_scale
    probe_imgs = [
        cv.resize(
            img,
            dsize=None,
            fx=probe_scale,
            fy=probe_scale,
            interpolation=cv.INTER_AREA,
        )
        for img in images
    ]
    focals = [float(c.focal) for c in cams]
    warped_scale = float(np.median(focals)) * aspect
    probe_cams = _clone_cameras(cams)
    for c in probe_cams:
        c.focal *= aspect
        c.ppx *= aspect
        c.ppy *= aspect
    warper = cv.PyRotationWarper("affine", warped_scale)
    corners: list[tuple[int, int]] = []
    warped: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for i, img in enumerate(probe_imgs):
        K = probe_cams[i].K().astype(np.float32)
        corner, image_wp = warper.warp(
            img, K, probe_cams[i].R, cv.INTER_LINEAR, cv.BORDER_REFLECT
        )
        mask = 255 * np.ones(img.shape[:2], np.uint8)
        _, mask_wp = warper.warp(
            mask, K, probe_cams[i].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT
        )
        corners.append(corner)
        warped.append(image_wp)
        masks.append(mask_wp)
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)
    gray_cans: list[np.ndarray] = []
    mask_cans: list[np.ndarray] = []
    for img, mask, corner in zip(warped, masks, corners):
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
        m = (np.asarray(mask) > 0).astype(np.uint8)
        cx, cy = int(corner[0]), int(corner[1])
        mh, mw = m.shape[:2]
        cg = np.zeros((H, W), dtype=np.float32)
        cm = np.zeros((H, W), dtype=np.uint8)
        ix0, iy0 = cx - x0, cy - y0
        sx0, sy0 = max(0, -ix0), max(0, -iy0)
        dx0, dy0 = max(0, ix0), max(0, iy0)
        sx1, sy1 = min(mw, W - ix0), min(mh, H - iy0)
        if sx1 > sx0 and sy1 > sy0:
            cg[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = g[
                sy0:sy1, sx0:sx1
            ]
            cm[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = m[
                sy0:sy1, sx0:sx1
            ]
        gray_cans.append(cg)
        mask_cans.append(cm)
    overlap = (mask_cans[0] > 0) & (mask_cans[1] > 0)
    if not np.any(overlap):
        return gray_cans, mask_cans, corners, -1, H, W
    xs = np.where(overlap.any(axis=0))[0]
    seam_x = int((int(xs.min()) + int(xs.max())) // 2)
    return gray_cans, mask_cans, corners, seam_x, H, W


def _overlap_band_shifts(
    gray_cans: list[np.ndarray],
    mask_cans: list[np.ndarray],
    seam_x: int,
    *,
    band: int = REFINE_SEAM_BAND,
    min_resp: float = REFINE_MIN_RESP,
) -> list[tuple[float, float, float, float]]:
    """Same-pixel phaseCorrelate samples near the mid-overlap seam.

    Returns list of (y_probe, dx_probe, dy_probe, response). dx/dy are the
    shift to apply to cam1 to match cam0 (OpenCV phaseCorrelate convention).
    """
    if seam_x < 0:
        return []
    overlap = (mask_cans[0] > 0) & (mask_cans[1] > 0)
    H, W = gray_cans[0].shape
    x0 = max(0, seam_x - band)
    x1 = min(W, seam_x + band)
    if x1 - x0 < 24:
        return []
    samples: list[tuple[float, float, float, float]] = []
    for frac in np.linspace(0.08, 0.92, 15):
        yc = int(H * frac)
        y0 = yc - 35
        y1 = yc + 35
        if y0 < 0 or y1 > H:
            continue
        mm = overlap[y0:y1, x0:x1]
        if float(mm.mean()) < 0.8:
            continue
        a = gray_cans[0][y0:y1, x0:x1].copy()
        b = gray_cans[1][y0:y1, x0:x1].copy()
        a[~mm] = 0
        b[~mm] = 0
        win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
        shift, resp = cv.phaseCorrelate(a * win, b * win)
        if resp < min_resp:
            continue
        samples.append((float(yc), float(shift[0]), float(shift[1]), float(resp)))
    return samples


def refine_overlap_affine(
    cameras: list,
    images: list[np.ndarray],
    *,
    work_scale: float,
    probe_scale: float = REFINE_PROBE_SCALE,
    max_passes: int = REFINE_MAX_PASSES,
) -> list:
    """Nudge camera[1] tx/ty/rotation so overlap content aligns at the seam.

    OpenCV's work-megapix affine BA is under-constrained on soft/halftone
    booklet overlaps. A page-edge-only ty nudge can zero the bottom light→dark
    step while leaving several full-res pixels of horizontal error and a small
    residual rotation — both of which break the angled halftone screen at the
    hard seam. Measuring same-pixel phase correlation in a narrow band around
    the mid-overlap cut (the cut compose will use) and refining tx, ty, and a
    tiny rotation matches ICE much more closely.
    """
    if len(cameras) != 2 or len(images) != 2:
        return cameras
    if work_scale <= 0 or probe_scale <= 0:
        return cameras

    cams = _clone_cameras(cameras)
    s_ratio = work_scale / probe_scale  # probe-px → work-px for R translations
    for _ in range(max_passes):
        gray_cans, mask_cans, _corners, seam_x, _H, _W = _warp_pair_to_canvas(
            cams, images, work_scale=work_scale, probe_scale=probe_scale
        )
        samples = _overlap_band_shifts(gray_cans, mask_cans, seam_x)
        if len(samples) < 4:
            break
        ys = np.array([s[0] for s in samples], dtype=np.float64)
        dxs = np.array([s[1] for s in samples], dtype=np.float64)
        dys = np.array([s[2] for s in samples], dtype=np.float64)
        # dx(y) ≈ dx0 + θ·y  → θ is residual rotation of cam1 vs cam0 (probe px).
        theta, _dx0 = np.polyfit(ys, dxs, 1)
        dtx_probe = float(np.mean(dxs))
        dty_probe = float(np.mean(dys))
        dtx_full = dtx_probe / probe_scale
        dty_full = dty_probe / probe_scale
        theta_deg = float(np.degrees(theta))

        trans_done = (
            abs(dtx_full) < REFINE_MIN_ABS_PX and abs(dty_full) < REFINE_MIN_ABS_PX
        )
        rot_done = abs(theta_deg) < REFINE_MIN_ROT_DEG
        if trans_done and rot_done:
            break
        if abs(dtx_full) > REFINE_MAX_ABS_PX or abs(dty_full) > REFINE_MAX_ABS_PX:
            break
        if abs(theta_deg) > REFINE_MAX_ROT_DEG:
            # Ambiguous content — skip rotation this pass, still allow translation.
            theta = 0.0

        R = np.asarray(cams[1].R, dtype=np.float64).reshape(3, 3).copy()
        if abs(theta) > 1e-12:
            c, s = float(np.cos(theta)), float(np.sin(theta))
            # Left-multiply in the same [[c,s],[-s,c]] convention BA uses.
            R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
        # phaseCorrelate shift applied to cam1; same work-scale units as BA.
        if not trans_done:
            R[0, 2] += dtx_probe * s_ratio
            R[1, 2] += dty_probe * s_ratio
        cams[1].R = R.astype(np.float32)
    return cams


def refine_vertical_translation(
    cameras: list,
    images: list[np.ndarray],
    *,
    work_scale: float,
    probe_scale: float = REFINE_PROBE_SCALE,
    max_passes: int = REFINE_MAX_PASSES,
) -> list:
    """Backward-compatible alias for :func:`refine_overlap_affine`."""
    return refine_overlap_affine(
        cameras,
        images,
        work_scale=work_scale,
        probe_scale=probe_scale,
        max_passes=max_passes,
    )


def _pair_to_canvas(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[np.ndarray], int, int, int, int]:
    """Paste warped panels into a shared canvas. Returns cans, mcans, x0, y0, W, H."""
    sizes = [(im.shape[1], im.shape[0]) for im in images]
    x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)
    cans: list[np.ndarray] = []
    mcans: list[np.ndarray] = []
    for img, mask, corner in zip(images, masks, corners):
        img = np.asarray(img.get() if hasattr(img, "get") else img)
        mask = np.asarray(mask.get() if hasattr(mask, "get") else mask)
        cx, cy = int(corner[0]), int(corner[1])
        h, w = img.shape[:2]
        c = np.zeros((H, W, 3), dtype=np.uint8)
        m = np.zeros((H, W), dtype=np.uint8)
        ix0, iy0 = cx - x0, cy - y0
        sx0, sy0 = max(0, -ix0), max(0, -iy0)
        dx0, dy0 = max(0, ix0), max(0, iy0)
        sx1, sy1 = min(w, W - ix0), min(h, H - iy0)
        if sx1 > sx0 and sy1 > sy0:
            c[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = img[
                sy0:sy1, sx0:sx1
            ]
            m[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = (
                (mask[sy0:sy1, sx0:sx1] > 0).astype(np.uint8) * 255
            )
        cans.append(c)
        mcans.append(m)
    return cans, mcans, x0, y0, W, H


def _overlap_mid_x(mcans: list[np.ndarray]) -> tuple[int, int, int]:
    ov = (mcans[0] > 0) & (mcans[1] > 0)
    if not np.any(ov):
        return -1, -1, -1
    xs = np.where(ov.any(axis=0))[0]
    ox0, ox1 = int(xs.min()), int(xs.max())
    return ox0, ox1, int((ox0 + ox1) // 2)


def _piecewise_shift_profile(
    gray0: np.ndarray,
    gray1: np.ndarray,
    overlap: np.ndarray,
    seam_x: int,
    *,
    strip: int = PIECEWISE_STRIP,
    step: int = PIECEWISE_STEP,
    half: int = PIECEWISE_HALF,
    min_resp: float = PIECEWISE_MIN_RESP,
    lf_sigma: float = PIECEWISE_LF_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust (y, dx, dy) samples via LF phaseCorrelate along the seam."""
    H = gray0.shape[0]
    h0 = cv.GaussianBlur(gray0.astype(np.float32), (0, 0), lf_sigma)
    h1 = cv.GaussianBlur(gray1.astype(np.float32), (0, 0), lf_sigma)
    ys: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []
    for yc in range(strip, H - strip, step):
        y0, y1 = yc - strip // 2, yc + strip // 2
        x0, x1 = seam_x - half, seam_x + half
        if x0 < 0 or x1 > gray0.shape[1]:
            continue
        mm = overlap[y0:y1, x0:x1]
        if float(mm.mean()) < 0.9:
            continue
        a = h0[y0:y1, x0:x1].copy()
        b = h1[y0:y1, x0:x1].copy()
        a[~mm] = 0
        b[~mm] = 0
        win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
        shift, resp = cv.phaseCorrelate(a * win, b * win)
        if resp < min_resp:
            continue
        ys.append(float(yc))
        dxs.append(float(shift[0]))
        dys.append(float(shift[1]))
    if len(ys) < 8:
        return (
            np.asarray(ys, dtype=np.float64),
            np.asarray(dxs, dtype=np.float64),
            np.asarray(dys, dtype=np.float64),
        )
    ys_a = np.asarray(ys, dtype=np.float64)
    dxs_a = np.asarray(dxs, dtype=np.float64)
    dys_a = np.asarray(dys, dtype=np.float64)
    med_y = float(np.median(dys_a))
    mad_y = float(np.median(np.abs(dys_a - med_y))) + 1e-6
    med_x = float(np.median(dxs_a))
    keep = (np.abs(dys_a - med_y) < 4.0 * 1.4826 * mad_y) & (
        np.abs(dxs_a - med_x) < 5.0
    )
    return ys_a[keep], dxs_a[keep], dys_a[keep]


def piecewise_align_pair(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    *,
    max_passes: int = PIECEWISE_MAX_PASSES,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """Remap the right panel so overlap content tracks the left along the seam.

    Global affine (BA + content refine) still leaves a slowly varying ±1–3px
    residual on soft/halftone booklet scans. Measuring that residual with
    low-pass phase correlation (avoids ~6px screen-period aliases) and
    applying a smooth 1D remap of the right panel matches ICE much more
    closely. Returns full-canvas images/masks with shared origin corners.
    """
    if len(images) != 2:
        return images, masks, corners

    cans, mcans, x0, y0, W, H = _pair_to_canvas(images, masks, corners)
    ox0, ox1, mid = _overlap_mid_x(mcans)
    if mid < 0:
        return images, masks, corners

    order = sorted(range(2), key=lambda i: corners[i][0])
    left_i, right_i = order[0], order[1]
    g_left = cv.cvtColor(cans[left_i], cv.COLOR_BGR2GRAY)
    right = cans[right_i]
    right_m = mcans[right_i]

    fade = ((mcans[0] > 0) & (mcans[1] > 0)).astype(np.float32)
    fade = cv.GaussianBlur(fade, (0, 0), 12)
    band = np.zeros(W, dtype=np.float32)
    half_band = max(180, (ox1 - ox0) // 2)
    band[max(0, mid - half_band) : min(W, mid + half_band)] = 1.0
    band = cv.GaussianBlur(band.reshape(1, -1), (0, 0), 50).ravel()
    fade = fade * band.reshape(1, -1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for _ in range(max_passes):
        g_right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        ov = (mcans[left_i] > 0) & (right_m > 0)
        ys, dxs, dys = _piecewise_shift_profile(g_left, g_right, ov, mid)
        if len(ys) < 8:
            break
        mean_abs = float(np.mean(np.abs(dys)) + np.mean(np.abs(dxs)))
        if mean_abs < PIECEWISE_CONVERGE_PX * 2.0:
            break
        ys_ext = np.concatenate([[0.0], ys, [float(H - 1)]])
        dy_row = np.interp(
            np.arange(H), ys_ext, np.concatenate([[dys[0]], dys, [dys[-1]]])
        ).astype(np.float32)
        dx_row = np.interp(
            np.arange(H), ys_ext, np.concatenate([[dxs[0]], dxs, [dxs[-1]]])
        ).astype(np.float32)
        dy_row = cv.GaussianBlur(
            dy_row.reshape(-1, 1), (0, 0), PIECEWISE_SMOOTH_SIGMA
        ).ravel()
        dx_row = cv.GaussianBlur(
            dx_row.reshape(-1, 1), (0, 0), PIECEWISE_SMOOTH_SIGMA
        ).ravel()
        # phaseCorrelate(src1=left, src2=right): warp right by +shift to match left.
        map_x = xx + dx_row.reshape(-1, 1) * fade
        map_y = yy + dy_row.reshape(-1, 1) * fade
        right = cv.remap(
            right, map_x, map_y, cv.INTER_CUBIC, borderMode=cv.BORDER_REFLECT
        )
        right_m = cv.remap(
            right_m, map_x, map_y, cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT
        )

    cans[right_i] = right
    mcans[right_i] = right_m
    # Both panels now live on the shared canvas at the panorama origin.
    new_corners = [(x0, y0), (x0, y0)]
    return cans, mcans, new_corners


def compose_seam_feather(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    *,
    feather_px: float = SEAM_FEATHER_PX,
    seam_x: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical cut (mid-overlap unless ``seam_x``) with a narrow smoothstep feather."""
    if len(images) != 2:
        return compose_hard_paste(images, masks, corners)

    cans, mcans, _x0, _y0, W, H = _pair_to_canvas(images, masks, corners)
    ox0, ox1, mid = _overlap_mid_x(mcans)
    if mid < 0:
        return compose_hard_paste(images, masks, corners)
    if seam_x is not None and ox0 <= int(seam_x) <= ox1:
        mid = int(seam_x)

    order = sorted(range(2), key=lambda i: corners[i][0])
    left_i, right_i = order[0], order[1]
    # If both corners are identical (post piecewise canvas), fall back to mask extent.
    if corners[0] == corners[1]:
        left_i, right_i = 0, 1
        # Prefer the panel that covers more of the left half as "left".
        mid_col = W // 2
        if int((mcans[1][:, :mid_col] > 0).sum()) > int(
            (mcans[0][:, :mid_col] > 0).sum()
        ):
            left_i, right_i = 1, 0

    cov_l = mcans[left_i] > 0
    cov_r = mcans[right_i] > 0
    half = max(int(round(feather_px)), 1)
    w_l = np.zeros((H, W), dtype=np.float32)
    w_r = np.zeros((H, W), dtype=np.float32)
    w_l[cov_l & ~cov_r] = 1.0
    w_r[cov_r & ~cov_l] = 1.0
    both = cov_l & cov_r
    if np.any(both):
        xs = np.arange(W, dtype=np.float32)
        t = np.clip((xs - (mid - half)) / max(2 * half, 1), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        # Broadcast weights across rows for overlap columns.
        w_l_col = 1.0 - t
        w_r_col = t
        for x in range(max(0, mid - half), min(W, mid + half + 1)):
            col = both[:, x]
            if not np.any(col):
                continue
            w_l[col, x] = float(w_l_col[x])
            w_r[col, x] = float(w_r_col[x])
        # Hard ownership outside the feather band.
        left_band = both.copy()
        left_band[:, mid - half :] = False
        right_band = both.copy()
        right_band[:, : mid + half + 1] = False
        w_l[left_band] = 1.0
        w_r[left_band] = 0.0
        w_l[right_band] = 0.0
        w_r[right_band] = 1.0

    s = np.maximum(w_l + w_r, 1e-6)
    w_l /= s
    w_r /= s
    out = (
        cans[left_i].astype(np.float32) * w_l[..., None]
        + cans[right_i].astype(np.float32) * w_r[..., None]
    )
    out_mask = ((cov_l | cov_r).astype(np.uint8)) * 255
    return np.clip(out, 0, 255).astype(np.uint8), out_mask


def straight_vertical_seam_masks(
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Partition a 2-panel overlap with a vertical cut (booklet left|right)."""
    if len(masks) != 2:
        return masks

    sizes = [(m.shape[1], m.shape[0]) for m in masks]
    x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)
    canvases: list[np.ndarray] = []
    for mask, corner in zip(masks, corners):
        m = (np.asarray(mask) > 0).astype(np.uint8)
        cx, cy = int(corner[0]), int(corner[1])
        h, w = m.shape[:2]
        canvas = np.zeros((H, W), dtype=np.uint8)
        ix0, iy0 = cx - x0, cy - y0
        sx0, sy0 = max(0, -ix0), max(0, -iy0)
        dx0, dy0 = max(0, ix0), max(0, iy0)
        sx1, sy1 = min(w, W - ix0), min(h, H - iy0)
        if sx1 > sx0 and sy1 > sy0:
            canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = m[
                sy0:sy1, sx0:sx1
            ]
        canvases.append(canvas)

    overlap = (canvases[0] > 0) & (canvases[1] > 0)
    if not np.any(overlap):
        return masks

    xs = np.where(overlap.any(axis=0))[0]
    x_mid = int((int(xs.min()) + int(xs.max())) // 2)
    order = sorted(range(2), key=lambda i: corners[i][0])
    left_i, right_i = order[0], order[1]

    owned = [np.zeros((H, W), dtype=np.uint8) for _ in range(2)]
    owned[left_i] = canvases[left_i].copy()
    owned[right_i] = canvases[right_i].copy()
    # Reassign overlap: left panel keeps x < mid, right keeps x >= mid.
    owned[left_i][:, x_mid:] = np.where(
        overlap[:, x_mid:], 0, owned[left_i][:, x_mid:]
    )
    owned[right_i][:, :x_mid] = np.where(
        overlap[:, :x_mid], 0, owned[right_i][:, :x_mid]
    )

    out: list[np.ndarray] = []
    for i, corner in enumerate(corners):
        cx, cy = int(corner[0]), int(corner[1])
        h, w = masks[i].shape[:2]
        ix0, iy0 = cx - x0, cy - y0
        local = np.zeros((h, w), dtype=np.uint8)
        sx0, sy0 = max(0, -ix0), max(0, -iy0)
        dx0, dy0 = max(0, ix0), max(0, iy0)
        sx1, sy1 = min(w, W - ix0), min(h, H - iy0)
        if sx1 > sx0 and sy1 > sy0:
            local[sy0:sy1, sx0:sx1] = owned[i][
                dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)
            ]
        out.append(local * 255)
    return out



def compose_hard_paste(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Copy pixels by seam ownership — no feather, no freq mix (no soft stripe)."""
    sizes = [(im.shape[1], im.shape[0]) for im in images]
    x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out_mask = np.zeros((H, W), dtype=np.uint8)
    for img, mask, corner in zip(images, masks, corners):
        img = np.asarray(img.get() if hasattr(img, "get") else img)
        mask = np.asarray(mask.get() if hasattr(mask, "get") else mask)
        m = mask > 0
        cx, cy = int(corner[0]), int(corner[1])
        h, w = img.shape[:2]
        ix0, iy0 = cx - x0, cy - y0
        sx0, sy0 = max(0, -ix0), max(0, -iy0)
        dx0, dy0 = max(0, ix0), max(0, iy0)
        sx1, sy1 = min(w, W - ix0), min(h, H - iy0)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        sl_y = slice(dy0, dy0 + (sy1 - sy0))
        sl_x = slice(dx0, dx0 + (sx1 - sx0))
        mloc = m[sy0:sy1, sx0:sx1]
        patch = img[sy0:sy1, sx0:sx1]
        out[sl_y, sl_x][mloc] = patch[mloc]
        out_mask[sl_y, sl_x] = np.maximum(out_mask[sl_y, sl_x], mloc.astype(np.uint8) * 255)
    return out, out_mask


def compose_frequency_split(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    *,
    lf_sigma: float = LF_SIGMA,
    feather_px: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Wide LF feather for tone; hard-seam HF so print screen stays sharp.

    Multiband softens flat skies; hard feather leaves a jagged visible join.
    Splitting frequency keeps ICE-like seams without the blur.
    """
    sizes = [(im.shape[1], im.shape[0]) for im in images]
    x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)

    lf_acc = np.zeros((H, W, 3), dtype=np.float64)
    lf_w = np.zeros((H, W), dtype=np.float64)
    hf_acc = np.zeros((H, W, 3), dtype=np.float64)
    hf_w = np.zeros((H, W), dtype=np.float64)
    out_mask = np.zeros((H, W), dtype=np.uint8)

    for img, mask, corner in zip(images, masks, corners):
        cx, cy = int(corner[0]), int(corner[1])
        if hasattr(img, "get"):
            img = img.get()
        if hasattr(mask, "get"):
            mask = mask.get()
        img = np.asarray(img)
        mask = np.asarray(mask)
        h, w = img.shape[:2]
        ix0, iy0 = cx - x0, cy - y0
        if ix0 >= W or iy0 >= H or ix0 + w <= 0 or iy0 + h <= 0:
            continue

        m = (mask > 0).astype(np.uint8)
        if m.max() == 0:
            continue

        # Dilate before distance transform so LF weights *cross* the hard seam
        # cut; otherwise each side is 100% one panel and the tone jump remains.
        feather_i = max(int(round(feather_px)), 1)
        ksize = feather_i * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
        m_lf = cv.dilate(m, kernel)
        dist = cv.distanceTransform(m_lf, cv.DIST_L2, 5)
        w_soft = np.clip(dist / max(feather_px, 1.0), 0.0, 1.0)
        w_soft = np.maximum(w_soft, m_lf.astype(np.float64) * 1e-3)

        img_f = img.astype(np.float64)
        lf = cv.GaussianBlur(img_f, (0, 0), lf_sigma)
        hf = img_f - lf

        # Clip placement to canvas
        src_x0 = max(0, -ix0)
        src_y0 = max(0, -iy0)
        dst_x0 = max(0, ix0)
        dst_y0 = max(0, iy0)
        src_x1 = min(w, W - ix0)
        src_y1 = min(h, H - iy0)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            continue

        ws = w_soft[src_y0:src_y1, src_x0:src_x1]
        mh = m[src_y0:src_y1, src_x0:src_x1]
        lf_p = lf[src_y0:src_y1, src_x0:src_x1]
        hf_p = hf[src_y0:src_y1, src_x0:src_x1]

        yslice = slice(dst_y0, dst_y0 + (src_y1 - src_y0))
        xslice = slice(dst_x0, dst_x0 + (src_x1 - src_x0))

        lf_acc[yslice, xslice] += lf_p * ws[..., None]
        lf_w[yslice, xslice] += ws
        # Hard HF on a straight seam (ICE-like thin join). Soft HF bands read
        # as a fuzzy stripe; LF feather above carries the tone match.
        hf_acc[yslice, xslice] += hf_p * mh[..., None]
        hf_w[yslice, xslice] += mh
        out_mask[yslice, xslice] = np.maximum(out_mask[yslice, xslice], mh * 255)

    denom = np.maximum(lf_w, 1e-6)
    lf_out = lf_acc / denom[..., None]
    # LF-only pixels (HF holes) read as a soft stripe — fill from nearest owner.
    missing = (out_mask > 0) & (hf_w < 0.5)
    if np.any(missing):
        owned = (hf_w >= 0.5).astype(np.uint8)
        # Nearest owned pixel for each hole.
        inv = np.where(owned, 0, 1).astype(np.uint8)
        dist, labels = cv.distanceTransformWithLabels(
            inv, cv.DIST_L2, 5, labelType=cv.DIST_LABEL_PIXEL
        )
        ys, xs = np.where(owned > 0)
        if len(ys):
            # labels point at pixel indices into the owned set layout used by OpenCV;
            # map via coordinate arrays from nonzero owned — use simpler dilate fill:
            kern = np.ones((5, 5), np.uint8)
            for _ in range(8):
                if not np.any((out_mask > 0) & (hf_w < 0.5)):
                    break
                owned_u8 = (hf_w >= 0.5).astype(np.uint8)
                dil = cv.dilate(owned_u8, kern)
                edge = (dil > 0) & (hf_w < 0.5) & (out_mask > 0)
                # Copy HF from a dilated max of weighted values / weight.
                for c in range(3):
                    num = cv.dilate(hf_acc[..., c], kern)
                    hf_acc[..., c] = np.where(edge, num, hf_acc[..., c])
                den = cv.dilate(hf_w, kern)
                hf_w = np.where(edge, np.maximum(den, 1.0), hf_w)
    hf_denom = np.maximum(hf_w, 1e-6)
    hf_out = hf_acc / hf_denom[..., None]
    result = np.clip(lf_out + hf_out, 0, 255).astype(np.uint8)
    return result, out_mask


def _overlap_strips(
    left: np.ndarray,
    right: np.ndarray,
    frac: float = OVERLAP_FRAC,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Right strip of left panel + left strip of right panel (same width)."""
    h0, w0 = left.shape[:2]
    h1, w1 = right.shape[:2]
    frac = float(np.clip(frac, OVERLAP_FRAC_MIN, OVERLAP_FRAC_MAX))
    sw = int(round(min(w0, w1) * frac))
    sw = max(sw, 64)
    sw = min(sw, w0 - 8, w1 - 8)
    h = min(h0, h1)
    # Top-align; flatbed booklet panels are usually same height.
    left_strip = left[:h, w0 - sw : w0]
    right_strip = right[:h, :sw]
    return left_strip, right_strip, sw, w0


def _gray32(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        g = bgr.astype(np.float32)
    else:
        g = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    return g


def _phase_shift(
    a: np.ndarray,
    b: np.ndarray,
    *,
    blur_sigma: float = 1.5,
) -> tuple[float, float, float]:
    """phaseCorrelate shift that moves ``b`` onto ``a``; returns (dx, dy, resp)."""
    ga = _gray32(a)
    gb = _gray32(b)
    if blur_sigma > 0:
        ga = cv.GaussianBlur(ga, (0, 0), blur_sigma)
        gb = cv.GaussianBlur(gb, (0, 0), blur_sigma)
    win = np.hanning(ga.shape[0])[:, None] * np.hanning(ga.shape[1])[None, :]
    (dx, dy), resp = cv.phaseCorrelate(ga * win, gb * win)
    return float(dx), float(dy), float(resp)


def _ecc_translation(
    left_strip: np.ndarray,
    right_strip: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float, float]:
    """Refine strip translation with ECC (MOTION_TRANSLATION only — no rotation)."""
    a = _gray32(left_strip)
    b = _gray32(right_strip)
    # Mild blur stabilizes ECC on screened print.
    a = cv.GaussianBlur(a, (0, 0), 1.0)
    b = cv.GaussianBlur(b, (0, 0), 1.0)
    a8 = np.clip(a, 0, 255).astype(np.uint8)
    b8 = np.clip(b, 0, 255).astype(np.uint8)
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    criteria = (
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        STRIP_ECC_ITERATIONS,
        STRIP_ECC_EPS,
    )
    try:
        cc, warp = cv.findTransformECC(
            a8,
            b8,
            warp,
            cv.MOTION_TRANSLATION,
            criteria,
            None,
            1,
        )
    except cv.error:
        return dx, dy, 0.0
    return float(warp[0, 2]), float(warp[1, 2]), float(cc)


def _strip_feature_translation(
    left_strip: np.ndarray,
    right_strip: np.ndarray,
) -> tuple[float, float, float] | None:
    """SIFT fallback inside the overlap strips → median (dx, dy), score."""
    g0 = _gray32(left_strip).astype(np.uint8)
    g1 = _gray32(right_strip).astype(np.uint8)
    sift = cv.SIFT_create(nfeatures=2000)
    k0, d0 = sift.detectAndCompute(g0, None)
    k1, d1 = sift.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None
    matcher = cv.BFMatcher(cv.NORM_L2)
    raw = matcher.knnMatch(d0, d1, k=2)
    good: list[cv.DMatch] = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None
    # Strips are nominally co-located; translation is median keypoint delta.
    dxs = []
    dys = []
    for m in good:
        p0 = k0[m.queryIdx].pt
        p1 = k1[m.trainIdx].pt
        dxs.append(p0[0] - p1[0])
        dys.append(p0[1] - p1[1])
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    # Inlier fraction within 3 px of median as a crude score.
    keep = (np.abs(np.asarray(dxs) - dx) < 3.0) & (
        np.abs(np.asarray(dys) - dy) < 3.0
    )
    score = float(keep.mean())
    if score < 0.3:
        return None
    return dx, dy, score


def _strip_shift_at_scale(
    left_strip: np.ndarray,
    right_strip: np.ndarray,
    *,
    scale: float,
    blur_sigma: float = 1.5,
) -> tuple[float, float, float]:
    """Phase-correlate strips, optionally downscaled; returns full-res (dx, dy, resp)."""
    if scale < 1.0:
        ls = cv.resize(
            left_strip,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_AREA,
        )
        rs = cv.resize(
            right_strip,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_AREA,
        )
    else:
        ls, rs = left_strip, right_strip
    dx_s, dy_s, resp = _phase_shift(ls, rs, blur_sigma=blur_sigma)
    return dx_s / scale, dy_s / scale, resp


def register_pair_translation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    overlap_frac: float = OVERLAP_FRAC,
) -> tuple[float, float, dict]:
    """Estimate (tx, ty) placing right's origin relative to left's origin.

    Translation only — no rotation / scale / perspective. Uses predicted
    overlap strips: phase correlation → ECC translation, with strip-local
    SIFT as a bootstrap when phase response is weak.
    """
    best: tuple[float, float, dict] | None = None
    # Prefer fractions near the true booklet overlap (~15–25%); wider strips
    # dilute the peak and can lock onto non-overlap structure.
    fracs = [overlap_frac]
    for f in (0.20, 0.22, 0.25, 0.18, 0.30):
        if all(abs(f - x) > 1e-6 for x in fracs):
            fracs.append(f)

    for frac in fracs:
        left_strip, right_strip, sw, w0 = _overlap_strips(left, right, frac)
        h, w = left_strip.shape[:2]
        long_edge = max(h, w)
        scale = 1.0
        if long_edge > STRIP_COARSE_LONG_EDGE:
            scale = STRIP_COARSE_LONG_EDGE / float(long_edge)

        dx, dy, resp = _strip_shift_at_scale(
            left_strip, right_strip, scale=scale, blur_sigma=1.5
        )
        method = "phase"

        if resp < STRIP_PHASE_MIN_RESP:
            feat = _strip_feature_translation(left_strip, right_strip)
            if feat is not None:
                dx, dy, feat_score = feat
                resp = max(resp, feat_score)
                method = "sift-strip"

        # Full-strip ECC refine (translation warp only).
        dx_e, dy_e, cc = _ecc_translation(left_strip, right_strip, dx, dy)
        if cc > 0.15:
            dx, dy = dx_e, dy_e
            method = method + "+ecc"

        # Nominal co-location of strips: right origin at (w0 - sw, 0).
        # phaseCorrelate(left, right) / ECC warp encode the shift that
        # warpAffine would apply to *sample* the right strip (dst=src(x+d)),
        # which moves content by -d. Panel corner placement must therefore
        # use the opposite sign so content moves into alignment.
        tx = float(w0 - sw) - float(dx)
        ty = -float(dy)
        info = {
            "overlap_frac": float(frac),
            "strip_w": int(sw),
            "phase_resp": float(resp),
            "ecc_cc": float(cc),
            "method": method,
            "dx_strip": float(dx),
            "dy_strip": float(dy),
        }
        # Score by phase response only — ECC cc stays high even on bad fracs.
        score = float(resp)
        if best is None or score > best[2].get("_score", -1.0):
            info["_score"] = score
            best = (tx, ty, info)

    if best is None:
        raise StitchError("Overlap-strip registration failed")
    tx, ty, info = best
    info.pop("_score", None)
    if info.get("phase_resp", 0.0) < STRIP_PHASE_MIN_RESP and info.get(
        "ecc_cc", 0.0
    ) < 0.15:
        raise StitchError(
            "Could not lock overlap strip (weak phase/ECC). "
            "Need a clearer horizontal overlap with distinctive features "
            f"[engine {ENGINE_REVISION}]"
        )
    return tx, ty, info


def _rigid_template_rect(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    left_mask: np.ndarray | None,
    tx: float,
    ty: float,
    margin: int = RIGID_STRIP_MARGIN,
) -> tuple[int, int, int, int] | None:
    """Overlap strip in left-panel coords (x0, y0, x1, y1), inset by ``margin``."""
    hL, wL = left_shape[:2]
    hR, wR = right_shape[:2]
    x0 = int(np.ceil(max(0.0, tx))) + margin
    x1 = int(np.floor(min(float(wL), tx + wR))) - margin
    y0 = int(np.ceil(max(0.0, ty))) + margin
    y1 = int(np.floor(min(float(hL), ty + hR))) - margin
    if left_mask is not None and x1 > x0:
        full = np.where((left_mask[:, x0:x1] > 0).all(axis=1))[0]
        if len(full) == 0:
            return None
        y0 = max(y0, int(full.min()) + margin)
        y1 = min(y1, int(full.max()) + 1 - margin)
    if x1 - x0 < 64 or y1 - y0 < 256:
        return None
    return x0, y0, x1, y1


def _ecc_strip(
    template: np.ndarray,
    target: np.ndarray,
    origin: tuple[int, int],
    A: np.ndarray,
    t: np.ndarray,
    motion: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Coarse-to-fine ECC of a left strip against the right panel.

    Model: ``p_right = A @ p_left + t`` (left-panel pixel coords). ``template``
    is the left strip whose top-left sits at ``origin`` in left coords.
    """
    o = np.asarray(origin, dtype=np.float64)
    cc = 0.0
    for scale, sigma in RIGID_ECC_PYRAMID:
        T = cv.GaussianBlur(template, (0, 0), sigma)
        I = cv.GaussianBlur(target, (0, 0), sigma)
        if scale != 1.0:
            T = cv.resize(T, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            I = cv.resize(I, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        # Template pixel u = (p_left - o) * s maps to target pixel v = p_right * s.
        warp = np.zeros((2, 3), dtype=np.float32)
        warp[:, :2] = A
        warp[:, 2] = (A @ o + t) * scale
        cc, warp = cv.findTransformECC(
            T,
            I,
            warp,
            motion,
            (
                cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                RIGID_ECC_ITERATIONS,
                RIGID_ECC_EPS,
            ),
            None,
            1,
        )
        A = warp[:, :2].astype(np.float64)
        t = warp[:, 2].astype(np.float64) / scale - A @ o
    return A, t, float(cc)


def register_pair_rigid(
    left: np.ndarray,
    right: np.ndarray,
    tx: float,
    ty: float,
    *,
    left_mask: np.ndarray | None = None,
    allow_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refine a translation lock into a rigid motion on the overlap strip.

    Returns ``(A, t, info)`` with ``p_right = A @ p_left + t``. ``A`` is a pure
    rotation (or identity): no scale, shear or perspective. Falls back to a
    translation-only ECC refine when the rotation is negligible, out of
    bounds, or rejected by the overlay residual check in the caller.
    """
    rect = _rigid_template_rect(left.shape, right.shape, left_mask, tx, ty)
    A0 = np.eye(2)
    t0 = np.array([-float(tx), -float(ty)])
    info: dict = {"rotation_deg": 0.0, "alignment": "translation"}
    if rect is None:
        return A0, t0, info
    x0, y0, x1, y1 = rect
    gL = _gray32(left)
    gR = _gray32(right)
    template = np.ascontiguousarray(gL[y0:y1, x0:x1])
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    def _moved(A: np.ndarray, t: np.ndarray) -> float:
        return float(np.linalg.norm((A @ centre + t) - (centre + t0)))

    best_A, best_t = A0, t0
    try:
        A_t, t_t, cc_t = _ecc_strip(
            template, gR, (x0, y0), A0, t0, cv.MOTION_TRANSLATION
        )
        if _moved(A_t, t_t) <= RIGID_MAX_SHIFT_PX:
            best_A, best_t = A_t, t_t
            info["ecc_cc_translation"] = cc_t
    except cv.error:
        pass
    if not allow_rotation:
        return best_A, best_t, info

    try:
        A_r, t_r, cc_r = _ecc_strip(
            template, gR, (x0, y0), best_A, best_t, cv.MOTION_EUCLIDEAN
        )
    except cv.error:
        return best_A, best_t, info
    theta = float(np.degrees(np.arctan2(A_r[1, 0], A_r[0, 0])))
    info["rotation_deg_estimate"] = theta
    info["ecc_cc_rigid"] = cc_r
    reach = float(max(left.shape[0], left.shape[1]))
    if abs(theta) > RIGID_MAX_ROT_DEG or _moved(A_r, t_r) > RIGID_MAX_SHIFT_PX:
        return best_A, best_t, info
    if abs(np.radians(theta)) * reach < RIGID_MIN_EFFECT_PX:
        return best_A, best_t, info
    info["rotation_deg"] = theta
    info["alignment"] = "rigid"
    info["_translation_fallback"] = (best_A, best_t)
    return A_r, t_r, info


def place_pair_rigid(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray | None,
    A: np.ndarray,
    t: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resample the left panel into the right panel's frame on a shared canvas.

    The right panel is pasted at an integer offset (never resampled); the
    left panel absorbs the sub-pixel translation and any rotation.
    """
    hL, wL = left.shape[:2]
    hR, wR = right.shape[:2]
    quad = np.array([[0, 0], [wL, 0], [0, hL], [wL, hL]], dtype=np.float64)
    quad_r = quad @ A.T + t
    xs = np.concatenate([quad_r[:, 0], [0.0, float(wR)]])
    ys = np.concatenate([quad_r[:, 1], [0.0, float(hR)]])
    ox = int(np.floor(-xs.min()))
    oy = int(np.floor(-ys.min()))
    W = int(np.ceil(xs.max())) + ox
    H = int(np.ceil(ys.max())) + oy
    Ai = np.linalg.inv(A)
    # Canvas q holds right pixel q - o; left pixel = Ai @ (q - o - t).
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = Ai
    M[:, 2] = -Ai @ (np.array([ox, oy], dtype=np.float64) + t)
    flags = cv.INTER_CUBIC | cv.WARP_INVERSE_MAP
    can_l = cv.warpAffine(left, M, (W, H), flags=flags, borderMode=cv.BORDER_REPLICATE)
    src_m = (
        np.full((hL, wL), 255, dtype=np.uint8)
        if left_mask is None
        else ((left_mask > 0).astype(np.uint8) * 255)
    )
    m_l = cv.warpAffine(
        src_m,
        M,
        (W, H),
        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    m_l = ((m_l >= 128).astype(np.uint8)) * 255
    can_r = np.zeros((H, W, 3), dtype=np.uint8)
    can_r[oy : oy + hR, ox : ox + wR] = right
    m_r = np.zeros((H, W), dtype=np.uint8)
    m_r[oy : oy + hR, ox : ox + wR] = 255
    return [can_l, can_r], [m_l, m_r]


def overlap_residual(
    cans: list[np.ndarray],
    mcans: list[np.ndarray],
    *,
    patch: int = RESIDUAL_PATCH,
) -> dict:
    """Overlay check: local (dx, dy) between the placed panels across the strip.

    Phase correlation + translation ECC on a patch grid (LF blur ignores the
    halftone screen). Perfect alignment gives zeros everywhere.
    """
    ov = (mcans[0] > 0) & (mcans[1] > 0)
    cols = np.where(ov.any(axis=0))[0]
    out = {"n": 0, "med_px": float("inf"), "p90_px": float("inf")}
    if len(cols) == 0:
        return out
    ox0, ox1 = int(cols.min()), int(cols.max())
    P = int(min(patch, (ox1 - ox0) // 2)) // 2 * 2
    if P < 64:
        return out
    g0 = cv.GaussianBlur(_gray32(cans[0]), (0, 0), 2.5)
    g1 = cv.GaussianBlur(_gray32(cans[1]), (0, 0), 2.5)
    win = cv.createHanningWindow((P, P), cv.CV_32F)
    H = ov.shape[0]
    xs = np.linspace(ox0 + P // 2 + 8, ox1 - P // 2 - 8, RESIDUAL_COLS)
    mags: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []
    for yc in range(P // 2 + 8, H - P // 2 - 8, RESIDUAL_STEP_Y):
        for xc in xs.astype(int):
            sl = (slice(yc - P // 2, yc + P // 2), slice(xc - P // 2, xc + P // 2))
            if not ov[sl].all():
                continue
            a = np.ascontiguousarray(g0[sl])
            b = np.ascontiguousarray(g1[sl])
            if float(a.std()) < 6.0 or float(b.std()) < 6.0:
                continue
            (dx, dy), resp = cv.phaseCorrelate(a, b, win)
            if resp < 0.15 or abs(dx) > 8 or abs(dy) > 8:
                continue
            warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            try:
                cc, warp = cv.findTransformECC(
                    a,
                    b,
                    warp,
                    cv.MOTION_TRANSLATION,
                    (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-4),
                    None,
                    1,
                )
            except cv.error:
                continue
            if cc < 0.8:
                continue
            dxs.append(float(warp[0, 2]))
            dys.append(float(warp[1, 2]))
            mags.append(float(np.hypot(warp[0, 2], warp[1, 2])))
    if len(mags) < 6:
        return out
    m = np.asarray(mags)
    return {
        "n": int(len(m)),
        "med_px": float(np.median(m)),
        "p90_px": float(np.percentile(m, 90)),
        "med_abs_dx": float(np.median(np.abs(dxs))),
        "med_abs_dy": float(np.median(np.abs(dys))),
    }


def choose_seam_x(
    cans: list[np.ndarray],
    mcans: list[np.ndarray],
    *,
    feather_px: float = SEAM_FEATHER_PX,
) -> int:
    """Column inside the overlap where the two panels agree best.

    Mean |left − right| per column (LF blur), smoothed over the feather width
    and searched in the central part of the overlap — the join a person
    would pick with a transparent overlay of the strip.
    """
    ov = (mcans[0] > 0) & (mcans[1] > 0)
    cols = np.where(ov.any(axis=0))[0]
    if len(cols) == 0:
        return -1
    ox0, ox1 = int(cols.min()), int(cols.max())
    g0 = cv.GaussianBlur(_gray32(cans[0][:, ox0 : ox1 + 1]), (0, 0), 2.0)
    g1 = cv.GaussianBlur(_gray32(cans[1][:, ox0 : ox1 + 1]), (0, 0), 2.0)
    sub = ov[:, ox0 : ox1 + 1]
    diff = np.where(sub, np.abs(g0 - g1), 0.0)
    count = sub.sum(axis=0).astype(np.float64)
    cost = diff.sum(axis=0) / np.maximum(count, 1.0)
    # Columns the overlap only partly covers (rotated panel ends) are unusable.
    cost[count < 0.95 * count.max()] = np.inf
    k = max(3, int(round(2 * feather_px)) | 1)
    finite = np.isfinite(cost)
    filled = np.where(finite, cost, np.nanmax(cost[finite]) if finite.any() else 0.0)
    smooth = np.convolve(filled, np.ones(k) / k, mode="same")
    smooth[~finite] = np.inf
    n = len(smooth)
    lo = int(n * (1.0 - SEAM_SEARCH_FRAC) / 2.0)
    hi = max(lo + 1, int(n * (1.0 + SEAM_SEARCH_FRAC) / 2.0))
    window = smooth[lo:hi]
    if not np.isfinite(window).any():
        return (ox0 + ox1) // 2
    return ox0 + lo + int(np.argmin(window))


def balance_overlap_gain(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Match right-panel mean RGB to left in the overlap (light color balance)."""
    if len(images) != 2:
        return images
    cans, mcans, _x0, _y0, _W, _H = _pair_to_canvas(images, masks, corners)
    ov = (mcans[0] > 0) & (mcans[1] > 0)
    if not np.any(ov) or int(ov.sum()) < 1024:
        return images
    order = sorted(range(2), key=lambda i: corners[i][0])
    left_i, right_i = order[0], order[1]
    if corners[0] == corners[1]:
        left_i, right_i = 0, 1
    gains = []
    for c in range(3):
        m0 = float(cans[left_i][:, :, c][ov].mean())
        m1 = float(cans[right_i][:, :, c][ov].mean())
        if m1 < 1.0:
            gains.append(1.0)
        else:
            gains.append(float(np.clip(m0 / m1, 0.85, 1.15)))
    out = [im.copy() for im in images]
    right = out[right_i].astype(np.float32)
    for c, g in enumerate(gains):
        right[:, :, c] *= g
    out[right_i] = np.clip(right, 0, 255).astype(np.uint8)
    return out


def stitch_two_planar(
    left: np.ndarray,
    right: np.ndarray,
    *,
    blend_type: str = BLEND_TYPE,
    blend_strength: int = BLEND_STRENGTH,
    overlap_frac: float = OVERLAP_FRAC,
    left_mask: np.ndarray | None = None,
    allow_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stitch one adjacent pair with the overlap-strip planar engine."""
    tx, ty, reg = register_pair_translation(
        left, right, overlap_frac=overlap_frac
    )
    A, t, rigid = register_pair_rigid(
        left, right, tx, ty, left_mask=left_mask, allow_rotation=allow_rotation
    )
    fallback = rigid.pop("_translation_fallback", None)
    images, masks = place_pair_rigid(left, right, left_mask, A, t)
    residual = overlap_residual(images, masks)
    if fallback is not None:
        # Keep the rotation only if the strip overlay really agrees better.
        imgs_t, masks_t = place_pair_rigid(left, right, left_mask, *fallback)
        residual_t = overlap_residual(imgs_t, masks_t)
        rigid["residual_translation"] = residual_t
        if residual_t["p90_px"] <= residual["p90_px"]:
            images, masks, residual = imgs_t, masks_t, residual_t
            A, t = fallback
            rigid["rotation_deg"] = 0.0
            rigid["alignment"] = "translation"
    rigid["residual"] = residual
    reg.update(rigid)
    corners: list[tuple[int, int]] = [(0, 0), (0, 0)]

    images = balance_overlap_gain(images, masks, corners)
    seam_x = choose_seam_x(images, masks)
    reg["seam_x"] = int(seam_x)

    if blend_type == "freq":
        sizes = [(im.shape[1], im.shape[0]) for im in images]
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        feather_px = max(
            16.0, float(np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100.0)
        )
        result_u8, mask_u8 = compose_frequency_split(
            images, masks, corners, lf_sigma=LF_SIGMA, feather_px=feather_px
        )
    elif blend_type == "seam":
        result_u8, mask_u8 = compose_seam_feather(
            images,
            masks,
            corners,
            feather_px=float(SEAM_FEATHER_PX),
            seam_x=seam_x if seam_x >= 0 else None,
        )
    elif blend_type == "no":
        masks_h = straight_vertical_seam_masks(masks, corners)
        result_u8, mask_u8 = compose_hard_paste(images, masks_h, corners)
    else:
        # Feather / multiband via OpenCV blender on translation-placed panels.
        sizes = [(im.shape[1], im.shape[0]) for im in images]
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100.0
        if blend_type == "multiband":
            blender = cv.detail_MultiBandBlender()
            bands = int(np.log(max(blend_width, 1.0)) / np.log(2.0) - 1.0)
            blender.setNumBands(max(1, bands))
        else:
            blender = cv.detail_FeatherBlender()
            blender.setSharpness(1.0 / max(blend_width, 1.0))
        blender.prepare(dst_sz)
        for img, mask, corner in zip(images, masks, corners):
            blender.feed(cv.UMat(img.astype(np.int16)), mask, corner)
        result, result_mask = blender.blend(None, None)
        result_u8 = cv.convertScaleAbs(result)
        mask_u8 = result_mask
        if mask_u8 is not None and len(mask_u8.shape) == 3:
            mask_u8 = mask_u8[:, :, 0]

    result_u8, mask_u8 = trim_opaque(result_u8, mask_u8)
    reg["A"] = [[float(v) for v in row] for row in A]
    reg["t"] = [float(v) for v in t]
    meta = {
        "registration": reg,
        "rotation_deg": float(reg.get("rotation_deg", 0.0)),
        "engine_revision": ENGINE_REVISION,
        "pipeline": "overlap-strip",
    }
    return result_u8, mask_u8, meta


def stitch_planar(
    paths: list[Path],
    *,
    work_megapix: float = WORK_MEGAPIX,
    seam_megapix: float = SEAM_MEGAPIX,
    compose_megapix: float = COMPOSE_MEGAPIX,
    conf_thresh: float = CONF_THRESH,
    match_conf: float = MATCH_CONF,
    blend_strength: int = BLEND_STRENGTH,
    blend_type: str = BLEND_TYPE,
    legacy_opencv_stitcher: bool = False,
    overlap_frac: float = OVERLAP_FRAC,
    allow_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (bgr_uint8, mask_uint8, meta_from_first).

    Default: constrained overlap-strip planar chain (rigid: translation plus
    a small bounded rotation; ``allow_rotation=False`` for translation only).
    Pass ``legacy_opencv_stitcher=True`` for the old ``cv.detail`` path.
    """
    if legacy_opencv_stitcher:
        return stitch_planar_legacy_opencv(
            paths,
            work_megapix=work_megapix,
            seam_megapix=seam_megapix,
            compose_megapix=compose_megapix,
            conf_thresh=conf_thresh,
            match_conf=match_conf,
            blend_strength=blend_strength,
            blend_type=blend_type,
        )

    if len(paths) < 2:
        raise StitchError("Need at least two images to stitch")

    full_imgs: list[np.ndarray] = []
    metas: list[dict] = []
    for path in paths:
        bgr, meta = load_bgr(path)
        if bgr is None or bgr.size == 0:
            raise StitchError(f"Cannot read image: {path}")
        full_imgs.append(bgr)
        metas.append(meta)

    # Successive pairwise onto an accumulating canvas (left → right).
    acc = full_imgs[0]
    acc_mask = 255 * np.ones((acc.shape[0], acc.shape[1]), dtype=np.uint8)
    regs: list[dict] = []
    for i in range(1, len(full_imgs)):
        acc, acc_mask, pair_meta = stitch_two_planar(
            acc,
            full_imgs[i],
            blend_type=blend_type,
            blend_strength=blend_strength,
            overlap_frac=overlap_frac,
            left_mask=acc_mask,
            allow_rotation=allow_rotation,
        )
        regs.append(pair_meta.get("registration") or {})

    meta = dict(metas[0])
    meta["paths_used"] = [str(p) for p in paths]
    meta["dropped"] = []
    meta["count"] = len(full_imgs)
    meta["engine_revision"] = ENGINE_REVISION
    meta["pipeline"] = "overlap-strip"
    meta["registrations"] = regs
    meta["rotations_deg"] = [float(r.get("rotation_deg", 0.0)) for r in regs]
    meta["match_conf_used"] = None
    meta["conf_thresh_used"] = None
    return acc, acc_mask, meta


def stitch_planar_legacy_opencv(
    paths: list[Path],
    *,
    work_megapix: float = WORK_MEGAPIX,
    seam_megapix: float = SEAM_MEGAPIX,
    compose_megapix: float = COMPOSE_MEGAPIX,
    conf_thresh: float = CONF_THRESH,
    match_conf: float = MATCH_CONF,
    blend_strength: int = BLEND_STRENGTH,
    blend_type: str = BLEND_TYPE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Legacy ``cv.detail`` affine stitcher (kept for A/B)."""
    if len(paths) < 2:
        raise StitchError("Need at least two images to stitch")

    finder = cv.SIFT_create()
    full_imgs: list[np.ndarray] = []
    full_sizes: list[tuple[int, int]] = []
    metas: list[dict] = []
    work_images: list[np.ndarray] = []
    seam_images = []

    work_scale = 1.0
    seam_scale = 1.0
    seam_work_aspect = 1.0
    is_work_scale_set = False
    is_seam_scale_set = False

    for path in paths:
        bgr, meta = load_bgr(path)
        if bgr is None or bgr.size == 0:
            raise StitchError(f"Cannot read image: {path}")
        full_imgs.append(bgr)
        full_sizes.append((bgr.shape[1], bgr.shape[0]))
        metas.append(meta)

        if not is_work_scale_set:
            if work_megapix < 0:
                work_scale = 1.0
            else:
                work_scale = min(
                    1.0,
                    np.sqrt(work_megapix * 1e6 / (bgr.shape[0] * bgr.shape[1])),
                )
            is_work_scale_set = True

        work_images.append(
            cv.resize(
                bgr,
                dsize=None,
                fx=work_scale,
                fy=work_scale,
                interpolation=cv.INTER_LINEAR_EXACT,
            )
        )

        if not is_seam_scale_set:
            if seam_megapix > 0:
                seam_scale = min(
                    1.0,
                    np.sqrt(seam_megapix * 1e6 / (bgr.shape[0] * bgr.shape[1])),
                )
            else:
                seam_scale = 1.0
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True

        seam_images.append(
            cv.resize(
                bgr,
                dsize=None,
                fx=seam_scale,
                fy=seam_scale,
                interpolation=cv.INTER_LINEAR_EXACT,
            )
        )

    # leaveBiggestComponent mutates features, so each attempt recomputes them.
    # Higher match_conf is stricter (Lowe ratio); lower conf_thresh is more
    # permissive for connected-component / bundle-adjust edge filtering.
    attempts: list[tuple[float, float]] = [(match_conf, conf_thresh)]
    for mc, ct in MATCH_FALLBACKS:
        if (mc, ct) in attempts:
            continue
        # Only add steps that are at least as permissive as the caller's request.
        if mc <= match_conf and ct <= conf_thresh:
            attempts.append((mc, ct))

    features = None
    pairwise = None
    idx_list: list[int] = []
    used_match = (match_conf, conf_thresh)
    for mc, ct in attempts:
        features = [
            cv.detail.computeImageFeatures2(finder, work_img)
            for work_img in work_images
        ]
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, False, mc)
        pairwise = matcher.apply2(features)
        matcher.collectGarbage()
        # Mutates features/pairwise in place to the largest connected component;
        # returned indices refer to the original image list.
        indices = cv.detail.leaveBiggestComponent(features, pairwise, ct)
        idx_list = [int(i) for i in indices]
        if len(idx_list) >= 2:
            conf_thresh = ct
            used_match = (mc, ct)
            break

    if len(idx_list) < 2:
        raise StitchError(
            "Could not find enough overlapping matches between images "
            "(need clear overlap, like ICE planar). Check order and overlap. "
            f"[engine {ENGINE_REVISION}; tried match_conf/conf_thresh={attempts}]"
        )

    seam_images = [seam_images[i] for i in idx_list]
    full_imgs = [full_imgs[i] for i in idx_list]
    full_sizes = [full_sizes[i] for i in idx_list]
    paths_used = [paths[i] for i in idx_list]
    metas = [metas[i] for i in idx_list]
    num_images = len(full_imgs)

    if len(idx_list) < len(paths):
        kept = set(idx_list)
        dropped_note = [p.name for i, p in enumerate(paths) if i not in kept]
    else:
        dropped_note = []

    estimator = cv.detail_AffineBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise, None)
    if not ok:
        raise StitchError("Affine estimation failed")

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = cv.detail_BundleAdjusterAffinePartial()
    adjuster.setConfThresh(conf_thresh)
    # Refine translation + rotation + scale; leave shear fixed (partial affine).
    refine_mask = np.zeros((3, 3), np.uint8)
    refine_mask[0, 0] = 1
    refine_mask[0, 1] = 1
    refine_mask[0, 2] = 1
    refine_mask[1, 1] = 1
    refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    ok, cameras = adjuster.apply(features, pairwise, cameras)
    if not ok:
        raise StitchError("Affine bundle adjustment failed")

    # Work-res BA leaves residual tx/ty/rotation on soft/halftone overlaps;
    # probe the mid-overlap band and nudge cam1 before the final warp.
    cameras = refine_overlap_affine(
        cameras, full_imgs, work_scale=work_scale
    )

    focals = sorted(float(cam.focal) for cam in cameras)
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (
            focals[len(focals) // 2] + focals[len(focals) // 2 - 1]
        ) / 2.0

    # No wave correction for flatbed scans — would invent curvature.

    corners = []
    masks_warped = []
    images_warped = []
    sizes = []

    warper = cv.PyRotationWarper("affine", warped_image_scale * seam_work_aspect)
    for i in range(num_images):
        mask = 255 * np.ones(
            (seam_images[i].shape[0], seam_images[i].shape[1]), np.uint8
        )
        K = cameras[i].K().astype(np.float32)
        K[0, 0] *= seam_work_aspect
        K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect
        K[1, 2] *= seam_work_aspect
        corner, image_wp = warper.warp(
            seam_images[i], K, cameras[i].R, cv.INTER_LINEAR, cv.BORDER_REFLECT
        )
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        _, mask_wp = warper.warp(
            mask, K, cameras[i].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT
        )
        masks_warped.append(mask_wp)

    images_warped_f = [img.astype(np.float32) for img in images_warped]

    # Large-block gain: smooth scanner tone drift without fine soft blotches.
    compensator = cv.detail_BlocksGainCompensator(EXPOS_BLOCK, EXPOS_BLOCK)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    # Keep full coverage masks for exposure; seam at full-res compose below.
    # (Upscaling a low-res seam leaves HF holes → LF-only soft stripe.)
    masks_warped = [
        np.asarray(m.get() if hasattr(m, "get") else m) for m in masks_warped
    ]
    lowres_seam_masks = None
    if num_images != 2:
        seam_finder = cv.detail_DpSeamFinder("COLOR")
        lowres_seam_masks = seam_finder.find(images_warped_f, corners, masks_warped)
        lowres_seam_masks = [
            np.asarray(m.get() if hasattr(m, "get") else m) for m in lowres_seam_masks
        ]

    # Full-resolution compose
    compose_scale = 1.0
    compose_work_aspect = 1.0
    corners = []
    sizes = []
    compose_images: list[np.ndarray] = []
    compose_masks: list[np.ndarray] = []
    blender = None
    dst_sz = None

    for idx in range(num_images):
        full_img = full_imgs[idx]
        if idx == 0:
            if compose_megapix > 0:
                compose_scale = min(
                    1.0,
                    np.sqrt(
                        compose_megapix
                        * 1e6
                        / (full_img.shape[0] * full_img.shape[1])
                    ),
                )
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper("affine", warped_image_scale)
            for i in range(num_images):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (
                    int(round(full_sizes[i][0] * compose_scale)),
                    int(round(full_sizes[i][1] * compose_scale)),
                )
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

        if abs(compose_scale - 1.0) > 1e-1:
            img = cv.resize(
                full_img,
                dsize=None,
                fx=compose_scale,
                fy=compose_scale,
                interpolation=cv.INTER_LINEAR_EXACT,
            )
        else:
            img = full_img

        K = cameras[idx].K().astype(np.float32)
        # Cubic compose warp keeps:halftone closer to the source than linear.
        corner, image_warped = warper.warp(
            img, K, cameras[idx].R, cv.INTER_CUBIC, cv.BORDER_REFLECT
        )
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_warped = warper.warp(
            mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT
        )
        compensator.apply(idx, corners[idx], image_warped, mask_warped)

        if lowres_seam_masks is not None:
            seam_mask = cv.resize(
                lowres_seam_masks[idx],
                (mask_warped.shape[1], mask_warped.shape[0]),
                0,
                0,
                cv.INTER_LINEAR,
            )
            seam_mask = np.where(seam_mask > 127, 255, 0).astype(np.uint8)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        compose_images.append(image_warped)
        compose_masks.append(mask_warped)

        if blend_type not in ("freq", "no", "seam"):
            image_warped_s = image_warped.astype(np.int16)
            if blender is None:
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100.0
                if blend_width < 1 or blend_type == "no":
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    bands = int(np.log(max(blend_width, 1.0)) / np.log(2.0) - 1.0)
                    blender.setNumBands(max(1, bands))
                else:
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1.0 / max(blend_width, 1.0))
                blender.prepare(dst_sz)
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

    # Non-rigid residual along the seam: remap right panel before cutting.
    if num_images == 2:
        compose_images, compose_masks, corners = piecewise_align_pair(
            compose_images, compose_masks, corners
        )
        if blend_type != "seam":
            # "seam" feathers across the mid-overlap; hard/freq need a cut first.
            compose_masks = straight_vertical_seam_masks(compose_masks, corners)

    if blend_type == "freq":
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        feather_px = max(
            16.0, float(np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100.0)
        )
        result_u8, mask_u8 = compose_frequency_split(
            compose_images,
            compose_masks,
            corners,
            lf_sigma=LF_SIGMA,
            feather_px=feather_px,
        )
    elif blend_type == "seam":
        result_u8, mask_u8 = compose_seam_feather(
            compose_images,
            compose_masks,
            corners,
            feather_px=float(SEAM_FEATHER_PX),
        )
    elif blend_type == "no":
        result_u8, mask_u8 = compose_hard_paste(compose_images, compose_masks, corners)
    else:
        # Re-feed blender with seamed masks for feather/multiband.
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100.0
        if blend_type == "multiband":
            blender = cv.detail_MultiBandBlender()
            bands = int(np.log(max(blend_width, 1.0)) / np.log(2.0) - 1.0)
            blender.setNumBands(max(1, bands))
        else:
            blender = cv.detail_FeatherBlender()
            blender.setSharpness(1.0 / max(blend_width, 1.0))
        blender.prepare(dst_sz)
        for img, mask, corner in zip(compose_images, compose_masks, corners):
            blender.feed(cv.UMat(img.astype(np.int16)), mask, corner)
        result, result_mask = blender.blend(None, None)
        result_u8 = cv.convertScaleAbs(result)
        mask_u8 = result_mask
        if mask_u8 is not None and len(mask_u8.shape) == 3:
            mask_u8 = mask_u8[:, :, 0]

    result_u8, mask_u8 = trim_opaque(result_u8, mask_u8)


    meta = dict(metas[0])
    meta["paths_used"] = [str(p) for p in paths_used]
    meta["dropped"] = dropped_note
    meta["count"] = num_images
    meta["engine_revision"] = ENGINE_REVISION
    meta["pipeline"] = "legacy-opencv-stitcher"
    meta["match_conf_used"] = float(used_match[0])
    meta["conf_thresh_used"] = float(used_match[1])
    return result_u8, mask_u8, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Planar stitch overlapping flatbed TIFF scans "
            "(overlap-strip rigid lock: translation + small placement "
            "rotation; no scale/perspective)."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Input images (TIFF preferred). Default: sort by filename.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output TIFF path (default: <first>-stitched.tif)",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Use CLI argument order instead of sorting by filename",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object to stdout (for the Mac app)",
    )
    parser.add_argument(
        "--work-megapix",
        type=float,
        default=WORK_MEGAPIX,
        help=f"Registration resolution in megapixels (default {WORK_MEGAPIX})",
    )
    parser.add_argument(
        "--blend",
        choices=("seam", "freq", "feather", "multiband", "no"),
        default=BLEND_TYPE,
        help=(
            f"Blend mode (default {BLEND_TYPE}; "
            "seam=narrow mid-overlap feather, freq=LF feather+HF hard seam)"
        ),
    )
    parser.add_argument(
        "--blend-strength",
        type=int,
        default=BLEND_STRENGTH,
        help=f"Blend width factor 0-100 (default {BLEND_STRENGTH})",
    )
    parser.add_argument(
        "--overlap-frac",
        type=float,
        default=OVERLAP_FRAC,
        help=(
            f"Predicted horizontal overlap fraction of panel width "
            f"(default {OVERLAP_FRAC})"
        ),
    )
    parser.add_argument(
        "--legacy-opencv-stitcher",
        action="store_true",
        help="Use the old cv.detail affine stitcher (A/B only)",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help=(
            "Translation only: skip the small rotation that keeps lines "
            "straight when the booklet moved between scans"
        ),
    )
    args = parser.parse_args(argv)
    as_json = args.json

    paths = [p.expanduser().resolve() for p in args.images]
    for p in paths:
        if not p.is_file():
            _emit(
                {
                    "ok": False,
                    "status": "error",
                    "message": f"Not a file: {p}",
                },
                as_json=as_json,
            )
            return 1

    if not args.keep_order:
        paths = sorted(paths, key=lambda p: p.name.lower())

    out = args.output.expanduser().resolve() if args.output else _default_output(paths)

    try:
        bgr, _mask, meta = stitch_planar(
            paths,
            work_megapix=args.work_megapix,
            blend_strength=args.blend_strength,
            blend_type=args.blend,
            legacy_opencv_stitcher=bool(args.legacy_opencv_stitcher),
            overlap_frac=float(args.overlap_frac),
            allow_rotation=not args.no_rotation,
        )
        dpi = meta.get("dpi") or (600.0, 600.0)
        if isinstance(dpi, (int, float)):
            dpi = (float(dpi), float(dpi))
        save_tiff(
            out,
            bgr,
            icc_profile=meta.get("icc_profile"),
            dpi=(float(dpi[0]), float(dpi[1])),
        )
    except StitchError as exc:
        _emit(
            {
                "ok": False,
                "status": "error",
                "message": str(exc),
                "engineRevision": ENGINE_REVISION,
            },
            as_json=as_json,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 — surface to app JSON
        _emit(
            {
                "ok": False,
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "engineRevision": ENGINE_REVISION,
            },
            as_json=as_json,
        )
        return 3

    h, w = bgr.shape[:2]
    pipeline = meta.get("pipeline") or "overlap-strip"
    rotations = meta.get("rotations_deg") or []
    msg = f"Stitched {meta['count']} panels → {w}×{h} [{pipeline}]"
    if any(abs(r) > 0 for r in rotations):
        msg += " (aligned rotation " + ", ".join(f"{r:+.3f}°" for r in rotations) + ")"
    if meta.get("dropped"):
        msg += f" (dropped unmatched: {', '.join(meta['dropped'])})"

    _emit(
        {
            "ok": True,
            "status": "stitched",
            "message": msg,
            "inputPaths": [str(p) for p in paths],
            "outputPath": str(out),
            "width": w,
            "height": h,
            "count": meta["count"],
            "tiff": all(_is_tiff(p) for p in paths),
            "engineRevision": ENGINE_REVISION,
            "pipeline": pipeline,
            "rotationsDeg": rotations,
            "matchConf": meta.get("match_conf_used"),
            "confThresh": meta.get("conf_thresh_used"),
        },
        as_json=as_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
