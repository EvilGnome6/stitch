#!/usr/bin/env python3
"""Planar TIFF stitcher for flatbed CD booklet / digipak scans.

Uses OpenCV's scan (affine) stitching path — not photo-panorama
homography — so panels stay square instead of skewing. Feature match
on 8-bit proxies; compose at full resolution with per-image gain
and a narrow feather blend (multiband softens halftone skies).
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
# OpenCV BestOf2NearestMatcher: higher match_conf = stricter Lowe ratio
# (keep if d1 < (1 - match_conf) * d2). 0.65 rejected low-texture booklet
# overlaps; 0.5 still filters weak pairs. conf_thresh gates pairwise edges
# in leaveBiggestComponent / bundle adjust — soft overlaps often land ~0.7.
CONF_THRESH = 0.5
MATCH_CONF = 0.5
# If the first pass finds no connected overlap, retry these (more permissive).
# Soft/halftone booklet overlaps often land ~0.7 pairwise conf — below the
# old OpenCV-style thresh of 1.0 — so keep a short ladder down to 0.3.
MATCH_FALLBACKS = (
    (0.4, 0.45),
    (0.3, 0.3),
)
# Bumped when match/seam defaults change so a stale app Resources engine is obvious.
ENGINE_REVISION = "2026-11-22-seam-match"
# Frequency-split: wide LF feather hides seams/tone jumps; hard HF keeps screen.
BLEND_STRENGTH = 5
# "seam"=narrow mid-overlap feather (ICE-like); "no"=hard paste; also freq/feather/multiband.
BLEND_TYPE = "seam"
SEAM_FEATHER_PX = 28  # ICE planar joins are soft (~seam sharpness), not hard cuts
LF_SIGMA = 12.0  # illumination only; larger values smear mid-freq into a soft stripe
EXPOS_BLOCK = 64  # large-ish blocks for smooth scanner drift
# Work-megapix BA leaves residual tx/ty/rotation on soft/halftone overlaps.
# Probe the warped overlap near the future seam and nudge cam1.R to fix.
REFINE_PROBE_SCALE = 0.5
REFINE_MAX_PASSES = 3
REFINE_MIN_ABS_PX = 0.35  # full-res px; ignore tiny residuals
REFINE_MAX_ABS_PX = 40.0  # guard against runaway corrections
REFINE_MIN_ROT_DEG = 0.004
REFINE_MAX_ROT_DEG = 0.15
REFINE_SEAM_BAND = 70  # probe-px half-width around mid-overlap seam
REFINE_MIN_RESP = 0.15  # phaseCorrelate response floor per band
# After global affine, residual dy(y) along the seam is still ±1–3px on
# soft/halftone booklets. A 1D remap of the right panel, locked to the
# ~6px screen via DoG/high-pass phase correlation, removes most of it.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Mid-overlap vertical cut with a narrow smoothstep feather (ICE-like)."""
    if len(images) != 2:
        return compose_hard_paste(images, masks, corners)

    cans, mcans, _x0, _y0, W, H = _pair_to_canvas(images, masks, corners)
    ox0, ox1, mid = _overlap_mid_x(mcans)
    if mid < 0:
        return compose_hard_paste(images, masks, corners)

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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (bgr_uint8, mask_uint8, meta_from_first)."""
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
    meta["match_conf_used"] = float(used_match[0])
    meta["conf_thresh_used"] = float(used_match[1])
    return result_u8, mask_u8, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Planar (affine) stitch overlapping flatbed TIFF scans."
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
    msg = f"Stitched {meta['count']} panels → {w}×{h}"
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
            "matchConf": meta.get("match_conf_used"),
            "confThresh": meta.get("conf_thresh_used"),
        },
        as_json=as_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
