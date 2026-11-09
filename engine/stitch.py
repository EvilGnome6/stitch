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
MATCH_FALLBACKS = (
    (0.4, 0.45),
)
# Frequency-split: wide LF feather hides seams/tone jumps; hard HF keeps screen.
BLEND_STRENGTH = 5
BLEND_TYPE = "no"  # no | freq | feather | multiband — "no"=hard paste after block gain
LF_SIGMA = 12.0  # illumination only; larger values smear mid-freq into a soft stripe
EXPOS_BLOCK = 64  # large-ish blocks for smooth scanner drift
# Work-megapix BA often leaves a few full-res px of vertical error (sparse
# matches + halftone ambiguity). Probe-compose and nudge cam1.ty to fix.
REFINE_PROBE_SCALE = 0.5
REFINE_MAX_PASSES = 2
REFINE_MIN_ABS_PX = 0.4  # full-res px; ignore tiny residuals


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


def refine_vertical_translation(
    cameras: list,
    images: list[np.ndarray],
    *,
    work_scale: float,
    probe_scale: float = REFINE_PROBE_SCALE,
    max_passes: int = REFINE_MAX_PASSES,
) -> list:
    """Nudge camera[1] ty so a probe hard-seam compose has no vertical step.

    OpenCV's work-megapix affine BA is under-constrained on soft/halftone
    overlaps and often leaves a few full-res pixels of vertical error that
    show up as a step at the booklet seam (especially at the bottom).
    """
    if len(cameras) != 2 or len(images) != 2:
        return cameras
    if work_scale <= 0 or probe_scale <= 0:
        return cameras

    cams = _clone_cameras(cameras)
    for _ in range(max_passes):
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
        aspect = probe_scale / work_scale
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
        # Same mid-overlap cut that compose will use — measure there, not at
        # whatever column happens to have the strongest tone gradient.
        sizes = [(im.shape[1], im.shape[0]) for im in warped]
        x0, y0, W, H = cv.detail.resultRoi(corners=corners, sizes=sizes)
        canvases = []
        for mask, corner in zip(masks, corners):
            m = (np.asarray(mask) > 0).astype(np.uint8)
            cx, cy = int(corner[0]), int(corner[1])
            mh, mw = m.shape[:2]
            canvas = np.zeros((H, W), dtype=np.uint8)
            ix0, iy0 = cx - x0, cy - y0
            sx0, sy0 = max(0, -ix0), max(0, -iy0)
            dx0, dy0 = max(0, ix0), max(0, iy0)
            sx1, sy1 = min(mw, W - ix0), min(mh, H - iy0)
            if sx1 > sx0 and sy1 > sy0:
                canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = m[
                    sy0:sy1, sx0:sx1
                ]
            canvases.append(canvas)
        overlap = (canvases[0] > 0) & (canvases[1] > 0)
        if not np.any(overlap):
            break
        xs = np.where(overlap.any(axis=0))[0]
        seam_x_panorama = int((int(xs.min()) + int(xs.max())) // 2)

        masks = straight_vertical_seam_masks(masks, corners)
        probe, probe_mask = compose_hard_paste(warped, masks, corners)
        # Map panorama seam into the trimmed probe coordinates.
        probe_trim, mask_trim = trim_opaque(probe, probe_mask)
        ys, xxs = np.where(probe_mask > 0)
        if len(xxs) == 0:
            break
        trim_x0 = int(xxs.min())
        seam_x = seam_x_panorama - trim_x0
        offset_probe = measure_vertical_seam_offset(probe_trim, seam_x=seam_x)
        if offset_probe is None:
            break
        dy_full = float(offset_probe) / probe_scale
        if abs(dy_full) < REFINE_MIN_ABS_PX:
            break
        # Guard against runaway corrections on ambiguous content.
        if abs(dy_full) > 80.0:
            break
        R1 = np.asarray(cams[1].R, dtype=np.float32).reshape(3, 3).copy()
        # R translations are in work-scale pixels; positive dy_full means the
        # right panel sits too low in the probe, so push cam1.ty up.
        R1[1, 2] += np.float32(dy_full * work_scale)
        cams[1].R = R1
    return cams


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
            break

    if len(idx_list) < 2:
        raise StitchError(
            "Could not find enough overlapping matches between images "
            "(need clear overlap, like ICE planar). Check order and overlap."
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

    # Work-res BA often leaves a few full-res px of vertical error on soft
    # overlaps; probe-compose and nudge cam1 translation before final warp.
    cameras = refine_vertical_translation(
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

        if blend_type not in ("freq", "no"):
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

    if num_images == 2:
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
        choices=("freq", "feather", "multiband", "no"),
        default=BLEND_TYPE,
        help=f"Blend mode (default {BLEND_TYPE}; freq=LF feather+HF hard seam)",
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
            {"ok": False, "status": "error", "message": str(exc)},
            as_json=as_json,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 — surface to app JSON
        _emit(
            {
                "ok": False,
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
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
        },
        as_json=as_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
