# Stitch

Mac-native drop target for planar-stitching overlapping flatbed TIFF scans (CD booklets / multipanel digipaks). Overlap-strip rigid registration (translation plus the small rotation from re-placing the booklet on the glass), light gain balance, narrow feather at the best-aligned column. No scale, no perspective, no panorama skew.

Same layout as [descreen](../descreen): Python **engine** + AppKit **app**.

## Layout

- `engine/` — `stitch.py` CLI (`--json` for the app)
- `app/Stitch/` — Mac app (drop / Open two or more TIFFs)
- `scripts/sync-engine.sh` — venv + copy into app Resources
- `scripts/build-app.sh` — build `app/build/Stitch.app`

## Quick start

Requires Homebrew Python 3 (`brew install python`) and Apple Command Line Tools (or Xcode).

```bash
./scripts/sync-engine.sh
./scripts/build-app.sh
open app/build/Stitch.app
```

`build-app.sh` always re-runs `sync-engine.sh` so the app does not keep a stale
`Resources/engine`.

Engine only:

```bash
cd engine && ./setup.sh
./stitch-engine panel_a.tif panel_b.tif panel_c.tif
./stitch-engine --json -o out.tif *.tif
```

Inputs are sorted by filename unless you pass `--keep-order`. Output defaults to `<first>-stitched.tif`.

Default pipeline: predicted overlap strips → phase correlation + ECC translation lock → rigid ECC refine on the strip (rotation bounded to ±2°, kept only when the overlay residual improves) → gain → seam at the overlap column where both panels agree best, soft feather. The right panel keeps its scan geometry; earlier panels are rotated onto it (as ICE does), so a straight line crossing the overlap stays straight. `--no-rotation` forces translation only. Pass `--legacy-opencv-stitcher` only for A/B against the old `cv.detail` path. No straighten / crop / autocomplete UI.

## License

Personal tooling. Algorithm lives in `engine/stitch.py`.
