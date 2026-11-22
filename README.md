# Stitch

Mac-native drop target for planar-stitching overlapping flatbed TIFF scans (CD booklets / multipanel digipaks). Overlap-strip translation registration, light gain balance, narrow mid-overlap feather. No rotation, no perspective, no panorama skew.

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

Default pipeline: predicted overlap strips → phase correlation + ECC translation → optional 1D seam polish → gain + soft feather. Pass `--legacy-opencv-stitcher` only for A/B against the old `cv.detail` path. No straighten / crop / autocomplete UI.

## License

Personal tooling. Algorithm lives in `engine/stitch.py`.
