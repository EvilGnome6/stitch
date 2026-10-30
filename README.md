# Stitch

Mac-native drop target for planar-stitching overlapping flatbed TIFF scans (CD booklets / multipanel digipaks). ICE-style scan motion: affine-partial alignment, block gain matching, multiband blend. No panorama skew.

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

Engine only:

```bash
cd engine && ./setup.sh
./stitch-engine panel_a.tif panel_b.tif panel_c.tif
./stitch-engine --json -o out.tif *.tif
```

Inputs are sorted by filename unless you pass `--keep-order`. Output defaults to `<first>-stitched.tif`.

Planar only (translation + small rotation + uniform scale). No straighten / crop / autocomplete UI.

## License

Personal tooling. Algorithm lives in `engine/stitch.py`.
