# Development

Use the same install path as users: `bash bin/install.sh` (or `make setup`).

## Napari plugin

The plugin is registered via:
- `src/napari_curvealign/napari.yaml` – manifest (commands, widgets)
- `pyproject.toml` – entry point: `napari-curvealign = "napari_curvealign:napari.yaml"`

After `uv pip install -e .`, run `uv run napari` and open **Plugins → napari-curvealign**.

If you only need plugin segmentation features (Cellpose/StarDist) without curvelets:

```bash
uv sync --extra segmentation
```

## Running tests

```bash
make test
```

Headless (no GUI): `QT_QPA_PLATFORM=offscreen make test`

Curvelet tests run automatically when curvelops is installed; otherwise they are skipped.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, place in `../utils/` |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`

## Synchronizing `ctfire_py` from `32-convert-ctfire`

`src/ctfire_py/` is a direct copy from the CT-FIRE fork repo
(`H:\GitHub.06.2022\tmequant_ctfire\tme-quant`, branch `32-convert-ctfire`).
We use a plain copy — **not** `git subtree` — because `32-convert-ctfire` is
an unstable feature branch that may be rebased or force-pushed.

**To update `ctfire_py` after new work lands on `32-convert-ctfire`:**

```bash
# 1. Note the current HEAD of the source branch
cd H:\GitHub.06.2022\tmequant_ctfire\tme-quant
git log --oneline -1   # e.g. abc1234 some commit message

# 2. In this repo (prototype branch), replace the folder
cd H:\GitHub.06.2022\tme-quant
git rm -r src/ctfire_py/
cp -r ../tmequant_ctfire/tme-quant/src/ctfire_py src/ctfire_py
git add src/ctfire_py/
git commit -m "sync: update ctfire_py from 32-convert-ctfire <SHA>"
```

Replace `<SHA>` with the actual commit hash noted in step 1 so the sync is
traceable.