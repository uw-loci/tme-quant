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

`src/ctfire_py/` is **tracked** in this repo (commit `d3d0d12`).
Changes to ctfire_py must only be made in the `32-convert-ctfire` branch of the
CT-FIRE fork repo (`H:\GitHub.06.2022\tmequant_ctfire\tme-quant`).
This branch only syncs from upstream — it never modifies ctfire_py files directly.

**To sync after new work lands on `32-convert-ctfire`:**

```bash
# 1. Update the fork repo and note the new HEAD
cd H:\GitHub.06.2022\tmequant_ctfire\tme-quant
git checkout 32-convert-ctfire && git pull
git log --oneline -1   # note the SHA, e.g. abc1234

# 2. In this repo, replace the folder and commit
cd H:\GitHub.06.2022\tme-quant
Remove-Item -Recurse -Force src/ctfire_py/
Copy-Item -Recurse H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src/ctfire_py src/ctfire_py
git add src/ctfire_py/
git commit -m "sync: update ctfire_py from 32-convert-ctfire <SHA>"
```

Replace `<SHA>` with the commit hash from step 1 so the sync is traceable.

**Merge conflict prevention:** because only `32-convert-ctfire` modifies ctfire_py,
merging both branches into `main` produces no conflicts — the prototype branch just
carries the baseline and defers all changes to `32-convert-ctfire`.