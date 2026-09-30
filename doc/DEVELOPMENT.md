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

## scikit-ops prototype

See [SCIKIT_OPS.md](SCIKIT_OPS.md) for the complete op catalog, invocation
examples, data conventions, and current GUI discovery status.

Install the optional development dependency:

```bash
uv sync --extra scikit-ops
```

The starter collection is in `src/tme_quant_ops`. Its two ops wrap
UI-independent functions from `pycurvelets.segmentation`:

- `segment_threshold`: image to label image
- `boundary_labels`: label image to boundary labels

Discover them without building an isolated environment:

```bash
uv run python -c 'import skop; print(skop.discover("tme_quant_ops"))'
uv run pytest -q tests/test_scikit_ops.py
```

Run them through Appose in the environment defined by
`envs/tme-quant/pixi.toml`:

```bash
uv run python examples/scikit_ops_demo.py
```

The first Appose run installs Pixi if needed and builds the environment. Later
runs reuse it. During development, `Runner(root=.../src)` adds this checkout to
the worker's import path. Add a released TME-Quant package or pinned Git
revision to `pixi.toml` before publishing the op collection.

### Current GUI-host limitation

As of the pinned scikit-ops revision, `skop.discover()` can discover this
third-party collection by package name, but the stock `skop-napari` panel and
`skop-fiji` service still default to the built-in `skop.ops` collection. The
ops and environment can therefore be developed and exercised through
`skop.Runner` now, but making them appear in both stock GUIs requires either:

1. contributing the ops to the upstream `skop.ops` collection; or
2. adding configurable collection/package discovery to both host projects.

Do not copy these modules into an installed `skop` package: that makes the
prototype depend on mutable site-packages state and cannot be distributed
reliably.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, place in `../utils/` |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`
