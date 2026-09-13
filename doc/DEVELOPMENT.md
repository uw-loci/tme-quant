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

### SHG–HE registration (BDcreation_reg / BDcreation_reg2)

| Suite | When it runs | Needs |
| --- | --- | --- |
| `tests/test_he_bdc_reg1.py` | CI | nothing extra |
| `tests/test_shg_he_registration.py` | CI | patient_001 and patient_02 fixtures shipped in the sdist |
| `tests/test_shg_he_registration_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `tests/matlab_parity/dumps` (in the sdist) |
| `tests/test_shg_he_registration_gt.py` | `TMEQ_RUN_MATLAB_PARITY=1` | dumps + patient_02 GT HE TIFFs (in the sdist) |

See `tests/matlab_parity/README.md` and `tests/artifacts/bdc_regression_viz/README.md`.

## Wheel vs sdist

| Command | Output |
| --- | --- |
| `make wheel` | `dist/*.whl` — `src/` only (users) |
| `make sdist` | `dist/*.tar.gz` — source + tests, dumps, figures, registration fixtures (devs) |

`tests/test_packaging.py` checks the contract.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, place in `../utils/` |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`