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
| `tests/test_shg_he_registration.py` | CI | patient fixtures in the clone (not in wheel/sdist) |
| `tests/test_shg_he_registration_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `tests/matlab_parity/dumps` (git only) |
| `tests/test_shg_he_registration_gt.py` | `TMEQ_RUN_MATLAB_PARITY=1` | dumps + patient_02 GT HE TIFFs (git only) |

See `tests/matlab_parity/README.md` and `tests/artifacts/bdc_registration_viz/README.md`.

### Tumor annotation (BDcreationHE / BDcreationHE2)

| Suite | When it runs | Needs |
| --- | --- | --- |
| `tests/test_he_bdc_annotation.py` | CI | nothing extra |
| `tests/test_tumor_annotation_from_he.py` | CI | patient_001 HE fixtures + CA_Boundary goldens (clone) |
| `tests/test_tumor_annotation_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `tests/matlab_parity/dumps/he2_*` and `he_*` (git only) |

Default `annotation_method="hsv"` is `BDcreationHE2.m`. `annotation_method="rgb_kmeans"` is `BDcreationHE.m` (unseeded MATLAB `kmeans`; Python replays `rng(28,'twister')`). Do not edit `_he_bdc_common.py` / `_he_bdc_reg1.py` for annotation work — new primitives live in `_he_bdc_annotation.py`.

## Wheel vs sdist

| Command | Output |
| --- | --- |
| `make wheel` | `dist/*.whl` — `src/` only (users) |
| `make sdist` | `dist/*.tar.gz` — source + docs + lightweight tests (no tests 1-9 / dumps) |

`tests/test_packaging.py` checks the contract.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, place in `../utils/` |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`