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

### Command-line pipeline (`examples/curvealign_pipeline.py`)

This script is the reference for the Fiji plugin and must work on every platform. See [CURVEALIGN_PIPELINE.md](CURVEALIGN_PIPELINE.md) for how to run it and set its parameters. `tests/test_curvealign_pipeline.py` runs unchanged on all of them. Without `TMEQ_RUN_CURVELETS=1`, only the fast tests run, and those don't need curvelops.

Linux, WSL and macOS, with CurveLab extracted to `../utils`:

```bash
bash bin/install.sh
TMEQ_RUN_CURVELETS=1 uv run pytest tests/test_curvealign_pipeline.py -v -rs
uv run python examples/curvealign_pipeline.py tests/test_images/real1.tif --output-dir /tmp/ca_out
```

- **WSL:** use a clone inside the Linux file system, not `/mnt/c` or `/mnt/h`. `bin/install.sh` deletes the project's `.venv` and rebuilds CurveLab in `../utils`, which would break a Windows setup sharing those folders.
- **macOS:** CI does not cover macOS, so run this locally before relying on the script there. Install the compilers first with `xcode-select --install`.

Windows, from an MSYS2 UCRT64 shell with your curvelops environment activated:

```bash
TMEQ_RUN_CURVELETS=1 PYTHONPATH=src python -m pytest tests/test_curvealign_pipeline.py -v -rs
```

### SHG–HE registration (BDcreation_reg / BDcreation_reg2)

| Suite | When it runs | Needs |
| --- | --- | --- |
| `tests/test_he_bdc_reg1.py` | CI | nothing extra |
| `tests/test_shg_he_registration.py` | CI | patient fixtures in the clone (not in wheel/sdist) |
| `tests/test_shg_he_registration_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `archive/matlab_parity/dumps` (git only) |
| `tests/test_shg_he_registration_gt.py` | `TMEQ_RUN_MATLAB_PARITY=1` | dumps + patient_02 GT HE TIFFs (git only) |

See [MATLAB_PARITY.md](MATLAB_PARITY.md), [BDC_REGISTRATION_COMPARISONS.md](BDC_REGISTRATION_COMPARISONS.md), and [ARTIFACTS.md](ARTIFACTS.md). Comparison figures are generated locally.

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