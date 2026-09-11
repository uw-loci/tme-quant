### tme-quant

Python translation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) with the goal of unifying cell and collagen analysis, provided as a modern Python `src/` package and a napari plugin.

### About this repo
- `src/pycurvelets`: Python implementation using the curvelet transform, including the SHG–HE registration port (`SHG_HE_registration`)
- `src/napari_curvealign`: napari plugin surface for interactive use
- `tests/`: pytest suite (data-driven tests and headless napari smoke test)
- `.github/workflows/ci.yml`: GitHub Actions workflow (runs core tests only)

### Wheel vs sdist vs git-dev
Three installable views of the same repo. Both the wheel and the sdist are
produced from a clone (`make wheel` / `make sdist`, or `python -m build`).

| View | How to get it | What it contains |
| --- | --- | --- |
| **Wheel** | `pip install tme-quant` or `make wheel` | `src/` only (`pycurvelets`, `napari_curvealign`, `data/*.npz`). Runtime. |
| **sdist** | `pip install tme-quant --no-binary tme-quant` or `make sdist` | Source + CI-runnable tests and the small patient_001 fixtures. `MANIFEST.in` excludes `tests/matlab_parity` and `tests/artifacts`. |
| **git-dev** | `git clone` + `uv sync` | The sdist contents plus the MATLAB dump harness, comparison figures, and (optional, gitignored) the local patient_02 tree. |

`tests/test_packaging.py` asserts this split. Bit-exact MATLAB parity
(`TMEQ_RUN_MATLAB_PARITY=1`) needs the git-dev tree; see
`tests/matlab_parity/README.md`.

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- napari uses PyQt6 (included in dependencies)

### Quick start (without curvelets)

```bash
uv sync
uv run napari
```

### Optional: curvelet backend (curvelops)

To enable curvelet-powered features and tests you must build FFTW 2.1.5 and CurveLab, then install with curvelops. Use the automated script:

**Prerequisites:** Clone this repo, install [uv](https://docs.astral.sh/uv/), and download CurveLab to `../utils` (see [doc/INSTALL.md](doc/INSTALL.md)).

macOS/Linux:
```bash
bash bin/install.sh
# or: make setup
```

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows: use MSYS2 (for `gcc`, `make`) or Visual Studio toolchain; build FFTW 2.1.5 and CurveLab from source, set `FFTW` and `FDCT` env vars to their install roots, then use `uv` commands as above.

### Development

See [doc/DEVELOPMENT.md](doc/DEVELOPMENT.md) for plugin setup and troubleshooting. Running tests:

- Headless (no GUI): set Qt to offscreen
  - macOS/Linux: `export QT_QPA_PLATFORM=offscreen`
  - Windows/PowerShell: `$env:QT_QPA_PLATFORM = 'offscreen'`

- Default test suite:
```bash
make test
```

- Enable curvelet-dependent tests:
```bash
export TMEQ_RUN_CURVELETS=1
QT_QPA_PLATFORM=offscreen uv run pytest -q tests/test_get_ct.py tests/test_new_curv.py tests/test_process_image.py -rs
```

- Enable strict MATLAB-reference parity assertions:
```bash
export TMEQ_VALIDATE_MATLAB=1
```

- SHG–HE registration (issue 33). CI always runs the cheap gates in
  `tests/test_shg_he_registration.py` and `tests/test_he_bdc_reg1.py`. The
  bit-exact dump suite (tests 1-9) is git-dev only:
```bash
export TMEQ_RUN_MATLAB_PARITY=1
QT_QPA_PLATFORM=offscreen uv run pytest -q -p no:napari \
    tests/test_shg_he_registration.py \
    tests/test_he_bdc_reg1.py \
    tests/test_shg_he_registration_matlab_parity.py \
    tests/test_shg_he_registration_gt.py
```

Notes:
- The napari test is an import-only smoke test (no `Viewer` is created); it runs headless.
- MATLAB parity tests are opt-in because they validate exact numerical agreement with historical MATLAB reference artifacts.

Testing policy:
- Tests must not write files to the repository root. Use a system
  temporary directory instead (e.g., `tempfile.TemporaryDirectory`).
- If a deterministic artifact is needed across runs, commit it under
  `tests/test_resources/` and read from there during tests.

### Continuous integration
- CI has two lanes in `.github/workflows/ci.yml`:
  - `test-basic`: default Python matrix without CurveLab secret requirements.
  - `test-curvelab`: secure lane that fetches/builds CurveLab + FFTW and runs curvelet-enabled tests.
- Both lanes run with `QT_QPA_PLATFORM=offscreen`.

### Working with secrets in GitHub Actions
This project uses GitHub Actions secrets for tasks that require authentication or access to private resources. Secrets can be configured in the [Settings tab of the repostiory](https://github.com/uw-loci/tme-quant/settings/secrets/actions). For more information about secrets, see:
- [Secrets as a concept](https://docs.github.com/en/actions/concepts/security/secrets)
- [Secrets in actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions).

Never hardcode secrets or access tokens in workflow files; always use the `secrets:` context.

### Troubleshooting
- Qt error ("No Qt bindings could be found"): ensure `uv sync` completed; pyproject includes PyQt6.
- Segfault on Viewer creation: avoid creating a `napari.Viewer()` in tests; we only import napari and run offscreen.
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
