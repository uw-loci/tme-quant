### tme-quant

Python translation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) with the goal of unifying cell and collagen analysis, provided as a modern Python `src/` package and a napari plugin.

### About this repo
- `src/pycurvelets`: Python implementation using the curvelet transform
- `src/napari_curvealign`: napari plugin surface for interactive use
- `tests/`: pytest suite (data-driven tests and headless napari smoke test)
- `.github/workflows/ci.yml`: GitHub Actions workflow (runs core tests only)

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Python 3.11+
- For napari: a Qt binding (PyQt or PySide)

### Quick start (without curvelets)

```bash
uv run napari
```

### Optional: curvelet backend (curvelops)

To enable curvelet-powered features and tests you must build and install FFTW 2.1.5 and CurveLab:

macOS/Linux outline:
```bash
make setup
```

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows: use MSYS2 (for `gcc`, `make`) or Visual Studio toolchain; build FFTW 2.1.5 and CurveLab from source, set `FFTW` and `FDCT` env vars to their install roots, then use `uv` commands as above.

### Development: running the tests

- Headless (no GUI): set Qt to offscreen
  - macOS/Linux: `export QT_QPA_PLATFORM=offscreen`
  - Windows/PowerShell: `$env:QT_QPA_PLATFORM = 'offscreen'`

- Core tests (no curvelets):
```bash
make test
```

- Full tests with curvelets (after installing `curvelops`):
```bash
export TMEQ_RUN_CURVELETS=1
make test
```

Notes:
- The napari test is an import-only smoke test (no `Viewer` is created); it runs headless.

Testing policy:
- Tests must not write files to the repository root. Use a system
  temporary directory instead (e.g., `tempfile.TemporaryDirectory`).
- If a deterministic artifact is needed across runs, commit it under
  `tests/test_resources/` and read from there during tests.

### Continuous integration
- CI installs the package without `curvelops` to avoid building FFTW/CurveLab on runners.
- CI environment:
  - `TMEQ_RUN_CURVELETS=0` (curvelet tests skipped)
  - `QT_QPA_PLATFORM=offscreen` (headless napari import)

### Troubleshooting
- Qt error ("No Qt bindings could be found"): install `pyqt` (or `pyside2`) from conda-forge.
- Segfault on Viewer creation: avoid creating a `napari.Viewer()` in tests; we only import napari and run offscreen.
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
