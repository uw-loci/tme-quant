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

- Core tests (no curvelets):
```bash
make test
```

- Full tests with curvelets (after installing `curvelops`): `make test` — curvelet tests run automatically when curvelops is available; otherwise skipped.

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
  - Curvelet tests skipped (curvelops not installed on CI)
  - `QT_QPA_PLATFORM=offscreen` (headless napari import)

### Troubleshooting
- Qt error ("No Qt bindings could be found"): ensure `uv sync` completed; pyproject includes PyQt6.
- Segfault on Viewer creation: avoid creating a `napari.Viewer()` in tests; we only import napari and run offscreen.
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
