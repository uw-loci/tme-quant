### tme-quant

Python translation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) with the goal of unifying cell and collagen analysis, provided as a modern Python `src/` package and a napari plugin.

### About this repo
- `src/pycurvelets`: Python implementation using the curvelet transform
- `src/napari_curvealign`: napari plugin surface for interactive use
- `curvealign_py/`: Complete CurveAlign Python API with organized architecture
  - `curvealign/core/`: Core analysis algorithms (visualization-free)
  - `curvealign/types/`: Organized type definitions (core, options, results)
  - `curvealign/visualization/`: Pluggable backends (standalone, napari, ImageJ)
- `tests/`: pytest suite (data-driven tests and headless napari smoke test)
- `.github/workflows/ci.yml`: GitHub Actions workflow (runs core tests only)

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Conda (recommended) or Python 3.10–3.13
- For napari: a Qt binding (PyQt or PySide)

### Quick start (without curvelets)
Install the package and the napari GUI. This path avoids the native curvelet build.

```bash
conda create -y -n napari-env -c conda-forge python=3.11 pip
conda activate napari-env

# project install (editable)
pip install -e .

# napari + Qt
conda install -y -c conda-forge napari pyqt qtpy
```

### Optional: curvelet backend (curvelops)
To enable curvelet-powered features and tests you must build and install FFTW 2.1.5 and CurveLab, then install `curvelops`.

macOS/Linux outline:
```bash
# 1) Build FFTW 2.1.5 (C only)
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
 tar xzf fftw-2.1.5.tar.gz
 cd fftw-2.1.5
 # If configure fails on macOS due to outdated config.{sub,guess}, update them from your system
 ./configure --prefix="$HOME/opt/fftw-2.1.5" --disable-fortran
 make -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
 make install
 export FFTW="$HOME/opt/fftw-2.1.5"

# 2) Build CurveLab 2.1.x
export FDCT="/path/to/CurveLab-2.1.x"
 cd "$FDCT/fdct_wrapping_cpp/src" && make
 cd "$FDCT/fdct/src" && make
 cd "$FDCT/fdct3d/src" && make

# 3) Install build tooling and curvelops
conda activate napari-env
 python -m pip install -U pip
 pip install pybind11 scikit-build-core cmake ninja
 export FFTW="$HOME/opt/fftw-2.1.5"
 export FDCT="/path/to/CurveLab-2.1.x"
 pip install -v "curvelops @ git+https://github.com/PyLops/curvelops@0.23"
```

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows: use MSYS2 (for `gcc`, `make`) or Visual Studio toolchain; build FFTW 2.1.5 and CurveLab from source, set `FFTW` and `FDCT` env vars to their install roots, then install `curvelops` as above. Supervisors can validate these steps on a Windows host.

### Development: running the tests
- Headless (no GUI): set Qt to offscreen
  - macOS/Linux: `export QT_QPA_PLATFORM=offscreen`
  - Windows/PowerShell: `$env:QT_QPA_PLATFORM = 'offscreen'`

- Core tests (no curvelets):
```bash
pytest -q
```

- Full tests with curvelets (after installing `curvelops`):
```bash
export TMEQ_RUN_CURVELETS=1
pytest -q
```

Notes:
- The napari test is an import-only smoke test (no `Viewer` is created); it runs headless.

Testing policy:
- Tests must not write files to the repository root. Use a system
  temporary directory instead (e.g., `tempfile.TemporaryDirectory`).
- If a deterministic artifact is needed across runs, commit it under
  `tests/test_resources/` and read from there during tests.

### CurveAlign Python API

The repository now includes a complete, modern Python API for CurveAlign (`curvealign_py/`):

#### Quick Start with CurveAlign API
```python
# Add to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "curvealign_py"))

# Use the API
import curvealign
result = curvealign.analyze_image(image)

# Optional visualization
from curvealign.visualization import standalone
overlay = standalone.create_overlay(image, result.curvelets)
```

#### Features
- **Core Analysis**: Visualization-free algorithms with minimal dependencies
- **Organized Types**: Clean type packages (core, options, results)
- **Pluggable Visualization**: Support for matplotlib, napari, and ImageJ
- **Framework Integration**: Ready for scientific Python workflows

See `curvealign_py/ARCHITECTURE.md` for complete documentation.

### Continuous integration
- CI installs the package without `curvelops` to avoid building FFTW/CurveLab on runners.
- CI environment:
  - `TMEQ_RUN_CURVELETS=0` (curvelet tests skipped)
  - `QT_QPA_PLATFORM=offscreen` (headless napari import)

### Troubleshooting
- Qt error ("No Qt bindings could be found"): install `pyqt` (or `pyside2`) from conda-forge.
- Segfault on Viewer creation: avoid creating a `napari.Viewer()` in tests; we only import napari and run offscreen.
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
