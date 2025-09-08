### tme-quant

Curvelet-based image analysis utilities and a napari plugin, organized with a modern Python src/ layout.

- `src/pycurvelets`: Python implementation using the curvelet transform
- `src/napari_curvealign`: napari plugin surface for interactive use
- `tests/`: pytest suite (data-driven tests and headless napari smoke test)
- `.github/workflows/ci.yml`: GitHub Actions workflow (runs core tests only)

### Requirements
- macOS or Linux
- Conda (recommended) or Python 3.10–3.13
- For napari: a Qt binding (PyQt or PySide)
- For optional curvelet backend (curvelops): FFTW 2.1.5 and CurveLab 2.1.x built locally

### Quick start (no curvelets)
This runs all tests that do not require the native curvelet backend.

```bash
# create env
conda create -y -n napari-env -c conda-forge python=3.11 pip
conda activate napari-env

# project install (editable)
pip install -e .

# napari + Qt (headless-safe via offscreen)
conda install -y -c conda-forge napari pyqt qtpy

# run tests headless
export QT_QPA_PLATFORM=offscreen
pytest -q
```

Expected: tests pass; curvelops-dependent tests are skipped by default unless enabled.

### Install the curvelet backend (curvelops)
curvelops requires FFTW 2.1.5 and CurveLab (FDCT/FDCT3D) libraries. These are not redistributed here. Build locally and set environment variables so the build can find them.

1) Build FFTW 2.1.5 (C only)
```bash
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
tar xzf fftw-2.1.5.tar.gz
cd fftw-2.1.5
# If configure fails on macOS due to outdated config.{sub,guess}, update from your system (optional):
# cp /usr/share/automake-*/config.sub ./ && cp /usr/share/automake-*/config.guess ./
./configure --prefix="$HOME/opt/fftw-2.1.5" --disable-fortran
make -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
make install
export FFTW="$HOME/opt/fftw-2.1.5"
```

2) Build CurveLab 2.1.x (user-specific path example below)
```bash
export FDCT="/Users/hydrablaster/Desktop/Eliceiri_lab/CurveLab-2.1.2"  # adjust to your path
cd "$FDCT/fdct_wrapping_cpp/src" && make
cd "$FDCT/fdct/src" && make
cd "$FDCT/fdct3d/src" && make
```

3) Install build tooling and curvelops
```bash
conda activate napari-env
python -m pip install -U pip
pip install pybind11 scikit-build-core cmake ninja

# env vars used by curvelops build
export FFTW="$HOME/opt/fftw-2.1.5"
export FDCT="/Users/hydrablaster/Desktop/Eliceiri_lab/CurveLab-2.1.2"  # adjust to your path

# install curvelops from source
pip install -v "curvelops @ git+https://github.com/PyLops/curvelops@0.23"
```

4) Run full test suite including curvelets
```bash
export QT_QPA_PLATFORM=offscreen
export TMEQ_RUN_CURVELETS=1
pytest -q
```

Notes:
- CurveLab and FFTW 2.x are separately licensed; you must accept their licenses and obtain/build them yourself.
- The test `new_curv` writes an `in_curves.csv` artifact in the project root. It is harmless and can be ignored.

### What the tests do
- `tests/test_relative_angles.py`: functional tests for relative-angle utilities (pure-Python).
- `tests/napari_curvealign_test.py`: headless import smoke test for napari (no Viewer creation; avoids OpenGL).
- `tests/test_new_curv.py`: curvelet-based tests; run only when `TMEQ_RUN_CURVELETS=1` and curvelops is installed.

### CI behavior (GitHub Actions)
- CI installs the package without curvelops to avoid building FFTW/CurveLab on runners.
- Environment variables on CI:
  - `TMEQ_RUN_CURVELETS=0` so curvelops tests are skipped.
  - `QT_QPA_PLATFORM=offscreen` so napari can import without a display.

### Troubleshooting
- Qt error ("No Qt bindings could be found"): install `pyqt` (or `pyside2`) from conda-forge.
- Segfault on Viewer creation: avoid creating a `napari.Viewer()` in tests; we only import napari and run offscreen.
- curvelops build errors: ensure `FFTW` and `FDCT` env vars point to install roots and that 2D/3D CurveLab libraries were built.

### Development
- Editable install: `pip install -e .`
- Lint/tests: `ruff` and `pytest`

### Branching and merging
Current status:
- Branch `napari_curve_align` already contains `main` (merged in), but `main` does not yet contain the latest `napari_curve_align` commits.

If you want to consolidate work on `main` now:
```bash
# merge napari_curve_align into main
git checkout main
git merge --no-ff napari_curve_align
git push origin main
# optional: delete the feature branch after merge
# git branch -d napari_curve_align
# git push origin :napari_curve_align
```

Otherwise, continue working on `napari_curve_align` and open a PR to `main` when ready.
