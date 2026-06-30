# MSYS2 UCRT64 Environment Setup

This guide documents how to build and test **tme-quant** natively on Windows
using [MSYS2](https://www.msys2.org/) with the UCRT64 subsystem.  It is an
alternative to the `uv`-based workflow described in [INSTALL.md](INSTALL.md)
and is useful when you need MinGW/GCC-built Python and C++ extensions (e.g.
for CurveLab / curvelops compatibility).

---

## Prerequisites

| Tool | How to get it |
|------|--------------|
| MSYS2 | <https://www.msys2.org/> — install to `C:\msys64` (default) |
| UCRT64 shell | Launch **MSYS2 UCRT64** from the Start menu |
| CurveLab 2.1.3 | <https://curvelet.org/download.php> — extract to `../utils/CurveLab-2.1.3` relative to `tme-quant` |

### Install MSYS2 packages

Open an **MSYS2 UCRT64** terminal and run:

```bash
pacman -Syu   # update package database and core

pacman -S --needed \
  mingw-w64-ucrt-x86_64-gcc \
  mingw-w64-ucrt-x86_64-gcc-fortran \
  mingw-w64-ucrt-x86_64-python \
  mingw-w64-ucrt-x86_64-python-pip \
  mingw-w64-ucrt-x86_64-python-numpy \
  mingw-w64-ucrt-x86_64-python-scipy \
  mingw-w64-ucrt-x86_64-python-matplotlib \
  mingw-w64-ucrt-x86_64-python-scikit-image \
  mingw-w64-ucrt-x86_64-python-scikit-learn \
  mingw-w64-ucrt-x86_64-python-pandas \
  mingw-w64-ucrt-x86_64-python-openpyxl \
  mingw-w64-ucrt-x86_64-python-h5py \
  mingw-w64-ucrt-x86_64-python-pytest \
  mingw-w64-ucrt-x86_64-pybind11 \
  make curl tar git
```

> **Why pacman instead of pip?**  Heavy scientific packages (scipy, numpy,
> scikit-image, etc.) require Fortran and LAPACK to compile from source.
> MSYS2 provides pre-built binaries via `pacman`, avoiding those build
> requirements entirely.

---

## Directory Layout

```
parent/
├── tme-quant/          # This repo
└── utils/
    ├── fftw-2.1.5/     # Built from source (Step 1)
    └── CurveLab-2.1.3/ # Downloaded from curvelet.org
```

---

## Step 1 — Build FFTW 2.1.5

CurveLab requires FFTW 2.x (not 3.x).  Download and build from source:

```bash
cd ../utils
curl -L http://www.fftw.org/fftw-2.1.5.tar.gz | tar xz
cd fftw-2.1.5

./configure \
  --prefix="$(pwd)" \
  --enable-type-prefix \
  --with-gcc
make -j$(nproc)
make install

# Verify
ls lib/libsfftw.a lib/libsrfftw.a lib/libdfftw.a lib/libdrfftw.a
```

## Step 2 — Build CurveLab 2.1.3

```bash
cd ../CurveLab-2.1.3

# Edit makefile.opt to set FFTW paths:
#   FFTW_DIR  = /path/to/utils/fftw-2.1.5
#   LDFLAGS   = -L$(FFTW_DIR)/lib
#   CPPFLAGS  = -I$(FFTW_DIR)/include

make lib -j$(nproc)

# Verify
ls fdct_wrapping_cpp/src/libfdct_wrapping.a
ls fdct3d/src/libfdct3d.a
ls fdct_usfft_cpp/src/libfdct_usfft.a
```

## Step 3 — Create the Python virtual environment

```bash
cd /path/to/tme-quant

# --system-site-packages lets the venv see the pacman-installed packages
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

## Step 4 — Install tme-quant (builds fiber_backend)

```bash
# Set environment for C++ compilation
export FFTW="/path/to/utils/fftw-2.1.5"
export FDCT="/path/to/utils/CurveLab-2.1.3"
export CPPFLAGS="-I$FFTW/include"
export LDFLAGS="-L$FFTW/lib"

# --no-deps avoids pip trying to build pyqt6/napari/opencv from source
# (no pre-built wheels exist for MSYS2 Python)
pip install --no-deps -e .
```

## Step 5 — Install curvelops

CurveLab headers use `M_PI`, which requires `-D_USE_MATH_DEFINES` on
UCRT/MinGW:

```bash
export CXXFLAGS="-D_USE_MATH_DEFINES"
pip install --no-deps "curvelops @ git+https://github.com/PyLops/curvelops@0.23.4"

# curvelops needs pylops at runtime
pip install pylops
```

## Step 6 — Verify

```bash
python -c "
import ctfire_py
print('HAS_FIBER_BACKEND:', ctfire_py.HAS_FIBER_BACKEND)   # True
from curvelops import fdct2d_wrapper
print('curvelops: OK')
import numpy, scipy, matplotlib, skimage, h5py
print('All imports OK')
"
```

---

## Running the tests

```bash
source .venv/bin/activate
export TMEQ_RUN_CURVELETS=1
python -m pytest tests/test_ct_fire.py -v
```

> **Important:** Use `python -m pytest` (not bare `pytest`) to ensure the
> venv Python runs the tests.  The system `pytest` from pacman may resolve
> to a different Python and miss venv-only packages like `curvelops`.

Expected result:

```
56 passed, 5 warnings in ~39s
```

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| `scipy` pip install fails — no Fortran compiler | Install via `pacman` (pre-built binary) |
| `pyqt6` / `napari` pip install hangs or fails | Use `--no-deps` for `pip install -e .`; install PyQt5 via `pacman` if GUI is needed |
| `M_PI` undeclared when building curvelops | Set `CXXFLAGS="-D_USE_MATH_DEFINES"` before pip install |
| `cm.get_cmap()` removed in matplotlib ≥ 3.11 | Use `matplotlib.colormaps["hsv"]` instead |
| `pytest` uses system Python instead of venv | Run tests via `python -m pytest` |

---

## Tested Environment

| Component | Version |
|-----------|---------|
| MSYS2 | Rolling (June 2026) |
| GCC (UCRT64) | 16.1.0 |
| Python | 3.14.6 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.11.0 |
| scikit-image | 0.26.0 |
| curvelops | 0.23.4 |
| pylops | 2.7.0 |
