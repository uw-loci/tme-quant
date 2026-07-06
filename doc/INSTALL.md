# Installation

## What needs what

There are two usage tiers, with different prerequisites:

| You want to use… | Requires |
|---|---|
| `fire_2d_angle` / `ctfire_py` (fiber extraction) | A **C++ compiler with OpenMP** (to build the required `fiber_backend` extension). No CurveLab/FFTW needed. |
| `ct_fire` / curvelet reconstruction | The above **plus** FFTW + CurveLab + `curvelops`. |

> **Note:** `fiber_backend` is a compiled C++ extension with **no pure-Python
> fallback**. It is built automatically during `pip install` / `uv pip install -e .`.
> If no C++ compiler or OpenMP is available the install **fails** with platform-specific
> instructions — install the toolchain below and re-run.

## Prerequisites

1. **C++ compiler with OpenMP** (required for `fiber_backend`)
   - **Windows:** install "Microsoft C++ Build Tools" (Visual Studio Build Tools →
     "Desktop development with C++"); OpenMP ships with MSVC. *Or* use MSYS2 UCRT64:
     `pacman -S mingw-w64-ucrt-x86_64-gcc` and install with the UCRT64 Python.
   - **macOS:** `xcode-select --install` then `brew install libomp`.
   - **Linux:** `sudo apt install build-essential libgomp1` (or your distro's equivalent).

2. **Clone this repository**
   ```bash
   git clone https://github.com/uw-loci/tme-quant.git
   cd tme-quant
   ```

3. **Install uv** (https://docs.astral.sh/uv/)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. **Download CurveLab** (only for `ct_fire`/curvelops; cannot be redistributed)
   - https://curvelet.org/download.php
   - Extract to `../utils/` relative to tme-quant (e.g. `../utils/CurveLab-2.1.3`)

## Install

```bash
bash bin/install.sh
```

The script checks for uv, downloads FFTW, detects CurveLab in `../utils`, builds both, syncs the env, runs `uv pip install -e .` (which also compiles `fiber_backend`), and verifies.

Verify the backend is importable:

```bash
uv run python -c "import ctfire_py; print(ctfire_py.HAS_FIBER_BACKEND, ctfire_py.HAS_CURVELOPS)"
# -> True True
```

## Run

```bash
uv run napari
```

**Plugins → napari-curvealign** (or **CurveAlign for Napari**)

## Directory layout

```
parent/
├── tme-quant/          # This repo
└── utils/
    ├── fftw-2.1.5/     # Created by install.sh
    └── CurveLab-2.1.3/ # You must download and extract here
```

## Without curvelops (fiber extraction only)

To use `fire_2d_angle` / `ctfire_py` without the curvelet backend (no FFTW/CurveLab
required, but the C++/OpenMP toolchain from Prerequisites step 1 is still required to
build `fiber_backend`):

```bash
uv pip install -e .   # compiles fiber_backend
uv run python -c "import ctfire_py; print(ctfire_py.HAS_FIBER_BACKEND)"  # -> True
```

To run napari without the curvelet backend:

```bash
uv sync
uv run napari
```

**Note:** The napari plugin will load, but curvelet-based analysis (e.g. fiber orientation) will not run. You get the UI with mock/placeholder results. For real curvelet analysis, use the full install above.
