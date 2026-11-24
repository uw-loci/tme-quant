# Installation

## Prerequisites

1. **Clone this repository**
   ```bash
   git clone https://github.com/uw-loci/tme-quant.git
   cd tme-quant
   ```

2. **Install uv** (https://docs.astral.sh/uv/)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **CurveLab** (optional; cannot be redistributed)
   - Needed only for the FDCT backend (`pycurvelets-fdct`)
   - https://curvelet.org/download.php
   - Extract it into `utils/` inside this repository (e.g. `tme-quant/utils/CurveLab-2.1.3`)
   - Without it, the install script still installs `pycurvelets-core` and `pycurvelets-udct`

## Install

```bash
bash bin/install.sh
```

The script resolves paths from its own location, so it can be launched from any working directory. It creates `.venv` at the repository root and installs `pycurvelets-core` (with the `segmentation` and `fiji` extras) and `pycurvelets-udct`.

If `utils/` contains one CurveLab source tree (`CurveLab-<version>/` with `fdct_wrapping_cpp/src/makefile` and `fdct3d/src/makefile`), the script also downloads and builds FFTW there, builds CurveLab, installs `pycurvelets-fdct` (which builds curvelops), and verifies that both the `fdct` and `udct` backends load. If no CurveLab directory is present, it skips FFTW, CurveLab, and `pycurvelets-fdct`, and verifies `udct` only. An incomplete CurveLab directory, or more than one CurveLab tree, stops the script.

The native Windows path is not verified end to end: building the `curvelops` Python extension requires MSVC (`cl.exe`), while the FFTW and CurveLab build steps use GNU tools. Their libraries must also be compatible with MSVC. WSL2 is the supported Windows route; build all dependencies inside WSL rather than reusing Windows-built libraries.

## Run

```bash
uv run python examples/simple_usage.py
```

For an interactive front end, install the Fiji plugin
([uw-loci/fiji-curvealign](https://github.com/uw-loci/fiji-curvealign)) or the
napari plugin ([uw-loci/napari-curvealign](https://github.com/uw-loci/napari-curvealign)).

## Directory layout

```
tme-quant/
├── .venv/              # Created by install.sh; shared by all projects
├── pycurvelets-core/   # pyproject.toml, src/pycurvelets, tests/
├── pycurvelets-fdct/   # pyproject.toml, src/pycurvelets_fdct, tests/
├── pycurvelets-udct/   # pyproject.toml, src/pycurvelets_udct, tests/
└── utils/
   ├── fftw-2.1.5/     # Created by install.sh when building FDCT
   └── CurveLab-2.1.3/ # Optional; required only for the FDCT backend
```

## Without CurveLab

`bash bin/install.sh` with no CurveLab tree installs core and UDCT and does not build FFTW. To install only the core library (no curvelet backend), from the repository root:

```bash
uv venv --python 3.11
uv pip install -e "./pycurvelets-core[segmentation]"
```

From another project, depend on a subdirectory of this repository:

```bash
pip install "pycurvelets-core @ git+https://github.com/uw-loci/tme-quant.git#subdirectory=pycurvelets-core"
```

**Note:** With only the core library, registration, annotation, segmentation and preprocessing work, but curvelet-based analysis (e.g. fiber orientation) will not run. `bash bin/install.sh` installs UDCT for that analysis without CurveLab. Extract CurveLab under `utils/` before running the script to add FDCT.
