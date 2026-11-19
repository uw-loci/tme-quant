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

3. **Download CurveLab** (required; cannot be redistributed)
   - https://curvelet.org/download.php
   - Extract it into `utils/` inside this repository (e.g. `tme-quant/utils/CurveLab-2.1.3`)

## Install

```bash
bash bin/install.sh
```

The script resolves paths from its own location, so it can be launched from any working directory. It creates `utils/` inside this repository, downloads FFTW there, detects and builds CurveLab there, creates `.venv` at the repository root, installs `pycurvelets-core` and `pycurvelets-fdct` into it (one `uv pip install -e` per project), and verifies that the `fdct` backend loads.

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
└── utils/
   ├── fftw-2.1.5/     # Created by install.sh
   └── CurveLab-2.1.3/ # You must download and extract here
```

## Without a curvelet backend

To install only the core library (no FFTW/CurveLab required), from the repository root:

```bash
uv venv --python 3.11
uv pip install -e "./pycurvelets-core[segmentation]"
```

From another project, depend on a subdirectory of this repository:

```bash
pip install "pycurvelets-core @ git+https://github.com/uw-loci/tme-quant.git#subdirectory=pycurvelets-core"
```

**Note:** Registration, annotation, segmentation and preprocessing work, but curvelet-based analysis (e.g. fiber orientation) will not run. For real curvelet analysis, use the full install above.
