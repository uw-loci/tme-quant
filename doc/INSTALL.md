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

The script resolves paths from its own location, so it can be launched from any working directory. It creates `utils/` inside this repository, downloads FFTW there, detects and builds CurveLab there, syncs the environment, and verifies the installation.

The native Windows path is not verified end to end: building the `curvelops` Python extension requires MSVC (`cl.exe`), while the FFTW and CurveLab build steps use GNU tools. Their libraries must also be compatible with MSVC. WSL2 is the supported Windows route; build all dependencies inside WSL rather than reusing Windows-built libraries.

## Run

```bash
uv run napari
```

**Plugins → napari-curvealign** (or **CurveAlign for Napari**)

## Directory layout

```
tme-quant/
└── utils/
   ├── fftw-2.1.5/     # Created by install.sh
   └── CurveLab-2.1.3/ # You must download and extract here
```

## Without curvelops

To run napari without the curvelet backend (no FFTW/CurveLab required):

```bash
uv sync --extra segmentation
uv run napari
```

**Note:** The napari plugin will load, but curvelet-based analysis (e.g. fiber orientation) will not run. For real curvelet analysis, use the full install above.
