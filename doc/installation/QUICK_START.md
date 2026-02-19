# Quick Start Guide

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

3. **Download CurveLab** (required for curvelet backend; cannot be redistributed)
   - Visit: https://curvelet.org/download.php
   - Agree to the license and download CurveLab
   - Extract to `../utils/` relative to the tme-quant directory
   - Example: if tme-quant is at `/home/user/projects/tme-quant`, place CurveLab at `/home/user/projects/utils/CurveLab-2.1.3`

## Installation (with curvelops)

```bash
bash bin/install.sh
```

The script will:
- Check for uv (exits if not found)
- Download FFTW 2.1.5 to `../utils` if not present
- Build FFTW and CurveLab (installing build tools if needed on macOS/Linux)
- Detect CurveLab in `../utils` (will exit with instructions if not found)
- Create and sync the uv environment with curvelops
- Install tme-quant in editable mode
- Run verification

## Usage

```bash
uv run napari
```

The CurveAlign widget appears in: **Plugins → napari-curvealign → CurveAlign Widget**

## Directory Layout

For the install script to work automatically, use this layout:

```
parent/
├── tme-quant/          # This repo (clone here)
└── utils/
    ├── fftw-2.1.5/     # Created by install.sh (downloaded and built)
    └── CurveLab-2.1.3/ # You must download and extract CurveLab here
```

## Without curvelets

To run napari without the curvelet backend (no FFTW/CurveLab required):

```bash
uv sync
uv run napari
```

## Manual installation

If you prefer to install without curvelops or need more control, see [SETUP_GUIDE.md](SETUP_GUIDE.md) for manual steps.
