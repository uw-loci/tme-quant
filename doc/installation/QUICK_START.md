# Quick Start Guide

## Prerequisites

1. **Clone this repository**
   ```bash
   git clone https://github.com/uw-loci/tme-quant.git
   cd tme-quant
   ```

2. **Download CurveLab** (required; cannot be redistributed)
   - Visit: https://curvelet.org/download.php
   - Agree to the license and download CurveLab
   - Extract to `../utils/` relative to the tme-quant directory
   - Example: if tme-quant is at `/home/user/projects/tme-quant`, place CurveLab at `/home/user/projects/utils/CurveLab-2.1.2`

## Installation

```bash
bash bin/install.sh
```

The script will:
- Install Miniforge if conda/mamba are not available
- Fail immediately if the `tme-quant` conda environment already exists (remove it first: `conda env remove -n tme-quant`)
- Download FFTW 2.1.5 to `../utils` if not present
- Build FFTW and CurveLab (installing build tools if needed)
- Detect CurveLab in `../utils` (will exit with instructions if not found)
- Create the `tme-quant` conda environment
- Install tme-quant with curvelops
- Run verification

## Usage

```bash
conda activate tme-quant
napari
```

The CurveAlign widget appears in: **Plugins → napari-curvealign → CurveAlign Widget**

## Directory Layout

For the install script to work automatically, use this layout:

```
parent/
├── tme-quant/          # This repo (clone here)
└── utils/
    ├── fftw-2.1.5/     # Created by install.sh (downloaded and built)
    └── CurveLab-2.1.2/ # You must download and extract CurveLab here
```

## Manual Installation

If you prefer to install without curvelops or need more control, see [SETUP_GUIDE.md](SETUP_GUIDE.md) for manual steps.
