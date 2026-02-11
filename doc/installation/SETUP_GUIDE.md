# tme-quant Setup Guide

This guide covers installation of tme-quant with the **curvelet backend** (curvelops) and Napari plugin. For one-click installation, use [QUICK_START.md](QUICK_START.md).

## Quick Start (Recommended)

Use the automated install script:

```bash
cd tme-quant
bash bin/install.sh
```

This script will:
1. Install Miniforge (if conda/mamba not found)
2. Download and build FFTW 2.1.5 in `../utils`
3. Detect CurveLab in `../utils`
4. Create the `tme-quant` conda environment
5. Install with curvelops and run verification

**Prerequisites:** Clone this repo and download CurveLab to `../utils`. See below.

## Manual Installation

### Step 1: Install conda (or Miniforge with Mamba)

We recommend **Miniforge** (includes Mamba, faster than conda):

```bash
# macOS (Apple Silicon)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-*-MacOSX-arm64.sh
bash Miniforge3-*-MacOSX-arm64.sh

# macOS (Intel)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-*-MacOSX-x86_64.sh
bash Miniforge3-*-MacOSX-x86_64.sh

# Linux (x86_64)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-*-Linux-x86_64.sh
bash Miniforge3-*-Linux-x86_64.sh
```

Or use Miniconda from https://docs.conda.io/en/latest/miniconda.html

### Step 2: Create Python Environment

```bash
conda create -y -n tme-quant python=3.11
conda activate tme-quant

# Prefer mamba for faster installs (if available)
mamba install -y -c conda-forge napari pyqt qtpy
# or: conda install -y -c conda-forge napari pyqt qtpy
```

### Step 3: Install FFTW (for curvelet backend)

**Best practice:** Place FFTW in `../utils` (parallel to tme-quant) so it matches the install script layout:

```bash
# Create utils directory next to tme-quant
cd /path/to/parent/of/tme-quant
mkdir -p utils
cd utils

# Download FFTW
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
tar xzf fftw-2.1.5.tar.gz
cd fftw-2.1.5

# Configure and build (PIC required for shared libs used by curvelops)
./configure --prefix="$(pwd)" --disable-fortran CFLAGS="-fPIC"
make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || 4)
make install

export FFTW="$(pwd)"
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"
```

**Note:** Set `FFTW` in your shell config (e.g. `~/.bashrc`) if you build CurveLab in a later session.

### Step 4: Install CurveLab

CurveLab requires a separate download due to licensing:
1. Visit: https://curvelet.org/download.php
2. Agree to the license and download CurveLab
3. Extract to `../utils` (e.g. `../utils/CurveLab-2.1.2`)

```bash
# Set the CurveLab path (adjust to your version)
export FDCT="/path/to/utils/CurveLab-2.1.x"

# Build CurveLab components (if needed)
cd "$FDCT/fdct_wrapping_cpp/src"
make FFTW_DIR="$FFTW"

cd "$FDCT/fdct/src"
make FFTW_DIR="$FFTW"

cd "$FDCT/fdct3d/src"
make FFTW_DIR="$FFTW"
```

### Step 5: Install tme-quant

```bash
cd tme-quant

# With curvelet backend (requires FFTW and CurveLab)
pip install -e ".[curvelops]"

# Or install curvelops separately
pip install "curvelops @ git+https://github.com/PyLops/curvelops@0.23"
pip install -e .
```

### Step 6: Verify Installation

```bash
python -c "
import pycurvelets as pc
print('Installed ✅')
print('HAS_CURVELETS =', pc.HAS_CURVELETS)
print('new_curv is', 'available' if hasattr(pc, 'new_curv') else 'missing')
"
```

## Package Installation Options

### Basic (no curvelet backend)
```bash
pip install -e .
```
Core API only; no curvelops. Install curvelops manually if you need it later.

### With Curvelet Backend
```bash
pip install -e ".[curvelops]"
```
Requires FFTW and CurveLab as above.

## Troubleshooting

### Conda / Mamba Not Found
- Install Miniforge: https://github.com/conda-forge/miniforge/releases
- Or run `bash bin/install.sh` which will install Miniforge automatically

### FFTW Build Errors
- **macOS:** Ensure Xcode Command Line Tools: `xcode-select --install`
- **Linux:** `sudo apt-get install build-essential`
- Always use `CFLAGS="-fPIC"` for shared library builds

### CurveLab Not Found
- Download from https://curvelet.org/
- Place in `../utils/` (e.g. `../utils/CurveLab-2.1.2`) for compatibility with install script
- Set `FDCT` to the CurveLab directory

### Environment Already Exists
The install script fails if `tme-quant` env exists. Remove it first:
```bash
conda env remove -n tme-quant
```

## Activation

After installation:
```bash
conda activate tme-quant
napari
```

The CurveAlign widget appears in: **Plugins → napari-curvealign → CurveAlign Widget**
