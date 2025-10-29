# tme-quant Setup Guide

This guide shows two simple ways to install: native (recommended) and Docker.

## Quick Start (Recommended)

The easiest way to get started is using our automated setup script:

```bash
# Using the setup script
cd tme-quant
bash bin/setup.sh

# Or using make
make setup
```

This script will:
1. Check/install Conda
2. Create a Python environment
3. Optionally build FFTW 2.1.5 (10–15 min)
4. Prompt for CurveLab path (optional)
5. Install and verify tme-quant

## Manual Installation

### Step 1: Install System Dependencies

#### macOS
```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install Miniconda (if needed)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

#### Linux
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ make wget curl

# Install Miniconda (if needed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Create Python Environment

```bash
# Create conda environment
conda create -y -n tme-quant python=3.11
conda activate tme-quant

# Or use virtualenv
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install FFTW (Optional, for curvelet backend)

FFTW is only needed if you want to use the curvelet backend (curvelops).

```bash
# Download FFTW
cd ~/opt
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
tar xzf fftw-2.1.5.tar.gz
cd fftw-2.1.5

# Configure and build
./configure --prefix="$(pwd)" --disable-fortran
make -j$(nproc)  # or: make -j$(sysctl -n hw.logicalcpu)
make install

# Set environment variables
export FFTW="$(pwd)"
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"
```

### Step 4: Install CurveLab

CurveLab requires a separate download due to licensing:
1. Visit: https://curvelet.org/download.php
2. Agree to the license and download CurveLab
3. Extract it to a location on your system

```bash
# Set the CurveLab path (adjust to your location)
export FDCT="/path/to/CurveLab-2.1.x"

# Build CurveLab components (if needed)
cd "$FDCT/fdct_wrapping_cpp/src"
make

cd "$FDCT/fdct3d/src"
make
```

### Step 5: Install tme-quant

```bash
cd tme-quant

# Install without curvelet backend (simplest)
pip install -e .

# Or install with curvelet backend (requires FFTW and CurveLab)
pip install -e ".[curvelops]"

# Or install everything
pip install -e ".[all]"
```

### Step 6: Verify Installation

```bash
# Run verification
python -c "import curvealign_py as ca; import ctfire_py as cf; print('Installation successful!')"
```

## Docker Installation (Alternative)

For a consistent, isolated environment, use Docker:

```bash
# Build the Docker image
docker build -t tme-quant:latest .

# Run the container
docker run -it \
  -v /path/to/curvelab:/app/curvelab \
  -v /path/to/your/data:/app/data \
  tme-quant:latest

# Or use docker-compose
docker-compose up -d
```

**Note:** 
- Docker Desktop is available for Mac (both Intel and Apple Silicon)
- You need to install Docker Desktop from https://www.docker.com/products/docker-desktop/
- Select the correct version for your Mac (Intel or Apple Silicon/M1)
- You still need to provide CurveLab separately due to licensing restrictions
- On Mac, Docker uses a lightweight VM so performance may be slightly slower than native

Alternatively, you can use the native setup script (`make setup`) which works better on Mac without requiring Docker.

## Package Installation Options

### Basic Installation
```bash
pip install -e .
```
Includes core CurveAlign and CT-FIRE APIs without optional backends.

### With Visualization
```bash
pip install -e ".[visualization]"
```
Adds matplotlib visualization support.

### With napari Plugin
```bash
pip install -e ".[napari]"
```
Installs napari for interactive analysis.

### With Curvelet Backend
```bash
pip install -e ".[curvelops]"
```
Enables authentic CurveLab FDCT transforms (requires FFTW and CurveLab).

### Full Installation
```bash
pip install -e ".[all]"
```
Installs all optional dependencies including curvelops, napari, and visualization.

## Troubleshooting

### Conda Not Found
If conda is not found, install Miniconda:
- Download from: https://docs.conda.io/en/latest/miniconda.html
- Or use the setup script which will install it automatically

### FFTW Build Errors
- On macOS: Ensure Xcode Command Line Tools are installed
- On Linux: Install `build-essential` and `gfortran`
- Use `--disable-fortran` flag to avoid Fortran dependency

### CurveLab Not Found
- Make sure you've downloaded CurveLab from https://curvelet.org/
- Set the `FDCT` environment variable to the correct path
- Verify the directory contains `fdct_wrapping_cpp` and `fdct3d` folders

### Import Errors
```bash
# Make sure you're in the correct environment
conda activate tme-quant

# Reinstall the package
pip install -e . --force-reinstall --no-cache-dir
```

## Getting Help

- Documentation: See `doc/` directory
- Issues: Open an issue on GitHub
- Examples: Check `simple_usage.py`

## Activation Script

After installation, use the provided activation script:

```bash
source bin/activate_env.sh
```

Or manually activate:
```bash
conda activate tme-quant
```

