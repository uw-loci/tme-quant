# Quick Start Guide

## Napari Plugin Installation (Recommended)

For the **CurveAlign Napari Plugin** with all GUI features:

### Prerequisites
- Python 3.9+ installed
- FFTW 2.1.5 in `../utils/fftw-2.1.5`
- CurveLab 2.1.2 in `../utils/CurveLab-2.1.2`

### Installation

```bash
cd tme-quant
bash bin/install.sh
```

This will:
- ✓ Create virtual environment
- ✓ Install tme-quant (adds curvelab extra if FFTW/CurveLab are found)

Then activate and launch:
```bash
source bin/activate.sh
napari
```

The CurveAlign widget will appear in: **Plugins → napari-curvealign → CurveAlign Widget**

For detailed instructions, see [NAPARI_PLUGIN_INSTALLATION.md](NAPARI_PLUGIN_INSTALLATION.md).

---

## Base Package Installation (API Only)

For the **Python API only** (without Napari):

```bash
cd tme-quant
bash bin/setup.sh
```

This will:
- ✓ Create or reuse the `tme-quant` conda environment
- ✓ Install tme-quant (plus dev tools by default)
- ✓ Add curvelab extra if FFTW/CurveLab are found

Lean install (skip dev tools):
```bash
cd tme-quant
INSTALL_DEV=0 bash bin/setup.sh
```

Then activate:
```bash
source bin/activate_env.sh
```

## Docker Alternative

If you prefer Docker:
```bash
make docker-build
make docker-run
```

**Note:** Docker Desktop is available for Mac (Intel and Apple Silicon).

## Usage Example

```python
import curvealign_py as curvealign
import numpy as np

# Analyze an image
image = np.random.rand(256, 256)
result = curvealign.analyze_image(image)

print(f"Found {len(result.curvelets)} fiber segments")
print(f"Mean angle: {result.stats['mean_angle']:.1f}°")
```

## Makefile Commands

```bash
make setup      # Automated setup
make install    # Install package
make install-all # Install with all features
make test       # Run tests
make docker-build # Build Docker image
make docker-run   # Run Docker container
```

For more details, see:
- [NAPARI_PLUGIN_INSTALLATION.md](NAPARI_PLUGIN_INSTALLATION.md) - **Complete Napari plugin installation guide**
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed base package installation guide
- [CURVELOPS_INTEGRATION.md](CURVELOPS_INTEGRATION.md) - CurveLab integration details
