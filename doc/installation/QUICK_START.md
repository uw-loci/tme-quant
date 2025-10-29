# Quick Start Guide

## Installation (30 seconds to start)

```bash
cd tme-quant
make setup
```

This will:
- ✓ Install Conda (if needed)
- ✓ Create Python environment
- ✓ Build FFTW (optional, 10-15 min)
- ✓ Help you configure CurveLab
- ✓ Install tme-quant

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

## Files in bin/ Directory

All setup scripts are organized in `bin/`:
- `bin/setup.sh` - Main setup script
- `bin/setup_curvelops_env.sh` - Curvelops environment helper
- `bin/activate_env.sh` - Environment activation

## Makefile Commands

```bash
make setup      # Automated installation
make install    # Install package
make install-all # Install with all features
make test       # Run tests
make docker-build # Build Docker image
make docker-run   # Run Docker container
```

For more details, see:
- [INSTALLATION_OVERVIEW.md](INSTALLATION_OVERVIEW.md) - Overview of build steps
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed installation guide
- [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md) - Summary of changes
- [CURVELOPS_INTEGRATION.md](CURVELOPS_INTEGRATION.md) - CurveLab integration details
