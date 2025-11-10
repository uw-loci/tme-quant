# tme-quant

Modern Python implementation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) for collagen fiber analysis, featuring a comprehensive API and CT-FIRE integration.

> **Note**: This branch (`api-curation`) focuses on the Python API. For the napari plugin, see the [`napari-curvealign`](https://github.com/uw-loci/tme-quant/tree/napari-curvealign) branch.

## Python API - Quick Start

The repository provides a complete, modern Python API for fiber analysis:

```python
import curvealign_py as curvealign

# Basic analysis
result = curvealign.analyze_image(image)
print(f"Found {len(result.curvelets)} fiber segments")
print(f"Mean angle: {result.stats['mean_angle']:.1f}°")

# Create visualizations
overlay = curvealign.overlay(image, result.curvelets)
angle_map = curvealign.angle_map(image, result.curvelets)

# CT-FIRE integration
result_ctfire = curvealign.analyze_image(image, mode="ctfire")
```

### API Features
- **Authentic FDCT**: Real CurveLab transforms via Curvelops integration
- **Unified Interface**: Both curvelet-based and CT-FIRE fiber extraction
- **Granular Architecture**: Modular design with clean separation of concerns  
- **Visualization Support**: Built-in support for matplotlib and extensible for other backends
- **Type Safety**: Comprehensive type definitions and validation
- **Framework Ready**: Designed for scientific Python workflows

## Repository Structure (high-level)
- `src/curvealign_py/`: CurveAlign Python API
- `src/ctfire_py/`: CT-FIRE Python API
- `bin/`: Automated setup scripts
- `tests/`: Comprehensive test suite with CI integration

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Conda (recommended) or Python 3.9–3.13

## Installation

### Quick Start (Native - recommended)

Use our automated setup script:

```bash
cd tme-quant
bash bin/setup.sh
# or using make
make setup
```

This script installs dependencies and guides CurveLab setup (optional).  
For details see [doc/installation/SETUP_GUIDE.md](doc/installation/SETUP_GUIDE.md).  
Docker users: see [docker/README.md](docker/README.md).

### Docker (alternative)

```bash
make docker-build
make docker-run
```

### Verify Installation (either method)
```python
import curvealign_py as curvealign
import ctfire_py as ctfire

# Test basic functionality
result = curvealign.analyze_image(your_image)
print(f"Analysis complete: {len(result.curvelets)} features detected")
```

### Curvelet backend (optional)
To enable authentic CurveLab FDCT via `curvelops`, you need FFTW 2.1.5 and CurveLab. The setup script will prompt for these; see [doc/installation/SETUP_GUIDE.md](doc/installation/SETUP_GUIDE.md) if you prefer manual steps.

## Testing (optional)
```bash
pytest -v
```

## Usage examples
See `simple_usage.py`.

## Troubleshooting

### Common Issues
- **Import errors**: Ensure package is installed: `pip install -e .`
- **curvelops build errors**: Verify `FFTW` and `FDCT` environment variables point to install roots

### Documentation
- API documentation: See individual module docstrings
- Architecture details: Comprehensive type system and modular design
- Examples: Check `simple_usage.py` for common usage patterns

## Support
Troubleshooting and details: see docs under `doc/installation/` and `doc/curvealign/`.
