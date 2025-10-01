# tme-quant

Modern Python implementation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) for collagen fiber analysis, featuring a comprehensive API, CT-FIRE integration, and napari plugin support.

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
- **Pluggable Visualization**: Support for matplotlib, napari, and ImageJ backends
- **Type Safety**: Comprehensive type definitions and validation
- **Framework Ready**: Designed for scientific Python workflows

## Repository Structure
- `src/curvealign_py/`: Modern CurveAlign Python API
- `src/ctfire_py/`: CT-FIRE Python API for individual fiber extraction
- `src/napari_curvealign/`: Interactive napari plugin
- `tests/`: Comprehensive test suite with API and integration tests
- `src/curvealign_matlab/`: Original MATLAB reference implementation

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Conda (recommended) or Python 3.10–3.13
- For napari: a Qt binding (PyQt or PySide)

## Installation

### Quick Start (Recommended)
Install the unified package with all Python APIs:

```bash
conda create -y -n tme-quant -c conda-forge python=3.11 pip
conda activate tme-quant

# Install the package (includes both CurveAlign and CT-FIRE APIs)
pip install -e .

# Optional: Install napari for interactive analysis
conda install -y -c conda-forge napari pyqt qtpy
```

### Verify Installation
```python
import curvealign_py as curvealign
import ctfire_py as ctfire

# Test basic functionality
result = curvealign.analyze_image(your_image)
print(f"Analysis complete: {len(result.curvelets)} features detected")
```

### Optional: curvelet backend (curvelops)
To enable curvelet-powered features and tests you must build and install FFTW 2.1.5 and CurveLab, then install `curvelops`.

macOS/Linux outline:
```bash
# 1) Build FFTW 2.1.5 (C only)
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
 tar xzf fftw-2.1.5.tar.gz
 cd fftw-2.1.5
 # If configure fails on macOS due to outdated config.{sub,guess}, update them from your system
 ./configure --prefix="$HOME/opt/fftw-2.1.5" --disable-fortran
 make -j$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
 make install
 export FFTW="$HOME/opt/fftw-2.1.5"

# 2) Build CurveLab 2.1.x
export FDCT="/path/to/CurveLab-2.1.x"
 cd "$FDCT/fdct_wrapping_cpp/src" && make
 cd "$FDCT/fdct/src" && make
 cd "$FDCT/fdct3d/src" && make

# 3) Install build tooling and curvelops
conda activate napari-env
 python -m pip install -U pip
 pip install pybind11 scikit-build-core cmake ninja
 export FFTW="$HOME/opt/fftw-2.1.5"
 export FDCT="/path/to/CurveLab-2.1.x"
 pip install -v "curvelops @ git+https://github.com/PyLops/curvelops@0.23"
```

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows: use MSYS2 (for `gcc`, `make`) or Visual Studio toolchain; build FFTW 2.1.5 and CurveLab from source, set `FFTW` and `FDCT` env vars to their install roots, then install `curvelops` as above. Supervisors can validate these steps on a Windows host.

## Testing

The repository includes comprehensive tests for all API components:

```bash
# Run all tests (CurveAlign, CT-FIRE, and integration)
pytest -v

# Run specific test suites
pytest tests/curvealign_py/ -v  # CurveAlign API tests
pytest tests/ctfire_py/ -v     # CT-FIRE API tests
pytest tests/test_unified_api.py -v  # Integration tests
```

### Headless Testing
For CI or headless environments:
```bash
export QT_QPA_PLATFORM=offscreen  # Linux/macOS
pytest -q
```

### Curvelet Backend Tests
After installing `curvelops` (see `CURVELOPS_INTEGRATION.md`):
```bash
# Setup Curvelops environment
source setup_curvelops_env.sh

# Run tests with real FDCT
export TMEQ_RUN_CURVELETS=1
pytest -q

# Test Curvelops integration specifically
python test_curvelops_final.py
```

## Advanced Usage

### Batch Processing
```python
import curvealign_py as curvealign

# Process multiple images
images = [load_image(path) for path in image_paths]
results = curvealign.batch_analyze(images, mode="curvelets")

# CT-FIRE batch analysis
results_ctfire = curvealign.batch_analyze(images, mode="ctfire")
```

### Custom Analysis Options
```python
# Configure analysis parameters
options = curvealign.CurveAlignOptions(
    keep=0.002,              # Coefficient threshold
    dist_thresh=150.0,       # Boundary distance
    minimum_nearest_fibers=6 # Feature computation
)

result = curvealign.analyze_image(image, options=options)
```

### Visualization Backends
```python
# Matplotlib backend (default)
overlay = curvealign.overlay(image, result.curvelets, backend="matplotlib")

# napari integration
curvealign.overlay(image, result.curvelets, backend="napari")

# ImageJ integration  
curvealign.overlay(image, result.curvelets, backend="imagej")
```

## Troubleshooting

### Common Issues
- **Qt error**: Install Qt bindings: `conda install -c conda-forge pyqt`
- **Import errors**: Ensure package is installed: `pip install -e .`
- **curvelops build errors**: Verify `FFTW` and `FDCT` environment variables point to install roots

### Documentation
- API documentation: See individual module docstrings
- Architecture details: Comprehensive type system and modular design
- Examples: Check `simple_usage.py` for common usage patterns

## Contributing
This repository follows modern Python packaging standards with comprehensive testing and CI integration. All APIs are designed for extensibility and scientific workflow integration.
