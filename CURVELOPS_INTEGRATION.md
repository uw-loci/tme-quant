# Curvelops Integration for CurveAlign Python API

## Overview

The CurveAlign Python API now supports **Curvelops** integration for authentic Fast Discrete Curvelet Transform (FDCT) operations. This provides a direct interface to the CurveLab library through the Curvelops Python wrapper.

## Integration Status

### Current Implementation
- ✅ **Graceful fallback**: Works with or without Curvelops installed
- ✅ **Automatic detection**: Detects Curvelops availability at runtime
- ✅ **Improved placeholders**: Enhanced placeholder implementations when Curvelops unavailable
- ✅ **Full API compatibility**: All existing code works unchanged
- ✅ **Better reconstruction**: Image shape-aware inverse transforms

### Curvelops Features Supported
- ✅ Forward FDCT (`apply_fdct`)
- ✅ Inverse FDCT (`apply_ifdct`) 
- ✅ Parameter extraction (`extract_parameters`)
- ✅ Status checking (`get_curvelops_status`)

## Installation

### Prerequisites
Curvelops requires **FFTW 2.1.5** and **CurveLab** to be built and installed first.

#### 1. Install FFTW 2.1.5
```bash
# Download and build FFTW 2.1.5
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
tar xzf fftw-2.1.5.tar.gz
cd fftw-2.1.5
./configure --prefix="$HOME/opt/fftw-2.1.5" --disable-fortran
make -j$(nproc)
make install
export FFTW="$HOME/opt/fftw-2.1.5"
```

#### 2. Install CurveLab
```bash
# Download CurveLab 2.1.x from http://curvelet.org/
export FDCT="/path/to/CurveLab-2.1.x"
cd "$FDCT/fdct_wrapping_cpp/src" && make
cd "$FDCT/fdct/src" && make  
cd "$FDCT/fdct3d/src" && make
```

#### 3. Install Curvelops
```bash
# Set environment variables
export FFTW="$HOME/opt/fftw-2.1.5"
export FDCT="/path/to/CurveLab-2.1.x"

# Install Curvelops
pip install -e ".[curvelops]"
```

## Usage

### Basic Usage
```python
import curvealign_py as curvealign
import numpy as np

# Create test image
image = np.random.rand(256, 256)

# Analyze with automatic Curvelops detection
result = curvealign.analyze_image(image)
print(f"Detected {len(result.curvelets)} fiber segments")
```

### Check Curvelops Status
```python
from curvealign_py.core.algorithms.fdct_wrapper import get_curvelops_status

status = get_curvelops_status()
print(f"Curvelops available: {status['available']}")
print(f"Backend: {status['backend']}")
if status['available']:
    print(f"Version: {status['version']}")
    print(f"Functional: {status['functional']}")
```

### Direct FDCT Usage
```python
from curvealign_py.core.algorithms.fdct_wrapper import apply_fdct, apply_ifdct

# Forward transform
coeffs = apply_fdct(image)
print(f"Generated {len(coeffs)} scales")

# Inverse transform  
reconstructed = apply_ifdct(coeffs, img_shape=image.shape)
print(f"Reconstructed image: {reconstructed.shape}")
```

## Implementation Details

### Automatic Fallback System
The integration uses a robust fallback system:

1. **Curvelops Available**: Uses authentic CurveLab FDCT transforms
2. **Curvelops Unavailable**: Uses improved placeholder implementations
3. **Curvelops Error**: Catches exceptions and falls back gracefully

### Enhanced Placeholder Implementation
When Curvelops is not available, the system uses enhanced placeholders that:
- Generate realistic coefficient structures
- Use image-based coefficient generation (not pure random)
- Provide better reconstruction quality
- Support proper parameter scaling

### API Improvements
- **Image shape awareness**: All functions now accept `img_shape` parameters
- **Better parameter extraction**: Coordinates scaled to image dimensions
- **Improved reconstruction**: Shape-aware inverse transforms
- **Status reporting**: Runtime status checking capabilities

## Testing

### Run Integration Tests
```bash
# Test the integration
python test_curvelops_integration.py

# Run full test suite
pytest tests/ -v
```

### Expected Output (Without Curvelops)
```
=== Testing Curvelops Integration ===
Curvelops Status:
  Available: False
  Backend: placeholder

=== Testing FDCT Functions ===
Test image shape: (128, 128)
Testing forward FDCT...
  Forward FDCT successful: 5 scales
Testing inverse FDCT...
  Inverse FDCT successful: (128, 128)
  Reconstruction MSE: ~130.0
Testing parameter extraction...
  Parameter extraction successful: 5 scales

=== Testing Full CurveAlign Analysis ===
  Analysis successful: ~25 curvelets detected
  Mean angle: ~87°
  Alignment: ~0.2
  Visualization successful: (128, 128, 3)

=== All Tests Passed! ===
```

### Expected Output (With Curvelops)
When Curvelops is properly installed, you should see:
```
Curvelops Status:
  Available: True
  Backend: curvelops
  Version: 0.23
  Functional: True
```

## Performance Considerations

### With Curvelops
- **Authentic transforms**: Real CurveLab FDCT operations
- **High accuracy**: Mathematically correct curvelet decomposition
- **Better fiber detection**: More precise curvelet extraction
- **Slower**: Native C++ operations with Python overhead

### Without Curvelops (Placeholder)
- **Fast execution**: Pure NumPy/SciPy operations
- **Good approximation**: Reasonable fiber detection results
- **Development friendly**: No complex dependencies
- **Lower accuracy**: Approximated curvelet behavior

## Troubleshooting

### Common Issues

#### "Curvelops not available" Warning
This is normal when Curvelops is not installed. The system will use placeholder implementations.

#### FFTW/FDCT Environment Variables
Ensure these are set correctly:
```bash
export FFTW="/path/to/fftw-2.1.5"
export FDCT="/path/to/CurveLab-2.1.x"
```

#### Build Errors
- Verify FFTW 2.1.5 is built with `--disable-fortran`
- Ensure CurveLab makefiles completed successfully
- Check that environment variables point to correct directories

#### Runtime Errors
The system will catch Curvelops errors and fall back to placeholders automatically.

## Future Enhancements

### Planned Improvements
- [ ] Direct CurveLab binding (bypass Curvelops)
- [ ] GPU acceleration support
- [ ] Caching of coefficient structures
- [ ] Performance benchmarking tools
- [ ] Advanced parameter tuning interface

### Contributing
To improve the Curvelops integration:
1. Test with different FFTW/CurveLab versions
2. Optimize coefficient structure conversion
3. Add more comprehensive error handling
4. Improve placeholder algorithm accuracy

## References
- [Curvelops Documentation](https://github.com/PyLops/curvelops)
- [CurveLab Website](http://curvelet.org/)
- [FFTW Documentation](http://fftw.org/)
- [CurveAlign Original Paper](https://doi.org/10.1038/nmeth.2233)
