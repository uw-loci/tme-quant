# CurveAlign Python API - Setup Notes

## Installation in tme-quant Repository

The CurveAlign Python API has been moved to the tme-quant repository. To use it:

### 1. Create New Virtual Environment
```bash
cd curvealign_py/
python3 -m venv curvealign_env
source curvealign_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install numpy scipy scikit-image tifffile matplotlib
pip install pytest pytest-cov  # for testing
```

### 3. Install the Package
```bash
pip install -e .
```

### 4. Test Installation
```bash
# Run tests
pytest tests/ -v

# Run simple usage example
cd ..
python3 simple_usage.py
```

## Usage in tme-quant Context

```python
# In your Python scripts within tme-quant
import sys
from pathlib import Path

# Add curvealign to path
sys.path.insert(0, str(Path(__file__).parent / "curvealign_py"))
import curvealign

# Use the API
result = curvealign.analyze_image(image)

# Or use granular imports for specific components
from curvealign.types.core import Curvelet, Boundary
from curvealign.types.config import CurveAlignOptions
from curvealign.core.processors import extract_curvelets, compute_features
from curvealign.visualization.backends import matplotlib_create_overlay
```

## Integration with Existing tme-quant Code

The CurveAlign API can work alongside the existing `src/pycurvelets` and `src/napari_curvealign` modules:

- **`curvealign_py/`**: Complete, standalone API with organized architecture
- **`src/pycurvelets`**: Original implementation (can coexist)
- **`src/napari_curvealign`**: napari plugin (can leverage the new API)

## Architecture Benefits in tme-quant

- **Unified Interface**: Single API for all CurveAlign functionality
- **Framework Integration**: Ready for napari plugin enhancement
- **Clean Dependencies**: Minimal core requirements
- **Extensible**: Easy to integrate with other tme-quant modules

See `ARCHITECTURE.md` for complete architectural documentation.
