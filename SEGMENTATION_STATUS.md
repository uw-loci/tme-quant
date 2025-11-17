# Segmentation Methods Status

## 📍 Location of Segmentation Modules

The segmentation functionality is located in:
- **Main module**: `src/napari_curvealign/segmentation.py`
- **Widget integration**: `src/napari_curvealign/widget.py` (Segmentation tab)

## ✅ Currently Available Methods

### 1. Threshold-based Segmentation
- **Status**: ✅ **Working**
- **Dependencies**: `scikit-image` (already installed)
- **Location in code**: `segmentation.py` → `_segment_threshold()`
- **Use case**: Simple binary segmentation, bright objects on dark background

### 2. Cellpose (Cytoplasm & Nuclei)
- **Status**: ✅ **Working** (just installed)
- **Dependencies**: `cellpose>=2.0`, `torch` (installed successfully)
- **Location in code**: `segmentation.py` → `_segment_cellpose()`
- **Use case**: 
  - **Cytoplasm**: Whole cell segmentation
  - **Nuclei**: Cell nuclei segmentation (DAPI, Hoechst staining)

## ❌ Not Available

### 3. StarDist (Nuclei)
- **Status**: ❌ **Failed to install**
- **Reason**: Compilation error on Python 3.13
- **Error**: C++ compilation failed (`stderr` undeclared identifier)
- **Location in code**: `segmentation.py` → `_segment_stardist()`

**Why it failed:**
StarDist requires compilation of C++ extensions, and the current version has compatibility issues with Python 3.13 on macOS. The build process failed during compilation.

**Workaround options:**

1. **Use the remote environment bridge** (recommended for Python 3.13+):
   - The code already supports this via `env_bridge.py`
   - Create a Python 3.12 environment with StarDist installed
   - Use `stardist_python_path` option to point to that environment
   - See `segmentation.py` lines 254-280 for remote execution support

2. **Use Cellpose instead**:
   - Cellpose (Nuclei) provides similar functionality to StarDist
   - Already working and installed

3. **Downgrade to Python 3.12** (if StarDist is critical):
   - StarDist is known to work with Python 3.9-3.12
   - Create a separate environment with Python 3.12

## 📦 Installation Summary

### What was installed:
```bash
✅ cellpose>=2.0
✅ torch (PyTorch backend)
✅ tensorflow>=2.6.0
✅ csbdeep
```

### What failed:
```bash
❌ stardist (compilation error on Python 3.13)
```

### What was already available:
```bash
✅ scikit-image (for threshold segmentation)
```

## 🔍 How to Check Available Methods

You can check which methods are available programmatically:

```python
from napari_curvealign.segmentation import check_available_methods

methods = check_available_methods()
print(methods)
# Output: {'threshold': True, 'cellpose': True, 'stardist': False, 'skimage': True}
```

## 🎯 Current Status in Napari Plugin

When you open the CurveAlign plugin in napari:
- **Segmentation tab** → **Method dropdown** will show:
  - ✅ "Threshold-based ✓" (available)
  - ✅ "Cellpose (Cytoplasm) ✓" (available)
  - ✅ "Cellpose (Nuclei) ✓" (available)
  - ❌ "StarDist (Nuclei) ✗" (not available)

The widget automatically checks availability and marks methods with ✓ or ✗.

## 🛠️ Next Steps

1. **For immediate use**: Use Threshold or Cellpose methods (both working)
2. **For StarDist**: 
   - Option A: Use Cellpose (Nuclei) as alternative
   - Option B: Set up remote environment bridge (see `env_bridge.py`)
   - Option C: Wait for StarDist Python 3.13 compatibility update

## 📝 Notes

- The segmentation module uses try/except imports to gracefully handle missing dependencies
- Methods are checked at runtime via `check_available_methods()`
- The widget UI automatically reflects availability with checkmarks
- All segmentation code is in `src/napari_curvealign/segmentation.py`

