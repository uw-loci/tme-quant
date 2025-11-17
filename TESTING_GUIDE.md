# CurveAlign Napari Plugin Testing Guide

## Installation

### 1. Install with napari dependencies

```bash
cd /Users/hydrablaster/Desktop/Eliceiri_lab/tme-quant

# Install in development mode with all dependencies
pip install -e ".[napari]"

# OR install with all optional dependencies
pip install -e ".[all]"
```

### 2. Verify installation

```bash
# Check if napari is installed
python -c "import napari; print(f'Napari version: {napari.__version__}')"

# Check if plugin is registered
python -c "from napari_curvealign import napari_experimental_provide_dock_widget; print('Plugin registered!')"

# Check optional dependencies
python -c "import roifile; print('roifile: OK')" 2>/dev/null || echo "roifile: Not installed (optional)"
python -c "import aicsimageio; print('aicsimageio: OK')" 2>/dev/null || echo "aicsimageio: Not installed (optional)"
```

---

## Quick Start

### Method 1: Launch from Command Line

```bash
# Start napari with the plugin
napari
```

Then in Napari:
1. Go to `Plugins` → `napari-curvealign` → `CurveAlign Widget`
2. The plugin widget will appear on the right side

### Method 2: Launch from Python Script

Create a test script `test_plugin.py`:

```python
import napari
import numpy as np
from skimage import data

# Create napari viewer
viewer = napari.Viewer()

# Load sample image (or use your own)
# Option 1: Use built-in sample
fiber_image = data.camera()  # Grayscale test image

# Option 2: Load your own image
# from skimage.io import imread
# fiber_image = imread('/path/to/your/collagen_image.tif')

# Add image to viewer
viewer.add_image(fiber_image, name='Fiber Image')

# Add the CurveAlign widget
viewer.window.add_plugin_dock_widget('napari-curvealign', 'CurveAlign Widget')

# Run napari
napari.run()
```

Run it:
```bash
python test_plugin.py
```

---

## Feature Testing Checklist

### ✅ 1. Basic Image Loading

**Test:** Load image through the plugin interface

```python
# test_basic_loading.py
import napari
import numpy as np

viewer = napari.Viewer()

# Add CurveAlign widget
widget = viewer.window.add_plugin_dock_widget('napari-curvealign', 'CurveAlign Widget')

# Create synthetic fiber image
image = np.random.rand(512, 512).astype(np.float32)
viewer.add_image(image, name='Test Image')

napari.run()
```

**Expected:**
- Widget appears on the right
- "Main", "Preprocessing", and "ROI Manager" tabs visible
- Image loads in viewer

---

### ✅ 2. Preprocessing Panel

**Test:** Apply filters to an image

**Steps:**
1. Load an image (use the sample script above)
2. Go to "Preprocessing" tab
3. Try each option:
   - ☐ Check "Apply Gaussian Smoothing" (adjust sigma: 1.0-5.0)
   - ☐ Check "Apply Tubeness Filter" (sigma: 1.0-3.0)
   - ☐ Check "Apply Frangi Filter" (sigma range: 1-10)
   - ☐ Check "Apply Thresholding" (try different methods: Otsu, Triangle, etc.)

4. Click "Run Analysis" in Main tab

**Expected:**
- Preprocessing is applied before analysis
- Processed image updates in viewer
- No crashes

**Test Script:**
```python
# test_preprocessing.py
import napari
import numpy as np
from skimage import data

viewer = napari.Viewer()
widget = viewer.window.add_plugin_dock_widget('napari-curvealign', 'CurveAlign Widget')

# Load sample image
image = data.camera()
viewer.add_image(image, name='Camera')

# Widget should be visible - manually test preprocessing options
print("Test preprocessing options in the GUI")
napari.run()
```

---

### ✅ 3. ROI Manager

**Test:** Create, save, and load ROIs

**Steps:**

#### 3a. Create ROIs
1. Go to "ROI Manager" tab
2. Click "Create Rectangle"
3. Draw a rectangle on the image
4. Click "Create Polygon"
5. Draw a polygon (click multiple points, press Enter to finish)
6. Check the ROI list updates

#### 3b. Save ROIs (Test all formats)
1. Select ROI(s) in the list
2. Click "Save ROI"
3. Test each format:
   - ☐ Save as JSON (`.json`) - **recommended for full data**
   - ☐ Save as Fiji ROI (`.zip` for multiple, `.roi` for single)
   - ☐ Save as CSV (`.csv`)
   - ☐ Save as TIFF mask (`.tif`)

#### 3c. Load ROIs
1. Clear ROIs (delete them)
2. Click "Load ROI"
3. Load each format you saved:
   - ☐ Load JSON
   - ☐ Load Fiji ROI/ZIP
   - ☐ Load CSV
   - ☐ Load TIFF mask

**Expected:**
- ROIs appear in viewer
- ROI list updates correctly
- All formats save/load without errors
- Round-trip preserves ROI data

**Test Script:**
```python
# test_roi_manager.py
import napari
import numpy as np

viewer = napari.Viewer()
widget = viewer.window.add_plugin_dock_widget('napari-curvealign', 'CurveAlign Widget')

# Add image
image = np.random.rand(512, 512)
viewer.add_image(image, name='Test')

# Access ROI manager (from widget)
# Note: You'll need to interact with GUI for full testing
print("Test ROI Manager in GUI:")
print("1. Go to ROI Manager tab")
print("2. Create rectangles and polygons")
print("3. Save in different formats")
print("4. Load them back")

napari.run()
```

---

### ✅ 4. CurveAlign Analysis

**Test:** Run curvelet analysis on image/ROI

**Steps:**
1. Load a fiber image (real collagen image recommended)
2. Optional: Create an ROI to analyze specific region
3. Go to "Main" tab
4. Select analysis mode:
   - ☐ "Curvelets" (faster, orientation analysis)
   - ☐ "CT-FIRE" (slower, individual fiber extraction)
5. Adjust parameters if needed:
   - Keep Boundary: on/off
   - Number of scales
   - Threshold
6. Click "Run Analysis"

**Expected:**
- Progress indicator appears
- Analysis completes without errors
- Results display (angle histogram, statistics)
- Output saved to specified directory

**Test Script:**
```python
# test_analysis.py
import napari
from skimage import data
import numpy as np

viewer = napari.Viewer()
widget = viewer.window.add_plugin_dock_widget('napari-curvealign', 'CurveAlign Widget')

# Create synthetic fiber-like image
x = np.linspace(0, 4*np.pi, 512)
y = np.linspace(0, 4*np.pi, 512)
X, Y = np.meshgrid(x, y)
fibers = np.sin(X) * np.cos(Y) + np.random.rand(512, 512) * 0.1
viewer.add_image(fibers, name='Synthetic Fibers')

print("Steps to test analysis:")
print("1. Select 'Curvelets' mode")
print("2. Click 'Run Analysis'")
print("3. Check for angle histogram output")
print("4. Try 'CT-FIRE' mode")

napari.run()
```

---

### ✅ 5. ROI Analysis

**Test:** Analyze specific ROIs

**Steps:**
1. Load fiber image
2. Create one or more ROIs
3. Go to "ROI Manager" tab
4. Select an ROI from the list
5. Click "Analyze Selected ROI"
6. OR click "Analyze All ROIs"
7. Click "Show ROI Table" to see results

**Expected:**
- Analysis runs on ROI region only
- Results table shows per-ROI statistics
- No crashes with different ROI shapes

---

### ✅ 6. Visualization

**Test:** Result visualization features

**Steps:**
1. Run analysis on an image
2. Check viewer for output layers:
   - ☐ Angle/orientation heatmap (HSV colormap)
   - ☐ Fiber overlays (if CT-FIRE mode)
   - ☐ Statistics display

**Expected:**
- Heatmap uses HSV colormap (colors represent angles)
- Overlays are visible and aligned
- Can toggle layer visibility

---

### ✅ 7. Batch Processing

**Test:** Batch analyze multiple images

**Steps:**
1. Go to "Main" tab
2. Click "Batch Process" (if implemented)
3. Select directory with multiple images
4. Run batch analysis

**Expected:**
- All images processed
- Results saved per image
- Progress indicator works

---

## Testing with Real Data

### Recommended Test Images

1. **Collagen SHG images** (if available)
2. **Fibronectin immunofluorescence**
3. **Sample data from MATLAB CurveAlign** (for validation)

### Download Test Data

```python
# download_test_data.py
from skimage import data
from skimage.io import imsave

# Save some test images
imsave('test_camera.tif', data.camera())
imsave('test_cells.tif', data.cell())

print("Test images saved!")
```

---

## Fiji Integration Testing

### Test Fiji Bridge

**Prerequisites:**
```bash
pip install napari-imagej pyimagej
```

**Steps:**
1. Start Fiji/ImageJ
2. Launch napari with plugin
3. Try preprocessing with Fiji bridge:
   - Check "Use Fiji/ImageJ bridge" in Preprocessing tab
   - Apply Tubeness/Frangi via Fiji

**Expected:**
- Fiji integration works (if napari-imagej configured)
- Falls back gracefully if Fiji not available

---

## Common Issues & Solutions

### Issue 1: Plugin not appearing in Plugins menu
**Solution:**
```bash
# Reinstall plugin
pip install -e ".[napari]" --force-reinstall

# Check napari plugin directory
python -c "from napari.settings import get_settings; print(get_settings().plugins.extension2reader)"
```

### Issue 2: Import errors for optional dependencies
**Solution:**
```bash
# Install missing dependencies
pip install roifile aicsimageio napari-imagej
```

### Issue 3: Analysis fails
**Check:**
- Image is 2D grayscale (not RGB)
- Image has reasonable size (not too small/large)
- Output directory is writable

### Issue 4: ROI save/load fails
**Check:**
- File permissions
- Valid file extensions
- `roifile` installed for Fiji format

---

## Automated Testing

### Create a test suite

```python
# test_suite.py
import napari
import numpy as np
from napari_curvealign.roi_manager import ROIManager, ROIShape
from napari_curvealign.preprocessing import preprocess_image, PreprocessingOptions

def test_roi_manager():
    """Test ROI Manager basic functionality."""
    print("Testing ROI Manager...")
    
    roi_manager = ROIManager()
    roi_manager.set_image_shape((512, 512))
    
    # Test add rectangle
    coords = np.array([[10, 10], [50, 50]])
    roi = roi_manager.add_roi(coords, ROIShape.RECTANGLE, "test_rect")
    assert roi.name == "test_rect"
    assert len(roi_manager.rois) == 1
    
    # Test save/load JSON
    roi_manager.save_rois("test_rois.json", format='json')
    roi_manager.rois = []
    roi_manager.load_rois("test_rois.json", format='json')
    assert len(roi_manager.rois) == 1
    
    print("✅ ROI Manager tests passed!")

def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    image = np.random.rand(256, 256).astype(np.float32)
    options = PreprocessingOptions(
        apply_gaussian=True,
        gaussian_sigma=1.0
    )
    
    processed = preprocess_image(image, options)
    assert processed.shape == image.shape
    assert processed.dtype == np.float32
    
    print("✅ Preprocessing tests passed!")

def test_roi_roundtrip():
    """Test ROI format round-trip."""
    print("Testing ROI round-trip...")
    
    roi_manager = ROIManager()
    roi_manager.set_image_shape((512, 512))
    
    # Create various ROI types
    rect = np.array([[10, 10], [50, 50]], dtype=float)
    poly = np.array([[100, 100], [120, 100], [120, 120], [100, 120]], dtype=float)
    
    roi_manager.add_roi(rect, ROIShape.RECTANGLE, "rect1")
    roi_manager.add_roi(poly, ROIShape.POLYGON, "poly1")
    
    # Test each format
    formats = ['json', 'csv']
    for fmt in formats:
        ext = {'json': '.json', 'csv': '.csv'}[fmt]
        filename = f"test_rois{ext}"
        
        roi_manager.save_rois(filename, format=fmt)
        original_count = len(roi_manager.rois)
        roi_manager.rois = []
        roi_manager.load_rois(filename, format=fmt)
        
        assert len(roi_manager.rois) == original_count, f"Format {fmt} failed round-trip"
        print(f"  ✅ {fmt.upper()} round-trip OK")
    
    print("✅ All ROI round-trip tests passed!")

if __name__ == "__main__":
    test_roi_manager()
    test_preprocessing()
    test_roi_roundtrip()
    print("\n🎉 All tests passed!")
```

Run tests:
```bash
python test_suite.py
```

---

## Performance Testing

### Test with different image sizes

```python
# test_performance.py
import time
import numpy as np
from napari_curvealign.preprocessing import preprocess_image, PreprocessingOptions

sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
options = PreprocessingOptions(apply_gaussian=True)

for size in sizes:
    image = np.random.rand(*size).astype(np.float32)
    
    start = time.time()
    processed = preprocess_image(image, options)
    elapsed = time.time() - start
    
    print(f"Size {size}: {elapsed:.3f} seconds")
```

---

## Next Steps

1. **Basic functionality:** ✅ Test all features in GUI
2. **ROI workflows:** ✅ Test create, save, load, analyze
3. **Format compatibility:** ✅ Test Fiji ROI import/export
4. **Real data:** Test with actual collagen/fiber images
5. **Integration:** Test Napari-Fiji bridge (if using napari-imagej)

---

## Reporting Issues

If you find bugs, note:
- What you were doing
- Expected vs actual behavior
- Error messages (check terminal output)
- napari version: `python -c "import napari; print(napari.__version__)"`
- Plugin version: Check `pyproject.toml`

---

## Quick Reference

**Launch Plugin:**
```bash
napari
# Then: Plugins → napari-curvealign → CurveAlign Widget
```

**Test ROI Manager:**
```python
from napari_curvealign.roi_manager import ROIManager, ROIShape
import numpy as np

rm = ROIManager()
rm.set_image_shape((512, 512))
roi = rm.add_roi(np.array([[10,10], [50,50]]), ROIShape.RECTANGLE, "test")
rm.save_rois("test.json", format='json')
```

**Test Preprocessing:**
```python
from napari_curvealign.preprocessing import preprocess_image, PreprocessingOptions
import numpy as np

img = np.random.rand(256, 256).astype(np.float32)
opts = PreprocessingOptions(apply_gaussian=True, gaussian_sigma=2.0)
result = preprocess_image(img, opts)
```

---

Happy Testing! 🧪🔬

