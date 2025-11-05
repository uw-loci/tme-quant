# 🧬 Automated Segmentation Feature

## Overview

The CurveAlign Napari plugin now includes automated segmentation capabilities, matching the functionality of MATLAB CurveAlign's TumorTrace and cell analysis modules. This allows you to automatically generate ROIs from cell/tumor segmentation instead of manually drawing them.

## 🎯 Purpose

**Problem**: Manually drawing ROIs around dozens or hundreds of cells is time-consuming and subjective.

**Solution**: Use deep learning models (Cellpose, StarDist) or threshold-based segmentation to automatically identify cells/tumors and create ROIs.

**Workflow**: Image → Segmentation → Auto-generate ROIs → Analyze fiber alignment around each cell

---

## 🚀 Quick Start

### 1. Install Segmentation Dependencies

```bash
cd /Users/hydrablaster/Desktop/Eliceiri_lab/tme-quant
source .venv/bin/activate

# Install segmentation tools
pip install -e ".[segmentation]"

# This installs:
# - cellpose (cytoplasm and nuclei segmentation)
# - stardist (nuclei segmentation)
# - tensorflow (deep learning backend)
```

### 2. Basic Usage in Napari

```python
import napari
from napari_curvealign.widget import CurveAlignWidget

# Load your image with cells/nuclei
viewer = napari.Viewer()
viewer.open('path/to/cell_image.tif')

# Add CurveAlign widget
widget = CurveAlignWidget(viewer)
viewer.window.add_dock_widget(widget, name='CurveAlign')

# Go to "Segmentation" tab:
# 1. Select method (e.g., "Cellpose (Nuclei)")
# 2. Adjust parameters (diameter, thresholds)
# 3. Click "Run Segmentation"
# 4. Review segmentation mask
# 5. Click "Create ROIs from Mask"
# 6. Switch to "ROI Manager" tab to see generated ROIs
# 7. Click "Analyze All ROIs" to analyze fibers around each cell

napari.run()
```

---

## 📚 Segmentation Methods

### 1. ✅ Threshold-based Segmentation

**Best for**: Simple binary segmentation, bright objects on dark background  
**Speed**: Very fast  
**Installation**: No extra dependencies (uses scikit-image)

**Parameters**:
- **Threshold Method**: Otsu, Triangle, Isodata, Mean, Minimum
- **Min Area**: Minimum object size in pixels (removes noise)
- **Max Area**: Maximum object size (optional, 0 = no limit)
- **Remove Border Objects**: Exclude objects touching image edges

**Example Use Case**: Segment bright tumor regions in SHG collagen images

```python
from napari_curvealign.segmentation import (
    SegmentationMethod, SegmentationOptions, segment_image
)

options = SegmentationOptions(
    method=SegmentationMethod.THRESHOLD,
    threshold_method="otsu",
    min_area=500,
    remove_border_objects=True
)

labels = segment_image(image, options)
print(f"Found {labels.max()} objects")
```

---

### 2. 🔬 Cellpose (Cytoplasm)

**Best for**: Segmenting whole cells with visible cytoplasm  
**Speed**: Moderate (GPU recommended)  
**Accuracy**: State-of-the-art for cell segmentation

**Parameters**:
- **Cell Diameter**: Typical cell size in pixels (0 = auto-detect)
- **Flow Threshold**: Confidence threshold for cell boundaries (0-3, lower = more lenient)
- **Cellprob Threshold**: Probability threshold (-6 to 6, higher = more stringent)

**Example**: Segment cancer cells in H&E images

---

### 3. 🧬 Cellpose (Nuclei)

**Best for**: Segmenting cell nuclei (DAPI, Hoechst staining)  
**Speed**: Moderate (GPU recommended)  
**Accuracy**: Excellent for round nuclei

**Parameters**: Same as Cellpose (Cytoplasm)

**Example**: Segment nuclei in immunofluorescence images

---

### 4. ⭐ StarDist (Nuclei)

**Best for**: Fast nuclei segmentation  
**Speed**: Fast (even without GPU)  
**Accuracy**: Very good for star-convex objects

**Parameters**:
- **Probability Threshold**: Confidence for object detection (0-1)
- **NMS Threshold**: Non-maximum suppression (0-1, removes overlapping detections)

**Example**: High-throughput nuclei counting

---

## 🔬 Biological Applications

### Application 1: Tumor Microenvironment Analysis

**Goal**: Analyze collagen fiber alignment around individual tumor cells

```python
# 1. Load H&E image (tumor cells visible)
viewer.open('tumor_he.tif')

# 2. Segment tumor cells
# - Go to Segmentation tab
# - Select "Cellpose (Cytoplasm)"
# - Set diameter = 100 (for large tumor cells)
# - Run Segmentation

# 3. Create ROIs from segmentation
# - Click "Create ROIs from Mask"
# - Each cell becomes an ROI

# 4. Load corresponding SHG collagen image
viewer.open('tumor_shg.tif')

# 5. Analyze fiber alignment around each cell
# - Go to ROI Manager tab
# - Click "Analyze All ROIs"

# Result: Per-cell fiber alignment measurements
```

### Application 2: TACS (Tumor-Associated Collagen Signatures)

**Goal**: Quantify TACS-3 (aligned perpendicular fibers) around tumors

```python
from napari_curvealign.segmentation import create_tumor_boundary_rois

# 1. Segment tumor boundary
# 2. Create inner and outer ROIs
boundary_rois = create_tumor_boundary_rois(
    labeled_mask,
    inner_distance=10,  # pixels from tumor edge
    outer_distance=50   # outer boundary distance
)

# 3. Analyze fiber alignment in each region
# - Compare inner vs. outer fiber orientation
# - Quantify radial vs. tangential alignment
```

### Application 3: High-Throughput Cell Analysis

**Goal**: Analyze hundreds of cells automatically

```python
# 1. Load multi-cell image
# 2. Run StarDist segmentation (fast!)
# 3. Generate 100+ ROIs automatically
# 4. Batch analyze all ROIs
# 5. Export results to CSV for statistical analysis
```

---

## 🎨 Workflow Diagrams

### Basic Workflow
```
┌─────────────┐
│  Load Image │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Segmentation   │  (Cellpose/StarDist/Threshold)
│  Tab            │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Run            │  → Creates labeled mask
│  Segmentation   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Create ROIs    │  → Converts masks to polygons
│  from Mask      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  ROI Manager    │  → List of generated ROIs
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Analyze All    │  → Fiber analysis per ROI
│  ROIs           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Export Results │  → CSV with per-cell measurements
└─────────────────┘
```

### Advanced: Multi-Channel Analysis
```
Channel 1 (DAPI)         Channel 2 (SHG Collagen)
     │                            │
     ▼                            │
Segment Nuclei                    │
(StarDist)                        │
     │                            │
     ▼                            │
Create ROIs  ◄───────────────┐    │
     │                       │    │
     │                       │    │
     ▼                       │    ▼
Expand ROIs                  │  Analyze Fibers
(10-50 pixels)               │  Around Cells
     │                       │    │
     └───────────────────────┘    │
                                  ▼
                           Per-Cell Fiber
                           Statistics
```

---

## 📊 Parameter Guidelines

### Cellpose Diameter Selection

| Cell Type | Typical Diameter (pixels) | Notes |
|-----------|---------------------------|-------|
| Nuclei (20x) | 20-40 | Small, round objects |
| Cells (10x) | 50-100 | Whole cells with cytoplasm |
| Large Cells (5x) | 100-200 | Tumor cells, muscle fibers |
| Auto-detect | 0 | Let Cellpose estimate |

### Threshold Method Selection

| Method | Best For | Characteristics |
|--------|----------|-----------------|
| **Otsu** | Bimodal histograms | Most common, works for two populations |
| **Triangle** | Skewed histograms | Good when objects are much brighter |
| **Isodata** | Multiple modes | Iterative, handles complex histograms |
| **Mean** | Simple separation | Fast, less robust |

### Post-Processing Options

- **Fill Holes**: Recommended for cells (removes internal gaps)
- **Smooth Contours**: Reduces jagged edges (better for visualization)
- **Remove Border Objects**: Avoids incomplete cells at edges
- **Min Area Filter**: Removes debris and noise (typically 100-500 pixels)

---

## 🐛 Troubleshooting

### Problem: "cellpose is not installed"

**Solution**:
```bash
pip install -e ".[segmentation]"
# OR
pip install cellpose stardist tensorflow
```

### Problem: Segmentation is too slow

**Solutions**:
1. Use StarDist instead of Cellpose (faster)
2. Install GPU support for TensorFlow/PyTorch
3. Reduce image size (downsample if appropriate)
4. Use threshold-based segmentation for simple cases

### Problem: Too many/too few objects detected

**Cellpose**:
- Too many → Increase `cellprob_threshold`
- Too few → Decrease `flow_threshold` or `cellprob_threshold`
- Wrong size → Adjust `diameter`

**StarDist**:
- Too many → Increase `prob_thresh` or `nms_thresh`
- Too few → Decrease `prob_thresh`

**Threshold**:
- Try different threshold methods (Otsu, Triangle, Isodata)
- Adjust `min_area` and `max_area`

### Problem: ROIs don't match cells well

**Solutions**:
1. Adjust segmentation parameters (see above)
2. Use post-processing (fill holes, smooth contours)
3. Increase `simplify_tolerance` for smoother polygons
4. Manually refine ROIs in ROI Manager

---

## 🧪 Testing the Feature

### Test 1: Threshold Segmentation

```python
# Run in Python or Napari console
import napari
from napari_curvealign.widget import CurveAlignWidget
from napari_curvealign.segmentation import (
    SegmentationMethod, SegmentationOptions, segment_image, masks_to_roi_data
)
import numpy as np

# Create test image with circular objects
image = np.zeros((512, 512))
from skimage.draw import disk
for _ in range(10):
    rr, cc = disk((np.random.randint(50, 462), np.random.randint(50, 462)), 30)
    image[rr, cc] = 255

# Test segmentation
options = SegmentationOptions(
    method=SegmentationMethod.THRESHOLD,
    min_area=500
)
labels = segment_image(image, options)
print(f"✅ Found {labels.max()} objects (expected ~10)")

# Test ROI conversion
rois = masks_to_roi_data(labels, min_area=500)
print(f"✅ Created {len(rois)} ROIs")

# Visualize
viewer = napari.Viewer()
viewer.add_image(image, name='Test Image')
viewer.add_labels(labels, name='Segmentation')
napari.run()
```

### Test 2: Integration with Widget

```python
import napari
from napari_curvealign.widget import CurveAlignWidget
import numpy as np
from skimage import data

# Use sample image
image = data.human_mitosis()

# Launch widget
viewer = napari.Viewer()
viewer.add_image(image, name='Cells')

widget = CurveAlignWidget(viewer)
viewer.window.add_dock_widget(widget, name='CurveAlign')

# Instructions printed to console:
print("\n" + "="*60)
print("📋 Manual Testing Instructions:")
print("="*60)
print("\n1. Go to 'Segmentation' tab")
print("2. Keep 'Threshold-based' selected")
print("3. Click 'Run Segmentation'")
print("4. You should see a colored label overlay")
print("5. Click 'Create ROIs from Mask'")
print("6. Go to 'ROI Manager' tab")
print("7. You should see ROIs in the list")
print("\n✅ If you see ROIs, segmentation is working!")
print("="*60 + "\n")

napari.run()
```

---

## 📖 API Reference

### segment_image()

```python
def segment_image(
    image: np.ndarray,
    options: Optional[SegmentationOptions] = None
) -> np.ndarray:
    """
    Segment an image to create a labeled mask.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 2D RGB)
    options : SegmentationOptions, optional
        Segmentation parameters
        
    Returns
    -------
    np.ndarray
        Labeled mask where each object has a unique integer label
    """
```

### masks_to_roi_data()

```python
def masks_to_roi_data(
    labeled_mask: np.ndarray,
    min_area: int = 100,
    simplify_tolerance: float = 1.0
) -> List[Dict]:
    """
    Convert a labeled segmentation mask to ROI data.
    
    Each labeled region becomes a polygon ROI.
    
    Parameters
    ----------
    labeled_mask : np.ndarray
        Labeled image where each object has a unique integer label
    min_area : int, default 100
        Minimum area for objects to be converted to ROIs
    simplify_tolerance : float, default 1.0
        Tolerance for polygon simplification
        
    Returns
    -------
    List[Dict]
        List of ROI dictionaries
    """
```

---

## 🔗 References

- **Cellpose Paper**: Stringer et al. (2021), Nature Methods
- **StarDist Paper**: Schmidt et al. (2018), MICCAI
- **MATLAB TumorTrace**: Original CurveAlign documentation
- **TACS**: Provenzano et al. (2006), BMC Medicine

---

## ✅ Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Threshold Segmentation | ✅ Complete | Works out of the box |
| Cellpose Integration | ✅ Complete | Requires `pip install cellpose` |
| StarDist Integration | ✅ Complete | Requires `pip install stardist` |
| Mask → ROI Conversion | ✅ Complete | Automatic polygon extraction |
| GUI Integration | ✅ Complete | 4th tab in widget |
| Multi-format ROI Export | ✅ Complete | JSON, Fiji, CSV, TIFF |
| Batch ROI Analysis | ✅ Complete | Via ROI Manager |

**🎉 Ready for production use with cell/tumor images!**

---

**Next**: Load your real collagen + cell images and start analyzing fiber-cell interactions!

