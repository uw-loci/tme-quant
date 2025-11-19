# CurveAlign Feature Comparison: MATLAB vs Napari Plugin

This document compares the functionality of the original MATLAB CurveAlign application with the current Napari plugin implementation.

## Overview

The Napari CurveAlign plugin aims to provide feature parity with the MATLAB CurveAlign 6.0 application while leveraging modern Python tools and the Napari visualization framework.

## Feature Status

### ✅ Fully Implemented

1. **Basic CurveAlign Analysis**
   - Curvelet transform extraction (FDCT)
   - Coefficient thresholding
   - Angle and position extraction
   - Overlay visualization
   - Angle map visualization

2. **Preprocessing**
   - Bio-Formats import (via aicsimageio/napari-imagej)
   - Tubeness filter
   - Frangi filter
   - Auto-thresholding (Otsu, Triangle, Isodata, Mean, Minimum)

3. **Segmentation**
   - Threshold-based segmentation
   - Cellpose (Cytoplasm and Nuclei)
   - StarDist (Nuclei)
   - Mask to ROI conversion

4. **ROI Management**
   - ROI creation (Rectangle, Ellipse, Polygon, Freehand)
   - ROI save/load (JSON, Fiji ROI, CSV, TIFF mask)
   - ROI analysis integration

5. **CT-FIRE Integration**
   - CT-FIRE mode selection in widget
   - Individual fiber extraction
   - Fiber metrics (length, width, angle, curvature)

6. **Visualization**
   - Overlay images
   - Angle maps (heatmaps)
   - Histogram generation
   - Results table display

7. **Circular Statistics**
   - Circular mean
   - Circular variance
   - Circular standard deviation
   - Mean resultant length (alignment metric)

### ⚠️ Partially Implemented

1. **Feature Extraction**
   - **Current**: Basic density and alignment features (density_nn, alignment_nn, density_box, alignment_box)
   - **MATLAB**: ~30 features including:
     - Multiple nearest neighbor distances (2, 4, 8, 16)
     - Multiple box sizes (32, 64, 128)
     - Mean and std of distances/alignments
     - Individual fiber features (when using CT-FIRE): total length, end-to-end length, curvature, width
   - **Status**: Core features implemented, but not all MATLAB features are exposed

2. **Boundary Analysis**
   - **Current**: Basic boundary distance calculation
   - **MATLAB**: Full boundary analysis including:
     - Distance to boundary
     - Relative angles to boundary
     - Extension points
     - Boundary association statistics
   - **Status**: Infrastructure exists in curvealign_py API, but widget integration is incomplete

3. **ROI Analysis**
   - **Current**: Basic ROI analysis
   - **MATLAB**: ROI-based density calculation, ROI-specific statistics
   - **Status**: ROI manager exists, but full ROI analysis pipeline needs enhancement

### ❌ Not Yet Implemented

1. **Batch Processing**
   - MATLAB: `LOCIca_cluster.m`, `batch_curveAlign.m`, parallel processing
   - Status: `batch_analyze()` exists in API but not exposed in widget

2. **Cell Analysis Module**
   - MATLAB: Deep learning-based cell segmentation and fiber-cell interaction analysis
   - Status: Segmentation exists, but cell-fiber interaction analysis not implemented

3. **Advanced Visualization**
   - MATLAB: Multiple visualization modes, custom colormaps, interactive plots
   - Status: Basic visualization implemented

4. **Statistics Export**
   - MATLAB: CSV, XLSX, MAT file export with comprehensive statistics
   - Status: Basic DataFrame display, but export functionality limited

5. **Image Registration**
   - MATLAB: Manual image registration for multi-channel alignment
   - Status: Not implemented

## Architecture Comparison

### MATLAB Structure
```
CurveAlign.m (GUI)
  ├── processImage.m (main pipeline)
  │   ├── getCT.m / getFIRE.m (feature extraction)
  │   ├── newCurv.m (curvelet transform)
  │   ├── getBoundary.m (boundary analysis)
  │   └── makeStatsO.m (statistics)
  ├── CAroi.m (ROI analysis)
  ├── Cellanalysis/ (cell analysis)
  └── ctFIRE/ (CT-FIRE module)
```

### Napari Plugin Structure
```
widget.py (UI/Controller)
  ├── new_curv.py (analysis orchestration)
  │   └── curvealign_py API (core analysis)
  ├── preprocessing.py (image preprocessing)
  ├── segmentation.py (automated ROI generation)
  ├── roi_manager.py (ROI lifecycle)
  └── fiji_bridge.py (Fiji/ImageJ integration)
```

## Key Differences

1. **API Design**: Napari plugin uses a modern Python API (`curvealign_py`) that separates concerns better than the MATLAB monolithic structure.

2. **Visualization**: Napari plugin leverages Napari's native visualization capabilities, while MATLAB uses custom plotting functions.

3. **Extensibility**: Python plugin is more easily extensible with modern Python tools (scikit-image, pandas, etc.).

4. **Dependencies**: MATLAB requires MATLAB Runtime, while Python plugin uses open-source libraries.

## Recommendations for Full Feature Parity

1. **Enhance Feature Extraction**: Extend `feature_processor.py` to compute all ~30 MATLAB features.

2. **Complete Boundary Analysis**: Integrate full boundary analysis from `curvealign_py` API into widget.

3. **Add Batch Processing UI**: Create a batch processing tab in the widget for analyzing multiple images.

4. **Implement Cell Analysis**: Add cell-fiber interaction analysis module.

5. **Enhance Export**: Add comprehensive export functionality (CSV, XLSX, JSON).

6. **Add Image Registration**: Implement manual image registration for multi-channel images.

## Current Status Summary

**Overall Feature Parity: ~75%**

- Core analysis: ✅ 100%
- Preprocessing: ✅ 100%
- Segmentation: ✅ 100%
- ROI Management: ⚠️ 80%
- Feature Extraction: ⚠️ 60%
- Boundary Analysis: ⚠️ 50%
- Visualization: ✅ 90%
- Batch Processing: ❌ 0%
- Cell Analysis: ❌ 0%

The plugin successfully implements the core CurveAlign functionality and provides a modern, extensible interface. The main gaps are in advanced features like batch processing and cell analysis, which can be added incrementally.

