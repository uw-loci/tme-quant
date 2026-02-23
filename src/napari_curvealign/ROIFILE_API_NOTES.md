# roifile 2025.x API Reference for CurveAlign

## Correct API Usage (As Implemented)

### Creating ROIs

The `roifile` 2025.x library uses a two-step process:
1. Create ROI with `frompoints()`
2. Set the `roitype` attribute

```python
import roifile as rf
import numpy as np

# Step 1: Create from points
points = np.array([[10, 10], [50, 50]])
fiji_roi = rf.ImagejRoi.frompoints(points, name="My ROI")

# Step 2: Set the type
fiji_roi.roitype = rf.ROI_TYPE.RECT     # Rectangle
fiji_roi.roitype = rf.ROI_TYPE.OVAL     # Ellipse/Oval
fiji_roi.roitype = rf.ROI_TYPE.POLYGON  # Polygon
fiji_roi.roitype = rf.ROI_TYPE.FREEHAND # Freehand
```

### ROI Types

Available in `rf.ROI_TYPE`:
- `RECT` (1) - Rectangle
- `OVAL` (2) - Ellipse/Oval
- `POLYGON` (7) - Polygon
- `FREEHAND` (8) - Freehand
- `LINE` (3) - Line
- `POLYLINE` (6) - Polyline
- `POINT` (10) - Point
- `ANGLE` (5) - Angle
- `FREELINE` (4) - Free line
- `TRACED` (9) - Traced
- `NOROI` (0) - No ROI

### Saving ROIs

```python
# Save single ROI
fiji_roi.tofile("my_roi.roi")

# Save multiple ROIs in ZIP
import tempfile
import zipfile
import shutil

temp_dir = tempfile.mkdtemp()
try:
    for i, roi in enumerate(rois):
        fiji_roi = create_fiji_roi(roi)  # Your conversion function
        fiji_roi.tofile(f"{temp_dir}/{roi.name}.roi")
    
    with zipfile.ZipFile("RoiSet.zip", 'w') as zipf:
        for filename in os.listdir(temp_dir):
            zipf.write(os.path.join(temp_dir, filename), filename)
finally:
    shutil.rmtree(temp_dir)
```

### Loading ROIs

```python
import zipfile

# Load single ROI
fiji_roi = rf.ImagejRoi.fromfile("my_roi.roi")

# Load from ZIP (CRITICAL: Use frombytes, not fromfile!)
with zipfile.ZipFile("RoiSet.zip", 'r') as zipf:
    for filename in zipf.namelist():
        if filename.endswith('.roi'):
            # Read bytes first!
            roi_bytes = zipf.read(filename)
            fiji_roi = rf.ImagejRoi.frombytes(roi_bytes)
```

### Converting to Our ROI Format

```python
def convert_fiji_to_roi(fiji_roi):
    """Convert Fiji ROI to our ROI format."""
    # Get type
    roi_type = fiji_roi.roitype
    
    # For RECT and OVAL, read bbox
    if roi_type == rf.ROI_TYPE.RECT:
        left = fiji_roi.left
        top = fiji_roi.top
        width = fiji_roi.width
        height = fiji_roi.height
        coords = np.array([[left, top], [left + width, top + height]], dtype=float)
        shape = ROIShape.RECTANGLE
        
    elif roi_type == rf.ROI_TYPE.OVAL:
        left = fiji_roi.left
        top = fiji_roi.top
        width = fiji_roi.width
        height = fiji_roi.height
        coords = np.array([[left, top], [left + width, top + height]], dtype=float)
        shape = ROIShape.ELLIPSE
        
    # For POLYGON and FREEHAND, use coordinates()
    elif roi_type == rf.ROI_TYPE.POLYGON:
        coords = np.asarray(fiji_roi.coordinates(), dtype=float)
        shape = ROIShape.POLYGON
        
    elif roi_type == rf.ROI_TYPE.FREEHAND:
        coords = np.asarray(fiji_roi.coordinates(), dtype=float)
        shape = ROIShape.FREEHAND
        
    return coords, shape
```

## Common Mistakes to Avoid

### ❌ WRONG: Using non-existent constructors
```python
# These don't exist in roifile 2025.x!
fiji_roi = rf.ImagejRoi.rect(left, top, width, height)  # ❌
fiji_roi = rf.ImagejRoi.oval(left, top, width, height)  # ❌
fiji_roi = rf.ImagejRoi.polygon(x_points, y_points)     # ❌
```

### ❌ WRONG: Passing roitype to frompoints
```python
# frompoints() doesn't accept roitype parameter
fiji_roi = rf.ImagejRoi.frompoints(points, roitype=rf.ROI_TYPE.RECT)  # ❌
```

### ❌ WRONG: Using fromfile on ZipExtFile
```python
with zipfile.ZipFile("RoiSet.zip", 'r') as zipf:
    with zipf.open("roi.roi") as f:
        fiji_roi = rf.ImagejRoi.fromfile(f)  # ❌ TypeError
```

### ✅ CORRECT: Set roitype after creation
```python
fiji_roi = rf.ImagejRoi.frompoints(points, name="ROI")
fiji_roi.roitype = rf.ROI_TYPE.RECT  # ✅
```

### ✅ CORRECT: Use frombytes for ZIP files
```python
with zipfile.ZipFile("RoiSet.zip", 'r') as zipf:
    roi_bytes = zipf.read("roi.roi")
    fiji_roi = rf.ImagejRoi.frombytes(roi_bytes)  # ✅
```

## API Signature Reference

```python
# frompoints signature (roifile 2025.x)
ImagejRoi.frompoints(
    points: ArrayLike | None = None,
    /,
    *,
    name: str | None = None,
    position: int | None = None,
    index: int | str | None = None,
    c: int | None = None,
    z: int | None = None,
    t: int | None = None
) -> ImagejRoi

# frombytes signature
ImagejRoi.frombytes(
    data: bytes,
    /,
    *,
    byteorder: Literal['>', '<'] | None = None
) -> ImagejRoi

# fromfile signature
ImagejRoi.fromfile(
    filename: str | PathLike,
    /,
    *,
    byteorder: Literal['>', '<'] | None = None
) -> ImagejRoi
```

## Attributes Available

After loading a ROI:
```python
fiji_roi = rf.ImagejRoi.fromfile("roi.roi")

# Type information
fiji_roi.roitype          # int: ROI type code
fiji_roi.name             # str: ROI name

# Bounding box (for RECT, OVAL)
fiji_roi.left             # int: left edge
fiji_roi.top              # int: top edge
fiji_roi.right            # int: right edge (computed)
fiji_roi.bottom           # int: bottom edge (computed)
fiji_roi.width            # int: width
fiji_roi.height           # int: height

# Coordinates (for POLYGON, FREEHAND)
fiji_roi.coordinates()    # Returns array of [x, y] points

# Other metadata
fiji_roi.c_position       # C position (channel)
fiji_roi.z_position       # Z position (slice)
fiji_roi.t_position       # T position (time)
fiji_roi.fill_color       # Fill color
fiji_roi.stroke_color     # Stroke color
```

## Version Information

- **roifile version tested**: 2025.5.10
- **Python version**: 3.13
- **Date**: 2025-11-05

## References

- roifile GitHub: https://github.com/cgohlke/roifile
- ImageJ ROI format spec: https://imagej.net/ij/developer/source/ij/io/RoiDecoder.java.html

