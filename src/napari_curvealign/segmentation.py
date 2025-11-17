"""
Automated segmentation module for ROI generation.

This module provides various segmentation methods to automatically generate ROIs
from images, matching the functionality of MATLAB CurveAlign's TumorTrace and
cell analysis modules.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Literal
from enum import Enum
from dataclasses import dataclass

try:
    from skimage import measure, morphology, filters
    from skimage.segmentation import clear_border
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cellpose
    from cellpose import models
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False

try:
    from stardist.models import StarDist2D
    HAS_STARDIST = True
except ImportError:
    HAS_STARDIST = False


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    THRESHOLD = "Threshold-based"
    CELLPOSE_CYTO = "Cellpose (Cytoplasm)"
    CELLPOSE_NUCLEI = "Cellpose (Nuclei)"
    STARDIST = "StarDist (Nuclei)"
    CUSTOM_MASK = "Custom Mask"


@dataclass
class SegmentationOptions:
    """Options for segmentation."""
    method: SegmentationMethod = SegmentationMethod.THRESHOLD
    
    # Threshold-based options
    threshold_method: str = "otsu"  # otsu, triangle, isodata, etc.
    min_area: int = 100  # Minimum object area in pixels
    max_area: Optional[int] = None  # Maximum object area (None = no limit)
    remove_border_objects: bool = True
    
    # Cellpose options
    cellpose_model_type: str = "cyto"  # cyto, nuclei, cyto2
    cellpose_diameter: Optional[float] = 30.0  # Cell diameter in pixels (None = auto)
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    
    # StarDist options
    stardist_model: str = "2D_versatile_fluo"  # Model name
    stardist_prob_thresh: float = 0.5
    stardist_nms_thresh: float = 0.4
    stardist_python_path: Optional[str] = None  # Path to Python with StarDist (for remote execution)
    
    # Post-processing
    fill_holes: bool = True
    smooth_contours: bool = True


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
        
    Examples
    --------
    >>> from skimage import data
    >>> image = data.coins()
    >>> options = SegmentationOptions(method=SegmentationMethod.THRESHOLD)
    >>> labels = segment_image(image, options)
    >>> print(f"Found {labels.max()} objects")
    """
    if options is None:
        options = SegmentationOptions()
    
    method = options.method
    
    if method == SegmentationMethod.THRESHOLD:
        return _segment_threshold(image, options)
    elif method in (SegmentationMethod.CELLPOSE_CYTO, SegmentationMethod.CELLPOSE_NUCLEI):
        return _segment_cellpose(image, options)
    elif method == SegmentationMethod.STARDIST:
        return _segment_stardist(image, options)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def _segment_threshold(image: np.ndarray, options: SegmentationOptions) -> np.ndarray:
    """
    Threshold-based segmentation (like MATLAB's TumorTrace histogram method).
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for threshold segmentation")
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        image_gray = 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]
    else:
        image_gray = image
    
    # Normalize to 0-1 range
    image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min() + 1e-10)
    
    # Apply threshold
    threshold_method = options.threshold_method.lower()
    if threshold_method == "otsu":
        thresh_value = filters.threshold_otsu(image_gray)
    elif threshold_method == "triangle":
        thresh_value = filters.threshold_triangle(image_gray)
    elif threshold_method == "isodata":
        thresh_value = filters.threshold_isodata(image_gray)
    elif threshold_method == "mean":
        thresh_value = filters.threshold_mean(image_gray)
    elif threshold_method == "minimum":
        thresh_value = filters.threshold_minimum(image_gray)
    else:
        # Default to Otsu
        thresh_value = filters.threshold_otsu(image_gray)
    
    # Create binary mask
    binary = image_gray > thresh_value
    
    # Remove small objects
    if options.min_area > 0:
        binary = morphology.remove_small_objects(binary, min_size=options.min_area)
    
    # Fill holes
    if options.fill_holes:
        binary = morphology.remove_small_holes(binary, area_threshold=options.min_area)
    
    # Remove border objects
    if options.remove_border_objects:
        binary = clear_border(binary)
    
    # Label connected components
    labeled = measure.label(binary)
    
    # Filter by max area if specified
    if options.max_area is not None:
        props = measure.regionprops(labeled)
        for prop in props:
            if prop.area > options.max_area:
                labeled[labeled == prop.label] = 0
        # Re-label to remove gaps
        labeled = measure.label(labeled > 0)
    
    return labeled


def _segment_cellpose(image: np.ndarray, options: SegmentationOptions) -> np.ndarray:
    """
    Cellpose-based segmentation for cells/nuclei.
    
    Updated for Cellpose 4.x API which uses CellposeModel instead of Cellpose.
    """
    if not HAS_CELLPOSE:
        raise ImportError(
            "Cellpose is not installed. Install with: pip install cellpose\n"
            "For GPU support, also install: pip install torch"
        )
    
    try:
        from cellpose.models import CellposeModel
    except ImportError:
        raise ImportError(
            "Could not import CellposeModel. Make sure you have Cellpose 4.x installed.\n"
            "Try: pip install --upgrade cellpose"
        )
    
    # Determine model type
    if options.method == SegmentationMethod.CELLPOSE_NUCLEI:
        model_type = "nuclei"
    else:
        model_type = options.cellpose_model_type
    
    # Create model (Cellpose 4.x API)
    # CellposeModel(model_type='cyto' or 'nuclei' or custom path)
    model = CellposeModel(model_type=model_type, device=None)  # device=None auto-selects CPU/GPU
    
    # Prepare channels
    # Cellpose 4.x: [0,0] for grayscale
    if image.ndim == 2:
        channels = [0, 0]  # Grayscale
    else:
        # For RGB: [2,3] if nuclei in blue, [0,0] for grayscale conversion
        channels = [0, 0]
    
    # Normalize image to 0-255 range if needed (Cellpose 4.x expects this)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    elif image.max() > 255:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Run segmentation (Cellpose 4.x API)
    # Returns: masks, flows, styles
    masks, flows, styles = model.eval(
        image,
        diameter=options.cellpose_diameter,
        channels=channels,
        flow_threshold=options.cellpose_flow_threshold,
        cellprob_threshold=options.cellpose_cellprob_threshold,
        normalize=True  # Cellpose 4.x parameter
    )
    
    # Post-process
    if options.min_area > 0 and HAS_SKIMAGE:
        binary = masks > 0
        binary = morphology.remove_small_objects(binary, min_size=options.min_area)
        masks = measure.label(binary)
    
    if options.max_area is not None and HAS_SKIMAGE:
        props = measure.regionprops(masks)
        for prop in props:
            if prop.area > options.max_area:
                masks[masks == prop.label] = 0
        masks = measure.label(masks > 0)
    
    return masks


def _segment_stardist(image: np.ndarray, options: SegmentationOptions) -> np.ndarray:
    """
    StarDist-based segmentation for nuclei.
    
    Note: StarDist requires Python 3.9-3.12. If you're using Python 3.13+,
    use the remote environment bridge (set stardist_python_path in options).
    """
    # Check if using remote environment
    if hasattr(options, 'stardist_python_path') and options.stardist_python_path:
        # Use environment bridge for cross-version support
        from .env_bridge import segment_stardist_remote
        
        labels = segment_stardist_remote(
            image,
            python_path=options.stardist_python_path,
            model_name=options.stardist_model,
            prob_thresh=options.stardist_prob_thresh,
            nms_thresh=options.stardist_nms_thresh
        )
        
        # Post-process
        if options.min_area > 0 and HAS_SKIMAGE:
            binary = labels > 0
            binary = morphology.remove_small_objects(binary, min_size=options.min_area)
            labels = measure.label(binary)
        
        if options.max_area is not None and HAS_SKIMAGE:
            props = measure.regionprops(labels)
            for prop in props:
                if prop.area > options.max_area:
                    labels[labels == prop.label] = 0
            labels = measure.label(labels > 0)
        
        return labels
    
    # Standard in-process StarDist (requires Python 3.9-3.12)
    if not HAS_STARDIST:
        raise ImportError(
            "StarDist is not installed or incompatible with current Python version.\n\n"
            "Option 1: Install StarDist (Python 3.9-3.12 only):\n"
            "  pip install stardist\n\n"
            "Option 2: Use remote environment (Python 3.13+):\n"
            "  1. Create Python 3.12 environment with StarDist\n"
            "  2. Set stardist_python_path in options\n"
            "  See: napari_curvealign.env_bridge.create_stardist_environment_guide()"
        )
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        image_gray = 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]
    else:
        image_gray = image
    
    # Normalize to 0-1 range (StarDist expects this)
    image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min() + 1e-10)
    
    # Load model
    model = StarDist2D.from_pretrained(options.stardist_model)
    
    # Run prediction
    labels, details = model.predict_instances(
        image_gray,
        prob_thresh=options.stardist_prob_thresh,
        nms_thresh=options.stardist_nms_thresh
    )
    
    # Post-process
    if options.min_area > 0 and HAS_SKIMAGE:
        binary = labels > 0
        binary = morphology.remove_small_objects(binary, min_size=options.min_area)
        labels = measure.label(binary)
    
    if options.max_area is not None and HAS_SKIMAGE:
        props = measure.regionprops(labels)
        for prop in props:
            if prop.area > options.max_area:
                labels[labels == prop.label] = 0
        labels = measure.label(labels > 0)
    
    return labels


def masks_to_roi_data(
    labeled_mask: np.ndarray,
    min_area: int = 100,
    simplify_tolerance: float = 1.0
) -> List[Dict]:
    """
    Convert a labeled segmentation mask to ROI data.
    
    Each labeled region becomes a polygon ROI with coordinates extracted
    from the region's contour.
    
    Parameters
    ----------
    labeled_mask : np.ndarray
        Labeled image where each object has a unique integer label
    min_area : int, default 100
        Minimum area for objects to be converted to ROIs
    simplify_tolerance : float, default 1.0
        Tolerance for polygon simplification (higher = simpler polygons)
        
    Returns
    -------
    List[Dict]
        List of ROI dictionaries with 'name', 'coordinates', and 'shape' keys
        
    Examples
    --------
    >>> labels = segment_image(image, options)
    >>> rois = masks_to_roi_data(labels, min_area=200)
    >>> print(f"Created {len(rois)} ROIs")
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for mask to ROI conversion")
    
    rois = []
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        if prop.area < min_area:
            continue
        
        # Get contour of this region
        # Create binary mask for this object
        binary_object = (labeled_mask == prop.label)
        
        # Find contours
        contours = measure.find_contours(binary_object, 0.5)
        
        if not contours:
            continue
        
        # Use the longest contour
        contour = max(contours, key=len)
        
        # Simplify contour if needed
        if simplify_tolerance > 0:
            try:
                from skimage.measure import approximate_polygon
                contour = approximate_polygon(contour, tolerance=simplify_tolerance)
            except ImportError:
                pass  # Use original contour
        
        # Convert to (row, col) format and ensure float dtype
        coordinates = np.asarray(contour, dtype=float)
        
        # Create ROI dictionary
        roi_dict = {
            'name': f'Cell_{prop.label}',
            'coordinates': coordinates,
            'shape': 'polygon',
            'area': prop.area,
            'centroid': prop.centroid,
            'bbox': prop.bbox
        }
        
        rois.append(roi_dict)
    
    return rois


def create_tumor_boundary_rois(
    labeled_mask: np.ndarray,
    inner_distance: int = 10,
    outer_distance: int = 50
) -> List[Dict]:
    """
    Create inner and outer boundary ROIs around segmented objects.
    
    This implements the TumorTrace-style ROI generation for analyzing
    fiber alignment at different distances from tumor/cell boundaries.
    
    Parameters
    ----------
    labeled_mask : np.ndarray
        Labeled segmentation mask
    inner_distance : int, default 10
        Distance in pixels for inner boundary ROI
    outer_distance : int, default 50
        Distance in pixels for outer boundary ROI
        
    Returns
    -------
    List[Dict]
        List of boundary ROI dictionaries
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required")
    
    from scipy import ndimage
    
    rois = []
    
    # Create binary mask of all objects
    binary_all = labeled_mask > 0
    
    # Create distance transform
    distance_from_edge = ndimage.distance_transform_edt(~binary_all)
    
    # Inner boundary: pixels within inner_distance of object edge
    inner_boundary = (distance_from_edge > 0) & (distance_from_edge <= inner_distance)
    
    # Outer boundary: pixels between inner_distance and outer_distance
    outer_boundary = (distance_from_edge > inner_distance) & (distance_from_edge <= outer_distance)
    
    # Convert boundaries to contours
    for name, boundary in [('Inner_Boundary', inner_boundary), ('Outer_Boundary', outer_boundary)]:
        contours = measure.find_contours(boundary, 0.5)
        if contours:
            # Use longest contour
            contour = max(contours, key=len)
            coordinates = np.asarray(contour, dtype=float)
            
            rois.append({
                'name': name,
                'coordinates': coordinates,
                'shape': 'polygon'
            })
    
    return rois


def check_available_methods() -> Dict[str, bool]:
    """
    Check which segmentation methods are available.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating availability of each method
    """
    return {
        'threshold': HAS_SKIMAGE,
        'cellpose': HAS_CELLPOSE,
        'stardist': HAS_STARDIST,
        'skimage': HAS_SKIMAGE
    }


def get_recommended_parameters(image: np.ndarray, method: SegmentationMethod) -> Dict:
    """
    Get recommended segmentation parameters based on image properties.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    method : SegmentationMethod
        Segmentation method
        
    Returns
    -------
    Dict
        Recommended parameter values
    """
    recommendations = {}
    
    # Estimate typical object size
    image_area = image.shape[0] * image.shape[1]
    
    if method == SegmentationMethod.CELLPOSE_CYTO:
        # For cytoplasm, cells are typically 50-200 pixels in diameter
        recommendations['cellpose_diameter'] = 100.0
        recommendations['min_area'] = 500
    elif method == SegmentationMethod.CELLPOSE_NUCLEI or method == SegmentationMethod.STARDIST:
        # For nuclei, objects are typically 20-50 pixels in diameter
        recommendations['cellpose_diameter'] = 30.0
        recommendations['min_area'] = 200
    elif method == SegmentationMethod.THRESHOLD:
        # For threshold, use conservative min area
        recommendations['min_area'] = max(100, image_area // 1000)
    
    return recommendations

