"""
CurveAlign Python API - High-level interface for collagen fiber analysis.

This module provides the main user-facing API for CurveAlign functionality,
including image analysis, ROI processing, and batch operations.
"""

from typing import Optional, Sequence, Iterable, Tuple, List, Union, Literal
from pathlib import Path
import numpy as np

from .types import (
    Curvelet, CtCoeffs, Boundary, AnalysisResult, ROIResult, 
    CurveAlignOptions, FeatureOptions, FeatureTable, BoundaryMetrics
)
from .core.processors import (
    extract_curvelets, reconstruct_image, 
    compute_features, measure_boundary_alignment
)


def analyze_image(
    image: np.ndarray,
    boundary: Optional[Boundary] = None,
    mode: Literal["curvelets", "ctfire"] = "curvelets",
    options: Optional[CurveAlignOptions] = None,
) -> AnalysisResult:
    """
    Analyze a single image for collagen fiber organization.
    
    This is the main high-level function for single image analysis, supporting
    both curvelet-based and CT-FIRE based fiber extraction methods.
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to analyze
    boundary : Boundary, optional
        Boundary definition for relative angle measurements
    mode : {"curvelets", "ctfire"}, default "curvelets"
        Fiber extraction method to use
    options : CurveAlignOptions, optional
        Analysis parameters and options
        
    Returns
    -------
    AnalysisResult
        Complete analysis results including curvelets, features, and stats
        
    Examples
    --------
    >>> import numpy as np
    >>> import curvealign
    >>> 
    >>> # Load your image
    >>> image = np.random.randn(256, 256)  # Replace with real image
    >>> 
    >>> # Basic analysis
    >>> result = curvealign.analyze_image(image)
    >>> print(f"Found {len(result.curvelets)} fiber segments")
    >>> 
    >>> # With custom options
    >>> options = curvealign.CurveAlignOptions(keep=0.002, dist_thresh=150)
    >>> result = curvealign.analyze_image(image, options=options)
    """
    if options is None:
        options = CurveAlignOptions()
    
    # Validate mode for backward compatibility with legacy tests
    if mode not in ("curvelets", "ctfire"):
        raise ValueError(f"Unknown mode: {mode}")
    
    if mode == "ctfire":
        # Use CT-FIRE for fiber extraction
        try:
            import sys
            from pathlib import Path
            import ctfire_py as ctfire
            
            # Convert CurveAlign options to CT-FIRE options
            ctfire_options = ctfire.CTFireOptions(
                run_mode="ctfire",
                keep=options.keep,
                scale=options.scale,
                thresh_flen=options.minimum_box_size * 2,  # Reasonable default
            )
            
            # Run CT-FIRE analysis
            ctfire_result = ctfire.analyze_image(image, ctfire_options)
            
            # Convert CT-FIRE fibers to CurveAlign curvelets
            curvelets = _convert_fibers_to_curvelets(ctfire_result.fibers)
            
            # Use CT-FIRE statistics but compute CurveAlign features for consistency
            features = compute_features(curvelets, options.to_feature_options())
            stats = ctfire_result.stats.copy()
            
        except ImportError:
            raise ImportError("CT-FIRE package not available. Ensure ctfire_py is installed.")
    else:
        # Extract curvelets
        curvelets, coeffs = extract_curvelets(
            image,
            keep=options.keep,
            scale=options.scale,
            group_radius=options.group_radius
        )
        
        # Compute features
        feature_options = options.to_feature_options()
        features = compute_features(curvelets, feature_options)
        
        # Compute summary statistics
        stats = _compute_summary_statistics(curvelets, features, None)
    
    # Boundary analysis if boundary provided
    boundary_metrics = None
    if boundary is not None:
        boundary_metrics = measure_boundary_alignment(
            curvelets,
            boundary,
            dist_thresh=options.dist_thresh,
            min_dist=options.min_dist,
            exclude_inside_mask=options.exclude_inside_mask
        )
        
        # Update stats with boundary metrics if not from CT-FIRE
        if mode != "ctfire":
            stats = _compute_summary_statistics(curvelets, features, boundary_metrics)
    
    return AnalysisResult(
        curvelets=curvelets,
        features=features,
        boundary_metrics=boundary_metrics,
        stats=stats
    )


def analyze_roi(
    image: np.ndarray,
    rois: Sequence[Boundary],
    options: Optional[CurveAlignOptions] = None,
) -> ROIResult:
    """
    Analyze multiple ROIs within a single image.
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to analyze
    rois : Sequence[Boundary]
        List of ROI boundaries to analyze
    options : CurveAlignOptions, optional
        Analysis parameters and options
        
    Returns
    -------
    ROIResult
        Results for each ROI and comparison statistics
    """
    if options is None:
        options = CurveAlignOptions()
    
    roi_results = []
    
    for i, roi in enumerate(rois):
        # For now, analyze each ROI as a boundary
        # TODO: Implement proper ROI cropping and analysis
        result = analyze_image(image, boundary=roi, options=options)
        roi_results.append(result)
    
    # Compute comparison statistics
    comparison_stats = _compute_roi_comparison_stats(roi_results)
    
    return ROIResult(
        roi_results=roi_results,
        comparison_stats=comparison_stats
    )


def batch_analyze(
    inputs: Iterable[Union[Path, np.ndarray]],
    boundaries: Optional[Iterable[Optional[Boundary]]] = None,
    mode: Literal["curvelets", "ctfire"] = "curvelets",
    options: Optional[CurveAlignOptions] = None,
) -> List[AnalysisResult]:
    """
    Analyze multiple images in batch.
    
    Parameters
    ----------
    inputs : Iterable[Path | np.ndarray]
        Image paths or arrays to analyze
    boundaries : Iterable[Boundary], optional
        Boundaries for each image (if provided)
    options : CurveAlignOptions, optional
        Analysis parameters and options
        
    Returns
    -------
    List[AnalysisResult]
        Analysis results for each input image
    """
    if options is None:
        options = CurveAlignOptions()
    
    results = []
    
    # Convert inputs to list for indexing
    input_list = list(inputs)
    boundary_list = list(boundaries) if boundaries else [None] * len(input_list)
    
    for i, input_item in enumerate(input_list):
        # Load image if path provided
        if isinstance(input_item, (str, Path)):
            from skimage import io
            image = io.imread(str(input_item))
        else:
            image = input_item
        
        # Get corresponding boundary
        boundary = boundary_list[i] if i < len(boundary_list) else None
        
        # Analyze image
        result = analyze_image(image, boundary=boundary, mode=mode, options=options)
        results.append(result)
    
    return results


def get_curvelets(
    image: np.ndarray,
    keep: float = 0.001,
    scale: Optional[int] = None,
    group_radius: Optional[float] = None,
) -> Tuple[List[Curvelet], CtCoeffs]:
    """
    Extract curvelets from an image (mid-level API).
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image
    keep : float, default 0.001
        Fraction of coefficients to keep
    scale : int, optional
        Specific scale to analyze
    group_radius : float, optional
        Radius for grouping nearby curvelets
        
    Returns
    -------
    Tuple[List[Curvelet], CtCoeffs]
        Extracted curvelets and coefficient structure
    """
    return extract_curvelets(image, keep, scale, group_radius)


def reconstruct(coeffs: CtCoeffs, scales: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Reconstruct image from curvelet coefficients (mid-level API).
    
    Parameters
    ----------
    coeffs : CtCoeffs
        Curvelet coefficient structure
    scales : Sequence[int], optional
        Specific scales to include in reconstruction
        
    Returns
    -------
    np.ndarray
        Reconstructed image
    """
    return reconstruct_image(coeffs, scales)


# Convenience visualization functions with backend selection
def overlay(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    backend: str = "matplotlib",
    **kwargs
) -> np.ndarray:
    """
    Create curvelet overlay using specified backend.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    curvelets : Sequence[Curvelet]
        Curvelets to overlay
    backend : str, default "matplotlib"
        Visualization backend to use
    **kwargs
        Additional arguments passed to backend
        
    Returns
    -------
    np.ndarray
        Overlay image
    """
    from .visualization import create_overlay
    return create_overlay(image, curvelets, backend=backend, **kwargs)


def angle_map(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    backend: str = "matplotlib",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create angle maps using specified backend.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    curvelets : Sequence[Curvelet]
        Curvelets with angle information
    backend : str, default "matplotlib"
        Visualization backend to use
    **kwargs
        Additional arguments passed to backend
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw and processed angle maps
    """
    from .visualization import create_angle_maps
    return create_angle_maps(image, curvelets, backend=backend, **kwargs)


def _compute_summary_statistics(
    curvelets: List[Curvelet],
    features: FeatureTable,
    boundary_metrics: Optional[BoundaryMetrics]
) -> dict:
    """Compute summary statistics for analysis result."""
    if not curvelets:
        return {
            'n_curvelets': 0,
            'mean_angle': 0.0,
            'angle_std': 0.0,
            'mean_weight': 0.0,
            'alignment': 0.0,
            'density': 0.0
        }
    
    angles = np.array([c.angle_deg for c in curvelets])
    weights = np.array([c.weight or 1.0 for c in curvelets])
    
    stats = {
        'n_curvelets': len(curvelets),
        'mean_angle': np.mean(angles),
        'angle_std': np.std(angles),
        'mean_weight': np.mean(weights),
    }
    # Backward-compatibility alias for older tests expecting 'std_angle'
    stats['std_angle'] = stats['angle_std']
    # Backward-compatibility alias for tests expecting 'total_curvelets'
    stats['total_curvelets'] = stats['n_curvelets']
    
    # Compute alignment from features if available
    if 'alignment_nn' in features:
        stats['alignment'] = np.mean(features['alignment_nn'])
    else:
        # Fallback: compute alignment from angle coherence
        stats['alignment'] = _compute_angle_coherence(angles)
    
    # Compute density from features if available
    if 'density_nn' in features:
        stats['density'] = np.mean(features['density_nn'])
    else:
        # Fallback: simple spatial density
        stats['density'] = len(curvelets) / (256 * 256)  # Assume 256x256 for fallback
    
    # Add boundary statistics if available
    if boundary_metrics:
        stats.update({
            f'boundary_{key}': value 
            for key, value in boundary_metrics.alignment_stats.items()
        })
    
    return stats


def _convert_fibers_to_curvelets(fibers):
    """Convert CT-FIRE fibers to CurveAlign curvelets for unified interface."""
    curvelets = []
    
    for fiber in fibers:
        # Use fiber center point and mean angle
        if fiber.points:
            center_idx = len(fiber.points) // 2
            center_row, center_col = fiber.points[center_idx]
            
            curvelet = Curvelet(
                center_row=int(center_row),
                center_col=int(center_col),
                angle_deg=fiber.angle_deg,
                weight=fiber.length  # Use length as weight
            )
            curvelets.append(curvelet)
    
    return curvelets


def _compute_roi_comparison_stats(roi_results: List[AnalysisResult]) -> dict:
    """Compute comparison statistics across ROIs."""
    if not roi_results:
        return {}
    
    # Extract key metrics
    n_curvelets = [len(r.curvelets) for r in roi_results]
    alignments = [r.stats.get('alignment', 0.0) for r in roi_results]
    densities = [r.stats.get('density', 0.0) for r in roi_results]
    
    return {
        'n_rois': len(roi_results),
        'total_curvelets': sum(n_curvelets),
        'mean_curvelets_per_roi': np.mean(n_curvelets),
        'std_curvelets_per_roi': np.std(n_curvelets),
        'mean_alignment': np.mean(alignments),
        'std_alignment': np.std(alignments),
        'mean_density': np.mean(densities),
        'std_density': np.std(densities),
    }


def _compute_angle_coherence(angles: np.ndarray) -> float:
    """Compute coherence of angles (alignment measure)."""
    if len(angles) <= 1:
        return 0.0
    
    # Convert to radians and use circular statistics
    angles_rad = np.radians(angles)
    complex_angles = np.exp(1j * 2 * angles_rad)  # Factor of 2 for fiber symmetry
    coherence = np.abs(np.mean(complex_angles))
    
    return coherence