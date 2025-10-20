"""
FIRE algorithm implementation.

This module implements the core FIRE (Fiber Extraction) algorithm
for extracting individual fibers from microscopy images.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, distance_transform_cdt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from ...types import Fiber, CTFireOptions


def extract_fibers_fire(
    image: np.ndarray,
    options: Optional[CTFireOptions] = None
) -> List[Fiber]:
    """
    Extract individual fibers using the FIRE algorithm.
    
    This implements the main FIRE algorithm from fire.m and fiberproc.m.
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image (may be curvelet-enhanced)
    options : CTFireOptions, optional
        FIRE algorithm parameters
        
    Returns
    -------
    List[Fiber]
        List of extracted individual fibers
    """
    if options is None:
        options = CTFireOptions()
    
    # Step 1: Smooth image
    smoothed_image = gaussian_filter(image.astype(float), sigma=options.sigma_im)
    
    # Step 2: Threshold image
    # Handle NaN values before thresholding
    if np.any(np.isnan(smoothed_image)):
        # Replace NaN values with the minimum finite value
        finite_mask = np.isfinite(smoothed_image)
        if np.any(finite_mask):
            min_finite = np.min(smoothed_image[finite_mask])
            smoothed_image = np.where(np.isfinite(smoothed_image), smoothed_image, min_finite)
        else:
            # If all values are NaN, create a uniform image
            smoothed_image = np.ones_like(smoothed_image) * 0.5
    
    if options.thresh_im is not None:
        threshold = options.thresh_im * np.max(smoothed_image)
        binary_image = smoothed_image > threshold
    elif options.thresh_im2 is not None:
        binary_image = smoothed_image > options.thresh_im2
    else:
        # Auto-threshold using Otsu
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(smoothed_image)
        binary_image = smoothed_image > threshold
    
    # Step 3: Distance transform
    if options.dtype == "euclidean":
        distance_map = distance_transform_edt(binary_image)
    else:  # cityblock
        distance_map = distance_transform_cdt(binary_image, metric='taxicab')
    
    # Step 4: Extract fiber skeleton
    skeleton = skeletonize(binary_image)
    
    # Step 5: Find connected components and extract fiber paths
    labeled_skeleton = label(skeleton)
    fiber_candidates = []
    
    for region in regionprops(labeled_skeleton):
        if region.area >= options.thresh_numv:  # Minimum vertices
            coords = region.coords
            fiber_path = _trace_fiber_path(coords, distance_map)
            
            if len(fiber_path) >= options.thresh_numv:
                fiber = _create_fiber_from_path(fiber_path, distance_map, options)
                if fiber.length >= options.thresh_flen:
                    fiber_candidates.append(fiber)
    
    # Step 6: Link fibers and remove duplicates
    linked_fibers = _link_similar_fibers(fiber_candidates, options)
    
    # Step 7: Remove short fibers after linking
    final_fibers = [f for f in linked_fibers if f.length >= options.thresh_flen]
    
    return final_fibers


def enhance_image_with_curvelets(
    image: np.ndarray,
    keep: float = 0.001,
    scale: Optional[int] = None
) -> np.ndarray:
    """
    Enhance image using curvelet transform for better fiber extraction.
    
    This implements the curvelet enhancement step from ctFIRE_1.m.
    
    Parameters
    ----------
    image : np.ndarray
        Original input image
    keep : float, default 0.001
        Fraction of curvelet coefficients to keep
    scale : int, optional
        Specific scale for enhancement
        
    Returns
    -------
    np.ndarray
        Curvelet-enhanced image
    """
    # Import curvealign processors for enhancement (installed from src/curvealign_py)
    try:
        from curvealign_py.core.processors import extract_curvelets, reconstruct_image
    except ImportError as e:
        raise ImportError("CurveAlign package not available for curvelet enhancement. Ensure 'curvealign_py' is installed from src.") from e
    
    # Extract curvelets and reconstruct with limited scales
    _, coeffs = extract_curvelets(image, keep=keep, scale=scale)
    
    # Reconstruct using selected scales for enhancement
    # Pass original image shape to ensure correct output dimensions
    if scale is not None:
        enhanced_image = reconstruct_image(coeffs, scales=[scale], img_shape=image.shape)
    else:
        # Use multiple scales for enhancement
        enhanced_image = reconstruct_image(coeffs, scales=[1, 2, 3], img_shape=image.shape)
    
    return enhanced_image


def _trace_fiber_path(coords: np.ndarray, distance_map: np.ndarray) -> List[tuple]:
    """
    Trace fiber path from skeleton coordinates.
    
    This implements fiber path tracing logic similar to FIRE's approach.
    """
    if len(coords) < 2:
        return []
    
    # Sort coordinates to create a path
    # Start from point with maximum distance (center of fiber)
    distances = [distance_map[coord[0], coord[1]] for coord in coords]
    start_idx = np.argmax(distances)
    
    # Create path by connecting nearest neighbors
    path = [tuple(coords[start_idx])]
    remaining = list(range(len(coords)))
    remaining.remove(start_idx)
    
    current_pos = coords[start_idx]
    
    while remaining:
        # Find nearest remaining point
        distances_to_current = [np.linalg.norm(coords[i] - current_pos) for i in remaining]
        nearest_idx = remaining[np.argmin(distances_to_current)]
        
        path.append(tuple(coords[nearest_idx]))
        current_pos = coords[nearest_idx]
        remaining.remove(nearest_idx)
    
    return path


def _create_fiber_from_path(path: List[tuple], distance_map: np.ndarray, options: CTFireOptions) -> Fiber:
    """
    Create Fiber object from traced path.
    
    Computes geometric properties following FIRE algorithm.
    """
    if len(path) < 2:
        return Fiber(
            points=path,
            length=0.0,
            width=0.0,
            angle_deg=0.0,
            straightness=0.0,
            endpoints=(path[0], path[0]) if path else ((0, 0), (0, 0)),
            curvature=0.0
        )
    
    # Compute length
    total_length = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        total_length += np.linalg.norm(np.array(p2) - np.array(p1))
    
    # Compute mean width from distance map
    widths = [distance_map[p[0], p[1]] * 2 for p in path]  # Diameter = 2 * radius
    mean_width = np.mean(widths)
    
    # Compute mean angle
    angles = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        dx = p2[1] - p1[1]  # col direction
        dy = p2[0] - p1[0]  # row direction
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad) % 180  # Fiber symmetry
        angles.append(angle_deg)
    
    mean_angle = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(angles))))) % 180
    
    # Compute straightness (end-to-end distance / path length)
    if len(path) >= 2:
        end_to_end = np.linalg.norm(np.array(path[-1]) - np.array(path[0]))
        straightness = end_to_end / total_length if total_length > 0 else 0.0
    else:
        straightness = 1.0
    
    # Compute curvature (simplified)
    curvature = 1.0 - straightness  # Simple approximation
    
    return Fiber(
        points=path,
        length=total_length,
        width=mean_width,
        angle_deg=mean_angle,
        straightness=straightness,
        endpoints=(path[0], path[-1]),
        curvature=curvature
    )


def _link_similar_fibers(fibers: List[Fiber], options: CTFireOptions) -> List[Fiber]:
    """
    Link fibers with similar orientations that are close to each other.
    
    This implements the fiber linking logic from fiberlink.m and fiberlinkgap.m.
    """
    if len(fibers) <= 1:
        return fibers
    
    linked_fibers = []
    used = set()
    
    for i, fiber in enumerate(fibers):
        if i in used:
            continue
        
        # Find fibers to link with this one
        linkable = [fiber]
        used.add(i)
        
        for j, other_fiber in enumerate(fibers):
            if j in used or j == i:
                continue
            
            # Check if fibers can be linked
            if _can_link_fibers(fiber, other_fiber, options):
                linkable.append(other_fiber)
                used.add(j)
        
        # Create linked fiber if multiple fibers were linked
        if len(linkable) > 1:
            linked_fiber = _merge_fibers(linkable)
            linked_fibers.append(linked_fiber)
        else:
            linked_fibers.append(fiber)
    
    return linked_fibers


def _can_link_fibers(fiber1: Fiber, fiber2: Fiber, options: CTFireOptions) -> bool:
    """Check if two fibers can be linked based on proximity and orientation."""
    # Check angle similarity
    angle_diff = abs(fiber1.angle_deg - fiber2.angle_deg)
    angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wraparound
    
    if angle_diff > options.thresh_linka * 180:  # Convert to degrees
        return False
    
    # Check distance between endpoints
    min_dist = float('inf')
    for ep1 in fiber1.endpoints:
        for ep2 in fiber2.endpoints:
            dist = np.linalg.norm(np.array(ep1) - np.array(ep2))
            min_dist = min(min_dist, dist)
    
    return min_dist <= options.thresh_linkd


def _merge_fibers(fibers: List[Fiber]) -> Fiber:
    """Merge multiple fibers into a single connected fiber."""
    if len(fibers) == 1:
        return fibers[0]
    
    # Combine all points
    all_points = []
    for fiber in fibers:
        all_points.extend(fiber.points)
    
    # Compute combined properties
    total_length = sum(f.length for f in fibers)
    mean_width = np.mean([f.width for f in fibers])
    
    # Compute mean angle using circular statistics
    angles_rad = [np.radians(f.angle_deg) for f in fibers]
    mean_angle_rad = np.angle(np.mean(np.exp(1j * np.array(angles_rad))))
    mean_angle_deg = np.degrees(mean_angle_rad) % 180
    
    # Compute overall straightness
    if len(all_points) >= 2:
        end_to_end = np.linalg.norm(np.array(all_points[-1]) - np.array(all_points[0]))
        straightness = end_to_end / total_length if total_length > 0 else 0.0
    else:
        straightness = 1.0
    
    return Fiber(
        points=all_points,
        length=total_length,
        width=mean_width,
        angle_deg=mean_angle_deg,
        straightness=straightness,
        endpoints=(all_points[0], all_points[-1]),
        curvature=1.0 - straightness
    )
