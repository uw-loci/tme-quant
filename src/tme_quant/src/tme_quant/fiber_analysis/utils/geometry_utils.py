# File: tme_quant/utils/geometry_utils.py

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


def find_nearest_boundary_point(
    fiber_point: np.ndarray,
    tumor_boundary: 'ROI',
    pixel_size: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Find the nearest point on the tumor boundary to a fiber point.
    
    Args:
        fiber_point: [x, y] coordinates of fiber point
        tumor_boundary: Tumor boundary ROI
        pixel_size: Pixel size in microns
        
    Returns:
        Tuple of (nearest_point, distance)
            - nearest_point: [x, y] coordinates on boundary
            - distance: Distance in microns
    """
    # Convert fiber point to Shapely Point
    point = Point(fiber_point)
    
    # Get boundary geometry
    boundary_geom = tumor_boundary.to_shapely_geometry()
    
    # Handle different boundary types
    if isinstance(boundary_geom, Polygon):
        # Use exterior ring for polygon
        boundary_line = boundary_geom.exterior
    elif isinstance(boundary_geom, LineString):
        boundary_line = boundary_geom
    else:
        raise ValueError(f"Unsupported boundary geometry type: {type(boundary_geom)}")
    
    # Find nearest point on boundary
    nearest_geom = nearest_points(point, boundary_line)[1]
    nearest_point = np.array([nearest_geom.x, nearest_geom.y])
    
    # Calculate distance
    distance = np.linalg.norm(fiber_point - nearest_point) * pixel_size
    
    return nearest_point, distance


def compute_boundary_normal(
    point_on_boundary: np.ndarray,
    tumor_boundary: 'ROI',
    epsilon: float = 1.0
) -> float:
    """
    Compute the normal vector angle at a point on the tumor boundary.
    
    The normal points inward (toward tumor interior).
    
    Args:
        point_on_boundary: [x, y] coordinates on boundary
        tumor_boundary: Tumor boundary ROI
        epsilon: Distance for numerical differentiation
        
    Returns:
        Normal angle in degrees (0-360)
    """
    # Get boundary geometry
    boundary_geom = tumor_boundary.to_shapely_geometry()
    
    if isinstance(boundary_geom, Polygon):
        boundary_line = boundary_geom.exterior
    else:
        boundary_line = boundary_geom
    
    # Get coordinates along boundary
    coords = np.array(boundary_line.coords)
    
    # Find closest segment
    point = Point(point_on_boundary)
    min_dist = float('inf')
    closest_idx = 0
    
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        dist = point.distance(segment)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # Get tangent vector from closest segment
    p1 = coords[closest_idx]
    p2 = coords[closest_idx + 1]
    tangent = p2 - p1
    tangent = tangent / np.linalg.norm(tangent)
    
    # Normal is perpendicular to tangent (rotated 90°)
    # For inward normal, choose direction toward centroid
    normal_option1 = np.array([-tangent[1], tangent[0]])
    normal_option2 = np.array([tangent[1], -tangent[0]])
    
    # Determine which normal points inward
    if isinstance(boundary_geom, Polygon):
        centroid = np.array([boundary_geom.centroid.x, boundary_geom.centroid.y])
        to_centroid = centroid - point_on_boundary
        
        # Choose normal that aligns better with direction to centroid
        if np.dot(normal_option1, to_centroid) > np.dot(normal_option2, to_centroid):
            normal = normal_option1
        else:
            normal = normal_option2
    else:
        # For non-polygon boundaries, default to option 1
        normal = normal_option1
    
    # Convert to angle
    normal_angle = np.degrees(np.arctan2(normal[1], normal[0]))
    
    # Normalize to 0-360
    if normal_angle < 0:
        normal_angle += 360
    
    return normal_angle


def _angle_between_orientations(
    fiber_angle: float,
    boundary_normal_angle: float,
) -> Dict[str, float]:
    """
    Pure-math helper: compute angle_to_tangent between a fiber orientation and
    a boundary normal angle.

    This is an internal building block used by ``compute_fiber_to_boundary_alignment``
    (the legacy Shapely path) and ``FiberObject.compute_boundary_relative_metrics``.
    It performs no geometry — callers are responsible for deriving the boundary
    normal angle before calling this function.

    For the full spatial pipeline that accepts raw boundary coordinates and
    returns all three TACS angles, use ``compute_tacs_angles`` instead.

    Parameters
    ----------
    fiber_angle : float
        Fiber orientation in degrees.
    boundary_normal_angle : float
        Boundary normal angle in degrees.

    Returns
    -------
    float
        ``angle_to_tangent`` — acute angle [0°, 90°]; 0° = fiber parallel to
        boundary (TACS-2 pattern).
    """
    # Normalize angles to 0-180 range for orientation
    def normalize_orientation(angle):
        angle = angle % 180
        return angle
    
    fiber_orientation = normalize_orientation(fiber_angle)
    normal_orientation = normalize_orientation(boundary_normal_angle)
    
    # Angle to normal (intermediate; 0° = fiber perpendicular to boundary)
    angle_to_normal = abs(fiber_orientation - normal_orientation)
    if angle_to_normal > 90:
        angle_to_normal = 180 - angle_to_normal

    # Angle to tangent (0° = fiber parallel to boundary — used for TACS)
    return float(90 - angle_to_normal)


def compute_fiber_to_boundary_alignment(
    fiber_centerline: np.ndarray,
    boundary: 'ROI',
    pixel_size: float = 1.0
) -> Dict[str, float]:
    """
    Compute comprehensive fiber-to-boundary alignment metrics.
    
    This is the main function that combines all metrics.
    
    Args:
        fiber_centerline: Nx2 array of fiber coordinates
        boundary: Tumor boundary ROI
        pixel_size: Pixel size in microns
        
    Returns:
        Dictionary with all alignment metrics
    """
    # Get fiber midpoint
    mid_idx = len(fiber_centerline) // 2
    fiber_point = fiber_centerline[mid_idx]
    
    # Find nearest boundary point
    nearest_point, distance = find_nearest_boundary_point(
        fiber_point, boundary, pixel_size
    )
    
    # Compute boundary normal
    normal_angle = compute_boundary_normal(nearest_point, boundary)
    
    # Compute fiber orientation [0°, 180°)
    fiber_vector = fiber_centerline[-1] - fiber_centerline[0]
    fiber_angle = float(np.degrees(np.arctan2(fiber_vector[1], fiber_vector[0])) % 180)
    
    # Compute relative angles
    angle_to_tangent = _angle_between_orientations(fiber_angle, normal_angle)

    # Compute alignment score (0 = perpendicular, 1 = parallel)
    alignment_score = abs(angle_to_tangent) / 90.0

    return {
        'distance': distance,
        'nearest_point': nearest_point,
        'boundary_normal_angle': normal_angle,
        'fiber_angle': fiber_angle,
        'angle_to_tangent': angle_to_tangent,
        'alignment_score': alignment_score
    }

def compute_angle_to_boundary_normal(
    fiber_orientation: float,
    boundary_point1: Tuple[float, float],
    boundary_point2: Tuple[float, float]
) -> float:
    """
    Compute angle between fiber orientation and tumor boundary normal.
    
    The boundary normal is perpendicular to the local boundary tangent.
    This is used for TACS (Tumor-Associated Collagen Signatures) classification.
    
    Args:
        fiber_orientation: Fiber orientation angle in degrees (0-180°)
        boundary_point1: First point on boundary near fiber (x, y)
        boundary_point2: Second point on boundary near fiber (x, y)
        
    Returns:
        Angle between fiber and boundary normal in degrees (0-90°)
        
    Notes:
        - Boundary tangent is computed from the two boundary points
        - Boundary normal is perpendicular (90°) to the tangent
        - Returns the acute angle (0-90°) to match TACS classification ranges
        
    Example:
        >>> # Horizontal boundary (tangent = 0°), vertical fiber (90°)
        >>> # Boundary normal is vertical (90°), so angle diff = 0° (parallel to normal)
        >>> angle = compute_angle_to_boundary_normal(90, (0, 0), (10, 0))
        >>> print(angle)  # ~0° (fiber parallel to boundary normal)
        
        >>> # Horizontal boundary, horizontal fiber (0°)
        >>> # Boundary normal is vertical (90°), so angle diff = 90° (perpendicular to normal)
        >>> angle = compute_angle_to_boundary_normal(0, (0, 0), (10, 0))
        >>> print(angle)  # ~90° (fiber perpendicular to boundary normal)
    """
    import numpy as np
    
    # Compute boundary tangent vector
    dx = boundary_point2[0] - boundary_point1[0]
    dy = boundary_point2[1] - boundary_point1[1]
    
    # Avoid division by zero
    if np.abs(dx) < 1e-10 and np.abs(dy) < 1e-10:
        # Points are too close, return NaN
        return np.nan
    
    # Normalize fiber orientation to [0°, 180°)
    fiber_orientation = float(fiber_orientation) % 180

    # Boundary tangent angle [0°, 180°)
    boundary_tangent = float(np.degrees(np.arctan2(dy, dx)) % 180)
    
    # Boundary normal is perpendicular to tangent
    boundary_normal = (boundary_tangent + 90) % 180
    
    # Compute angular difference between fiber and boundary normal
    angle_diff = np.abs(fiber_orientation - boundary_normal)
    
    # Normalize to [0, 90] (we want the acute angle)
    # Because orientations are in [0, 180), the difference can be up to 180
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    
    return angle_diff


def compute_angle_to_boundary_normal_simplified(
    fiber_orientation: float,
    boundary_tangent_angle: float
) -> float:
    """
    Simplified version when boundary tangent angle is already known.
    
    Args:
        fiber_orientation: Fiber orientation angle (0-180°)
        boundary_tangent_angle: Pre-computed boundary tangent angle (0-180°)
        
    Returns:
        Angle to boundary normal (0-90°)
        
    Example:
        >>> # Horizontal boundary (tangent = 0°), vertical fiber (90°)
        >>> angle = compute_angle_to_boundary_normal_simplified(90, 0)
        >>> print(angle)  # 0° (parallel to normal)
    """
    import numpy as np

    # Normalize both inputs to [0°, 180°)
    fiber_orientation   = float(fiber_orientation)   % 180
    boundary_tangent_angle = float(boundary_tangent_angle) % 180

    # Boundary normal is perpendicular to tangent
    boundary_normal = (boundary_tangent_angle + 90) % 180

    # Angular difference
    angle_diff = abs(fiber_orientation - boundary_normal)
    
    # Normalize to [0, 90]
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    
    return angle_diff



def compute_fiber_properties(
    centerline: np.ndarray,
    image: np.ndarray,
    pixel_size: float = 1.0,
    width_range: Tuple[float, float] = (0.5, 20.0),
) -> Dict[str, float]:
    """
    Compute geometric properties of a single fiber from its centerline.

    Parameters
    ----------
    centerline:
        Nx2 array of (row, col) coordinates tracing the fiber.
    image:
        2-D grayscale image (used to estimate fiber width via intensity profile).
    pixel_size:
        Microns per pixel (applied to length, width).
    width_range:
        (min, max) acceptable fiber width in microns.

    Returns
    -------
    Dict with keys: length, width, straightness, angle, curvature.
    """
    if centerline is None or len(centerline) < 2:
        return {
            'length': 0.0, 'width': 1.0,
            'straightness': 0.0, 'angle': 0.0, 'curvature': 0.0,
        }

    coords = np.asarray(centerline, dtype=np.float64)

    # Arc length along centerline
    diffs   = np.diff(coords, axis=0)
    arc_length = float(np.sum(np.linalg.norm(diffs, axis=1))) * pixel_size

    # Straightness (end-to-end / arc length)
    end_to_end   = float(np.linalg.norm(coords[-1] - coords[0])) * pixel_size
    straightness = float(np.clip(end_to_end / max(arc_length, 1e-10), 0.0, 1.0))

    # Angle of end-to-end vector [0°, 180°)
    vec   = coords[-1] - coords[0]
    angle = float(np.degrees(np.arctan2(vec[0], vec[1])) % 180)

    # Curvature (mean turning angle per unit length)
    if len(diffs) >= 2:
        seg_angles  = np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))
        turn_angles = np.diff(seg_angles)
        turn_angles = (turn_angles + 180) % 360 - 180   # wrap to [-180, 180]
        curvature   = float(np.sum(np.abs(turn_angles))) / max(arc_length, 1e-10)
    else:
        curvature = 0.0

    # Width via intensity profile FWHM
    width = _estimate_fiber_width(coords, image, pixel_size, width_range)

    return {
        'length':       arc_length,
        'width':        width,
        'straightness': straightness,
        'angle':        angle,
        'curvature':    curvature,
    }


def _estimate_fiber_width(
    centerline: np.ndarray,
    image: np.ndarray,
    pixel_size: float,
    width_range: Tuple[float, float],
) -> float:
    """Estimate fiber width from perpendicular intensity cross-sections (FWHM)."""
    h, w_img = image.shape[:2]
    widths = []
    sample_idx = np.linspace(1, len(centerline) - 2,
                             min(5, max(1, len(centerline) - 2)), dtype=int)
    for idx in sample_idx:
        r, c = centerline[idx]
        dr = centerline[min(idx + 1, len(centerline) - 1)][0] - centerline[max(idx - 1, 0)][0]
        dc = centerline[min(idx + 1, len(centerline) - 1)][1] - centerline[max(idx - 1, 0)][1]
        norm = np.sqrt(dr**2 + dc**2) + 1e-10
        perp_r, perp_c = -dc / norm, dr / norm
        half_px = int(width_range[1] / pixel_size)
        offsets = np.arange(-half_px, half_px + 1)
        rows = np.clip(r + offsets * perp_r, 0, h - 1).astype(int)
        cols = np.clip(c + offsets * perp_c, 0, w_img - 1).astype(int)
        profile = image[rows, cols].astype(np.float64)
        if profile.max() > 0:
            above = (profile / profile.max()) >= 0.5
            widths.append(float(np.clip(np.sum(above) * pixel_size, *width_range)))
    return float(np.median(widths)) if widths else float(np.clip(1.0, *width_range))


# ─────────────────────────────────────────────────────────────────────────────
# 8-Connected boundary traversal primitives
# (faithfully ported from pycurvelets utils.connectivity and utils.math)
# ─────────────────────────────────────────────────────────────────────────────

def _get_first_neighbor(
    coords: np.ndarray,
    idx: int,
    visited: np.ndarray,
    direction: int,
) -> int:
    """
    Return the index of the first unvisited 8-connected neighbour in *coords*.

    Port of pycurvelets ``get_first_neighbor``.  Operates on dense boundary
    pixel traces stored as (row, col) integer pairs.

    Parameters
    ----------
    coords : (N, 2) int array
        Dense boundary pixels as (row, col).
    idx : int
        Current position index in *coords*.
    visited : (N,) bool array
        Marks already-visited indices.
    direction : {1, 2}
        1 = backward sweep (S-first neighbour order).
        2 = forward  sweep (N-first neighbour order).

    Returns
    -------
    int
        Index of the first unvisited 8-connected neighbour, or *idx* if none
        found (matching MATLAB fallback behaviour).
    """
    pt = coords[idx]
    if direction == 1:
        offsets = [(1, 0), (1, -1), (0, -1), (-1, -1),
                   (-1, 0), (-1, 1), (0, 1), (1, 1)]
    else:  # direction == 2
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0), (1, -1), (0, -1), (-1, -1)]

    rows = coords[:, 0]
    cols = coords[:, 1]
    for dr, dc in offsets:
        matches = np.where((rows == pt[0] + dr) & (cols == pt[1] + dc))[0]
        if matches.size > 0:
            nbr = int(matches[0])
            if not visited[nbr]:
                return nbr
    return idx


def _find_connected_pts(
    coords: np.ndarray,
    idx: int,
    num: int,
) -> np.ndarray:
    """
    Return *num* 8-connected pixels centred at *idx* along a dense boundary.

    Port of pycurvelets ``find_connected_pts``.  Returns a NaN-filled array
    when connectivity runs out before reaching *num* points.

    Parameters
    ----------
    coords : (N, 2) array
        Dense boundary pixels as (row, col).
    idx : int
        Seed index placed in the middle of the returned window.
    num : int
        Total number of points to return.  Should be odd.

    Returns
    -------
    (num, 2) ndarray — NaN-filled where not enough connected neighbours exist.
    """
    _nan = np.full((num, 2), np.nan)
    con_pts = _nan.copy()
    hnum = (num - 1) // 2
    con_pts[hnum] = coords[idx]

    visited = np.zeros(len(coords), dtype=bool)

    # backward pass
    cur = idx
    for i in range(hnum - 1, -1, -1):
        visited[cur] = True
        nxt = _get_first_neighbor(coords, cur, visited, direction=1)
        if nxt == cur:          # stuck — not enough connected pixels
            return _nan.copy()
        con_pts[i] = coords[nxt]
        cur = nxt

    # forward pass
    cur = idx
    for i in range(hnum + 1, num):
        visited[cur] = True
        nxt = _get_first_neighbor(coords, cur, visited, direction=2)
        if nxt == cur:
            return _nan.copy()
        con_pts[i] = coords[nxt]
        cur = nxt

    return con_pts


def _circ_r(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
    d: float = 0.0,
) -> float:
    """
    Mean resultant vector length for circular data.

    Port of pycurvelets ``circ_r``  (Zar, 2010, eq. 26.16).

    Parameters
    ----------
    alpha : array-like
        Angles in radians.
    w : array-like or None
        Weights.  Defaults to uniform weights.
    d : float
        Bin spacing in radians for bias correction of grouped data.  0 = off.

    Returns
    -------
    float — mean resultant length r ∈ [0, 1].
    """
    alpha = np.asarray(alpha, dtype=float)
    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w, dtype=float)
    r = float(np.abs(np.sum(w * np.exp(1j * alpha))) / np.sum(w))
    if d != 0.0:
        r *= d / (2.0 * np.sin(d / 2.0))
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Public boundary-tangent and relative-angle utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_boundary_tangent_angle(
    coords: np.ndarray,
    idx: int,
    num: int = 21,
) -> float:
    """
    Robust polynomial-fit tangent angle at a point on a dense boundary trace.

    Port of pycurvelets ``find_outline_slope``.

    **Use this function for dense pixel-level boundary traces** (e.g. CurveAlign
    output, skeletonised mask borders where every adjacent pair of points is
    8-connected).  For sparse polygon vertex arrays use the simpler 2-point
    method already provided by ``compute_angle_to_boundary_normal``.

    Parameters
    ----------
    coords : (N, 2) int array
        Dense boundary pixels stored as (row, col) pairs.
    idx : int
        Index of the query point in *coords*.
    num : int
        Number of 8-connected pixels to sample around *idx*.  Default 21.

    Returns
    -------
    float
        Tangent angle in degrees [0°, 180°), or NaN when fewer than *num*
        connected pixels are found around *idx*.
    """
    con_pts = _find_connected_pts(np.asarray(coords, dtype=int), idx, num)
    if np.any(np.isnan(con_pts)):
        return np.nan

    rise = con_pts[-1, 1] - con_pts[0, 1]   # col (x) difference
    run  = con_pts[-1, 0] - con_pts[0, 0]   # row (y) difference

    if run == 0:
        rough_slope = 90.0
    else:
        rough_slope = float(np.degrees(np.arctan(rise / run)) % 180)

    # Choose fitting axis based on dominant direction
    if rough_slope < 45 or rough_slope > 135:
        # Mostly horizontal — fit col (y) as function of row (x)
        x_fit = np.linspace(con_pts[0, 0], con_pts[-1, 0], 50)
        coeffs = np.polyfit(con_pts[:, 0], con_pts[:, 1], 2)
        y_fit  = np.polyval(coeffs, x_fit)
    else:
        # Mostly vertical — fit row (x) as function of col (y)
        y_fit  = np.linspace(con_pts[0, 1], con_pts[-1, 1], 50)
        coeffs = np.polyfit(con_pts[:, 1], con_pts[:, 0], 2)
        x_fit  = np.polyval(coeffs, y_fit)

    d_run  = x_fit[25] - x_fit[23]
    d_rise = y_fit[25] - y_fit[23]
    return float(np.degrees(np.arctan2(d_rise, d_run)) % 180)


def find_nearest_boundary_index(
    coords: np.ndarray,
    px: float,
    py: float,
) -> int:
    """
    Return the index in *coords* nearest to the query point (px, py).

    Works for both dense pixel traces and sparse polygon vertex arrays.

    Parameters
    ----------
    coords : (N, 2) array
        Points stored as (row, col) or (x, y) — must share the coordinate
        space of (px, py).
    px, py : float
        Query point in the same coordinate system as *coords*.

    Returns
    -------
    int — index of the nearest point in *coords*.
    """
    arr = np.asarray(coords, dtype=float)
    dists = (arr[:, 0] - px) ** 2 + (arr[:, 1] - py) ** 2
    return int(np.argmin(dists))


def compute_relative_fiber_angles(
    obj_center: Tuple[float, float],
    obj_angle: float,
    roi_coords: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
    index2object: Optional[int] = None,
    angle_option: int = 0,
    dense_boundary: bool = False,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """
    Compute all relative orientation angles between a TME object and an ROI.

    Unified port of pycurvelets ``get_relative_angles``, extended with a
    ``dense_boundary`` flag so the same function handles both sparse polygon
    ROIs and dense pixel-level boundary traces from CurveAlign.

    Three angle types are returned, each covering a distinct question:

    ``angle_to_boundary_tangent``
        How is the object oriented relative to the *local* boundary surface?
        0° = parallel (TACS-2 pattern); 90° = perpendicular (TACS-3 pattern).
    ``angle_to_roi_orientation``
        How aligned is the object with the ROI's *global* shape axis?
        Requires only the two objects' orientations and centroids.
    ``angle_to_centers_line``
        Does the object point *toward* the ROI centroid or *across* it?

    Parameters
    ----------
    obj_center : (x, y) float tuple
        Object centroid in (x, y) / (col, row) coordinates.
    obj_angle : float
        Absolute object orientation in degrees [0°, 180°].
    roi_coords : (N, 2) ndarray
        ROI boundary coordinates in **(row, col)** order (skimage convention).
    image_size : (height, width) or None
        When provided a binary mask is created and ``regionprops`` is used to
        derive the ROI global orientation (MATLAB-compatible path).
        When None a central-moment approximation is used instead.
    index2object : int or None
        Index into *roi_coords* of the boundary point nearest to the object.
        Auto-computed via ``find_nearest_boundary_index`` when None.
    angle_option : {0, 1, 2, 3}
        0 = all three angles, 1 = boundary-tangent only,
        2 = ROI-orientation only, 3 = centers-line only.
    dense_boundary : bool
        If True *roi_coords* is a dense 8-connected pixel trace and
        ``compute_boundary_tangent_angle`` (polynomial fit) is used.
        If False (default) *roi_coords* is a sparse polygon and the fast
        2-point tangent method is used.

    Returns
    -------
    relative_angles : dict
        ``angle_to_boundary_tangent`` — acute angle [0°, 90°].
        ``angle_to_roi_orientation``  — acute angle [0°, 90°].
        ``angle_to_centers_line``     — acute angle [0°, 90°].
        Any uncomputed entry (controlled by *angle_option*) is None.
    roi_measurements : dict
        ``center``      — (x, y) ROI centroid.
        ``orientation`` — ROI global orientation in degrees [0°, 180°].
        ``area``        — ROI pixel area (0 when *image_size* is None).
        ``boundary``    — *roi_coords* passed through.

    Notes
    -----
    Coordinate convention follows pycurvelets / skimage:
    *roi_coords* rows are ``(row, col)`` = ``(y, x)``;
    *obj_center* is ``(x, y)`` = ``(col, row)``.

    Raises
    ------
    ValueError
        If *image_size* is provided and *roi_coords* does not define exactly
        one connected region.
    """
    coords = np.asarray(roi_coords, dtype=float)
    obj_cx, obj_cy = float(obj_center[0]), float(obj_center[1])   # x, y
    obj_angle = float(obj_angle) % 180   # enforce [0°, 180°)

    # ── ROI global measurements ───────────────────────────────────────────
    if image_size is not None:
        from skimage.measure import regionprops, label
        from skimage.draw import polygon2mask
        h_img, w_img = image_size
        mask  = polygon2mask((h_img, w_img), coords)
        lbl   = label(mask.astype(np.uint8))
        props = regionprops(lbl)
        if len(props) != 1:
            raise ValueError(
                "roi_coords must define exactly one connected region when "
                "image_size is provided."
            )
        prop       = props[0]
        roi_center = np.array(prop.centroid)[::-1]          # (row,col) → (x,y)
        roi_angle  = float(-90.0 + np.degrees(prop.orientation))
        if roi_angle < 0.0:
            roi_angle += 180.0
        roi_area = int(prop.area)
    else:
        # Central-moment approximation — no image mask required
        rc         = coords.mean(axis=0)                    # (mean_row, mean_col)
        roi_center = rc[::-1].copy()                        # → (x, y)
        centered   = coords - rc
        m20 = float(np.mean(centered[:, 0] ** 2))          # var(row)
        m02 = float(np.mean(centered[:, 1] ** 2))          # var(col)
        m11 = float(np.mean(centered[:, 0] * centered[:, 1]))
        prop_orient = 0.5 * np.arctan2(-m11, m20 - m02)    # radians
        roi_angle   = float(-90.0 + np.degrees(prop_orient))
        if roi_angle < 0.0:
            roi_angle += 180.0
        roi_area = 0

    roi_measurements: Dict[str, Any] = {
        'center':      roi_center,
        'orientation': roi_angle,
        'area':        roi_area,
        'boundary':    roi_coords,
    }

    relative_angles: Dict[str, Optional[float]] = {
        'angle_to_boundary_tangent': None,
        'angle_to_roi_orientation':  None,
        'angle_to_centers_line':     None,
    }

    # Auto-compute nearest boundary index when not supplied.
    # coords are (row, col) = (y, x); obj_center is (x, y) — swap to match.
    if index2object is None:
        index2object = find_nearest_boundary_index(coords, obj_cy, obj_cx)

    # ── angle_to_boundary_tangent  (angle2boundaryEdge) ───────────────────
    if angle_option in (0, 1):
        boundary_pt = coords[index2object]

        if dense_boundary:
            # Polynomial fit on 8-connected dense pixel trace
            tangent_angle = compute_boundary_tangent_angle(
                coords.astype(int), index2object
            )
            if np.isnan(tangent_angle):
                temp_ang: Optional[float] = None
            else:
                # _circ_r([2α, 2β]) = |cos(α−β)|; arcsin gives angle_to_normal
                r = _circ_r([
                    np.radians(2.0 * obj_angle),
                    np.radians(2.0 * tangent_angle),
                ])
                angle_to_normal = float(
                    np.degrees(np.arcsin(np.clip(r, 0.0, 1.0)))
                )
                temp_ang = 90.0 - angle_to_normal   # → angle_to_tangent
        else:
            # Sparse polygon — image-edge guard (mirrors get_relative_angles)
            on_edge = False
            if image_size is not None:
                h_img, w_img = image_size
                bp_r, bp_c = int(boundary_pt[0]), int(boundary_pt[1])
                on_edge = (
                    bp_r <= 1 or bp_c <= 1
                    or bp_r >= h_img or bp_c >= w_img
                )
            if on_edge:
                temp_ang = 0.0
            else:
                # Find nearest polygon edge by midpoint proximity
                n = len(coords)
                best_dist = np.inf
                best_i = 0
                for i in range(n):
                    j = (i + 1) % n
                    mr = (coords[i, 0] + coords[j, 0]) / 2.0
                    mc = (coords[i, 1] + coords[j, 1]) / 2.0
                    dist = (obj_cy - mr) ** 2 + (obj_cx - mc) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i
                j = (best_i + 1) % n
                # Convert (row, col) → (x, y) for compute_angle_to_boundary_normal
                p1_xy = (float(coords[best_i, 1]), float(coords[best_i, 0]))
                p2_xy = (float(coords[j, 1]),      float(coords[j, 0]))
                angle_to_normal = compute_angle_to_boundary_normal(
                    obj_angle, p1_xy, p2_xy
                )
                if angle_to_normal is None or np.isnan(angle_to_normal):
                    temp_ang = None
                else:
                    temp_ang = float(90.0 - angle_to_normal)  # → angle_to_tangent

        relative_angles['angle_to_boundary_tangent'] = temp_ang

    # ── angle_to_roi_orientation  (angle2boundaryCenter) ──────────────────
    if angle_option in (0, 2):
        diff = abs(obj_angle - roi_angle)
        if diff > 90.0:
            diff = 180.0 - diff
        relative_angles['angle_to_roi_orientation'] = float(diff)

    # ── angle_to_centers_line  (angle2centersLine) ────────────────────────
    if angle_option in (0, 3):
        # Following original variable naming convention (dx=Δy, dy=Δx)
        delta_y = obj_cy - float(roi_center[1])   # y_obj − y_roi
        delta_x = obj_cx - float(roi_center[0])   # x_obj − x_roi
        if abs(delta_x) < 1e-10:
            centers_line_angle = 90.0 if delta_y >= 0.0 else -90.0
        else:
            centers_line_angle = float(np.degrees(np.arctan(delta_y / delta_x)))
        if centers_line_angle < 0.0:
            centers_line_angle = abs(centers_line_angle)
        else:
            centers_line_angle = 180.0 - centers_line_angle
        result = abs(centers_line_angle - obj_angle)
        if result > 90.0:
            result = 180.0 - result
        relative_angles['angle_to_centers_line'] = float(result)

    return relative_angles, roi_measurements


__all__ = [
    'find_nearest_boundary_point',
    'compute_boundary_normal',
    'compute_fiber_to_boundary_alignment',
    'compute_angle_to_boundary_normal',
    'compute_angle_to_boundary_normal_simplified',
    'compute_boundary_tangent_angle',
    'find_nearest_boundary_index',
    'compute_relative_fiber_angles',
]