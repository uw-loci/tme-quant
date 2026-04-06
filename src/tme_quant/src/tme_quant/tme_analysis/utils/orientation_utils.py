"""
Pixel-level boundary-relative orientation utilities for TME analysis.

Provides functions that iterate over an orientation map from CurveAlign (or
any other method) and compute per-pixel angles relative to a given ROI
boundary, together with TACS-like classification.

Key functions
-------------
nearest_boundary_segment
    Return the edge of a polygon that is closest to a query point.  Used to
    obtain the local boundary tangent for the angle calculation.
discretize_roi_boundary
    Convert a sparse polygon ROI to a dense 8-connected pixel-level boundary
    trace suitable for polynomial-fit tangent estimation.
compute_orientation_relative_to_roi
    Main entry point: iterate every valid pixel in an orientation map, filter
    to the TACS zone around *roi*, and return per-pixel angle statistics.
    Supports both the fast 2-point tangent (sparse polygon) and the more
    accurate polynomial-fit tangent (dense boundary trace).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import numpy as np

from tme_quant.core.roi_manager import ROIObject
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
    compute_boundary_tangent_angle,
    find_nearest_boundary_index,
    compute_relative_fiber_angles,
)
from tme_quant.fiber_analysis.tacs import classify_fiber_segment_tacs_like


# ─────────────────────────────────────────────────────────────────────────────

def discretize_roi_boundary(
    coords: np.ndarray,
    step: float = 1.0,
) -> np.ndarray:
    """
    Convert a sparse polygon ROI to a dense pixel-level boundary trace.

    Linearly interpolates along each polygon edge so that consecutive points
    are at most *step* pixels apart, producing the kind of dense 8-connected
    trace expected by ``compute_boundary_tangent_angle`` (polynomial fit).

    Use this when:

    * You have a hand-drawn or algorithmically generated polygon ROI (sparse
      vertices) but want the more accurate polynomial-fit tangent direction
      from ``compute_boundary_tangent_angle`` rather than the fast 2-point
      method.
    * You are calling ``compute_orientation_relative_to_roi`` with
      ``dense_boundary=True`` and your ROI is a polygon, not an already-dense
      CurveAlign pixel trace.

    Parameters
    ----------
    coords : (N, 2) ndarray
        Polygon vertices in **(row, col)** order (skimage / array convention).
        The polygon is treated as closed — the last vertex is connected back
        to the first.
    step : float
        Maximum spacing in pixels between consecutive output points.
        Default 1.0 (every pixel).  Increase for faster (less dense) traces.

    Returns
    -------
    (M, 2) ndarray of float
        Dense boundary in **(row, col)** order, suitable for passing directly
        to ``compute_boundary_tangent_angle`` or as *roi_coords* to
        ``compute_relative_fiber_angles`` with ``dense_boundary=True``.

    Example
    -------
    >>> polygon = roi.coordinates[:, ::-1]          # (x,y) → (row,col)
    >>> dense   = discretize_roi_boundary(polygon)
    >>> compute_relative_fiber_angles(
    ...     obj_center    = (cx, cy),
    ...     obj_angle     = fiber_angle,
    ...     roi_coords    = dense,
    ...     dense_boundary= True,
    ... )
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    pts: list[np.ndarray] = []
    for i in range(n):
        p0 = coords[i]
        p1 = coords[(i + 1) % n]
        seg_len = float(np.linalg.norm(p1 - p0))
        num = max(2, int(np.ceil(seg_len / step)))
        # linspace excludes the endpoint to avoid duplicating vertices
        ts = np.linspace(0.0, 1.0, num, endpoint=False)
        pts.append(p0[np.newaxis] + ts[:, np.newaxis] * (p1 - p0)[np.newaxis])
    return np.vstack(pts)


# ─────────────────────────────────────────────────────────────────────────────

def nearest_boundary_segment(
    coords: np.ndarray,
    px: float,
    py: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return the two polygon vertices that form the boundary edge nearest to
    the query point (px, py).

    The nearest edge is found by comparing each edge's midpoint to the query
    point (fast O(n) scan).  Used to obtain the local boundary tangent for
    ``compute_angle_to_boundary_normal``.

    .. note::
        This function is intentionally kept as a fast O(n) midpoint scan and
        is designed for the **pixel-map loop** in
        ``compute_orientation_relative_to_roi``, where it is called once per
        valid pixel.  For **single-fiber analysis** on a dense 8-connected
        pixel-trace boundary (e.g. CurveAlign output), prefer
        ``fiber_analysis.utils.compute_boundary_tangent_angle`` which uses a
        polynomial fit over 21 connected neighbours and is more robust near
        corners and high-curvature regions.

    Parameters
    ----------
    coords : (N, 2) ndarray
        Polygon vertex coordinates stored as (x, y) pairs.
    px, py : float
        Query point in the same coordinate system as *coords*.

    Returns
    -------
    tuple of two (x, y) float pairs
        The start and end vertex of the nearest edge.
    """
    n = len(coords)
    best_dist = np.inf
    best_i = 0

    for i in range(n):
        j = (i + 1) % n
        mx = (coords[i, 0] + coords[j, 0]) / 2
        my = (coords[i, 1] + coords[j, 1]) / 2
        d = (px - mx) ** 2 + (py - my) ** 2
        if d < best_dist:
            best_dist = d
            best_i = i

    j = (best_i + 1) % n
    return (
        (float(coords[best_i, 0]), float(coords[best_i, 1])),
        (float(coords[j, 0]),      float(coords[j, 1])),
    )


# ─────────────────────────────────────────────────────────────────────────────

def compute_orientation_relative_to_roi(
    orientation_map: np.ndarray,
    alignment_map:   Optional[np.ndarray],
    roi:             ROIObject,
    pixel_size:      float = 1.0,
    tacs_zone_width: float = 100.0,
    subsample:       int   = 2,
    dense_boundary:  bool  = False,
    dense_coords:    Optional[np.ndarray] = None,
) -> dict:
    """
    Compute boundary-relative fiber orientation for every valid pixel in
    *orientation_map* that falls within the TACS zone of *roi*.

    Parameters
    ----------
    orientation_map : (H, W) float32
        Per-pixel dominant fiber orientation in degrees.  NaN = unreliable
        pixel (will be skipped).
    alignment_map : (H, W) float32 or None
        Per-pixel alignment strength in [0, 1].  Two sources are accepted:

        * **CurveAlign** — the per-window angular energy concentration stored
          in ``CurveAlignResult.alignment_map`` (all pixels in a window share
          the same value).  Different from the per-region alignment score
          computed over a fiber group (see ``curvealign.compute_region_alignment``).
        * **OrientationJ** — the per-pixel structure tensor coherency
          ``(λ_max − λ_min) / (λ_max + λ_min)`` from ``OrientationJResult.coherency_map``.

        Pass ``None`` to skip alignment recording.
    roi : ROIObject
        The boundary to measure relative to.  Must be a closed shape
        (polygon, rectangle, ellipse) with valid ``coordinates``.
    pixel_size : float
        Microns per pixel.  Used to convert pixel distances to µm.
    tacs_zone_width : float
        Maximum distance in µm from the boundary to include a pixel.
    subsample : int
        Step size along each axis.  ``subsample=2`` processes every 2nd pixel
        in x and y, reducing compute roughly 4×.
    dense_boundary : bool
        If False (default) the fast 2-point tangent from the nearest polygon
        edge is used via ``nearest_boundary_segment`` +
        ``compute_angle_to_boundary_normal``.  Accurate enough for smoothly
        curved boundaries with many vertices.

        If True the polynomial-fit tangent from
        ``compute_boundary_tangent_angle`` is used instead.  This is more
        robust near corners and high-curvature sections.  Requires a dense
        8-connected pixel-level boundary trace; supply one via *dense_coords*
        or it will be auto-generated from ``roi.coordinates`` using
        ``discretize_roi_boundary``.
    dense_coords : (M, 2) ndarray or None
        Pre-computed dense boundary in **(row, col)** order.  Only used when
        ``dense_boundary=True``.  When None and ``dense_boundary=True`` the
        sparse polygon is discretized automatically via
        ``discretize_roi_boundary``.  Pass a pre-computed array to avoid
        re-discretizing on repeated calls.

    Returns
    -------
    dict with keys:

    points : list[dict]
        One record per included pixel, each containing:
        ``x``, ``y``, ``orientation``, ``alignment``, ``dist_to_boundary``,
        ``angle_to_tangent``, ``tacs_like``.
    mean_angle_to_tangent : float
    std_angle_to_tangent : float
    tacs_distribution : dict
        Counts of each TACS-like label (``'TACS-1-like'``, …).
    n_points_in_zone : int
    roi_label : str

    Returns an empty dict if *roi* has fewer than 3 vertices.
    """
    coords = roi.coordinates
    if coords is None or len(coords) < 3:
        return {}

    # Prepare dense boundary trace when polynomial-fit tangent is requested
    _dense: Optional[np.ndarray] = None
    if dense_boundary:
        if dense_coords is not None:
            _dense = np.asarray(dense_coords, dtype=float)
        else:
            # roi.coordinates are (x, y); discretize_roi_boundary expects (row, col)
            poly_rc = coords[:, ::-1]
            _dense = discretize_roi_boundary(poly_rc, step=1.0)

    h, w = orientation_map.shape
    points: list[dict] = []
    tacs_dist: Counter = Counter()

    for y in range(0, h, subsample):
        for x in range(0, w, subsample):
            angle = orientation_map[y, x]
            if np.isnan(angle):
                continue

            # Bounding-box fast reject before the expensive per-edge loop
            b = roi.geometry.bounds
            if b is not None:
                margin = tacs_zone_width / pixel_size
                if (x < b[0] - margin or x > b[2] + margin or
                        y < b[1] - margin or y > b[3] + margin):
                    continue

            # Precise distance to the nearest polygon edge
            min_dist_px = np.inf
            n = len(coords)
            for i in range(n):
                j = (i + 1) % n
                ax, ay = coords[i, 0], coords[i, 1]
                bx, by = coords[j, 0], coords[j, 1]
                seg_len = np.hypot(bx - ax, by - ay)
                if seg_len < 1e-9:
                    d = np.hypot(x - ax, y - ay)
                else:
                    t = max(0.0, min(1.0,
                        ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / seg_len ** 2
                    ))
                    d = np.hypot(
                        x - (ax + t * (bx - ax)),
                        y - (ay + t * (by - ay)),
                    )
                if d < min_dist_px:
                    min_dist_px = d

            dist_um = min_dist_px * pixel_size
            if dist_um > tacs_zone_width:
                continue

            # Local boundary tangent → angle_to_tangent
            if dense_boundary and _dense is not None:
                # Polynomial-fit over 21 8-connected neighbours (row, col)
                bidx = find_nearest_boundary_index(_dense, float(y), float(x))
                tangent_angle = compute_boundary_tangent_angle(
                    _dense.astype(int), bidx
                )
                if np.isnan(tangent_angle):
                    continue
                # angle_to_normal via the same circular-stats formula used in
                # compute_relative_fiber_angles (dense path)
                from tme_quant.fiber_analysis.utils.geometry_utils import (
                    _circ_r,
                )
                r = _circ_r([
                    np.radians(2.0 * float(angle)),
                    np.radians(2.0 * tangent_angle),
                ])
                angle_to_normal = float(
                    np.degrees(np.arcsin(np.clip(r, 0.0, 1.0)))
                )
                angle_to_tangent = float(90.0 - angle_to_normal)
            else:
                # Fast 2-point tangent from nearest polygon edge
                pt1, pt2 = nearest_boundary_segment(coords, x, y)
                angle_to_normal = compute_angle_to_boundary_normal(
                    fiber_orientation=angle,
                    boundary_point1=pt1,
                    boundary_point2=pt2,
                )
                if np.isnan(angle_to_normal):
                    continue
                angle_to_tangent = 90.0 - angle_to_normal
            tacs_like = classify_fiber_segment_tacs_like(
                angle_to_tangent=angle_to_tangent,
                distance_to_boundary=dist_um,
                tacs_zone_width=tacs_zone_width,
            )

            rec = {
                'x':                x,
                'y':                y,
                'orientation':      float(angle),
                'alignment':        (float(alignment_map[y, x])
                                     if alignment_map is not None else None),
                'dist_to_boundary': dist_um,
                'angle_to_tangent': angle_to_tangent,
                'tacs_like':        tacs_like,
            }
            points.append(rec)
            if tacs_like:
                tacs_dist[tacs_like] += 1

    if not points:
        return {
            'points':                [],
            'mean_angle_to_tangent': np.nan,
            'std_angle_to_tangent':  np.nan,
            'tacs_distribution':     {},
            'n_points_in_zone':      0,
            'roi_label':             roi.label,
        }

    tangents = [p['angle_to_tangent'] for p in points]

    return {
        'points':                points,
        'mean_angle_to_tangent': float(np.mean(tangents)),
        'std_angle_to_tangent':  float(np.std(tangents)),
        'tacs_distribution':     dict(tacs_dist),
        'n_points_in_zone':      len(points),
        'roi_label':             roi.label,
    }


__all__ = [
    'nearest_boundary_segment',
    'discretize_roi_boundary',
    'compute_orientation_relative_to_roi',
]
