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
compute_orientation_relative_to_roi
    Main entry point: iterate every valid pixel in an orientation map, filter
    to the TACS zone around *roi*, and return per-pixel angle statistics.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import numpy as np

from tme_quant.core.roi_manager import ROIObject
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
)
from tme_quant.fiber_analysis.tacs import classify_fiber_segment_tacs_like


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
        Per-pixel alignment / coherency strength [0, 1].  May be None.
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

    Returns
    -------
    dict with keys:

    points : list[dict]
        One record per included pixel, each containing:
        ``x``, ``y``, ``orientation``, ``alignment``, ``dist_to_boundary``,
        ``angle_to_normal``, ``angle_to_tangent``, ``tacs_like``.
    mean_angle_to_tangent : float
    mean_angle_to_normal : float
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

            # Local boundary tangent at the nearest edge → angle to normal
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
                'angle_to_normal':  angle_to_normal,
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
            'mean_angle_to_normal':  np.nan,
            'tacs_distribution':     {},
            'n_points_in_zone':      0,
            'roi_label':             roi.label,
        }

    tangents = [p['angle_to_tangent'] for p in points]
    normals  = [p['angle_to_normal']  for p in points]

    return {
        'points':                points,
        'mean_angle_to_tangent': float(np.mean(tangents)),
        'mean_angle_to_normal':  float(np.mean(normals)),
        'std_angle_to_tangent':  float(np.std(tangents)),
        'tacs_distribution':     dict(tacs_dist),
        'n_points_in_zone':      len(points),
        'roi_label':             roi.label,
    }


__all__ = [
    'nearest_boundary_segment',
    'compute_orientation_relative_to_roi',
]
