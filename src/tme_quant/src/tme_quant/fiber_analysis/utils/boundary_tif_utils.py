# -*- coding: utf-8 -*-
"""
Boundary-TIF curvelet/fiber measurement utilities.

Adapted from pycurvelets ``get_tif_boundary`` (MATLAB conversion).

Associates boundary coordinate traces with curvelet/fiber candidates and
computes per-fiber boundary-relative measurements:

- **nearest boundary distance** — Euclidean distance to the closest boundary pixel.
- **nearest region distance** — whether the fiber centre falls inside the ROI mask.
- **nearest boundary angle** — alignment of fiber to boundary tangent, expressed in
  pycurvelets circular-statistics convention (≈ 90° − |fiber_angle − tangent_angle|).
  *This is NOT the TACS* ``angle_to_boundary_tangent`` *convention.*  Convert via
  ``angle_to_boundary_tangent = 90 − nearest_boundary_angle`` before passing to
  TACS classification.
- **extension point distance / angle** — always NaN; incomplete in the original
  pycurvelets implementation (the computed values were never stored).  Preserved
  for column-name parity.

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .geometry_utils import _circ_r, compute_boundary_tangent_angle
from .fiber_dataframe_utils import round_mlab


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize_line_segment(point_1, point_2):
    """Rasterize the straight-line segment between two 2-D points.

    Port of pycurvelets ``get_segment_pixels``.  Uses a fractional-step
    accumulator (not Bresenham's integer algorithm) to match the MATLAB
    reference implementation and preserve numerical parity.

    Parameters
    ----------
    point_1, point_2 : array_like of shape (2,)
        Endpoints as ``(row, col)`` coordinates.

    Returns
    -------
    (segment_points, absolute_angle) : (list of [int, int], float)
        *segment_points* — integer pixel coordinates along the segment.
        *absolute_angle* — angle of the segment in radians (range −π/2 to π/2).
        Returns ``None`` when ``point_1 == point_2`` (identical endpoints).

    Examples
    --------
    >>> _rasterize_line_segment([2, 3], [6, 8])[0]
    [[2, 3], [3, 4], [4, 5], [5, 7], [6, 8]]
    """
    absolute_angle = np.nan

    point_1 = round_mlab(point_1)
    point_2 = round_mlab(point_2)

    rise = point_2[1] - point_1[1]
    run = point_2[0] - point_1[0]

    maxrr = max(abs(rise), abs(run))
    if maxrr == 0:
        return None

    if run == 0:
        absolute_angle = np.pi / 2 if rise > 0 else -np.pi / 2
    else:
        absolute_angle = np.arctan(rise / run)

    fraction_rise = 0.5 * rise / maxrr
    fraction_run = 0.5 * run / maxrr

    spt = point_1
    x = spt[0]
    y = spt[1]
    segment_points = [spt]

    while True:
        if spt[0] == point_2[0] and spt[1] == point_2[1]:
            break
        y = y + fraction_rise
        x = x + fraction_run
        round_y = round_mlab(y)
        round_x = round_mlab(x)
        if round_x != spt[0] or round_y != spt[1]:
            spt = [round_x, round_y]
            segment_points.append(spt)

    return segment_points, absolute_angle


def _get_fiber_line_points(center, angle, box_size):
    """Enumerate pixel coordinates along the fiber orientation axis.

    Port of pycurvelets ``get_points_on_line``.  Extends the fiber from its
    centre point by ``box_size`` pixels in both directions along the fiber
    angle, then rasterizes the resulting segment.

    Parameters
    ----------
    center : sequence of float
        Fiber centre as ``(row, col)``.
    angle : float
        Fiber orientation in degrees.
    box_size : float
        Half-length of the extension in pixels.

    Returns
    -------
    (line_points, ortho_points) : (list of [int, int], list)
        *line_points* — rasterized pixel coordinates along the fiber axis.
        *ortho_points* — always empty (orthogonal extension not implemented).
    """
    slope = -np.tan(np.deg2rad(angle))

    if np.isinf(slope):
        dist_y = 0
        dist_x = box_size
    else:
        dist_y = box_size / np.sqrt(1.0 + slope * slope)
        dist_x = dist_y * slope

    p1 = [center[0] - dist_x, center[1] - dist_y]
    p2 = [center[0] + dist_x, center[1] + dist_y]

    result = _rasterize_line_segment(p1, p2)
    line_curv = result[0] if result is not None else []
    return line_curv, []


def _compute_fiber_boundary_relative_angle(
    coords: np.ndarray,
    idx: int,
    fiber_angle: float,
    img_height: int,
    img_width: int,
) -> Tuple[float, np.ndarray]:
    """Compute the circular-mean alignment angle between a fiber and boundary tangent.

    Port of pycurvelets ``get_relative_angle``.  Uses
    ``compute_boundary_tangent_angle`` (port of ``find_outline_slope``) and
    ``_circ_r`` (port of ``circ_r``).

    **Angle convention (pycurvelets):**  The returned angle is
    ``arcsin(circ_r([2·fiber_angle_rad, 2·boundary_angle_rad]))`` in degrees,
    which equals ``90° − |fiber_angle − boundary_tangent_angle|``.  This is the
    *complement* of the TACS ``angle_to_boundary_tangent`` convention.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        Dense pixel-level boundary coordinates ``(row, col)``.
    idx : int
        Index of the nearest boundary point for this fiber.
    fiber_angle : float
        Fiber orientation in degrees.
    img_height, img_width : int
        Image dimensions; boundary points on the image edge are excluded.

    Returns
    -------
    (angle_degrees, boundary_point) : (float, ndarray of shape (2,))
        *angle_degrees* — 0 when the boundary point is on the image edge;
        NaN when ``compute_boundary_tangent_angle`` returns NaN.
        *boundary_point* — the ``coords[idx]`` row/col pair.
    """
    boundary_angle = compute_boundary_tangent_angle(coords, idx)
    boundary_point = coords[int(idx), :]

    if (
        boundary_point[0] == 0
        or boundary_point[1] == 0
        or boundary_point[0] == img_height - 1
        or boundary_point[1] == img_width - 1
    ):
        return 0.0, boundary_point

    temp_angle = _circ_r(
        [np.radians(2.0 * fiber_angle), np.radians(2.0 * boundary_angle)]
    )
    temp_angle = float(np.degrees(np.arcsin(temp_angle)))
    return temp_angle, boundary_point


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_tif_boundary(
    coordinates,
    img: np.ndarray,
    fiber_df: pd.DataFrame,
    dist_thresh: float,
    min_dist,
) -> Tuple[np.ndarray, list, int, pd.DataFrame]:
    """Associate boundary coordinates with fibers and compute boundary-relative metrics.

    Port of pycurvelets ``get_tif_boundary``.  For every curvelet/fiber candidate
    in *fiber_df* this function computes:

    1. **nearest boundary distance** — Euclidean distance to the closest boundary pixel.
    2. **nearest region distance** — 1 if the fiber centre is inside the ROI (pixel
       value 255 or 1 in *img*), 0 otherwise.
    3. **nearest boundary angle** — alignment to boundary tangent in pycurvelets
       convention: ``arcsin(circ_r([2·α_fiber, 2·α_tangent]))`` in degrees
       (≈ 90° − |α_fiber − α_tangent|).  Only computed for fibers within *dist_thresh*.
    4. **extension point distance / angle** — always NaN; the computation in the
       original pycurvelets function was incomplete (values computed but never stored).
       Preserved for output-column parity.

    It also computes *num_img_points* — the number of image pixels that fall within
    *dist_thresh* of the boundary (after subsampling by the step size).

    Parameters
    ----------
    coordinates : dict or ndarray
        Boundary pixel coordinates.  Either a ``dict`` mapping arbitrary keys to
        ``(N_i, 2)`` arrays (one per boundary segment), which are stacked internally,
        or a pre-stacked ``(N, 2)`` ndarray.  Each row is ``(row, col)``.
    img : ndarray of shape (H, W)
        Image used for region membership test.  Pixels with value 255 or 1 are
        considered inside the ROI.
    fiber_df : pd.DataFrame
        One row per fiber.  Required columns:

        - ``angle``       — fiber orientation in degrees.
        - ``center_row`` / ``center_col``  or  ``center_1`` / ``center_2``
          — fiber centre coordinates.

    dist_thresh : float
        Pixels within this distance of the boundary are evaluated.  Fibers
        farther than *dist_thresh* receive NaN for ``nearest_boundary_angle``.
    min_dist : scalar or sequence
        When falsy (``[]``, ``0``, ``None``, ``False``), only *dist_thresh* is
        applied (main code path).  When truthy, inner threshold filtering is
        requested but not implemented in the original — all metric arrays
        remain NaN.  Preserved for signature parity.

    Returns
    -------
    result_mat : ndarray of shape (n_fibers, 7)
        Numeric results.  Column order matches *result_mat_names*.
    result_mat_names : list of str
        Column labels for *result_mat*::

            ["nearest_boundary_distance",
             "nearest_region_distance",
             "nearest_boundary_angle",
             "extension_point_distance",   # always NaN
             "extension_point_angle",       # always NaN
             "boundary_point_row",
             "boundary_point_col"]

    num_img_points : int
        Estimated number of image pixels within *dist_thresh* of the boundary
        (subsampled count × step size).
    result_df : pd.DataFrame
        Same data as *result_mat* with named columns.
    """
    img_height, img_width = img.shape[:2]

    # Normalise fiber centre column names
    df = fiber_df
    if "center_row" not in df.columns:
        if "center_1" in df.columns and "center_2" in df.columns:
            df = df.rename(columns={"center_1": "center_row", "center_2": "center_col"})
        else:
            raise ValueError(
                "fiber_df must have 'center_row'/'center_col' or 'center_1'/'center_2'."
            )

    all_center_points = (
        np.round(df[["center_row", "center_col"]].values).astype(int)
    )

    # Stack boundary segments into a single coordinate array
    if isinstance(coordinates, np.ndarray):
        coords = coordinates
    else:
        coords = np.vstack([coordinates[k] for k in coordinates])
    coords = np.asarray(coords, dtype=float)

    # For each fiber, find the nearest boundary point
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(coords)
    distances, indices = neighbors.kneighbors(all_center_points)
    idx_dist = indices.flatten()
    dist = distances.flatten()

    # Region membership via pixel lookup
    center_rows = np.clip(all_center_points[:, 0], 0, img_height - 1)
    center_cols = np.clip(all_center_points[:, 1], 0, img_width - 1)
    reg_dist = img[center_rows, center_cols]

    # Subsample image points for num_img_points estimation
    step_size = img_width // 20
    linear_indices = np.arange(0, img_height * img_width, step_size)
    # NOTE: (img_width, img_height) ordering replicates MATLAB column-major sub2ind
    cols_unravel, rows_unravel = np.unravel_index(linear_indices, (img_width, img_height))
    all_img_points = np.column_stack((rows_unravel, cols_unravel))

    sorted_coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]
    subsampled_boundary = sorted_coords[::3, :]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(
        subsampled_boundary
    )
    dist_to_boundary, _ = nbrs.kneighbors(all_img_points)
    dist_to_boundary = dist_to_boundary.flatten()

    if not min_dist:
        in_mask = dist_to_boundary <= (dist_thresh + 1e-12)
    else:
        in_mask = (dist_to_boundary <= dist_thresh) & (dist_to_boundary > min_dist)

    in_points = all_img_points[in_mask]
    num_img_points = len(in_points) * step_size

    # Per-fiber metrics
    n = len(df)
    nearest_boundary_dist = np.full(n, np.nan)
    nearest_region_dist = np.full(n, np.nan)
    nearest_boundary_angle = np.full(n, np.nan)
    extension_point_dist = np.full(n, np.nan)       # always NaN (see module docstring)
    extension_point_angle = np.full(n, np.nan)      # always NaN (see module docstring)
    measurement_boundary = np.full((n, 2), np.nan)

    if not min_dist:
        for i in range(n):
            nearest_region_dist[i] = int((reg_dist[i] == 255) | (reg_dist[i] == 1))
            nearest_boundary_dist[i] = dist[i]

            if dist[i] <= dist_thresh:
                angle_val, bnd_pt = _compute_fiber_boundary_relative_angle(
                    coords, idx_dist[i], df.iloc[i]["angle"], img_height, img_width
                )
                nearest_boundary_angle[i] = angle_val
            else:
                bnd_pt = np.full(2, np.nan)

            # Extension-point intersection (fiber axis vs boundary).
            # NOTE: extension_point_dist[i] and extension_point_angle[i] are computed
            # locally below but are *never assigned* back to the output arrays.  This
            # replicates the original pycurvelets behaviour (incomplete implementation);
            # both columns are always NaN.  See module docstring.
            fiber_center = (df.iloc[i]["center_row"], df.iloc[i]["center_col"])
            line_curvelets, _ = _get_fiber_line_points(fiber_center, df.iloc[i]["angle"], dist_thresh)
            line_curvelets = np.array(line_curvelets)

            if line_curvelets.ndim == 2 and line_curvelets.shape[0] > 0:
                a = line_curvelets
                b = coords
                a_view = a.view([("", a.dtype)] * a.shape[1])
                b_view = b.view([("", b.dtype)] * b.shape[1])
                intersection_line, inter_a, _ = np.intersect1d(
                    a_view, b_view, return_indices=True
                )
                intersection_line = a[inter_a]

                if intersection_line.size != 0:
                    nbrs_inter = NearestNeighbors(
                        n_neighbors=1, algorithm="brute", metric="euclidean"
                    ).fit(intersection_line)
                    line_distance, idx_line_distance = nbrs_inter.kneighbors([fiber_center])
                    # NOTE: line_distance and idx_line_distance are intentionally not
                    # stored to extension_point_dist[i] / extension_point_angle[i].
                    # This matches the original pycurvelets bug; the values remain NaN.
                else:
                    extension_point_dist[i] = np.nan
                    extension_point_angle[i] = np.nan
            else:
                extension_point_dist[i] = np.nan
                extension_point_angle[i] = np.nan

            # Boundary coords are in [col, row] (MATLAB x,y) order; swap to [row, col]
            # so that result columns boundary_point_row/col are correctly labeled.
            measurement_boundary[i] = bnd_pt[[1, 0]] if not np.any(np.isnan(bnd_pt)) else bnd_pt

    result_mat = np.column_stack([
        nearest_boundary_dist,
        nearest_region_dist,
        nearest_boundary_angle,
        extension_point_dist,
        extension_point_angle,
        measurement_boundary,
    ])

    result_mat_names = [
        "nearest_boundary_distance",
        "nearest_region_distance",
        "nearest_boundary_angle",
        "extension_point_distance",
        "extension_point_angle",
        "boundary_point_row",
        "boundary_point_col",
    ]

    result_df = pd.DataFrame(result_mat, columns=result_mat_names)
    return result_mat, result_mat_names, num_img_points, result_df


__all__ = ["extract_tif_boundary"]
