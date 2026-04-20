# -*- coding: utf-8 -*-
"""
Fiber-to-ROI alignment utilities.

Adapted from pycurvelets ``get_alignment_to_roi`` (MATLAB conversion).

**API differences from pycurvelets:**

- ``ROIList`` (pycurvelets dataclass) is replaced by plain NumPy arrays +
  explicit ``img_height`` / ``img_width`` arguments, removing the pycurvelets
  dependency.  Pass ``roi_coords`` as a single (N, 2) boundary array in
  **(row, col)** order (skimage convention).

- Column names follow tme_quant conventions (see REFACTORING_GUIDE.md §6):
    * ``angle_to_boundary_edge``   → ``angle_to_boundary_tangent``
    * ``angle_to_boundary_center`` → ``angle_to_roi_orientation``
    * ``angle_to_center_line``     → ``angle_to_centers_line``

- Angle convention:
    * pycurvelets ``angle_to_boundary_edge`` = arcsin(circ_r([2α, 2β]))
      = 90° when fiber is parallel to boundary (high alignment).
    * tme_quant  ``angle_to_boundary_tangent`` = 90° − (above)
      = 0° when fiber is parallel to boundary (TACS-2 pattern).

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from tme_quant.fiber_analysis.utils.geometry_utils import (
    _circ_r,
    compute_boundary_tangent_angle,
    find_nearest_boundary_index,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_fiber_alignment_to_roi(
    roi_coords: np.ndarray,
    img_height: int,
    img_width: int,
    fiber_structure: pd.DataFrame,
    distance_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate per-fiber alignment measurements relative to an ROI boundary.

    Adapted from pycurvelets ``get_alignment_to_roi``.  Computes three
    orientation angles and basic spatial metadata for every fibre that lies
    within *distance_threshold* of the ROI boundary.

    Parameters
    ----------
    roi_coords : (N, 2) ndarray
        ROI boundary coordinates in **(row, col)** order (skimage convention).
        Must have ≥ 3 points and form a closed polygon.
    img_height : int
        Image height in pixels.  Used to detect boundary points that touch
        image edges (edge-touching points receive ``angle_to_boundary_tangent = 0``
        following the original MATLAB behaviour).
    img_width : int
        Image width in pixels.
    fiber_structure : pd.DataFrame
        One row per fibre.  Required columns:

        * ``center_row``  — row coordinate of the fibre centre (y, pixels).
        * ``center_col``  — column coordinate of the fibre centre (x, pixels).
        * ``angle``       — fibre orientation in degrees [0°, 180°).

        Column aliases ``center_1`` / ``center_2`` (pycurvelets convention)
        are accepted and internally mapped.

    distance_threshold : float or None
        Maximum distance (pixels) from a fibre centre to the ROI boundary for
        inclusion in the output.  When ``None`` all fibres are included
        (pre-selected mode — mirrors pycurvelets behaviour when no threshold
        is given).

    Returns
    -------
    result_df : pd.DataFrame
        One row per included fibre with columns:

        * ``angle_to_boundary_tangent`` — acute angle [0°, 90°] between the
          fibre and the local boundary tangent.  0° = parallel (TACS-2);
          90° = perpendicular (TACS-3).
          pycurvelets name: ``angle_to_boundary_edge`` (90° − this value).
        * ``angle_to_roi_orientation``  — acute angle [0°, 90°] between the
          fibre orientation and the ROI's global orientation axis.
          pycurvelets name: ``angle_to_boundary_center``.
        * ``angle_to_centers_line``     — acute angle [0°, 90°] between the
          fibre and the line connecting the fibre centre to the ROI centroid.
          pycurvelets name: ``angle_to_center_line``.
        * ``fiber_center_row``          — fibre centre row (y).
        * ``fiber_center_col``          — fibre centre column (x).
        * ``fiber_angle``               — fibre orientation angle in degrees.
        * ``distance``                  — distance to ROI boundary in pixels
          (``None`` in pre-selected mode).
        * ``boundary_point_row``        — row of nearest boundary point.
        * ``boundary_point_col``        — column of nearest boundary point.

    fiber_count : int
        Number of fibres included in *result_df*.

    Raises
    ------
    ValueError
        If *roi_coords* is ``None``, has fewer than 3 points, or
        *fiber_structure* is empty.

    Notes
    -----
    **Dense boundary assumption:** ``angle_to_boundary_tangent`` is computed
    via ``compute_boundary_tangent_angle``, which requires *roi_coords* to be
    a **dense 8-connected pixel trace** (e.g. CurveAlign boundary output).
    If *roi_coords* is a sparse polygon (a handful of vertices), the
    8-connected neighbour search will fail to collect ``num=21`` points and
    ``angle_to_boundary_tangent`` will be ``None`` for every fibre.
    For sparse polygon ROIs use ``compute_relative_fiber_angles`` with
    ``dense_boundary=False`` instead.
    """
    if roi_coords is None or len(roi_coords) < 3:
        raise ValueError("roi_coords must be a valid boundary array with ≥ 3 points.")

    if fiber_structure is None or len(fiber_structure) == 0:
        raise ValueError("fiber_structure cannot be None or empty.")

    df = fiber_structure.copy()

    # Accept pycurvelets column aliases
    if "center_row" not in df.columns:
        if "center_1" in df.columns and "center_2" in df.columns:
            df = df.rename(columns={"center_1": "center_row", "center_2": "center_col"})
        else:
            raise ValueError(
                "fiber_structure must contain 'center_row'/'center_col' "
                "or 'center_1'/'center_2' columns."
            )

    coords = np.asarray(roi_coords, dtype=float)
    fiber_centers = df[["center_row", "center_col"]].to_numpy(dtype=np.float64)
    fiber_angles  = df["angle"].to_numpy(dtype=np.float64)

    select_fiber_flag = distance_threshold is not None

    # ── Fibre selection ───────────────────────────────────────────────────
    if select_fiber_flag:
        roi_tree = KDTree(coords)
        dist_arr, idx_arr = roi_tree.query(fiber_centers)     # (n_fibers, 1) each
        dist_arr = dist_arr[:, 0]
        idx_arr  = idx_arr[:, 0]
        fiber_indices = np.where(dist_arr <= distance_threshold)[0]
        logger.info(
            "compute_fiber_alignment_to_roi: %d fibres within %.1f px of boundary.",
            len(fiber_indices), distance_threshold,
        )
    else:
        fiber_indices = np.arange(len(df))
        dist_arr = np.full(len(df), np.nan)
        idx_arr  = np.zeros(len(df), dtype=int)
        logger.info(
            "compute_fiber_alignment_to_roi: pre-selected mode, %d fibres.",
            len(fiber_indices),
        )

    if len(fiber_indices) == 0:
        empty = pd.DataFrame(columns=[
            "angle_to_boundary_tangent", "angle_to_roi_orientation",
            "angle_to_centers_line", "fiber_center_row", "fiber_center_col",
            "fiber_angle", "distance", "boundary_point_row", "boundary_point_col",
        ])
        return empty, 0

    # ── ROI global properties (orientation, centroid) — computed once ─────
    from skimage.draw import polygon2mask
    from skimage.measure import regionprops, label as ski_label

    mask   = polygon2mask((img_height, img_width), coords)
    lbl    = ski_label(mask.astype(np.uint8))
    props  = regionprops(lbl)

    if len(props) != 1:
        raise ValueError(
            f"roi_coords must define exactly one connected region; "
            f"got {len(props)} regions."
        )

    prop        = props[0]
    roi_centroid_yx = np.array(prop.centroid)     # (row, col) = (y, x)
    roi_cx, roi_cy  = float(roi_centroid_yx[1]), float(roi_centroid_yx[0])   # (x, y)
    roi_angle   = float(-90.0 + np.degrees(prop.orientation))
    if roi_angle < 0.0:
        roi_angle += 180.0

    # ── Per-fibre angle computations ──────────────────────────────────────
    sel_centers = fiber_centers[fiber_indices]   # (n_sel, 2) in (row, col)
    sel_angles  = fiber_angles[fiber_indices]

    # angle_to_roi_orientation — vectorised (same formula as pycurvelets)
    angle_diffs = np.abs(sel_angles - roi_angle)
    angle_to_roi_orientation = np.where(
        angle_diffs > 90.0, 180.0 - angle_diffs, angle_diffs
    )

    # angle_to_centers_line — vectorised (same formula as pycurvelets)
    # Note: pycurvelets uses dx = col_fiber − col_roi; dy = row_fiber − row_roi
    dx = sel_centers[:, 1] - roi_cx   # Δcol (= Δx)
    dy = sel_centers[:, 0] - roi_cy   # Δrow (= Δy)
    cl_angles = np.degrees(np.arctan2(dx, dy))
    cl_angles  = np.where(cl_angles < 0, np.abs(cl_angles), 180.0 - cl_angles)
    angle_to_centers_line = np.abs(cl_angles - sel_angles)
    angle_to_centers_line = np.where(
        angle_to_centers_line > 90.0,
        180.0 - angle_to_centers_line,
        angle_to_centers_line,
    )

    # angle_to_boundary_tangent — per-fibre (needs local tangent at nearest boundary pt)
    angles_to_tangent = []
    boundary_point_rows = []
    boundary_point_cols = []

    for k, fi in enumerate(fiber_indices):
        if select_fiber_flag:
            boundary_idx = int(idx_arr[fi])
        else:
            boundary_idx = find_nearest_boundary_index(
                coords, float(fiber_centers[fi, 0]), float(fiber_centers[fi, 1])
            )

        bp = coords[boundary_idx]   # (row, col)
        boundary_point_rows.append(float(bp[0]))
        boundary_point_cols.append(float(bp[1]))

        # Image-edge guard (mirrors pycurvelets get_alignment_to_roi)
        on_edge = (
            bp[0] <= 1 or bp[1] <= 1
            or bp[0] >= img_height or bp[1] >= img_width
        )
        if on_edge:
            angles_to_tangent.append(0.0)
            continue

        tangent_angle = compute_boundary_tangent_angle(
            coords.astype(int), boundary_idx
        )
        if tangent_angle is None or np.isnan(tangent_angle):
            angles_to_tangent.append(None)
            continue

        # Compute angle_to_boundary_tangent via circ_r (same formula as pycurvelets)
        # pycurvelets stores arcsin(circ_r) as "angle_to_boundary_edge"
        # tme_quant convention: angle_to_boundary_tangent = 90° − arcsin(circ_r)
        r = _circ_r(np.array([
            np.radians(2.0 * sel_angles[k]),
            np.radians(2.0 * tangent_angle),
        ]))
        angle_to_normal = float(np.degrees(np.arcsin(np.clip(r, 0.0, 1.0))))
        angles_to_tangent.append(90.0 - angle_to_normal)

    # ── Assemble output DataFrame ─────────────────────────────────────────
    records = []
    for k, fi in enumerate(fiber_indices):
        records.append({
            "angle_to_boundary_tangent": angles_to_tangent[k],
            "angle_to_roi_orientation":  float(angle_to_roi_orientation[k]),
            "angle_to_centers_line":     float(angle_to_centers_line[k]),
            "fiber_center_row":          float(sel_centers[k, 0]),
            "fiber_center_col":          float(sel_centers[k, 1]),
            "fiber_angle":               float(sel_angles[k]),
            "distance":                  float(dist_arr[fi]) if select_fiber_flag else None,
            "boundary_point_row":        boundary_point_rows[k],
            "boundary_point_col":        boundary_point_cols[k],
        })

    result_df = pd.DataFrame(records)
    return result_df, len(result_df)


__all__ = ["compute_fiber_alignment_to_roi"]
