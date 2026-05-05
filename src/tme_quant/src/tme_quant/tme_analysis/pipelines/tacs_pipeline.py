"""
TACS zone analysis pipeline.

Combines the pixel-map orientation analysis (``compute_orientation_relative_to_roi``)
with the per-object fiber analysis (``compute_relative_fiber_angles``) into a
single workflow that runs both paths on the same ROI and merges the results.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from tme_quant.core.roi_manager import ROIObject
from tme_quant.fiber_analysis.utils.geometry_utils import compute_relative_fiber_angles
from tme_quant.fiber_analysis.tacs import classify_fiber_segment_tacs_like
from tme_quant.tme_analysis.utils.orientation_utils import (
    discretize_roi_boundary,
    compute_orientation_relative_to_roi,
)


def _report(cb: Optional[Callable], step: int, total: int, msg: str) -> None:
    if cb is not None:
        cb(step, total, msg)


def analyze_tacs_zone(
    orientation_map:   np.ndarray,
    alignment_map:     Optional[np.ndarray],
    roi:               ROIObject,
    fiber_objects:     Optional[List] = None,
    pixel_size:        float = 1.0,
    tacs_zone_width:   float = 100.0,
    subsample:         int   = 2,
    dense_boundary:    bool  = False,
    image_size:        Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Run both the pixel-map and per-object TACS analyses on the same ROI.

    This function combines:

    * ``compute_orientation_relative_to_roi`` — dense pixel-level analysis of
      an orientation map (CurveAlign / OrientationJ output).
    * ``compute_relative_fiber_angles`` — per-object analysis of individually
      detected fiber objects (CTFire / ridge detector output).

    Either input is optional; pass ``fiber_objects=None`` to skip the
    per-object path, or ``orientation_map`` full of NaN to skip the pixel path.

    Parameters
    ----------
    orientation_map : (H, W) float32
        Per-pixel fiber orientation from CurveAlign or OrientationJ.
        Set to ``np.full((H, W), np.nan)`` to skip the pixel-map path.
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
        The tumor boundary ROI.
    fiber_objects : list of FiberObject or None
        Individually detected fibers.  Each must have ``.angle``,
        ``center_point`` (or ``centerline``), and optionally ``.orientation``.
        Pass None to skip the per-object path.
    pixel_size : float
        Microns per pixel.
    tacs_zone_width : float
        TACS zone half-width in µm.  Pixels / objects beyond this distance
        from the boundary are excluded.
    subsample : int
        Pixel-map subsampling factor (see ``compute_orientation_relative_to_roi``).
    dense_boundary : bool
        When True both the pixel-map and per-object analyses use the
        polynomial-fit tangent direction.  The polygon ROI is discretized
        once via ``discretize_roi_boundary`` and shared between both calls.
    image_size : (H, W) or None
        Passed to ``compute_relative_fiber_angles`` to enable regionprops-based
        ROI orientation.  Set to the image dimensions when available.

    Returns
    -------
    dict with keys:

    pixel_result : dict
        Full output of ``compute_orientation_relative_to_roi``.
    fiber_results : list[dict]
        One entry per fiber object in the TACS zone, each containing:
        ``fiber_id``, ``angle``, ``dist_to_boundary``,
        ``angle_to_boundary_tangent``, ``angle_to_roi_orientation``,
        ``angle_to_centers_line``, ``tacs_like``.
    combined_mean_angle_to_tangent : float
        Mean angle_to_tangent pooled across pixel + fiber paths
        (NaN if neither produced results).
    dense_boundary_used : bool
    roi_label : str

    Notes
    -----
    Workflow diagram::

        orientation_map ──► compute_orientation_relative_to_roi ──► pixel_result
                                         │
                                 (dense_boundary=True)
                                         │
        polygon ROI ──► discretize_roi_boundary ──► dense_coords
                                         │
        fiber_objects ──► compute_relative_fiber_angles ──► fiber_results

    Example
    -------
    >>> result = analyze_tacs_zone(
    ...     orientation_map = curvealign_angles,    # (H, W) float32
    ...     alignment_map   = orientationj_coherency, # (H, W) from OrientationJ, or None
    ...     roi             = tumor_roi,
    ...     fiber_objects   = ctfire_fibers,        # list[FiberObject] or None
    ...     pixel_size      = 0.5,
    ...     tacs_zone_width = 100.0,
    ...     subsample       = 2,
    ...     dense_boundary  = True,   # more accurate tangents
    ...     image_size      = orientation_map.shape,
    ... )
    >>> print(result['pixel_result']['mean_angle_to_tangent'])
    >>> print(result['combined_mean_angle_to_tangent'])
    """
    # ── 1. Discretize polygon once if dense_boundary is requested ─────────
    _report(progress_callback, 1, 3, "Discretizing ROI boundary…")
    dense_coords: Optional[np.ndarray] = None
    if dense_boundary and roi.coordinates is not None and len(roi.coordinates) >= 3:
        poly_rc = roi.coordinates[:, ::-1]   # (x,y) → (row,col)
        dense_coords = discretize_roi_boundary(poly_rc, step=1.0)

    # ── 2. Pixel-map analysis ─────────────────────────────────────────────
    _report(progress_callback, 2, 3, "Computing pixel-level orientation map…")
    pixel_result = compute_orientation_relative_to_roi(
        orientation_map = orientation_map,
        alignment_map   = alignment_map,
        roi             = roi,
        pixel_size      = pixel_size,
        tacs_zone_width = tacs_zone_width,
        subsample       = subsample,
        dense_boundary  = dense_boundary,
        dense_coords    = dense_coords,
    )

    # ── 3. Per-object fiber analysis ──────────────────────────────────────
    n_fibers = len(fiber_objects) if fiber_objects else 0
    _report(progress_callback, 3, 3, f"Classifying {n_fibers} fibers in TACS zone…")
    fiber_results: list[dict] = []
    if fiber_objects and roi.coordinates is not None and len(roi.coordinates) >= 3:
        # roi_coords for compute_relative_fiber_angles must be (row, col)
        roi_rc = (dense_coords
                  if dense_coords is not None
                  else roi.coordinates[:, ::-1])

        for fib in fiber_objects:
            # Resolve fiber center: prefer center_point, fall back to centerline
            center = getattr(fib, 'center_point', None)
            if center is None:
                cl = getattr(fib, 'centerline', None)
                if cl is not None and len(cl) >= 1:
                    center = cl[len(cl) // 2]
            if center is None:
                continue   # cannot locate fiber — skip

            fiber_row, fiber_col = float(center[0]), float(center[1])

            # ── Inside-ROI guard ───────────────────────────────────────────
            # Fibers whose center lies *inside* the ROI are excluded: they
            # are not part of the peri-tumoral stroma and their distance to
            # the nearest boundary edge would be near-zero, making the
            # angle-to-tangent geometry unreliable.
            # ROIObject.contains_point expects (x, y) == (col, row).
            if roi.contains_point(fiber_col, fiber_row):
                continue

            fiber_angle = float(getattr(fib, 'angle', 0.0)) % 180

            # ── Fast distance pre-filter (pixels → µm) ────────────────────
            # Avoid the more expensive compute_relative_fiber_angles call for
            # fibers that are clearly outside the TACS zone.
            _dists  = np.linalg.norm(
                roi_rc - np.array([fiber_row, fiber_col]),
                axis=1,
            )
            _argmin = int(np.argmin(_dists))
            dist_px = float(_dists[_argmin])
            nearest_bp = roi_rc[_argmin]          # (row, col)
            dist_um = dist_px * pixel_size
            if dist_um > tacs_zone_width:
                continue

            rel_angles, roi_meas = compute_relative_fiber_angles(
                obj_center     = (fiber_row, fiber_col),
                obj_angle      = fiber_angle,
                roi_coords     = roi_rc,
                image_size     = image_size,
                dense_boundary = dense_boundary,
            )

            ang_tan = rel_angles.get('angle_to_boundary_tangent')
            if ang_tan is None:
                continue

            tacs_like = classify_fiber_segment_tacs_like(
                angle_to_tangent    = ang_tan,
                distance_to_boundary= dist_um,
                tacs_zone_width     = tacs_zone_width,
            )

            fiber_results.append({
                'fiber_id':                getattr(fib, 'object_id', None),
                'center_point':            (fiber_row, fiber_col),
                'angle':                   fiber_angle,
                'dist_to_boundary':        dist_um,
                'nearest_boundary_point':  (float(nearest_bp[0]), float(nearest_bp[1])),
                'angle_to_boundary_tangent': ang_tan,
                'angle_to_roi_orientation':  rel_angles.get('angle_to_roi_orientation'),
                'angle_to_centers_line':     rel_angles.get('angle_to_centers_line'),
                'tacs_like':               tacs_like,
            })

    # ── 4. Pool mean angle_to_tangent across both paths ───────────────────
    all_tangents: list[float] = []
    if pixel_result.get('points'):
        all_tangents.extend(p['angle_to_tangent'] for p in pixel_result['points'])
    all_tangents.extend(f['angle_to_boundary_tangent'] for f in fiber_results)
    combined_mean = float(np.mean(all_tangents)) if all_tangents else float('nan')

    return {
        'pixel_result':                  pixel_result,
        'fiber_results':                 fiber_results,
        'combined_mean_angle_to_tangent': combined_mean,
        'dense_boundary_used':           dense_boundary,
        'roi_label':                     roi.label,
    }


__all__ = ['analyze_tacs_zone']
