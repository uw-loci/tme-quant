# -*- coding: utf-8 -*-
"""
CurveAlign curvelets-mode fiber-orientation pipeline.

Uses grouped curvelet orientation estimates (via ``build_fiber_structure_from_curvelets``)
as the fiber representation.  Each position in the output ``fiber_structure`` represents
the dominant orientation of a curvelet-grouped local region — not an individually
extracted fiber.

Port of the orchestration logic from pycurvelets ``process_image.py``.
All analysis sub-functions are already ported; this module assembles them
into a clean in-memory pipeline that returns a result dict (no file I/O).

When CT-FIRE-based individual fiber extraction becomes available, it will be
exposed via a separate ``curvealign_ctfireMode_pipeline`` module.

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from skimage.draw import polygon2mask
from skimage.measure import label, regionprops

from tme_quant.fiber_analysis.config import FiberFeatureParams
from tme_quant.fiber_analysis.utils.boundary_tif_utils import (
    extract_boundary_coords_from_mask,
    extract_tif_boundary,
)
from tme_quant.fiber_analysis.utils.fiber_dataframe_utils import (
    build_fiber_structure_from_curvelets,
)
from tme_quant.tme_analysis.utils.alignment_utils import compute_fiber_alignment_to_roi


def _report(cb: Optional[Callable], step: int, total: int, msg: str) -> None:
    if cb is not None:
        cb(step, total, msg)


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class CurveAlignPipelineResult:
    """Typed return value of :func:`curvealign_curvelets_mode_pipeline`.

    Replaces the raw ``dict`` previously returned so callers can access fields
    by attribute rather than by string key, and so the plugin can import a
    concrete type instead of unwrapping a dict.

    All attribute names match the former dict keys exactly, preserving
    backward-compatible access patterns (``result.fiber_structure`` replaces
    ``result["fiber_structure"]``).

    Attributes
    ----------
    fiber_structure : pd.DataFrame
        Per-fiber curvelet orientation DataFrame with columns
        ``center_row``, ``center_col``, ``angle``, ``weight``.
    fiber_features_df : pd.DataFrame
        Consolidated per-fiber feature table (position, angle, density,
        alignment, boundary metrics when available).
    density_df : pd.DataFrame
        Density statistics computed by ``build_fiber_structure_from_curvelets``.
    alignment_df : pd.DataFrame
        Alignment statistics computed by ``build_fiber_structure_from_curvelets``.
    roi_measurements_df : pd.DataFrame or None
        Per-fiber angle measurements per ROI (``None`` when no boundary).
    roi_summary_df : pd.DataFrame or None
        Per-ROI summary statistics (``None`` when no boundary).
    in_curvs_flag : ndarray[bool] or None
        Boolean mask selecting fibers within the TACS boundary zone.
        ``None`` when no boundary analysis was performed.
    nearest_angles : ndarray or None
        Per-fiber angle relative to the nearest boundary tangent.
        ``None`` when no boundary analysis was performed.
    boundary_measurement : bool
        ``True`` when boundary analysis was performed; ``False`` otherwise.
    params : dict
        Serialised pipeline parameters (image shape, keep, scale, radius,
        distance_threshold, tif_boundary, exclude_fibers_in_mask, min_dist).
    """
    fiber_structure:    pd.DataFrame
    fiber_features_df:  pd.DataFrame
    density_df:         pd.DataFrame
    alignment_df:       pd.DataFrame
    roi_measurements_df: Optional[pd.DataFrame]
    roi_summary_df:     Optional[pd.DataFrame]
    in_curvs_flag:      Optional[np.ndarray]
    nearest_angles:     Optional[np.ndarray]
    boundary_measurement: bool
    roi_coordinates:    Optional[dict] = None  # boundary ROI coord dict (key → (N,2) ndarray)
    params:             dict = field(default_factory=dict)


# ── Private helpers ───────────────────────────────────────────────────────────

def _concat_roi_df(
    roi_measurements: Optional[pd.DataFrame],
    details_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_row: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accumulate one ROI's results into the running detail and summary tables.

    Port of pycurvelets ``concat_roi_df``; column names updated to tme_quant
    convention (``angle_to_boundary_tangent``, ``angle_to_roi_orientation``,
    ``angle_to_centers_line``).
    """
    if roi_measurements is not None and not roi_measurements.empty:
        details_df = (
            roi_measurements.copy()
            if details_df.empty
            else pd.concat([details_df, roi_measurements], ignore_index=True)
        )
        summary_row["mean_angle_to_boundary_tangent"] = roi_measurements[
            "angle_to_boundary_tangent"
        ].mean()
        summary_row["mean_angle_to_roi_orientation"] = roi_measurements[
            "angle_to_roi_orientation"
        ].mean()
        summary_row["mean_angle_to_centers_line"] = roi_measurements[
            "angle_to_centers_line"
        ].mean()
        summary_row["number_of_fibers"] = len(roi_measurements)
    else:
        summary_row["mean_angle_to_boundary_tangent"] = np.nan
        summary_row["mean_angle_to_roi_orientation"] = np.nan
        summary_row["mean_angle_to_centers_line"] = np.nan
        summary_row["number_of_fibers"] = 0

    summary_df = (
        pd.DataFrame([summary_row])
        if summary_df.empty
        else pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    )
    return details_df, summary_df


def _process_single_roi(
    roi_index: int,
    roi_coords: np.ndarray,
    fiber_structure: pd.DataFrame,
    distance_threshold: Optional[float],
    img_shape: tuple,
) -> tuple[int, Optional[pd.DataFrame], dict]:
    """Compute per-fiber alignment angles for one ROI.

    Port of pycurvelets ``process_single_roi``.

    Bug fix vs. original: ``polygon2mask`` called with ``(height, width)`` and
    ``(row, col)`` coords — pycurvelets incorrectly passed ``(width, height)``,
    transposing the mask and producing wrong centroid/orientation values.
    """
    img_height, img_width = img_shape[:2]
    roi_coords_array = np.asarray(roi_coords)

    roi_mask = polygon2mask((img_height, img_width), roi_coords_array)
    roi_regions = regionprops(label(roi_mask.astype(int)))

    if len(roi_regions) != 1:
        raise ValueError(f"ROI {roi_index} does not correspond to a single region")

    roi_props = roi_regions[0]
    orientation_deg = -np.degrees(roi_props.orientation)
    if orientation_deg < 0:
        orientation_deg += 180

    summary_row = {
        "ROI_id": roi_index + 1,
        "center_row": roi_props.centroid[0],
        "center_col": roi_props.centroid[1],
        "orientation": orientation_deg,
        "area": roi_props.area,
    }

    roi_measurements = None
    try:
        roi_measurements, _ = compute_fiber_alignment_to_roi(
            roi_coords_array, img_height, img_width, fiber_structure, distance_threshold
        )
    except Exception:
        pass

    return roi_index, roi_measurements, summary_row


def _process_tif_rois(
    coordinates: dict,
    fiber_structure: pd.DataFrame,
    distance_threshold: Optional[float],
    img_shape: tuple,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all TIFF-mask ROIs sequentially, return accumulated DataFrames.

    Replaces the ``multiprocessing.Pool`` in pycurvelets ``process_tiff_boundary_rois``
    with a sequential loop — simpler, cross-platform, and testable in isolation.
    """
    details_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    for idx, roi_coords in enumerate(coordinates.values()):
        _, roi_measurements, summary_row = _process_single_roi(
            idx, roi_coords, fiber_structure, distance_threshold, img_shape
        )
        details_df, summary_df = _concat_roi_df(
            roi_measurements, details_df, summary_df, summary_row
        )

    return details_df, summary_df


def _analyze_global_boundary(
    coordinates: Optional[dict],
    boundary_img: Optional[np.ndarray],
    fiber_structure: pd.DataFrame,
    distance_threshold: Optional[float],
    min_dist,
    exclude_fibers_in_mask: bool,
) -> dict:
    """Compute global fiber-boundary relationships via ``extract_tif_boundary``.

    Port of pycurvelets ``analyze_global_boundary``; prints removed.
    """
    _, _, _, res_df = extract_tif_boundary(
        coordinates=coordinates,
        img=boundary_img,
        fiber_df=fiber_structure,
        dist_thresh=distance_threshold,
        min_dist=min_dist,
    )

    nearest_angles = res_df["nearest_boundary_angle"]

    if len(min_dist) == 0:
        in_curvs_flag = res_df["nearest_boundary_distance"] <= distance_threshold
    else:
        in_curvs_flag = (res_df["nearest_boundary_distance"] <= distance_threshold) & (
            res_df["nearest_boundary_distance"] > min_dist
        )

    if exclude_fibers_in_mask:
        mask_condition = res_df["nearest_region_distance"] == 0
        if len(min_dist) == 0:
            in_curvs_flag = (
                res_df["nearest_boundary_distance"] <= distance_threshold
            ) & mask_condition
        else:
            in_curvs_flag = (
                (res_df["nearest_boundary_distance"] <= distance_threshold)
                & (res_df["nearest_boundary_distance"] > min_dist)
                & mask_condition
            )

    measured_boundary = res_df[
        [
            "nearest_boundary_distance",
            "nearest_region_distance",
            "nearest_boundary_angle",
            "extension_point_distance",
            "extension_point_angle",
            "boundary_point_col",
            "boundary_point_row",
        ]
    ]

    return {
        "nearest_angles": nearest_angles,
        "in_curvs_flag": in_curvs_flag.values,
        "out_curvs_flag": (~in_curvs_flag).values,
        "distances": res_df["nearest_boundary_distance"],
        "measured_boundary": measured_boundary,
        "bins": np.arange(2.5, 90, 5),
    }


def _build_fiber_features_df(
    fiber_structure: pd.DataFrame,
    density_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    measured_boundary: Optional[pd.DataFrame],
    boundary_measurement: bool,
) -> pd.DataFrame:
    """Consolidate per-fiber features into a single DataFrame (no file I/O).

    Port of pycurvelets ``save_fiber_features`` with the Excel-save step removed
    and column names updated to tme_quant convention.
    """
    center_row = (
        fiber_structure["center_row"]
        if "center_row" in fiber_structure.columns
        else fiber_structure["center_1"]
    )
    center_col = (
        fiber_structure["center_col"]
        if "center_col" in fiber_structure.columns
        else fiber_structure["center_2"]
    )

    fib_feat_df = pd.DataFrame(
        {
            "fiber_key": list(range(len(fiber_structure))),
            "center_row": center_row.values,
            "center_col": center_col.values,
            "fiber_absolute_angle": fiber_structure["angle"].values,
            "fiber_weight": fiber_structure.get(
                "weight", pd.Series([np.nan] * len(fiber_structure))
            ).values,
        }
    )

    for src_df in (density_df, alignment_df):
        if not src_df.empty:
            for col in src_df.columns:
                fib_feat_df[col] = src_df[col].values

    boundary_cols = [
        "nearest_distance_to_boundary",
        "inside_epicenter_region",
        "nearest_relative_boundary_angle",
        "extension_point_distance",
        "extension_point_angle",
        "boundary_point_col",
        "boundary_point_row",
    ]
    boundary_col_mapping = {
        "nearest_boundary_distance": "nearest_distance_to_boundary",
        "nearest_region_distance": "inside_epicenter_region",
        "nearest_boundary_angle": "nearest_relative_boundary_angle",
        "extension_point_distance": "extension_point_distance",
        "extension_point_angle": "extension_point_angle",
        "boundary_point_col": "boundary_point_col",
        "boundary_point_row": "boundary_point_row",
    }

    if boundary_measurement and measured_boundary is not None:
        for src_col, dest_col in boundary_col_mapping.items():
            fib_feat_df[dest_col] = (
                measured_boundary[src_col].values
                if src_col in measured_boundary.columns
                else np.nan
            )
    else:
        for col_name in boundary_cols:
            fib_feat_df[col_name] = np.nan

    return fib_feat_df


# ── Public API ────────────────────────────────────────────────────────────────

def curvealign_curvelets_mode_pipeline(
    image: np.ndarray,
    fiber_structure: Optional[pd.DataFrame] = None,
    keep: float = 0.05,
    scale: int = 1,
    radius: float = 4.0,
    feature_params: Optional[FiberFeatureParams] = None,
    coordinates: Optional[dict] = None,
    boundary_img: Optional[np.ndarray] = None,
    distance_threshold: Optional[float] = None,
    tif_boundary: int = 0,
    exclude_fibers_in_mask: bool = False,
    min_dist=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Optional[CurveAlignPipelineResult]:
    """Run the CurveAlign curvelets-mode fiber-orientation pipeline on a single image.

    Uses grouped curvelet orientation estimates as the fiber representation.
    Each row of the returned ``fiber_structure`` DataFrame represents the
    dominant orientation of a curvelet-grouped local region — not an
    individually extracted fiber.

    Port of pycurvelets ``process_image``.  All file I/O, GUI calls, and
    multiprocessing have been removed; visualization is deferred to
    ``fiber_analysis.visualization.draw_utils``.

    Parameters
    ----------
    image : ndarray of shape (H, W)
        Greyscale image to analyse.
    fiber_structure : pd.DataFrame or None
        Pre-computed curvelet orientation DataFrame (``center_row``,
        ``center_col``, ``angle`` columns).  When ``None``, grouped curvelet
        orientation estimation is run automatically via
        ``build_fiber_structure_from_curvelets``.
    keep : float
        Fraction of curvelet coefficients to keep (used only when
        ``fiber_structure`` is ``None``).
    scale : int
        Curvelet scale index (used only when ``fiber_structure`` is ``None``).
    radius : float
        Grouping radius in pixels (used only when ``fiber_structure`` is ``None``).
    feature_params : FiberFeatureParams or None
        Density / alignment computation parameters.  ``None`` uses defaults.
    coordinates : dict or None
        Pre-computed ROI boundary coordinates mapping string keys to ``(N, 2)``
        ``[row, col]`` ndarrays.  When ``None`` and ``tif_boundary == 3``, they
        are extracted automatically from ``boundary_img``.
    boundary_img : ndarray or None
        Binary mask image.  Required when ``tif_boundary == 3`` and
        ``coordinates`` is ``None``.
    distance_threshold : float or None
        Maximum fiber-to-boundary distance (pixels) for boundary inclusion.
    tif_boundary : int
        Boundary mode.  ``0`` — no boundary analysis.  ``3`` — TIFF mask
        (ROI boundary extracted from ``boundary_img``).  ``1`` / ``2`` — CSV
        boundary modes (not yet ported; raises ``NotImplementedError``).
    exclude_fibers_in_mask : bool
        When ``True``, fibers whose centres lie inside the mask are excluded
        from boundary-angle statistics.
    min_dist : list, float, or None
        Minimum fiber-to-boundary distance threshold.  ``None`` / empty list
        means no lower bound.

    Returns
    -------
    CurveAlignPipelineResult or None
        ``None`` when no fibers are detected.  Otherwise a
        :class:`CurveAlignPipelineResult` with attributes mirroring the
        former dict keys (``fiber_structure``, ``density_df``,
        ``alignment_df``, ``fiber_features_df``, ``roi_measurements_df``,
        ``roi_summary_df``, ``in_curvs_flag``, ``nearest_angles``,
        ``boundary_measurement``, ``params``).

    Notes
    -----
    Bug fix vs. original ``process_single_roi``: ``polygon2mask`` is now called
    with ``(height, width)`` and ``(row, col)`` coords; pycurvelets incorrectly
    used ``(width, height)`` which transposed the ROI mask.
    """
    if min_dist is None:
        min_dist = []

    # ── 1. Fiber extraction ───────────────────────────────────────────────────
    _report(progress_callback, 1, 4, "Extracting curvelet fiber structure…")
    density_df = pd.DataFrame()
    alignment_df = pd.DataFrame()

    if fiber_structure is None:
        fiber_structure, density_df, alignment_df, _ = build_fiber_structure_from_curvelets(
            image, keep=keep, scale=scale, radius=radius, feature_params=feature_params
        )

    if fiber_structure is None or fiber_structure.empty:
        return None

    # ── 2. Boundary mode decision ─────────────────────────────────────────────
    _report(progress_callback, 2, 4, "Preparing boundary analysis…")
    boundary_measurement = bool(coordinates) or (
        tif_boundary == 3 and boundary_img is not None
    )

    if tif_boundary in (1, 2):
        raise NotImplementedError(
            "CSV boundary mode (tif_boundary=1 or 2) is not yet ported."
        )

    # ── 3. Boundary analysis ──────────────────────────────────────────────────
    _report(progress_callback, 3, 4, "Analyzing ROI boundaries…")
    in_curvs_flag: Optional[np.ndarray] = None
    nearest_angles: Optional[pd.Series] = None
    roi_measurements_df: Optional[pd.DataFrame] = None
    roi_summary_df: Optional[pd.DataFrame] = None
    measured_boundary: Optional[pd.DataFrame] = None

    if boundary_measurement and tif_boundary == 3:
        if coordinates is None and boundary_img is not None:
            coordinates = extract_boundary_coords_from_mask(boundary_img)

        if coordinates:
            roi_measurements_df, roi_summary_df = _process_tif_rois(
                coordinates, fiber_structure, distance_threshold, image.shape
            )

        if coordinates:
            boundary_results = _analyze_global_boundary(
                coordinates,
                boundary_img,
                fiber_structure,
                distance_threshold,
                min_dist,
                exclude_fibers_in_mask,
            )
            nearest_angles = boundary_results["nearest_angles"]
            in_curvs_flag = boundary_results["in_curvs_flag"]
            measured_boundary = boundary_results["measured_boundary"]
    elif not boundary_measurement:
        in_curvs_flag = np.ones(len(fiber_structure), dtype=bool)

    # ── 4. Consolidated feature table ─────────────────────────────────────────
    _report(progress_callback, 4, 4, "Assembling fiber feature table…")
    fiber_features_df = _build_fiber_features_df(
        fiber_structure, density_df, alignment_df, measured_boundary, boundary_measurement
    )

    _params = {
        "image_shape": list(image.shape),
        "keep": keep,
        "scale": scale,
        "radius": radius,
        "distance_threshold": distance_threshold,
        "tif_boundary": tif_boundary,
        "exclude_fibers_in_mask": exclude_fibers_in_mask,
        "min_dist": min_dist if min_dist else [],
    }

    return CurveAlignPipelineResult(
        fiber_structure=fiber_structure,
        density_df=density_df,
        alignment_df=alignment_df,
        fiber_features_df=fiber_features_df,
        roi_measurements_df=roi_measurements_df,
        roi_summary_df=roi_summary_df,
        in_curvs_flag=in_curvs_flag,
        nearest_angles=nearest_angles,
        boundary_measurement=boundary_measurement,
        roi_coordinates=coordinates if boundary_measurement else None,
        params=_params,
    )


__all__ = ["CurveAlignPipelineResult", "curvealign_curvelets_mode_pipeline"]
