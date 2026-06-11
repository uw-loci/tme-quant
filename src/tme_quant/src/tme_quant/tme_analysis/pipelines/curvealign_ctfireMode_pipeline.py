# -*- coding: utf-8 -*-
"""
CT-FIRE mode individual-fiber pipeline.

Uses CT-FIRE (via the external ``ctfire_py`` package) to extract individually
traced fiber segments from an SHG image.  Each row of the returned
``fiber_structure`` DataFrame represents one detected fiber with position,
orientation, length, curvature, and width.

This is the CT-FIRE counterpart to ``curvealign_curveletsMode_pipeline``.
Pipeline structure is identical; only Stage 1 (fiber extraction) differs.

``ctfire_py`` is an external PoC bridge — it must be installed separately in the
active Python environment (see Prerequisites below).  When the C++ backend is
properly integrated into ``tme_quant.fiber_analysis``, this module will delegate
to ``CTFireExtraction`` instead.

Prerequisites
-------------
**Full CT-FIRE mode** (``use_ct_reconstruction=True``, default):

1. Build the C++ extension for MSYS2 UCRT64 (.venv-curvelops)::

       source .venv-curvelops/bin/activate
       pip install pybind11
       cd H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src/ctfire_py/CPP
       make -f Makefile.ucrt64
       cp fiber_backend.*.so ../

2. Install the ctfire repo as an editable package::

       pip install -e H:/GitHub.06.2022/tmequant_ctfire/tme-quant --no-deps

3. Verify::

       python -c "from ctfire_py.ct_fire import ct_fire; print('ctfire_py OK')"

**FIRE-only mode** (``use_ct_reconstruction=False``):

* ``ctfire_py`` (same install as above) — **required**.
* curvelops / curvelet transform library — **NOT required**.
  ``fire_2d_angle()`` is called directly on the normalised input image;
  the curvelet reconstruction step is bypassed entirely.

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

import tempfile
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
    compute_fiber_density_and_alignment,
)
from tme_quant.tme_analysis.utils.alignment_utils import compute_fiber_alignment_to_roi


# ── Import guard ──────────────────────────────────────────────────────────────

def _check_ctfire_available() -> bool:
    try:
        import ctfire_py.ct_fire  # noqa: F401
        return True
    except ImportError:
        return False


_CTFIRE_INSTALL_MSG = (
    "ctfire_py is not installed in this environment.\n"
    "To install it in .venv-curvelops (MSYS2 UCRT64):\n"
    "  1. Build the C++ backend:\n"
    "       cd H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src/ctfire_py/CPP\n"
    "       make -f Makefile.ucrt64 && cp fiber_backend.*.so ../\n"
    "  2. Install the package:\n"
    "       pip install -e H:/GitHub.06.2022/tmequant_ctfire/tme-quant --no-deps\n"
    "See CLAUDE.md § CT-FIRE mode pipeline for full instructions."
)


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class CTFirePipelineResult:
    """Typed return value of :func:`curvealign_ctfire_mode_pipeline`.

    Attributes
    ----------
    fiber_structure : pd.DataFrame
        Per-fiber CT-FIRE DataFrame.  Columns: ``center_row``, ``center_col``,
        ``angle``, ``total_length``, ``end_length``, ``curvature``, ``width``.
    fiber_features_df : pd.DataFrame
        Consolidated per-fiber feature table (position, angle, CT-FIRE
        morphology, density, alignment, boundary metrics when available).
    density_df : pd.DataFrame
        Per-fiber density statistics from ``compute_fiber_density_and_alignment``.
    alignment_df : pd.DataFrame
        Per-fiber alignment statistics from ``compute_fiber_density_and_alignment``.
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
    roi_coordinates : dict or None
        Boundary ROI coordinate dict (key → ``(N, 2)`` ndarray, row/col).
    params : dict
        Serialised pipeline parameters.
    """
    fiber_structure:      pd.DataFrame
    fiber_features_df:    pd.DataFrame
    density_df:           pd.DataFrame
    alignment_df:         pd.DataFrame
    roi_measurements_df:  Optional[pd.DataFrame]
    roi_summary_df:       Optional[pd.DataFrame]
    in_curvs_flag:        Optional[np.ndarray]
    nearest_angles:       Optional[np.ndarray]
    boundary_measurement: bool
    roi_coordinates:      Optional[dict] = None
    params:               dict = field(default_factory=dict)


# ── Private helpers ───────────────────────────────────────────────────────────

def _report(cb: Optional[Callable], step: int, total: int, msg: str) -> None:
    if cb is not None:
        cb(step, total, msg)


def _run_ctfire_on_image(
    image: np.ndarray,
    ctfire_params_dict: Optional[dict],
    fiber_mode: int,
) -> pd.DataFrame:
    """Run CT-FIRE on an in-memory image; return fiber_structure DataFrame.

    Lazy-imports ``ctfire_py`` and ``pycurvelets`` (both available when the
    ctfire repo is installed as editable via Option 1 in CLAUDE_CTFIRE.md).

    Parameters
    ----------
    image :
        Greyscale 2-D array (H × W).
    ctfire_params_dict :
        CT-FIRE algorithm parameters dict.  ``None`` uses ``DEFAULT_CTFIRE_PARAMS``
        from ``pycurvelets.get_fire``.
    fiber_mode :
        1 = individual segments; 2 / 3 = complete merged fibers.

    Returns
    -------
    pd.DataFrame
        Columns: ``angle``, ``center_row``, ``center_col``, ``total_length``,
        ``end_length``, ``curvature``, ``width``.  Empty DataFrame when no
        fibers are found.
    """
    from ctfire_py.ct_fire import ct_fire as _ct_fire
    from pycurvelets.get_fire import _build_fiber_dataframe, DEFAULT_CTFIRE_PARAMS
    from pycurvelets.models import FeatureControlParameters

    resolved_params = (
        ctfire_params_dict if ctfire_params_dict is not None
        else DEFAULT_CTFIRE_PARAMS.copy()
    )

    # ct_fire requires a valid save_path even with save_images=False; use a
    # temporary directory that is cleaned up automatically.
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _fiber_out, ctfire_output = _ct_fire(
                image_path=None,
                image_name="pipeline_image",
                save_path=tmp_dir,
                control_params={
                    "show_plots": False,
                    "save_images": False,
                    "output_format": "tif",
                },
                ctfire_params=resolved_params,
                img=image,
            )
    except MemoryError:
        import warnings
        warnings.warn(
            "CT-FIRE ran out of memory (std::bad_alloc) — the image likely has too many "
            "fiber seed candidates. Try reducing image size, lowering coefficient_percentile, "
            "or increasing thresh_LMPdist in ctfire_params['value'].",
            RuntimeWarning,
            stacklevel=4,
        )
        return pd.DataFrame()

    if not ctfire_output or "data" not in ctfire_output:
        return pd.DataFrame()

    LL1 = ctfire_output.get("cP", {}).get("LL1", 0.0)
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=2, minimum_box_size=32, fiber_midpoint_estimate=1
    )
    fiber_df = _build_fiber_dataframe(
        ctfire_output["data"], LL1, fiber_mode, feature_cp,
        ctfire_params=resolved_params,
        img_shape=image.shape,
    )
    return fiber_df if fiber_df is not None else pd.DataFrame()


def _run_fire_only_on_image(
    image: np.ndarray,
    ctfire_params_dict: Optional[dict],
    fiber_mode: int,
) -> pd.DataFrame:
    """Run FIRE fiber extraction directly on a normalised image (no curvelets).

    Calls ``fire_2d_angle()`` on the normalised input image, bypassing the
    curvelet reconstruction step that ``ct_fire()`` performs.

    This function does **not** require curvelops or any curvelet transform
    library.  Only ``ctfire_py`` (and its C++ ``fiber_backend`` extension)
    must be installed.

    Parameters
    ----------
    image :
        Greyscale 2-D array (H × W).  Normalised internally to [0, 255]
        float32, matching the convention expected by ``fire_2d_angle``.
    ctfire_params_dict :
        CT-FIRE parameter dict.  The ``"value"`` sub-dict is forwarded to
        ``fire_2d_angle`` as its ``p`` argument.  ``None`` uses
        ``DEFAULT_CTFIRE_PARAMS``.  Note: ``coefficient_percentile`` and
        ``num_scales`` are ignored in this mode (no curvelet step).
    fiber_mode :
        1 = individual segments; 2 / 3 = complete merged fibers.

    Returns
    -------
    pd.DataFrame
        Columns: ``angle``, ``center_row``, ``center_col``, ``total_length``,
        ``end_length``, ``curvature``, ``width``.  Empty DataFrame when no
        fibers are found.
    """
    from ctfire_py.fire_2d_angle import fire_2d_angle as _fire_2d_angle
    from pycurvelets.get_fire import _build_fiber_dataframe, DEFAULT_CTFIRE_PARAMS
    from pycurvelets.models import FeatureControlParameters

    resolved_params = (
        ctfire_params_dict if ctfire_params_dict is not None
        else DEFAULT_CTFIRE_PARAMS.copy()
    )
    fire_params = resolved_params["value"].copy()

    # Normalise to [0, 255] float32 — same convention as ct_fire.py uses before
    # passing to fire_2d_angle, so thresh_im2 thresholds work correctly.
    img = image.astype(np.float32)
    if img.max() > 0:
        img = img / img.max() * 255.0

    # fire_2d_angle expects a 3-D array of shape (1, H, W).
    im3 = img[np.newaxis, :, :]

    try:
        data = _fire_2d_angle(p=fire_params, im=im3, plotflag=0)
    except MemoryError:
        import warnings
        warnings.warn(
            "FIRE ran out of memory (std::bad_alloc) — the image likely has too many "
            "fiber seed candidates. Try reducing image size or increasing "
            "thresh_LMPdist in ctfire_params['value'].",
            RuntimeWarning,
            stacklevel=4,
        )
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    # LL1: post-tracing minimum fiber arc-length filter (same key as ct_fire.py).
    # Default 30 matches ct_fire.py's default; set ctfire_params["LL1"] to override.
    LL1 = resolved_params.get("LL1", 30)
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=2, minimum_box_size=32, fiber_midpoint_estimate=1
    )
    fiber_df = _build_fiber_dataframe(
        data, LL1, fiber_mode, feature_cp,
        ctfire_params=resolved_params,
        img_shape=image.shape,
    )
    return fiber_df if fiber_df is not None else pd.DataFrame()


def _concat_roi_df(
    roi_measurements: Optional[pd.DataFrame],
    details_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_row: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accumulate one ROI's results into running detail and summary tables."""
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
    """Compute per-fiber alignment angles for one ROI."""
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
    """Process all TIFF-mask ROIs sequentially."""
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
    """Compute global fiber-boundary relationships via ``extract_tif_boundary``."""
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
    """Consolidate CT-FIRE per-fiber features into a single DataFrame.

    Includes CT-FIRE morphology columns (``total_length``, ``end_length``,
    ``curvature``, ``width``) that are absent from the curvelet pipeline.
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
            "fiber_key":           list(range(len(fiber_structure))),
            "center_row":          center_row.values,
            "center_col":          center_col.values,
            "fiber_absolute_angle": fiber_structure["angle"].values,
            "total_length":        fiber_structure.get(
                "total_length", pd.Series([np.nan] * len(fiber_structure))
            ).values,
            "end_length":          fiber_structure.get(
                "end_length", pd.Series([np.nan] * len(fiber_structure))
            ).values,
            "curvature":           fiber_structure.get(
                "curvature", pd.Series([np.nan] * len(fiber_structure))
            ).values,
            "fiber_width":         fiber_structure.get(
                "width", pd.Series([np.nan] * len(fiber_structure))
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
        "nearest_boundary_distance":  "nearest_distance_to_boundary",
        "nearest_region_distance":    "inside_epicenter_region",
        "nearest_boundary_angle":     "nearest_relative_boundary_angle",
        "extension_point_distance":   "extension_point_distance",
        "extension_point_angle":      "extension_point_angle",
        "boundary_point_col":         "boundary_point_col",
        "boundary_point_row":         "boundary_point_row",
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

def curvealign_ctfire_mode_pipeline(
    image: np.ndarray,
    ctfire_params: Optional[dict] = None,
    use_ct_reconstruction: bool = True,
    fiber_mode: int = 2,
    feature_params: Optional[FiberFeatureParams] = None,
    coordinates: Optional[dict] = None,
    boundary_img: Optional[np.ndarray] = None,
    distance_threshold: Optional[float] = None,
    tif_boundary: int = 0,
    exclude_fibers_in_mask: bool = False,
    min_dist=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Optional[CTFirePipelineResult]:
    """Run the CT-FIRE individual-fiber pipeline on a single image.

    Uses CT-FIRE (via the external ``ctfire_py`` package) to trace individual
    fiber segments.  Each row of the returned ``fiber_structure`` DataFrame is
    one detected fiber, with full morphology (length, curvature, width) in
    addition to position and orientation.

    Parameters
    ----------
    image : ndarray of shape (H, W)
        Greyscale image to analyse.
    ctfire_params : dict or None
        CT-FIRE algorithm parameters.  ``None`` uses ``DEFAULT_CTFIRE_PARAMS``
        from ``pycurvelets.get_fire``.  Key sub-dicts: ``"value"`` (FIRE graph
        parameters) and ``"widcon"`` (width calculation settings).
    use_ct_reconstruction : bool, default ``True``
        When ``True`` (default), runs the full CT-FIRE pipeline: curvelet
        reconstruction via ``ct_reconstruction()`` followed by
        ``fire_2d_angle()``.

        When ``False``, skips the curvelet preprocessing step and calls
        ``fire_2d_angle()`` directly on the normalised input image.
        **This mode does not require curvelops or any curvelet transform
        library** — only ``ctfire_py`` must be installed.  Use this when
        the image has already been pre-processed, for non-SHG images, or
        to isolate FIRE behaviour from curvelet preprocessing effects.
    fiber_mode : int
        Fiber representation mode passed to ``_build_fiber_dataframe``.
        1 = individual segments (shorter, more fragments);
        2 / 3 = complete merged fibers (default: 2).
    feature_params : FiberFeatureParams or None
        Density / alignment neighbourhood parameters.  ``None`` uses defaults.
    coordinates : dict or None
        Pre-computed ROI boundary coordinates mapping string keys to ``(N, 2)``
        ``[row, col]`` ndarrays.  When ``None`` and ``tif_boundary == 3``, they
        are extracted automatically from ``boundary_img``.

        Providing ``coordinates`` alone (without ``boundary_img``) enables
        **partial** boundary analysis: ``roi_measurements_df`` and
        ``roi_summary_df`` are populated, but ``nearest_angles`` and
        ``in_curvs_flag`` remain ``None`` (they require ``boundary_img`` for
        region-membership lookup via :func:`extract_tif_boundary`).
    boundary_img : ndarray or None
        Binary mask image used for two purposes: (1) automatic coordinate
        extraction when ``coordinates`` is ``None`` (via
        :func:`extract_boundary_coords_from_mask`); (2) fiber region-membership
        lookup in :func:`extract_tif_boundary`, which populates
        ``nearest_angles`` and ``in_curvs_flag``.  Optional when
        ``coordinates`` is already supplied — in that case only purpose (2) is
        skipped and ROI-level stats are still computed.
    distance_threshold : float or None
        Maximum fiber-to-boundary distance (pixels) for boundary inclusion.
    tif_boundary : int
        Boundary mode.  0 — no boundary analysis.  3 — TIFF mask.
        1 / 2 — CSV modes (not yet ported; raises ``NotImplementedError``).
        Supplying ``coordinates`` directly bypasses this flag: boundary ROI
        analysis runs regardless of the ``tif_boundary`` value.
    exclude_fibers_in_mask : bool
        When ``True``, fibers whose centres lie inside the mask are excluded
        from boundary-angle statistics.
    min_dist : list, float, or None
        Minimum fiber-to-boundary distance threshold.
    progress_callback : callable or None
        Optional ``(step, total, message)`` progress reporter.

    Returns
    -------
    CTFirePipelineResult or None
        ``None`` when no fibers are detected or ``ctfire_py`` is unavailable.

    Raises
    ------
    ImportError
        When ``ctfire_py`` is not installed.  The error message includes
        step-by-step installation instructions.
    NotImplementedError
        When ``tif_boundary`` is 1 or 2 (CSV modes not yet ported).
    """
    if not _check_ctfire_available():
        raise ImportError(_CTFIRE_INSTALL_MSG)

    if min_dist is None:
        min_dist = []

    if feature_params is None:
        feature_params = FiberFeatureParams()

    # ── 1. Fiber extraction ───────────────────────────────────────────────────
    if use_ct_reconstruction:
        _report(progress_callback, 1, 4, "Running CT-FIRE fiber extraction (with curvelet reconstruction)…")
        fiber_structure = _run_ctfire_on_image(image, ctfire_params, fiber_mode)
    else:
        _report(progress_callback, 1, 4, "Running FIRE fiber extraction (no curvelet reconstruction)…")
        fiber_structure = _run_fire_only_on_image(image, ctfire_params, fiber_mode)

    if fiber_structure is None or fiber_structure.empty:
        return None

    # ── 2. Density / alignment features ──────────────────────────────────────
    _report(progress_callback, 2, 4, "Computing fiber density and alignment…")
    density_df = pd.DataFrame()
    alignment_df = pd.DataFrame()
    try:
        density_df, alignment_df = compute_fiber_density_and_alignment(
            fiber_structure, feature_params
        )
    except Exception:
        pass

    # ── 3. Boundary mode decision ─────────────────────────────────────────────
    _report(progress_callback, 3, 4, "Analyzing ROI boundaries…")

    if tif_boundary in (1, 2):
        raise NotImplementedError(
            "CSV boundary mode (tif_boundary=1 or 2) is not yet ported."
        )

    boundary_measurement = bool(coordinates) or (
        tif_boundary == 3 and boundary_img is not None
    )

    in_curvs_flag: Optional[np.ndarray] = None
    nearest_angles: Optional[pd.Series] = None
    roi_measurements_df: Optional[pd.DataFrame] = None
    roi_summary_df: Optional[pd.DataFrame] = None
    measured_boundary: Optional[pd.DataFrame] = None

    if boundary_measurement:
        if coordinates is None and boundary_img is not None:
            coordinates = extract_boundary_coords_from_mask(boundary_img)

        if coordinates:
            roi_measurements_df, roi_summary_df = _process_tif_rois(
                coordinates, fiber_structure, distance_threshold, image.shape
            )
            if boundary_img is not None:
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
        "image_shape":            list(image.shape),
        "use_ct_reconstruction":  use_ct_reconstruction,
        "fiber_mode":             fiber_mode,
        "distance_threshold":     distance_threshold,
        "tif_boundary":           tif_boundary,
        "exclude_fibers_in_mask": exclude_fibers_in_mask,
        "min_dist":               min_dist if min_dist else [],
    }

    return CTFirePipelineResult(
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


__all__ = ["CTFirePipelineResult", "curvealign_ctfire_mode_pipeline"]
