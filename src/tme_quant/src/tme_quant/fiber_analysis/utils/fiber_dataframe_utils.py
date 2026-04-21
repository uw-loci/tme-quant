# -*- coding: utf-8 -*-
"""
Fiber density and local alignment DataFrame utilities.

Adapted from pycurvelets ``process_fibers`` (MATLAB conversion).

This module fills the gap between the low-level per-fiber geometry utilities
already in ``geometry_utils`` and the high-level extraction results in
``fiber_analysis/results.py``.  It operates on a **DataFrame representation**
of the fiber population (one row per fiber), which is the format produced by
CT-FIRE and CurveAlign pipelines.

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..config import FiberFeatureParams
from .geometry_utils import _circ_r


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (ported from pycurvelets utils.math)
# ─────────────────────────────────────────────────────────────────────────────

def round_mlab(num) -> int | list:
    """MATLAB-compatible rounding (round-half-away-from-zero).

    Port of pycurvelets ``round_mlab``.  Python's built-in ``round()`` uses
    banker's rounding (round-half-to-even), which diverges from MATLAB for
    exactly half-integer values.  Use this function when numerical parity
    with MATLAB reference outputs is required.

    Parameters
    ----------
    num : int, float, list, tuple, or array-like
        Value(s) to round.

    Returns
    -------
    int or list of int
    """
    import math
    if isinstance(num, (list, tuple)):
        return [int(math.floor(float(x) + 0.5)) for x in num]
    if hasattr(num, "__iter__"):  # numpy array or other iterable
        return [int(math.floor(float(x) + 0.5)) for x in num]
    return int(math.floor(float(num) + 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_fiber_density_and_alignment(
    fiber_structure: pd.DataFrame,
    params: FiberFeatureParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-fiber local density and alignment features.

    Adapted from pycurvelets ``process_fibers``.  Two complementary
    neighbourhood strategies are used:

    **kNN-based** (4 scales: 1×, 2×, 4×, 8× ``minimum_nearest_fibers``):
      - *density*: mean Euclidean distance to the k nearest fibres (px).
      - *alignment*: circular mean resultant length R ∈ [0, 1] of the k
        nearest fibre angles (0 = random, 1 = perfectly aligned).

    **Box-filter-based** (3 scales: 1×, 2×, 4× ``minimum_box_size``):
      - *density*: count of fibre centres falling inside the square box.
      - *alignment*: circular mean resultant length R of all fibres in box.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        One row per fibre.  Required columns:

        - ``angle``       — fibre orientation in degrees.
        - ``center_row``  — row coordinate of the fibre centre (y, pixels).
        - ``center_col``  — column coordinate of the fibre centre (x, pixels).

        Column aliases ``center_1`` / ``center_2`` (pycurvelets convention)
        are accepted and internally mapped to ``center_row`` / ``center_col``.

    params : FiberFeatureParams
        Controls neighbourhood sizes.  See ``FiberFeatureParams`` docstring.

    Returns
    -------
    density_df : pd.DataFrame
        Shape ``(n_fibers, 9)``.  Columns (names depend on param values):

        - ``distance_to_nearest_{k}_fibers`` — kNN density (×4 scales).
        - ``distance_to_nearest_fiber_mean`` — mean across 4 kNN scales.
        - ``distance_to_nearest_fiber_std``  — std across 4 kNN scales.
        - ``fibers_within_box_density{b}``   — box-filter count (×3 scales).

    alignment_df : pd.DataFrame
        Shape ``(n_fibers, 9)``.  Columns (names depend on param values):

        - ``alignment_of_nearest_{k}_fibers`` — kNN alignment (×4 scales).
        - ``alignment_mean``                  — mean across 4 kNN scales.
        - ``alignment_std``                   — std across 4 kNN scales.
        - ``fiber_alignment_in_box_{b}``      — box-filter alignment (×3 scales).

    Raises
    ------
    ValueError
        If ``fiber_structure`` is empty or missing required columns.
    """
    if fiber_structure is None or len(fiber_structure) == 0:
        raise ValueError("fiber_structure must be a non-empty DataFrame.")

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

    required = {"angle", "center_row", "center_col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fiber_structure is missing columns: {missing}")

    df["weight"] = np.nan

    fiber_number = len(df)
    x = df["center_row"].to_numpy(dtype=np.float64)
    y = df["center_col"].to_numpy(dtype=np.float64)
    centers = np.column_stack((x, y))
    angles = df["angle"].to_numpy(dtype=np.float64)

    min_k = params.minimum_nearest_fibers
    min_b = params.minimum_box_size

    nearest_fibers = [2 ** i * min_k for i in range(4)]   # k at 4 scales
    box_sizes = [2 ** i * min_b for i in range(3)]        # box at 3 scales
    fiber_sizes = np.ceil(np.array(box_sizes) / 2)

    n_cols = len(nearest_fibers) + len(box_sizes)
    density_list = np.full((fiber_number, n_cols), np.nan)
    alignment_list = np.full((fiber_number, n_cols), np.nan)

    # kNN pre-computation (single fit, all scales share it)
    K = nearest_fibers[-1] + 1
    nbrs = NearestNeighbors(
        n_neighbors=min(K, fiber_number),
        metric="euclidean",
        algorithm="brute",
    ).fit(centers)
    nn_dist, nn_idx = nbrs.kneighbors(centers)

    for i in range(fiber_number):
        neighbor_angles = angles[nn_idx[i, :]]

        # kNN-based features (4 scales)
        for j, k in enumerate(nearest_fibers):
            if k < nn_dist.shape[1]:
                density_list[i, j] = nn_dist[i, 1 : k + 1].mean()
                alignment_list[i, j] = _circ_r(
                    neighbor_angles[1 : k + 1] * 2.0 * np.pi / 180.0
                )
            # else: stays NaN (fewer fibres than neighbours requested)

        # Box-filter-based features (3 scales)
        for j, half in enumerate(fiber_sizes):
            square_mask = (
                (x > x[i] - half) & (x < x[i] + half) &
                (y > y[i] - half) & (y < y[i] + half)
            )
            vals = df.loc[square_mask, "angle"].to_numpy(dtype=np.float64)
            col_idx = len(nearest_fibers) + j
            density_list[i, col_idx] = len(vals)
            alignment_list[i, col_idx] = _circ_r(vals * 2.0 * np.pi / 180.0)

    density_df = pd.DataFrame({
        f"distance_to_nearest_{nearest_fibers[0]}_fibers":  density_list[:, 0],
        f"distance_to_nearest_{nearest_fibers[1]}_fibers":  density_list[:, 1],
        f"distance_to_nearest_{nearest_fibers[2]}_fibers":  density_list[:, 2],
        f"distance_to_nearest_{nearest_fibers[3]}_fibers":  density_list[:, 3],
        "distance_to_nearest_fiber_mean": np.nanmean(density_list[:, :4], axis=1),
        "distance_to_nearest_fiber_std":  np.nanstd( density_list[:, :4], axis=1, ddof=1),
        f"fibers_within_box_density{box_sizes[0]}":  density_list[:, 4],
        f"fibers_within_box_density{box_sizes[1]}":  density_list[:, 5],
        f"fibers_within_box_density{box_sizes[2]}":  density_list[:, 6],
    })

    alignment_df = pd.DataFrame({
        f"alignment_of_nearest_{nearest_fibers[0]}_fibers":  alignment_list[:, 0],
        f"alignment_of_nearest_{nearest_fibers[1]}_fibers":  alignment_list[:, 1],
        f"alignment_of_nearest_{nearest_fibers[2]}_fibers":  alignment_list[:, 2],
        f"alignment_of_nearest_{nearest_fibers[3]}_fibers":  alignment_list[:, 3],
        "alignment_mean": np.nanmean(alignment_list[:, :4], axis=1),
        "alignment_std":  np.nanstd( alignment_list[:, :4], axis=1, ddof=1),
        f"fiber_alignment_in_box_{box_sizes[0]}":  alignment_list[:, 4],
        f"fiber_alignment_in_box_{box_sizes[1]}":  alignment_list[:, 5],
        f"fiber_alignment_in_box_{box_sizes[2]}":  alignment_list[:, 6],
    })

    return density_df, alignment_df


def flatten_numeric(series: "pd.Series") -> np.ndarray:
    """Convert a Series of numbers or 1-element arrays into a flat float array.

    Port of pycurvelets ``flatten_numeric`` (unchanged signature).  Handles
    object-dtype Series where each element may be a scalar, a list, or a
    single-element ndarray (e.g. outputs from vectorised curvelet operations).

    Parameters
    ----------
    series : pd.Series
        Input Series.

    Returns
    -------
    np.ndarray of float64, 1-D
    """
    arr = series.to_numpy()
    if arr.dtype == object:
        flat = []
        for x in arr:
            if isinstance(x, (list, np.ndarray)):
                if np.ndim(x) == 0:
                    flat.append(float(x))
                elif len(np.ravel(x)) == 1:
                    flat.append(float(np.ravel(x)[0]))
                else:
                    flat.extend(np.ravel(x).astype(float))
            else:
                flat.append(float(x))
        return np.array(flat, dtype=float)
    return arr.astype(float).ravel()


def build_fiber_structure_from_curvelets(
    image: np.ndarray,
    keep: float = 0.05,
    scale: int = 1,
    radius: float = 4.0,
    feature_params: "FiberFeatureParams | None" = None,
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]":
    """Extract curvelet fiber candidates and compute density/alignment features.

    Orchestrates ``extract_curvelet_fiber_candidates`` and
    ``compute_fiber_density_and_alignment`` into a single call, mirroring
    ``get_ct`` from the pycurvelets CurveAlign pipeline.

    This function belongs to the **CurveAlign orientation pipeline** (population-
    level density and alignment statistics), not the CT-FIRE individual fiber
    extraction pipeline.

    Parameters
    ----------
    image : np.ndarray
        2-D image array ``(H, W)`` to be analysed.
    keep : float
        Fraction of curvelet coefficients to retain (default ``0.05``).
        Maps to ``CurveletControlParameters.keep``.
    scale : int
        Curvelet scale index to use (default ``1``).
        Maps to ``CurveletControlParameters.scale``.
    radius : float
        Grouping radius in pixels for curvelet candidates (default ``4.0``).
        Maps to ``CurveletControlParameters.radius``.
    feature_params : FiberFeatureParams, optional
        Controls neighbourhood sizes for density/alignment computation.
        Defaults to ``FiberFeatureParams()`` (uses library defaults).

    Returns
    -------
    fiber_structure : pd.DataFrame
        One row per curvelet candidate.  Columns: ``angle``, ``center_row``,
        ``center_col``, ``width``.
    density_df : pd.DataFrame
        Shape ``(n_fibers, 9)`` — kNN and box-filter density features.
        Empty DataFrame when no candidates are found.
    alignment_df : pd.DataFrame
        Shape ``(n_fibers, 9)`` — kNN and box-filter alignment features.
        Empty DataFrame when no candidates are found.
    curvelet_coefficients : object
        Raw curvelet coefficient structure returned by
        ``extract_curvelet_fiber_candidates``.

    Raises
    ------
    ImportError
        When curvelops is not installed (raised by
        ``extract_curvelet_fiber_candidates``).
    ValueError
        When ``image`` is not a 2-D array.
    """
    from .curvelet_utils import extract_curvelet_fiber_candidates

    if feature_params is None:
        feature_params = FiberFeatureParams()

    fiber_structure, coefficients, _ = extract_curvelet_fiber_candidates(
        image, keep=keep, scale=scale, radius=radius
    )

    if len(fiber_structure) == 0:
        return fiber_structure, pd.DataFrame(), pd.DataFrame(), coefficients

    density_df, alignment_df = compute_fiber_density_and_alignment(
        fiber_structure, feature_params
    )
    return fiber_structure, density_df, alignment_df, coefficients


__all__ = [
    "build_fiber_structure_from_curvelets",
    "compute_fiber_density_and_alignment",
    "flatten_numeric",
    "round_mlab",
]
