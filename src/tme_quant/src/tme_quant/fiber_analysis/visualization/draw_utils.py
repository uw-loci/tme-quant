# -*- coding: utf-8 -*-
"""
Fiber/curvelet overlay drawing utilities.

Adapted from pycurvelets ``draw_curvs`` and ``draw_map`` (MATLAB conversion).

- ``draw_curvs`` — draws fiber center points and orientation lines onto a
  matplotlib Axes object.
- ``draw_map`` — produces a 2-D angle heatmap as a pair of ndarrays; does not
  require matplotlib.

No Qt / napari dependencies.  See REFACTORING_GUIDE.md §2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib  # noqa: F401 — core dep; imported to confirm availability at load time
import matplotlib.pyplot as plt

from ..utils.geometry_utils import _circ_r


def draw_curvs(
    fiber_data,
    ax,
    length: float,
    color_flag: int,
    angles,
    mark_size: float,
    line_width: float,
    boundary_measurement: bool,
) -> None:
    """Draw curvelet/fiber center points and orientation lines on a matplotlib Axes.

    Port of pycurvelets ``draw_curvs``.  Renders each fiber as a filled center
    dot and a line segment of the given ``length`` aligned to the fiber angle.

    Parameters
    ----------
    fiber_data : pd.DataFrame or array_like
        Fiber data.  As a DataFrame it must contain ``angle`` and either
        ``center_row`` / ``center_col`` or ``center_1`` / ``center_2`` columns.
        As an array it must be shape ``(N, 2)`` with rows as ``[row, col]``.
    ax : matplotlib.axes.Axes
        Target Axes onto which fibers are drawn.
    length : float
        Half-length (pixels) of the orientation line extending from each center.
    color_flag : int
        0 → green lines/dots (used/included fibers);
        1 → red lines/dots (excluded fibers).
    angles : array_like
        Fiber angles in degrees.  Used only when ``boundary_measurement=False``.
    mark_size : float
        Marker size for center dots.
    line_width : float
        Line width for orientation lines.
    boundary_measurement : bool
        When ``True`` reads angle from DataFrame and draws both color_flags.
        When ``False`` uses the ``angles`` parameter and only draws color_flag==0.

    Notes
    -----
    Bug fix vs. original: ``center_1`` / ``center_2`` aliases are normalised before
    both code paths, whereas pycurvelets only normalised them for the non-boundary
    branch (causing a KeyError in the boundary branch with alias inputs).
    """
    # Normalise column aliases before either branch
    if hasattr(fiber_data, "columns") and "center_row" not in fiber_data.columns:
        if "center_1" in fiber_data.columns:
            fiber_data = fiber_data.rename(
                columns={"center_1": "center_row", "center_2": "center_col"}
            )

    if hasattr(fiber_data, "iterrows"):
        centers = fiber_data[["center_row", "center_col"]].values
    else:
        centers = np.array(fiber_data)

    if len(centers) == 0:
        return

    if boundary_measurement:
        for i in range(len(fiber_data)):
            xc = fiber_data["center_col"].iloc[i]
            yc = fiber_data["center_row"].iloc[i]
            color = "g" if color_flag == 0 else "r"
            ax.plot(xc, yc, f"{color}.", markersize=mark_size)
            ca = np.deg2rad(fiber_data["angle"].iloc[i])
            xc1 = xc - length * np.cos(ca)
            xc2 = xc + length * np.cos(ca)
            yc1 = yc + length * np.sin(ca)
            yc2 = yc - length * np.sin(ca)
            ax.plot([xc1, xc2], [yc1, yc2], f"{color}-", linewidth=line_width)
    else:
        if color_flag == 0:
            for center, angle in zip(centers, angles):
                xc = center[1]   # col
                yc = center[0]   # row
                ax.plot(xc, yc, "r.", markersize=mark_size)
                ca = np.deg2rad(angle)
                xc1 = xc - length * np.cos(ca)
                xc2 = xc + length * np.cos(ca)
                yc1 = yc + length * np.sin(ca)
                yc2 = yc - length * np.sin(ca)
                ax.plot([xc1, xc2], [yc1, yc2], "g-", linewidth=line_width)


def draw_map(
    fiber_structure,
    angles,
    img: np.ndarray,
    boundary_measurement: bool,
    map_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a per-pixel angle heatmap from fiber orientation data.

    Port of pycurvelets ``draw_map``.  Encodes each fiber's angle as a grey
    level at its centre pixel, optionally applies a circular-statistics
    neighbourhood filter, then a max-filter with density normalisation, and
    finally a Gaussian blur.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        One row per fiber.  Required columns: ``angle`` and either
        ``center_row`` / ``center_col`` or ``center_1`` / ``center_2``.
    angles : ndarray
        Fiber angles in degrees (same order as rows of ``fiber_structure``).
    img : ndarray of shape (H, W)
        Reference image used only for its dimensions.
    boundary_measurement : bool
        When ``True``, scales angles from 0–90° → 0–255 and skips the
        circular-statistics filter step.
        When ``False``, scales 0–180° → 0–255 and applies the filter.
    map_params : dict
        Filter parameters:

        - ``STDfilter_size`` — neighbourhood half-size for circular-statistics
          filter (default 24, used when ``boundary_measurement=False``).
        - ``SQUAREmaxfilter_size`` — half-size for the density max-filter
          (default 12).
        - ``GAUSSIANdiscfilter_sigma`` — Gaussian blur sigma (default 4).

    Returns
    -------
    rawmap : ndarray of float64, shape (H, W)
        Angle values (0–255) at fiber centre pixels; NaN elsewhere.
    procmap : ndarray of uint8, shape (H, W)
        Smoothed, max-filtered map clipped to 0–255.

    Notes
    -----
    Only change vs. original: ``circ_r`` (pycurvelets import) replaced by
    ``_circ_r`` from ``geometry_utils`` — identical semantics.
    """
    J, I = img.shape  # height, width

    rawmap = np.full((J, I), np.nan, dtype=np.float64)

    if "center_row" in fiber_structure.columns:
        centers = fiber_structure[["center_row", "center_col"]].values
    else:
        centers = fiber_structure[["center_1", "center_2"]].values

    for center, angle in zip(centers, angles):
        xc = int(np.round(center[1]))  # col
        yc = int(np.round(center[0]))  # row
        if xc < 0 or xc >= I or yc < 0 or yc >= J:
            continue
        if boundary_measurement:
            rawmap[yc, xc] = 255.0 * (angle / 90.0)
        else:
            rawmap[yc, xc] = 255.0 * (angle / 180.0)

    map2 = rawmap.copy()
    ind = np.where(~np.isnan(rawmap))
    y, x = ind[0], ind[1]

    if not boundary_measurement:
        fSize2 = map_params.get("STDfilter_size", 24)
        map2 = np.full((J, I), np.nan, dtype=np.float64)
        for i in range(len(y)):
            mask = (
                (x > x[i] - fSize2)
                & (x < x[i] + fSize2)
                & (y > y[i] - fSize2)
                & (y < y[i] + fSize2)
            )
            vals = rawmap[y[mask], x[mask]]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 2:
                map2[y[i], x[i]] = _circ_r(vals * np.pi / 127.5) * 255

    fSize = map_params.get("SQUAREmaxfilter_size", 12)
    fSize2 = int(np.ceil(fSize / 2))
    map4 = np.full(img.shape, np.nan, dtype=np.float64)

    for i in range(len(y)):
        val = map2[y[i], x[i]]
        if np.isnan(val):
            continue
        r0 = max(0, y[i] - fSize2)
        r1 = min(J, y[i] + fSize2 + 1)
        c0 = max(0, x[i] - fSize2)
        c1 = min(I, x[i] + fSize2 + 1)
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rg, cg = np.meshgrid(rows, cols, indexing="ij")
        num_fibs = np.sum(~np.isnan(rawmap[rg, cg]))
        if num_fibs > 0:
            map4[rg, cg] = np.fmax(map4[rg, cg], val / num_fibs)

    sig = map_params.get("GAUSSIANdiscfilter_sigma", 4)
    procmap = gaussian_filter(np.nan_to_num(map4, nan=0.0), sigma=sig, mode="nearest")
    procmap = np.clip(procmap, 0, 255).astype(np.uint8)

    return rawmap, procmap


def compute_angle_histogram(
    fiber_structure: pd.DataFrame,
    nearest_angles,
    in_curvs_flag,
    boundary_measurement: bool,
    tif_boundary: int,
    bins: np.ndarray,
) -> dict:
    """Compute angle histogram from fiber/boundary angle data.

    Port of pycurvelets ``save_histogram`` with file-save and plot stripped.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        Fiber data with ``angle`` column.
    nearest_angles : array-like or None
        Angles relative to boundary (used when available).
    in_curvs_flag : bool ndarray or None
        Mask selecting boundary-included fibers; applied when ``tif_boundary==3``.
    boundary_measurement : bool
        Whether boundary analysis was performed.
    tif_boundary : int
        Boundary mode (0=none, 3=TIFF mask).
    bins : ndarray
        Histogram bin edges.

    Returns
    -------
    dict
        ``{"counts": ndarray, "bin_centers": ndarray, "hist_data": ndarray (2, N)}``.
    """
    if boundary_measurement:
        if tif_boundary == 3:
            if nearest_angles is not None and in_curvs_flag is not None:
                values = nearest_angles[in_curvs_flag]
            elif nearest_angles is not None:
                values = nearest_angles
            else:
                values = fiber_structure["angle"].values
        else:
            values = (
                nearest_angles
                if nearest_angles is not None
                else fiber_structure["angle"].values
            )
    else:
        values = (
            nearest_angles
            if nearest_angles is not None
            else fiber_structure["angle"].values
        )

    n, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_data = np.vstack([n, bin_centers])
    return {"counts": n, "bin_centers": bin_centers, "hist_data": hist_data}


def generate_fiber_overlay(
    img: np.ndarray,
    fiber_structure: pd.DataFrame,
    coordinates,
    in_curvs_flag,
    out_curvs_flag,
    nearest_angles,
    measured_boundary,
    fiber_mode: int,
    tif_boundary: int,
    boundary_measurement: bool,
    make_associations: bool = False,
) -> tuple:
    """Create fiber overlay figure on a grayscale image.

    Port of pycurvelets ``generate_overlay`` with ``plt.savefig``/``plt.close``
    and all ``print()`` calls stripped.  The caller is responsible for closing
    the returned figure when done.

    Parameters
    ----------
    img : ndarray of shape (H, W)
        Grayscale background image.
    fiber_structure : pd.DataFrame
        Fiber data (``center_row``, ``center_col``, ``angle`` columns).
    coordinates : dict or None
        Boundary coordinate dict ``{key: (N, 2) ndarray [row, col]}``.
    in_curvs_flag : bool ndarray
        Mask selecting boundary-included fibers.
    out_curvs_flag : bool ndarray
        Mask selecting boundary-excluded fibers.
    nearest_angles : array-like or None
        Fiber angles relative to boundary.
    measured_boundary : pd.DataFrame or None
        Boundary point measurements (``boundary_point_row/col`` columns).
    fiber_mode : int
        Fiber extraction mode; controls orientation-line half-length:
        0 → 4 px (curvelet), 1 → 2.5 px (CT-FIRE segment), ≥2 → 10 px (CT-FIRE fiber).
    tif_boundary : int
        Boundary mode (0=none, 3=TIFF mask; 1/2 not yet ported).
    boundary_measurement : bool
        Whether boundary analysis was performed.
    make_associations : bool
        When ``True`` and ``tif_boundary==3``, draws lines from each included
        fiber centre to its nearest boundary point.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes with the overlay rendered.
    """
    if fiber_mode == 0:
        fiber_len = 4.0
    elif fiber_mode == 1:
        fiber_len = 2.5
    else:
        fiber_len = 10.0

    fig, ax = plt.subplots(
        figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100
    )
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    if boundary_measurement:
        if tif_boundary < 3:
            if coordinates:
                coords_array = np.array(list(coordinates.values())[0])
                ax.plot(coords_array[:, 1], coords_array[:, 0], "y-")
                ax.plot(coords_array[:, 1], coords_array[:, 0], "*y", markersize=3)
        elif tif_boundary == 3:
            for roi_coords in coordinates.values():
                roi_coords_array = np.array(roi_coords)
                ax.plot(
                    roi_coords_array[:, 1],
                    roi_coords_array[:, 0],
                    "y-",
                    linewidth=1,
                )

    marksize = 3
    linewidth = 1

    if tif_boundary == 3:
        if np.any(in_curvs_flag):
            draw_curvs(
                fiber_structure[in_curvs_flag],
                ax,
                fiber_len,
                color_flag=0,
                angles=nearest_angles[in_curvs_flag],
                mark_size=marksize,
                line_width=linewidth,
                boundary_measurement=boundary_measurement,
            )
        if np.any(out_curvs_flag):
            draw_curvs(
                fiber_structure[out_curvs_flag],
                ax,
                fiber_len,
                color_flag=1,
                angles=nearest_angles[out_curvs_flag],
                mark_size=marksize,
                line_width=linewidth,
                boundary_measurement=boundary_measurement,
            )
        if boundary_measurement and make_associations and measured_boundary is not None:
            fiber_centers = (
                fiber_structure[["center_row", "center_col"]].values
                if "center_row" in fiber_structure.columns
                else fiber_structure[["center_1", "center_2"]].values
            )
            in_curvs = fiber_centers[in_curvs_flag]
            in_bndry = measured_boundary[
                ["boundary_point_row", "boundary_point_col"]
            ].values[in_curvs_flag]
            for center, bndry_pt in zip(in_curvs, in_bndry):
                if not np.isnan(bndry_pt[0]) and not np.isnan(bndry_pt[1]):
                    ax.plot(
                        [center[1], bndry_pt[1]],
                        [center[0], bndry_pt[0]],
                        "b-",
                        linewidth=0.5,
                    )
    elif tif_boundary == 0:
        draw_curvs(
            fiber_structure,
            ax,
            fiber_len,
            color_flag=0,
            angles=nearest_angles,
            mark_size=marksize,
            line_width=linewidth,
            boundary_measurement=boundary_measurement,
        )
    # tif_boundary 1/2: not yet ported

    return fig, ax


def generate_fiber_heatmap(
    img: np.ndarray,
    fiber_structure: pd.DataFrame,
    in_curvs_flag,
    angles,
    distances,
    tif_boundary: int,
    boundary_measurement: bool,
    map_params: Optional[dict] = None,
) -> tuple:
    """Create fiber angle heatmap overlaid on a grayscale image.

    Port of pycurvelets ``generate_heatmap`` with ``plt.savefig``, CSV save,
    and all ``print()`` calls stripped.  The caller is responsible for closing
    the returned figure when done.

    Parameters
    ----------
    img : ndarray of shape (H, W)
        Grayscale background image.
    fiber_structure : pd.DataFrame
        Fiber data; filtered by ``in_curvs_flag`` before map generation.
    in_curvs_flag : bool ndarray
        Mask selecting fibers to include in the map.
    angles : ndarray
        Fiber angles corresponding to rows of ``fiber_structure``.
    distances : ndarray or None
        Distances to nearest boundary point (retained for caller use; not
        used in figure rendering after CSV save was removed).
    tif_boundary : int
        Boundary mode (0=none, 3=TIFF mask; 1/2 not yet ported).
    boundary_measurement : bool
        Controls angle scaling (0–90° when ``True``; 0–180° when ``False``)
        and colormap thresholds.
    map_params : dict or None
        Filter parameters passed to ``draw_map``:
        ``STDfilter_size`` (default 24), ``SQUAREmaxfilter_size`` (default 12),
        ``GAUSSIANdiscfilter_sigma`` (default 4).  ``None`` uses all defaults.

    Returns
    -------
    (fig, rawmap, procmap) : tuple
        Matplotlib Figure with heatmap overlay, raw angle map (float64, NaN
        outside fiber centres), and processed map (uint8, smoothed).
    """
    from matplotlib.colors import ListedColormap

    if map_params is None:
        map_params = {}

    map_fibers = fiber_structure[in_curvs_flag]
    map_angles = angles[in_curvs_flag]

    raw_map, proc_map = draw_map(
        map_fibers, map_angles, img, boundary_measurement, map_params
    )

    fig, ax = plt.subplots(
        figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100
    )
    ax.imshow(img, cmap="gray")

    if boundary_measurement:
        tg = int(10 * 255 / 90)
        ty = int(45 * 255 / 90)
        tr = int(60 * 255 / 90)
    else:
        tg = 32
        ty = 64
        tr = 128

    colors = np.zeros((256, 3))
    colors[tg:ty, 1] = 1.0
    colors[ty:tr, 0:2] = 1.0
    colors[tr:, 0] = 1.0
    cmap = ListedColormap(colors)

    ax.imshow(proc_map, cmap=cmap, alpha=0.5)
    ax.axis("off")

    return fig, raw_map, proc_map


__all__ = [
    "draw_curvs",
    "draw_map",
    "compute_angle_histogram",
    "generate_fiber_overlay",
    "generate_fiber_heatmap",
]
