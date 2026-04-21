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

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib  # noqa: F401 — core dep; imported to confirm availability at load time

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


__all__ = ["draw_curvs", "draw_map"]
