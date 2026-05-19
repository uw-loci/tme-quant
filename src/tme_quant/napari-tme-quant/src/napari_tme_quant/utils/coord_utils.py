"""Coordinate translation between tme_quant (row, col) and napari (y, x) / (z, y, x).

tme_quant always uses (row, col) = (y, x).
napari uses (y, x) for 2D layers and (z, y, x) for 3D.

All translations must go through this module — never inline them in widgets or
controllers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence


def row_col_to_yx(pos: "Sequence[float]") -> tuple[float, float]:
    """Convert a (row, col) position to a napari (y, x) tuple.

    Parameters
    ----------
    pos : (row, col) — as returned by FiberObject.get_position()

    Returns
    -------
    (y, x) — ready for use as napari layer coordinates
    """
    return (float(pos[0]), float(pos[1]))


def yx_to_row_col(yx: "Sequence[float]") -> tuple[float, float]:
    """Convert a napari (y, x) coordinate to a (row, col) pair."""
    return (float(yx[0]), float(yx[1]))


def row_col_z_to_zyx(pos: "Sequence[float]") -> tuple[float, float, float]:
    """Convert a (z, row, col) position to a napari (z, y, x) tuple."""
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def fiber_df_to_napari_points(df: pd.DataFrame) -> np.ndarray:
    """Convert a fiber DataFrame to a napari Points layer data array.

    Expects columns ``center_row`` and ``center_col`` (canonical names from
    REFACTORING_GUIDE §8.3). Returns an (N, 2) float64 array in (y, x) order.

    Parameters
    ----------
    df : DataFrame with at least ``center_row`` and ``center_col`` columns.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype float64 — napari (y, x) coords
    """
    return df[["center_row", "center_col"]].to_numpy(dtype=np.float64)


def fiber_df_to_napari_shapes(df: pd.DataFrame) -> list[np.ndarray]:
    """Convert a fiber DataFrame to a list of 2-point shape arrays for napari Shapes.

    Each shape is a (2, 2) array: [[y_start, x_start], [y_end, x_end]],
    representing a line segment along the fiber orientation.

    This is a simplified representation (center ± half-length along the angle).
    A full centerline representation is used when ``centerline`` data is available.

    Parameters
    ----------
    df : DataFrame with ``center_row``, ``center_col``, ``angle``, ``length`` columns.
    """
    shapes = []
    for _, row in df.iterrows():
        cy, cx = float(row["center_row"]), float(row["center_col"])
        angle_rad = np.deg2rad(float(row.get("angle", 0.0)))
        half_len = float(row.get("length", 10.0)) / 2.0
        dy = half_len * np.cos(angle_rad)
        dx = half_len * np.sin(angle_rad)
        shapes.append(np.array([[cy - dy, cx - dx], [cy + dy, cx + dx]]))
    return shapes
