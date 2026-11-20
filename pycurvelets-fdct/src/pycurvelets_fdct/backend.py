"""FDCT backend: CurveLab's wrapping transform through ``curvelops``.

The transform settings and the wedge geometry reproduce MATLAB CurveAlign's
``newCurv.m`` (``fdct_wrapping`` with 16 coarse angles and wavelets at the
finest scale), so ``new_curv`` output matches the MATLAB reference.
"""

from __future__ import annotations

import math

import numpy as np
from curvelops import fdct2d_wrapper
from pycurvelets.backends import Coefficients, ScaleGeometry, WedgeGeometry

NBANGLES_COARSE = 16
# CurveLab's ``allcurvelets`` flag: 0 puts wavelets at the finest scale.
ALL_CURVELETS = 0
# CurveLab numbers the wedges of a scale clockwise from this angle (degrees).
START_ANGLE_DEG = 225


def number_of_scales(shape: tuple[int, int]) -> int:
    """Number of FDCT scales MATLAB CurveAlign uses for an image of ``shape``."""
    return math.floor(math.log2(min(shape)) - 3)


def wedge_angle_deg(wedge: int, angle_step_deg: float) -> float:
    """Fiber orientation measured by wedge ``wedge`` (0-based) of a scale.

    The angle is the centre of the wedge's angular sector, folded into the
    range CurveAlign uses before it groups curvelets.
    """
    sector_start = START_ANGLE_DEG - angle_step_deg * wedge
    sector_end = START_ANGLE_DEG - angle_step_deg * (wedge + 1)
    angle = np.mean([sector_start, sector_end])
    if angle < 0:
        angle += 360
    if angle > 225:
        angle -= 180
    if angle < 45:
        angle += 180
    return float(angle)


class FDCTBackend:
    """CurveLab FDCT (wrapping) implementation of ``CurveletBackend``."""

    name = "fdct"

    def forward(self, img: np.ndarray) -> Coefficients:
        """Forward 2D FDCT of ``img`` as ``[scale][wedge]`` complex arrays."""
        return fdct2d_wrapper.fdct2d_forward_wrap(
            number_of_scales(img.shape), NBANGLES_COARSE, ALL_CURVELETS, img
        )

    def scale_geometry(self, shape: tuple[int, int], scale: int) -> ScaleGeometry:
        """Wedge angles and coefficient positions at one scale.

        Only the first half of the wedges is returned: FDCT wedge ``w`` and
        ``w + n/2`` cover the same orientation, as in ``newCurv.m``.
        """
        n_rows, n_cols = shape
        spacing_rows, spacing_cols, _, _, sizes_rows, sizes_cols = (
            fdct2d_wrapper.fdct2d_param_wrap(
                n_rows, n_cols, number_of_scales(shape), NBANGLES_COARSE, ALL_CURVELETS
            )
        )
        n_wedges = len(sizes_rows[scale])
        angle_step_deg = 360 / n_wedges

        wedges = []
        for w in range(n_wedges // 2):
            nx = sizes_rows[scale][w]
            ny = sizes_cols[scale][w]
            cx = math.ceil((nx + 1) / 2)
            cy = math.ceil((ny + 1) / 2)
            ix, iy = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
            rows = 1 + n_rows * (spacing_rows[scale][w] * (ix - cx) + 0.5)
            cols = 1 + n_cols * (spacing_cols[scale][w] * (iy - cy) + 0.5)
            # CurveLab coefficient arrays are the transpose of this (nx, ny)
            # grid, so Fortran order here lines up with C order on the arrays.
            wedges.append(
                WedgeGeometry(
                    angle_deg=wedge_angle_deg(w, angle_step_deg),
                    rows=rows.ravel(order="F"),
                    cols=cols.ravel(order="F"),
                )
            )
        return ScaleGeometry(angle_step_deg=angle_step_deg, wedges=wedges)

    def inverse(self, coefficients: Coefficients, shape: tuple[int, int]) -> np.ndarray:
        """Inverse 2D FDCT; returns a complex array of ``shape``."""
        return fdct2d_wrapper.fdct2d_inverse_wrap(
            shape[0],
            shape[1],
            number_of_scales(shape),
            NBANGLES_COARSE,
            ALL_CURVELETS,
            coefficients,
        )
