"""UDCT curvelet backend: the uniform discrete curvelet transform from the
``curvelets`` package (``curvelets.numpy.UDCT``, MIT licensed, on PyPI).

Conventions, chosen so new_curv treats this backend like FDCT:

* Scales: the same count as FDCT, ``floor(log2(min(M, N)) - 3)``. UDCT's
  frequency bands do not depend on the image size, and with this count the
  scale ``new_curv`` analyses by default (second finest) covers roughly
  0.17-0.35 cycles/pixel in both backends.
* Wedges per direction at curvelet scale j (1 = coarsest):
  ``3 * 2**ceil((j - 1) / 2)``. UDCT needs multiples of 3; doubling every
  other scale follows FDCT's parabolic scaling.
* ``high_frequency_mode="curvelet"`` (the default). In curvelets 1.2 the
  "wavelet" mode puts most of a low-frequency signal's energy in its top
  scale, which would break new_curv's count-from-the-finest scale indexing.
* ``transform_kind="real"``: each wedge window is one-sided in frequency, so
  every wedge is a distinct orientation (no mirrored pairs) and
  ``scale_geometry`` lists all of them.
* Shapes: UDCT needs each axis divisible by its decimation ratios
  (``2**(num_scales - 1)`` overall), and otherwise silently reconstructs
  wrongly. Images are padded symmetrically at the bottom and right to that
  multiple and cropped after the inverse. Coefficient positions are pixel
  positions in the padded image; new_curv's edge filter drops the ones that
  fall in the padding.
* Angles: degrees counterclockwise from the column axis (rows pointing down),
  taken from each window's frequency centroid and reported in (45, 225] like
  FDCT, because new_curv treats an angle of 0 as "no curvelet".
"""

from __future__ import annotations

import itertools
import logging
import math
from functools import lru_cache

import numpy as np
from curvelets.numpy import UDCT
from pycurvelets.backends import Coefficients, ScaleGeometry, WedgeGeometry

logger = logging.getLogger(__name__)

MIN_SCALES = 3
BASE_WEDGES_PER_DIRECTION = 3
# Angles are reported in (ANGLE_RANGE_START_DEG, ANGLE_RANGE_START_DEG + 180].
ANGLE_RANGE_START_DEG = 45.0


def number_of_scales(shape: tuple[int, int]) -> int:
    """Scale count for ``shape``, including the lowpass scale (FDCT's formula)."""
    n = math.floor(math.log2(min(shape)) - 3)
    if n < MIN_SCALES:
        raise ValueError(
            f"image shape {tuple(shape)} is too small for the udct backend: "
            f"it gives {n} scales, at least {MIN_SCALES} are needed (min side >= 64)"
        )
    return n


def wedges_per_direction(curvelet_scale: int) -> int:
    """Wedges per direction at curvelet scale ``curvelet_scale`` (1 = coarsest)."""
    return BASE_WEDGES_PER_DIRECTION * 2 ** math.ceil((curvelet_scale - 1) / 2)


def angular_wedges_config(num_scales: int) -> np.ndarray:
    """UDCT ``angular_wedges_config``: one row per curvelet scale, one column per axis."""
    return np.array([[wedges_per_direction(j)] * 2 for j in range(1, num_scales)])


def padded_shape(shape: tuple[int, int], num_scales: int) -> tuple[int, int]:
    """Smallest shape >= ``shape`` whose sides are multiples of ``2**(num_scales - 1)``."""
    multiple = 2 ** (num_scales - 1)
    return tuple(int(math.ceil(side / multiple) * multiple) for side in shape)


def orientation_deg(window, grid_shape: tuple[int, int]) -> float:
    """Orientation (degrees, in (45, 225]) of the structures a UDCT window responds to.

    The window's frequency centroid is averaged on doubled angles, so a window
    straddling the frequency axis still gets one direction; the spatial
    orientation is perpendicular to that frequency direction.
    """
    row_freq = np.fft.fftfreq(grid_shape[0])
    col_freq = np.fft.fftfreq(grid_shape[1])
    rows, cols = np.unravel_index(np.asarray(window.indices), grid_shape)
    weights = np.asarray(window.values, dtype=float) ** 2
    phi = np.arctan2(row_freq[rows], col_freq[cols])
    mean_phi_deg = np.degrees(np.angle(np.sum(weights * np.exp(2j * phi))) / 2)
    theta = (90.0 - mean_phi_deg) % 180.0
    if theta <= ANGLE_RANGE_START_DEG:
        theta += 180.0
    return float(theta)


@lru_cache(maxsize=8)
def _transform(grid_shape: tuple[int, int], num_scales: int) -> UDCT:
    """UDCT for a padded shape. Building the windows is the expensive part."""
    config = angular_wedges_config(num_scales)
    logger.debug("building UDCT for %s with wedges per direction %s", grid_shape, config[:, 0].tolist())
    return UDCT(grid_shape, angular_wedges_config=config, high_frequency_mode="curvelet", transform_kind="real")


class UDCTBackend:
    """Uniform discrete curvelet transform (curvelets.numpy.UDCT)."""

    name = "udct"

    def _plan(self, shape: tuple[int, int]) -> tuple[UDCT, tuple[int, int]]:
        shape = tuple(int(s) for s in shape)
        if len(shape) != 2:
            raise ValueError(f"the udct backend is 2-D only, got shape {shape}")
        num_scales = number_of_scales(shape)
        grid = padded_shape(shape, num_scales)
        return _transform(grid, num_scales), grid

    def forward(self, img: np.ndarray) -> Coefficients:
        img = np.asarray(img, dtype=np.float64)
        transform, grid = self._plan(img.shape)
        pad = ((0, grid[0] - img.shape[0]), (0, grid[1] - img.shape[1]))
        padded = np.pad(img, pad, mode="symmetric") if any(p[1] for p in pad) else img
        nested = transform.forward(padded)
        # [scale][direction][wedge] -> [scale][wedge], directions in order.
        return [[np.asarray(w) for direction in scale for w in direction] for scale in nested]

    def scale_geometry(self, shape: tuple[int, int], scale: int) -> ScaleGeometry:
        return _scale_geometry(self._plan(shape)[1], number_of_scales(tuple(shape)), scale)

    def inverse(self, coefficients: Coefficients, shape: tuple[int, int]) -> np.ndarray:
        transform, _ = self._plan(shape)
        nested = []
        for s, flat in enumerate(coefficients):
            sizes = [len(direction) for direction in transform.windows[s]]
            if sum(sizes) != len(flat):
                raise ValueError(
                    f"udct inverse: scale {s} has {len(flat)} wedges, the transform for shape "
                    f"{tuple(shape)} expects {sum(sizes)}"
                )
            starts = np.cumsum([0, *sizes])
            nested.append(
                [[np.asarray(w, dtype=np.complex128) for w in flat[a:b]] for a, b in itertools.pairwise(starts)]
            )
        reconstructed = np.real(transform.backward(nested))
        return reconstructed[: shape[0], : shape[1]]


@lru_cache(maxsize=32)
def _scale_geometry(grid: tuple[int, int], num_scales: int, scale: int) -> ScaleGeometry:
    transform = _transform(grid, num_scales)
    if not 0 <= scale < len(transform.windows):
        raise ValueError(f"scale {scale} out of range; the transform has {len(transform.windows)} scales")
    wedges = []
    for d, direction in enumerate(transform.windows[scale]):
        step_rows, step_cols = (int(r) for r in np.asarray(transform.decimation_ratios[scale][d]).ravel())
        n_rows, n_cols = grid[0] // step_rows, grid[1] // step_cols
        # Coefficient (i, j) sits at 0-based pixel (i * step_rows, j * step_cols).
        rows, cols = np.meshgrid(
            1 + step_rows * np.arange(n_rows, dtype=float),
            1 + step_cols * np.arange(n_cols, dtype=float),
            indexing="ij",
        )
        for window in direction:
            wedges.append(WedgeGeometry(orientation_deg(window, grid), rows.ravel(), cols.ravel()))
    return ScaleGeometry(angle_step_deg=180.0 / len(wedges), wedges=wedges)
