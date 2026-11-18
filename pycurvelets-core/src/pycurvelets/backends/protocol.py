"""The interface every curvelet transform backend implements.

``pycurvelets`` never imports a transform library itself. A backend package
(``pycurvelets-fdct``, ``pycurvelets-udct``) provides a class that satisfies
:class:`CurveletBackend` and registers it under the ``pycurvelets.backends``
entry-point group; :mod:`pycurvelets.backends.registry` finds and loads it.

Coefficients are exchanged as a nested list indexed ``[scale][wedge]``:
``coefficients[0]`` is the coarsest scale and ``coefficients[-1]`` the finest.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

Coefficients = list[list[np.ndarray]]


class WedgeGeometry(NamedTuple):
    """Where one wedge's coefficients sit in the image and what they measure.

    Attributes
    ----------
    angle_deg : float
        Fiber orientation, in degrees, that this wedge responds to. Same
        convention as ``new_curv`` output before angle grouping.
    rows, cols : np.ndarray
        1-D float arrays with one entry per coefficient of the wedge, in the
        order of ``np.ravel(coefficient_array)`` (C order). Values are image
        positions with the MATLAB 1-based convention; ``new_curv`` rounds them.
    """

    angle_deg: float
    rows: np.ndarray
    cols: np.ndarray


class ScaleGeometry(NamedTuple):
    """Geometry of the wedges ``new_curv`` reads at one scale.

    Attributes
    ----------
    angle_step_deg : float
        Angular spacing between neighbouring wedges, in degrees. ``new_curv``
        uses it as the bin width when it averages the angles of grouped
        curvelets.
    wedges : list of WedgeGeometry
        One entry per wedge to analyse. Entry ``k`` describes
        ``coefficients[scale][k]``. Backends whose wedges come in mirrored
        pairs list only one wedge of each pair.
    """

    angle_step_deg: float
    wedges: list[WedgeGeometry]


@runtime_checkable
class CurveletBackend(Protocol):
    """A 2D discrete curvelet transform usable by ``pycurvelets``.

    Implementations must be deterministic: the same image always gives the
    same coefficients, and ``scale_geometry`` must describe the arrays that
    ``forward`` returns for an image of that shape.
    """

    name: str

    def forward(self, img: np.ndarray) -> Coefficients:
        """Return the curvelet coefficients of a 2D image as ``[scale][wedge]``."""
        ...

    def scale_geometry(self, shape: tuple[int, int], scale: int) -> ScaleGeometry:
        """Describe the wedges at index ``scale`` of ``forward`` output for ``shape``."""
        ...

    def inverse(self, coefficients: Coefficients, shape: tuple[int, int]) -> np.ndarray:
        """Reconstruct an image of ``shape`` from ``[scale][wedge]`` coefficients."""
        ...
