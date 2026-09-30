"""Fiber-to-boundary analysis ops."""

from __future__ import annotations

from typing import Annotated, NamedTuple

from skop import op
from skop.types import PointsData, VectorsData


class TACSResult(NamedTuple):
    associations: VectorsData
    fiber_count: int
    mean_distance: float
    mean_relative_angle: float
    tacs3_fraction: float


@op(env="tme-quant")
def analyze_tacs(
    fibers: VectorsData,
    boundary: PointsData,
    min_distance: Annotated[
        float, {"min": 0.0, "max": 10_000.0, "step": 1.0}
    ] = 0.0,
    max_distance: Annotated[
        float, {"min": 0.0, "max": 10_000.0, "step": 1.0}
    ] = 200.0,
    tacs3_threshold: Annotated[
        float, {"min": 0.0, "max": 90.0, "step": 1.0}
    ] = 60.0,
) -> TACSResult:
    """Measure fiber orientation relative to an ordered boundary.

    Args:
        fibers: Fiber centers and directions as vectors.
        boundary: Ordered boundary coordinates in ``(row, column)`` order.
        min_distance: Minimum included fiber-to-boundary distance in pixels.
        max_distance: Maximum included fiber-to-boundary distance in pixels.
        tacs3_threshold: Minimum perpendicular angle counted as TACS-3-like.

    Returns:
        Association vectors and aggregate TACS measurements.
    """
    from pycurvelets.fiber_ops import analyze_tacs as _analyze_tacs

    return TACSResult(
        *_analyze_tacs(
            fibers,
            boundary,
            min_distance=min_distance,
            max_distance=max_distance,
            tacs3_threshold=tacs3_threshold,
        )
    )
