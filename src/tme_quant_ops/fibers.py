"""Fiber measurement and curvelet extraction ops."""

from __future__ import annotations

from typing import Annotated, NamedTuple

from skop import Axes, op
from skop.types import ImageData, VectorsData


_Image2D = Annotated[ImageData, Axes("y", "x")]


class CurveletResult(NamedTuple):
    fibers: VectorsData
    fiber_count: int
    global_alignment: float


class NeighborhoodResult(NamedTuple):
    fiber_count: int
    mean_nearest_distance: float
    mean_local_alignment: float
    global_alignment: float


@op(env="tme-quant-curvelets", exclusive=True)
def extract_curvelets(
    image: _Image2D,
    keep: Annotated[float, {"min": 0.001, "max": 1.0, "step": 0.001}] = 0.05,
    scale: Annotated[int, {"min": 1, "max": 10, "step": 1}] = 1,
    grouping_radius: Annotated[
        float, {"min": 0.0, "max": 100.0, "step": 0.5}
    ] = 10.0,
    vector_length: Annotated[
        float, {"min": 1.0, "max": 100.0, "step": 1.0}
    ] = 10.0,
) -> CurveletResult:
    """Extract collagen orientations using the CurveLab backend.

    Args:
        image: A 2-D grayscale image.
        keep: Fraction of strongest curvelet coefficients to retain.
        scale: Curvelet scale selected for analysis.
        grouping_radius: Radius used to group nearby curvelets.
        vector_length: Display length of returned fiber vectors.

    Returns:
        Fiber vectors, their count, and global axial alignment.
    """
    import numpy as np

    from pycurvelets.fiber_ops import fibers_to_vectors, summarize_neighborhoods
    from pycurvelets.models import CurveletControlParameters
    from pycurvelets.new_curv import new_curv

    fibers, _, _ = new_curv(
        image,
        CurveletControlParameters(keep=keep, scale=scale, radius=grouping_radius),
    )
    if len(fibers) == 0:
        vectors = np.empty((0, 2, 2), dtype=float)
        return CurveletResult(vectors, 0, np.nan)
    vectors = fibers_to_vectors(
        fibers[["center_row", "center_col"]].to_numpy(),
        fibers["angle"].to_numpy(),
        length=vector_length,
    )
    summary = summarize_neighborhoods(vectors)
    return CurveletResult(vectors, summary.fiber_count, summary.global_alignment)


@op(env="tme-quant")
def summarize_fiber_neighborhoods(
    fibers: VectorsData,
    neighbors: Annotated[int, {"min": 1, "max": 64, "step": 1}] = 4,
) -> NeighborhoodResult:
    """Summarize spacing and axial alignment around fiber vectors.

    Args:
        fibers: Fiber centers and directions as vectors.
        neighbors: Number of nearest neighboring fibers to consider.

    Returns:
        Fiber count, spacing, local alignment, and global alignment.
    """
    from pycurvelets.fiber_ops import summarize_neighborhoods

    return NeighborhoodResult(*summarize_neighborhoods(fibers, neighbors=neighbors))
