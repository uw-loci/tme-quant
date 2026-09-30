"""Boundary-related ops."""

from __future__ import annotations

from typing import Annotated

from skop import Axes, op
from skop.types import LabelsData, PointsData


_Labels2D = Annotated[LabelsData, Axes("y", "x")]


@op(env="tme-quant")
def boundary_labels(
    labels: _Labels2D,
    thickness: Annotated[int, {"min": 1, "max": 25, "step": 1}] = 1,
) -> LabelsData:
    """Extract the inner boundaries of objects in a 2-D label image.

    Args:
        labels: Integer label image with zero as background.
        thickness: Boundary thickness in pixels.

    Returns:
        A binary label image containing the object boundaries.
    """
    from pycurvelets.segmentation import boundary_labels as _boundary_labels

    return _boundary_labels(labels, thickness=thickness)


@op(env="tme-quant")
def largest_boundary_points(
    labels: _Labels2D,
    simplify_tolerance: Annotated[
        float, {"min": 0.0, "max": 25.0, "step": 0.25}
    ] = 1.0,
) -> PointsData:
    """Extract the longest ordered contour from a 2-D label image.

    Args:
        labels: Integer label image with zero as background.
        simplify_tolerance: Polygon simplification tolerance in pixels.

    Returns:
        Ordered boundary coordinates in ``(row, column)`` order.
    """
    from pycurvelets.segmentation import largest_boundary_points as _points

    return _points(labels, simplify_tolerance=simplify_tolerance)
