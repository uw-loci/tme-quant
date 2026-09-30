"""Segmentation ops.

Only NumPy and scikit-ops are imported during discovery. Runtime dependencies
belong inside each op body so discovery works in a minimal host environment.
"""

from __future__ import annotations

from typing import Annotated

from skop import Axes, op
from skop.types import ImageData, LabelsData


_Image2D = Annotated[ImageData, Axes("y", "x", "c?")]


@op(env="tme-quant")
def segment_threshold(
    image: _Image2D,
    method: Annotated[
        str,
        {"choices": ["otsu", "triangle", "isodata", "mean", "minimum"]},
    ] = "otsu",
    min_area: Annotated[int, {"min": 0, "max": 1_000_000, "step": 1}] = 100,
    fill_holes: bool = True,
    remove_border_objects: bool = True,
) -> LabelsData:
    """Segment a 2-D image using global thresholding.

    Args:
        image: A grayscale or RGB(A) image.
        method: Global threshold-selection method.
        min_area: Remove foreground objects smaller than this many pixels.
        fill_holes: Fill holes smaller than ``min_area`` pixels.
        remove_border_objects: Remove objects touching the image border.

    Returns:
        An integer label image with zero as background.
    """
    from pycurvelets.segmentation import segment_threshold as _segment_threshold

    return _segment_threshold(
        image,
        method=method,
        min_area=min_area,
        fill_holes=fill_holes,
        remove_border_objects=remove_border_objects,
    )
