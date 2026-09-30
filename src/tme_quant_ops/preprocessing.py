"""Fiber-enhancing preprocessing ops."""

from __future__ import annotations

from typing import Annotated

from skop import Axes, op
from skop.types import ImageData


_Image2D = Annotated[ImageData, Axes("y", "x", "c?")]


@op(env="tme-quant")
def enhance_tubeness(
    image: _Image2D,
    sigma: Annotated[float, {"min": 0.1, "max": 20.0, "step": 0.1}] = 1.0,
) -> ImageData:
    """Enhance bright tubular structures with a Meijering filter.

    Args:
        image: A 2-D grayscale or RGB(A) image.
        sigma: Structure scale in pixels.

    Returns:
        A floating-point enhanced image.
    """
    from pycurvelets.preprocessing import tubeness

    return tubeness(image, sigma=sigma)


@op(env="tme-quant")
def enhance_frangi(
    image: _Image2D,
    sigma_min: Annotated[float, {"min": 0.1, "max": 20.0, "step": 0.1}] = 1.0,
    sigma_max: Annotated[float, {"min": 0.1, "max": 50.0, "step": 0.1}] = 10.0,
    sigma_step: Annotated[float, {"min": 0.1, "max": 10.0, "step": 0.1}] = 1.0,
    beta: Annotated[float, {"min": 0.0, "max": 2.0, "step": 0.05}] = 0.5,
    gamma: Annotated[float, {"min": 0.0, "max": 100.0, "step": 0.5}] = 15.0,
) -> ImageData:
    """Enhance bright tubular structures with a Frangi filter.

    Args:
        image: A 2-D grayscale or RGB(A) image.
        sigma_min: Smallest structure scale in pixels.
        sigma_max: Largest structure scale in pixels.
        sigma_step: Spacing between evaluated scales.
        beta: Frangi blobness correction constant.
        gamma: Frangi structureness correction constant.

    Returns:
        A floating-point enhanced image.
    """
    from pycurvelets.preprocessing import frangi

    return frangi(
        image,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_step=sigma_step,
        beta=beta,
        gamma=gamma,
    )
