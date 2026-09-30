"""UI-independent image enhancement functions."""

from __future__ import annotations

import numpy as np


def tubeness(image: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    """Enhance bright tubular structures with the Meijering filter."""
    from skimage.filters import meijering

    image = _grayscale(image)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return np.asarray(
        meijering(image, sigmas=[sigma], black_ridges=False), dtype=np.float32
    )


def frangi(
    image: np.ndarray,
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 10.0,
    sigma_step: float = 1.0,
    beta: float = 0.5,
    gamma: float = 15.0,
) -> np.ndarray:
    """Enhance bright tubular structures with the Frangi filter."""
    from skimage.filters import frangi as skimage_frangi

    image = _grayscale(image)
    if sigma_min <= 0 or sigma_max < sigma_min or sigma_step <= 0:
        raise ValueError("sigmas must satisfy 0 < sigma_min <= sigma_max and step > 0")
    sigmas = np.arange(sigma_min, sigma_max + sigma_step / 2, sigma_step)
    return np.asarray(
        skimage_frangi(
            image,
            sigmas=sigmas,
            beta=beta,
            gamma=gamma,
            black_ridges=False,
        ),
        dtype=np.float32,
    )


def _grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return np.asarray(
            0.2125 * image[..., 0]
            + 0.7154 * image[..., 1]
            + 0.0721 * image[..., 2]
        )
    raise ValueError("expected a 2-D grayscale or RGB(A) image")


__all__ = ["frangi", "tubeness"]
