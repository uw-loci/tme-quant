"""
MATLAB-compatible ``imresize`` (bicubic / keys cubic kernel).

Adapted from https://github.com/fatheral/matlab_imresize (MIT license).
Used for BDcreation_reg2 / BDcreationHE2 parity with MathWorks ``imresize``.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def _derive_size_from_scale(img_shape: tuple[int, ...], scale: list[float]) -> list[int]:
    output_shape: list[int] = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def _derive_scale_from_size(img_shape_in: tuple[int, ...], img_shape_out: tuple[int, ...]) -> list[float]:
    scale: list[float] = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def _triangle(x: np.ndarray) -> np.ndarray:
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    return np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)


def _cubic(x: np.ndarray) -> np.ndarray:
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    return np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
        (1 < absx) & (absx <= 2),
    )


def _contributions(
    in_length: int,
    out_length: int,
    scale: float,
    kernel,
    k_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    if scale < 1:
        def h(x: np.ndarray) -> np.ndarray:
            return scale * kernel(scale * x)

        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(p) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def _imresizevec(
    inimg: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    dim: int,
) -> np.ndarray:
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    else:
        raise ValueError(f"dim must be 0 or 1, got {dim}")
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    return outimg


def _resize_along_dim(
    a: np.ndarray,
    dim: int,
    weights: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return _imresizevec(a, weights, indices, dim)


def matlab_imresize(
    image: np.ndarray,
    *,
    scalar_scale: float | None = None,
    output_shape: tuple[int, int] | None = None,
    method: str = "bicubic",
) -> np.ndarray:
    """
    Resize 2-D grayscale or channel-last RGB to match MATLAB ``imresize``.

    Parameters
    ----------
    image
        ``(H, W)`` or ``(H, W, C)`` array, float or uint8.
    scalar_scale
        Uniform scale factor (e.g. ``2/3`` for ppm=3 cap).
    output_shape
        Target ``(rows, cols)``.
    method
        ``'bicubic'`` (keys cubic) or ``'bilinear'``.
    """
    if method == "bicubic":
        kernel = _cubic
    elif method == "bilinear":
        kernel = _triangle
    else:
        raise ValueError(f"unidentified kernel method: {method!r}")

    kernel_width = 4.0
    if scalar_scale is not None and output_shape is not None:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    i = np.asarray(image)
    if scalar_scale is not None:
        scalar_scale_f = float(scalar_scale)
        scale = [scalar_scale_f, scalar_scale_f]
        output_size = _derive_size_from_scale(i.shape, scale)
    elif output_shape is not None:
        scale = _derive_scale_from_size(i.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError("either scalar_scale OR output_shape should be defined")

    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights_list: list[np.ndarray] = []
    indices_list: list[np.ndarray] = []
    for k in range(2):
        w, ind = _contributions(i.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights_list.append(w)
        indices_list.append(ind)

    b = np.copy(i)
    flag2d = False
    if b.ndim == 2:
        b = np.expand_dims(b, axis=2)
        flag2d = True
    for k in range(2):
        dim = int(order[k])
        b = _resize_along_dim(b, dim, weights_list[dim], indices_list[dim])
    if flag2d:
        b = np.squeeze(b, axis=2)
    return b.astype(np.float64) if b.dtype != np.uint8 else b
