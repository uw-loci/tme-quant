"""
MATLAB-compatible ``imresize`` (bicubic / keys cubic kernel).

Adapted from https://github.com/fatheral/matlab_imresize (MIT license).
Used for BDcreation_reg2 / BDcreationHE2 parity with MathWorks ``imresize``.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def _derive_size_from_scale(img_shape: tuple[int, ...], scale: list[float]) -> list[int]:
    """Output size for a per-axis scale, using MATLAB's ``ceil``.

    Parameters
    ----------
    img_shape : tuple of int
        Input ``(rows, cols, ...)``.
    scale : list of float
        Scale for each of the first two axes.

    Returns
    -------
    list of int
        ``[ceil(scale * rows), ceil(scale * cols)]``.
    """
    output_shape: list[int] = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def _derive_scale_from_size(img_shape_in: tuple[int, ...], img_shape_out: tuple[int, ...]) -> list[float]:
    """Per-axis scale that turns ``img_shape_in`` into ``img_shape_out``.

    Parameters
    ----------
    img_shape_in : tuple of int
        Input shape.
    img_shape_out : tuple of int
        Output shape.

    Returns
    -------
    list of float
        ``[out / in]`` for the first two axes.
    """
    scale: list[float] = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def _triangle(x: np.ndarray) -> np.ndarray:
    """Bilinear interpolation kernel used by MATLAB ``imresize``.

    Parameters
    ----------
    x : numpy.ndarray
        Distance from the sample, in input pixels.

    Returns
    -------
    numpy.ndarray
        Kernel weights. Nonzero on ``[-1, 1]``.
    """
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    return np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)


def _cubic(x: np.ndarray) -> np.ndarray:
    """Keys bicubic kernel used by MATLAB ``imresize``.

    Parameters
    ----------
    x : numpy.ndarray
        Distance from the sample, in input pixels.

    Returns
    -------
    numpy.ndarray
        Kernel weights. Nonzero on ``[-2, 2]``.
    """
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
    """Antialias weights and source indices for one axis of ``imresize``.

    Parameters
    ----------
    in_length : int
        Input length along this axis.
    out_length : int
        Output length along this axis.
    scale : float
        ``out_length / in_length``.
    kernel : callable
        ``_cubic`` or ``_triangle``.
    k_width : float
        Kernel half-width in input pixels before antialias scaling.

    Returns
    -------
    weights : numpy.ndarray
        Normalized kernel weights for each output pixel.
    indices : numpy.ndarray
        Input indices those weights read, reflected at the borders.
    """
    if scale < 1:
        def h(x: np.ndarray) -> np.ndarray:
            """Antialiased kernel: scale the weights when shrinking.

            Parameters
            ----------
            x : numpy.ndarray
                Distance from the output sample.

            Returns
            -------
            numpy.ndarray
                Scaled kernel weights.
            """
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
    """Apply one axis of the separable ``imresize`` weighted sum.

    Parameters
    ----------
    inimg : numpy.ndarray
        Image with a channel axis.
    weights : numpy.ndarray
        Weights from :func:`_contributions`.
    indices : numpy.ndarray
        Source indices from :func:`_contributions`.
    dim : int
        ``0`` for rows, ``1`` for columns.

    Returns
    -------
    numpy.ndarray
        Image resized along ``dim``. uint8 input is rounded back to uint8.

    Raises
    ------
    ValueError
        If ``dim`` is not 0 or 1.
    """
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
    """Resize ``a`` along one axis. See :func:`_imresizevec`.

    Parameters
    ----------
    a : numpy.ndarray
        Image with a channel axis.
    dim : int
        ``0`` for rows, ``1`` for columns.
    weights : numpy.ndarray
        Weights from :func:`_contributions`.
    indices : numpy.ndarray
        Source indices from :func:`_contributions`.

    Returns
    -------
    numpy.ndarray
        Image resized along ``dim``.
    """
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
    method : {'bicubic', 'bilinear'}, optional
        Interpolation kernel. Default is ``'bicubic'``.

    Returns
    -------
    numpy.ndarray
        Resized image. uint8 input stays uint8; other dtypes come back float64.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or both size arguments are set, or neither is.
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
