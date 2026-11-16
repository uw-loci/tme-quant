"""MATLAB Image Processing Toolbox ports shared by registration and annotation.

One place for the helpers that match MathWorks numerics: rounding, ``imresize``,
color conversion, ``graythresh``, ``imfilter``, ``histeq``, ``strel('disk')``,
``imwarp``, and the unseeded ``kmeans`` replay. Registration and annotation call
these functions. They do not redefine them.

The module is private. :mod:`pycurvelets` still exports only the pipeline entry
points (``shg_he_registration``, ``tumor_annotation_from_he``, and the
``BDcreation*`` aliases). The ``matlab_`` prefix stays on each function because
the result is the MATLAB call, which is not what NumPy or scikit-image return
for the same name.

``matlab_imresize`` is adapted from https://github.com/fatheral/matlab_imresize
(MIT).

The ITK v3 ``imregtform`` engine stays in :mod:`pycurvelets._itk_v3_matlab_engine`.
``matlab_imregtform_v3``, ``matlab_default_settings``, and ``matlab_T_to_A`` are
that engine, not image helpers, and they need the ITK classes defined beside them.
"""

from __future__ import annotations

from importlib import resources
from math import ceil
from typing import Any

import numpy as np
from scipy import ndimage

# MATLAB ``graythresh`` histograms a uint8 image: one bin per level 0..255.
_GRAYTHRESH_BINS = 256

# Full-precision Rec.601 weights MATLAB's rgb2gray uses (not 0.2989/0.5870/0.1140).
_RGB2GRAY_COEF = (0.298936021293775, 0.587043074451121, 0.114020904255103)

__all__ = [
    "gaussian_filter_matlab_like",
    "matlab_decorrstretch_uint8",
    "matlab_fspecial_disk",
    "matlab_fspecial_gaussian",
    "matlab_graythresh",
    "matlab_grayxform",
    "matlab_histeq",
    "matlab_im2bw",
    "matlab_im2double",
    "matlab_im2uint8",
    "matlab_imadjust_auto_integer",
    "matlab_imadjust_double_rgb",
    "matlab_imfilter",
    "matlab_imfilter_keep_class",
    "matlab_imhist_counts",
    "matlab_imresize",
    "matlab_imwarp_bilinear",
    "matlab_kmeans",
    "matlab_rgb2gray",
    "matlab_rgb2gray_uint8",
    "matlab_rgb2hsv",
    "matlab_round",
    "matlab_srgb2lab_uint8",
    "matlab_std_std_2d",
    "matlab_strel_disk",
    "matlab_stretchlim",
]

# ---------------------------------------------------------------------------
# imresize (bicubic / bilinear). Adapted from fatheral/matlab_imresize (MIT).
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Color, threshold, and filtering.
# ---------------------------------------------------------------------------

def matlab_rgb2hsv(rgb: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert float RGB in ``[0, 1]`` to MATLAB ``rgb2hsv`` order.

    Hue, saturation, and value are each in ``[0, 1]``. Hue wraps the same way
    as MathWorks ``rgb2hsv``, which is what the CurveAlign hue bands assume.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image. Channel axis is ``axis`` (default last).
    axis : int, optional
        Axis of the color channels. Default is ``-1``.

    Returns
    -------
    numpy.ndarray
        HSV image, same layout as the input.

    Raises
    ------
    ValueError
        If ``rgb`` is not an RGB image.
    """
    arr = np.moveaxis(np.asarray(rgb, dtype=np.float64), axis, -1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB image, got shape {rgb.shape}.")
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    m = np.minimum(np.minimum(r, g), b)
    delta = v - m
    s = np.zeros_like(v)
    nonzero_v = v > 0
    s[nonzero_v] = delta[nonzero_v] / v[nonzero_v]

    h = np.zeros_like(v)
    mask = delta > 0
    # Avoid division by zero; only fill where delta > 0.
    r_eq = mask & (v == r)
    g_eq = mask & (v == g) & ~r_eq
    b_eq = mask & (v == b) & ~r_eq & ~g_eq
    h[r_eq] = np.mod((g[r_eq] - b[r_eq]) / delta[r_eq], 6.0) / 6.0
    h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq] + 2.0) / 6.0
    h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq] + 4.0) / 6.0
    h = np.clip(h, 0.0, 1.0)
    out = np.stack([h, s, v], axis=-1)
    if axis != -1:
        out = np.moveaxis(out, -1, axis)
    return out


def matlab_round(x: float | np.ndarray) -> float | np.ndarray:
    """Round half away from zero, matching MATLAB ``round``.

    ``numpy.round`` sends halves to the nearest even integer. MATLAB sends
    them away from zero, which changes structuring-element radii.

    Parameters
    ----------
    x : float or numpy.ndarray
        Value or array to round.

    Returns
    -------
    float or numpy.ndarray
        Scalar in, scalar out. Array in, array out.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    if np.ndim(x) == 0:
        return float(out)
    return out


def matlab_rgb2gray(rgb: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert RGB to grayscale with MATLAB's full-precision coefficients.

    MATLAB ``rgb2gray`` uses the first row of ``inv([1 0.956 0.621; 1 -0.272
    -0.647; 1 -1.106 1.703])``. The rounded ``[0.2989, 0.5870, 0.1140]`` set
    differs by up to about ``1e-4``. A 2-D array is returned unchanged.

    Parameters
    ----------
    rgb : numpy.ndarray
        Grayscale ``(H, W)`` or RGB with the channel axis at ``axis``.
    axis : int, optional
        Channel axis for RGB input. Default is ``-1``.

    Returns
    -------
    numpy.ndarray
        Float64 grayscale image.

    Raises
    ------
    ValueError
        If an array with a channel axis has fewer than 3 channels.
    """
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB with >=3 channels, got shape {rgb.shape}.")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cr, cg, cb = _RGB2GRAY_COEF
    return cr * r + cg * g + cb * b


def matlab_graythresh(image: np.ndarray) -> float:
    """Otsu threshold matching MATLAB ``graythresh`` on a double image.

    MATLAB does ``counts = imhist(im2uint8(I(:)), 256)``: one bin per uint8
    level after rounding ``I * 255``. Otsu then runs on bin indices ``1..256``,
    ties are the mean of the arg-max indices, and the threshold is
    ``(idx - 1) / 255``. The bin count is fixed at :data:`_GRAYTHRESH_BINS`
    because that is the only histogram ``graythresh`` builds.

    Parameters
    ----------
    image : numpy.ndarray
        Double image, typically intensities in ``[0, 1]``.

    Returns
    -------
    float
        Threshold in ``[0, 1]``. ``0.0`` when every between-class variance
        is NaN or infinite.

    Raises
    ------
    ValueError
        If ``image`` is empty.
    """
    nbins = _GRAYTHRESH_BINS
    arr = np.asarray(image, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot compute graythresh for empty image.")
    # im2uint8 for double input: clip to [0, 1], scale by 255, round half away
    # from zero (values are non-negative here so np.floor(x + 0.5) matches).
    levels = np.floor(np.clip(arr, 0.0, 1.0) * (nbins - 1) + 0.5).astype(np.int64)
    counts = np.bincount(levels, minlength=nbins).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot compute graythresh for empty histogram.")
    p = counts / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, nbins + 1, dtype=np.float64))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    # MATLAB max() ignores NaN (0/0 at omega==0 or 1) but propagates Inf.
    not_nan = ~np.isnan(sigma_b_squared)
    if not not_nan.any():
        return 0.0
    maxval = np.max(sigma_b_squared[not_nan])
    if not np.isfinite(maxval):
        return 0.0
    idx = np.mean(np.flatnonzero(sigma_b_squared == maxval)) + 1.0  # 1-based
    return float((idx - 1.0) / (nbins - 1))


def matlab_fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """Square Gaussian kernel matching ``fspecial('gaussian', size, sigma)``.

    Parameters
    ----------
    size : int
        Side length in pixels. Must be at least 1.
    sigma : float
        Standard deviation. Must be greater than 0.

    Returns
    -------
    numpy.ndarray
        Kernel of shape ``(size, size)`` that sums to 1.

    Raises
    ------
    ValueError
        If ``size`` or ``sigma`` is out of range.
    """
    if size < 1:
        raise ValueError(f"Kernel size must be >= 1, got {size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(x, x)
    h = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return h / np.sum(h)


def matlab_imfilter(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    boundary: str = "zero",
) -> np.ndarray:
    """Correlate ``image`` with ``kernel``, matching MATLAB ``imfilter``.

    MATLAB centres an even-sized kernel on the top-left of the two middle
    samples. SciPy centres it on the bottom-right, so even axes use
    ``origin=-1``.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.
    kernel : numpy.ndarray
        Filter kernel. Must have the same number of dimensions as ``image``.
    boundary : {'zero', 'replicate'}, optional
        ``'zero'`` pads with 0 (MATLAB's default). ``'replicate'`` repeats
        the edge pixel. Default is ``'zero'``.

    Returns
    -------
    numpy.ndarray
        Filtered image, float64.

    Raises
    ------
    ValueError
        If the kernel rank does not match the image, or ``boundary`` is unknown.
    """
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != img.ndim:
        raise ValueError(f"kernel ndim {k.ndim} != image ndim {img.ndim}")
    # MATLAB places the kernel centre at floor((size+1)/2) (1-based), i.e. the
    # top-left of the two middle elements for even sizes, whereas scipy uses
    # size//2 (bottom-right). origin=-1 on even axes realigns them; this
    # matters for fspecial('gaussian', floor(ppm)) with ppm = 2 (2x2 kernel).
    origin = tuple(-1 if (s % 2 == 0) else 0 for s in k.shape)
    if boundary == "zero":
        return ndimage.correlate(
            img, k, mode="constant", cval=0.0, origin=origin
        ).astype(np.float64)
    if boundary == "replicate":
        return ndimage.correlate(img, k, mode="nearest", origin=origin).astype(np.float64)
    raise ValueError(f"boundary must be 'zero' or 'replicate', got {boundary!r}")

# ---------------------------------------------------------------------------
# Class conversion, histogram equalization, and disk kernels.
# ---------------------------------------------------------------------------

def matlab_im2double(image: np.ndarray) -> np.ndarray:
    """Scale an image to ``[0, 1]`` the way MATLAB ``im2double`` does.

    Parameters
    ----------
    image : numpy.ndarray
        uint8, uint16, bool, or float image.

    Returns
    -------
    numpy.ndarray
        Float64 image. uint8 is divided by 255, uint16 by 65535. Float input
        is not rescaled.
    """
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float64) / 65535.0
    if a.dtype == np.bool_:
        return a.astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def matlab_im2bw(image: np.ndarray, thresh: float) -> np.ndarray:
    """Threshold an image the way ``im2bw(I, level)`` does.

    Parameters
    ----------
    image : numpy.ndarray
        Image to threshold. A boolean image is copied.
    thresh : float
        Level in ``[0, 1]`` after ``im2double``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, true where the scaled image is strictly above ``thresh``.
    """
    a = np.asarray(image)
    if a.dtype == np.bool_:
        return a.copy()
    return matlab_im2double(a) > float(thresh)


def matlab_fspecial_disk(radius: float) -> np.ndarray:
    """Circular averaging kernel matching ``fspecial('disk', radius)``.

    Parameters
    ----------
    radius : float
        Disk radius in pixels. Must be greater than 0.

    Returns
    -------
    numpy.ndarray
        Kernel that sums to 1.

    Raises
    ------
    ValueError
        If ``radius`` is not positive.
    RuntimeError
        If the kernel sums to zero.
    """
    rad = float(radius)
    if rad <= 0:
        raise ValueError(f"disk radius must be > 0, got {rad}")
    crad = int(np.ceil(rad - 0.5))
    yy, xx = np.meshgrid(
        np.arange(-crad, crad + 1, dtype=np.float64),
        np.arange(-crad, crad + 1, dtype=np.float64),
        indexing="ij",
    )
    maxxy = np.maximum(np.abs(xx), np.abs(yy))
    minxy = np.minimum(np.abs(xx), np.abs(yy))
    rad2 = rad * rad
    max_p = maxxy + 0.5
    max_m = maxxy - 0.5
    min_p = minxy + 0.5
    min_m = minxy - 0.5
    m1 = np.where(
        rad2 < max_p**2 + min_m**2,
        min_m,
        np.sqrt(np.maximum(rad2 - max_p**2, 0.0)),
    )
    m2 = np.where(
        rad2 > max_m**2 + min_p**2,
        min_p,
        np.sqrt(np.maximum(rad2 - max_m**2, 0.0)),
    )
    # asin domain; float noise can push m/rad slightly outside [-1, 1].
    a1 = np.arcsin(np.clip(m1 / rad, -1.0, 1.0))
    a2 = np.arcsin(np.clip(m2 / rad, -1.0, 1.0))
    ring = (
        (rad2 < max_p**2 + min_p**2) & (rad2 > max_m**2 + min_m**2)
    ) | ((minxy == 0) & (max_m < rad) & (max_p >= rad))
    sgrid = (
        rad2 * (0.5 * (a2 - a1) + 0.25 * (np.sin(2.0 * a2) - np.sin(2.0 * a1)))
        - max_m * (m2 - m1)
        + (m1 - minxy + 0.5)
    ) * ring
    sgrid = sgrid + ((max_p**2 + min_p**2) < rad2)
    sgrid[crad, crad] = min(np.pi * rad2, np.pi / 2.0)
    if crad > 0 and rad > (crad - 0.5) and rad2 < (crad - 0.5) ** 2 + 0.25:
        m1s = np.sqrt(rad2 - (crad - 0.5) ** 2)
        m1n = m1s / rad
        sg0 = 2.0 * (rad2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2.0 * np.arcsin(m1n))) - m1s * (crad - 0.5))
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0
        sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0
        sgrid[crad, 1] = sgrid[crad, 1] - sg0
        sgrid[1, crad] = sgrid[1, crad] - sg0
    sgrid[crad, crad] = min(float(sgrid[crad, crad]), 1.0)
    total = float(sgrid.sum())
    if total == 0.0:
        raise RuntimeError("fspecial('disk') kernel summed to 0")
    return sgrid / total


def _histeq_transform(counts: np.ndarray, hgram: np.ndarray, n_pixels: int) -> np.ndarray:
    """Build the intensity map used inside MATLAB ``histeq``.

    Parameters
    ----------
    counts : numpy.ndarray
        Input histogram.
    hgram : numpy.ndarray
        Desired histogram. Its length is the number of output levels.
    n_pixels : int
        Number of pixels. Used as the error sentinels in ``histeq.m``.

    Returns
    -------
    numpy.ndarray
        Lookup values in ``[0, 1]``, one per input bin.
    """
    nn = np.asarray(counts, dtype=np.float64).ravel()
    desired = np.asarray(hgram, dtype=np.float64).ravel()
    m = desired.size
    cum = np.cumsum(nn)
    cumd = np.cumsum(desired)
    # nn with first and last bins zeroed, then /2 (histeq.m ``tol``).
    tol = nn.copy()
    tol[0] = 0.0
    tol[-1] = 0.0
    tol *= 0.5
    err = cumd[:, None] - cum[None, :] + tol[None, :]
    thresh = -float(n_pixels) * np.sqrt(np.finfo(np.float64).eps)
    err = np.where(err < thresh, float(n_pixels), err)
    t_idx = np.argmin(err, axis=0)
    return t_idx.astype(np.float64) / (m - 1)


def matlab_grayxform(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a lookup table the way MATLAB ``grayxform`` does.

    ``transform`` holds values in ``[0, 1]``. A uint8 image with a 256-entry
    table maps ``floor(255 * T(I) + 0.5)``. A double image indexes
    ``floor(I * (N - 1) + 0.5)`` and clamps values outside ``[0, 1]`` to the
    end bins.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 or double image.
    transform : numpy.ndarray
        Lookup table. At least two entries.

    Returns
    -------
    numpy.ndarray
        Mapped image. uint8 in, uint8 out. Double in, double out.

    Raises
    ------
    ValueError
        If ``transform`` has fewer than two entries.
    """
    t = np.asarray(transform, dtype=np.float64).ravel()
    if t.size < 2:
        raise ValueError("grayxform transform must have at least 2 entries")
    a = np.asarray(image)
    n_levels = t.size - 1
    if a.dtype == np.uint8:
        if n_levels == 255:
            mapped = 255.0 * t[a.astype(np.int64)] + 0.5
            return np.clip(np.floor(mapped), 0, 255).astype(np.uint8)
        scale = n_levels / 255.0
        index = np.floor(scale * a.astype(np.float64) + 0.5).astype(np.int64)
        index = np.clip(index, 0, n_levels)
        mapped = 255.0 * t[index] + 0.5
        return np.clip(np.floor(mapped), 0, 255).astype(np.uint8)
    x = a.astype(np.float64)
    idx = np.empty(x.shape, dtype=np.int64)
    inside = (x >= 0.0) & (x <= 1.0)
    idx[inside] = np.floor(x[inside] * n_levels + 0.5).astype(np.int64)
    idx[x > 1.0] = n_levels
    idx[x < 0.0] = 0
    idx = np.clip(idx, 0, n_levels)
    return t[idx]


def matlab_histeq(image: np.ndarray, n_levels: int = 64) -> np.ndarray:
    """Histogram-equalize an image the way MATLAB ``histeq`` does.

    MATLAB's default is 64 output levels for every class, including uint8.
    ``BDcreationHE.m`` calls ``histeq`` with that default. The input histogram
    always uses 256 bins. The output keeps the input class.

    Parameters
    ----------
    image : numpy.ndarray
        uint8 or double image.
    n_levels : int, optional
        Output levels. Default is 64. Must be at least 2.

    Returns
    -------
    numpy.ndarray
        Equalized image, same dtype as ``image``.

    Raises
    ------
    ValueError
        If ``n_levels`` is less than 2.
    """
    a = np.asarray(image)
    npts = 256
    m = int(n_levels)
    if m < 2:
        raise ValueError(f"histeq N must be >= 2, got {m}")
    counts = matlab_imhist_counts(a, npts)
    hgram = np.full(m, a.size / m, dtype=np.float64)
    transform = _histeq_transform(counts, hgram, a.size)
    return matlab_grayxform(a, transform)


def matlab_imfilter_keep_class(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    boundary: str = "zero",
) -> np.ndarray:
    """Filter an image and return the same numeric class.

    The filter runs in double precision. Integer results are rounded half
    away from zero and clipped to the dtype range.

    Parameters
    ----------
    image : numpy.ndarray
        uint8, uint16, or double image.
    kernel : numpy.ndarray
        Convolution kernel.
    boundary : {'zero', 'replicate', 'symmetric'}, optional
        How ``imfilter`` fills the border. Default is ``'zero'``.

    Returns
    -------
    numpy.ndarray
        Filtered image, same dtype as ``image``.
    """
    a = np.asarray(image)
    out = matlab_imfilter(a.astype(np.float64), kernel, boundary=boundary)
    if a.dtype == np.uint8:
        return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)
    if a.dtype == np.uint16:
        return np.clip(np.floor(out + 0.5), 0, 65535).astype(np.uint16)
    return out.astype(np.float64)

# ---------------------------------------------------------------------------
# Nested std, bilinear warp, and the fspecial+imfilter Gaussian.
# ---------------------------------------------------------------------------

def matlab_std_std_2d(channel: np.ndarray) -> float:
    """Nested sample standard deviation, matching MATLAB ``std(std(channel))``.

    MATLAB ``std`` of a matrix is one value per column. The outer ``std``
    reduces that vector to a scalar. That is not the same as ``std`` of the
    flattened image, and it is what ``BDcreation_reg2`` / ``BDcreationHE2``
    use for ``imadjust``.

    Parameters
    ----------
    channel : numpy.ndarray
        One 2-D color channel.

    Returns
    -------
    float
        Nested sample standard deviation. ``0.0`` for an empty channel.

    Raises
    ------
    ValueError
        If ``channel`` is not 2-D.
    """
    if channel.ndim != 2:
        raise ValueError(f"Expected 2-D channel, got shape {channel.shape}.")
    if channel.size == 0:
        return 0.0
    # Sample std (ddof=1) matches MATLAB ``std`` for vectors/matrices.
    col_std = np.std(channel, axis=0, ddof=1)
    col_std = col_std[~np.isnan(col_std)]
    if col_std.size <= 1:
        flat = channel.ravel()
        return float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
    return float(np.std(col_std, ddof=1))

def matlab_imwarp_bilinear(
    src: np.ndarray,
    out_shape: tuple[int, int],
    A: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    MATLAB-parity bilinear warp replicating ``imwarp(moving, Rmoving, tform, 'OutputView',
    Rfixed, 'FillValues', fill_value)`` with ``imref2d(size)`` defaults.

    Parameters
    ----------
    src : np.ndarray
        Source image ``(H_in, W_in)`` or ``(H_in, W_in, C)``, float.
    out_shape : (int, int)
        Output ``(H_out, W_out)``.
    A : np.ndarray, shape (2, 3) or (3, 3)
        Affine mapping **fixed (output) 0-based intrinsic coords -> moving (input)
        0-based intrinsic coords**, in column-vector form:
        ``[x_moving; y_moving; 1] = A @ [x_fixed; y_fixed; 1]``.
        Equivalent to applying the inverse of a forward (moving->fixed) affine.
    fill_value : float
        Value used for output pixels whose source sample lies outside the
        MATLAB-parity half-pixel-extended input domain, and for out-of-bound
        neighbours encountered during bilinear blending.

    Returns
    -------
    np.ndarray
        Warped image with the same number of channels as ``src`` and shape
        ``(H_out, W_out[, C])``, dtype float64.

    Notes
    -----
    Matches the ``imref2d``/``imwarp`` conventions critical for pixel-exact
    parity with MATLAB's ``BDcreation_reg2.m``:

    - Output pixel ``(r, c)`` center is at world ``(c, r)`` (0-based world
      coords, which differ from MATLAB's 1-based world only by a constant
      offset that cancels out when both domains share the same convention).
    - A source sample ``(x, y)`` is "inside" iff ``0 <= x <= W_in - 1`` and
      ``0 <= y <= H_in - 1`` (0-based pixel-*centre* range, i.e. MATLAB's
      ``1 <= x <= W``); anything else gets ``fill_value`` outright. There is
      **no** blending with the fill value at the border. This was verified
      against ``images.internal.interp2d(..., 'linear', fill)`` with
      ``SmoothEdges=false`` (imwarp's default) on a dense probe grid
      (``archive/matlab_parity/probe_interp2d.m``): inside samples never touch
      the fill, samples in the half-pixel band ``[0.5, 1)`` are pure fill.
    - Padding with ``fill_value`` below only guards the ``x == W_in - 1``
      edge, where the right neighbour has weight exactly 0.
    """
    src_arr = np.asarray(src, dtype=np.float64)
    if src_arr.ndim not in (2, 3):
        raise ValueError(f"src must be 2D or 3D, got shape {src_arr.shape}")

    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.shape == (2, 3):
        A_full = np.vstack([A_arr, [0.0, 0.0, 1.0]])
    elif A_arr.shape == (3, 3):
        A_full = A_arr
    else:
        raise ValueError(f"A must be 2x3 or 3x3, got shape {A_arr.shape}")

    H_in, W_in = src_arr.shape[:2]
    H_out, W_out = int(out_shape[0]), int(out_shape[1])

    # Build output intrinsic grid (0-based pixel indices).
    c_grid, r_grid = np.meshgrid(
        np.arange(W_out, dtype=np.float64),
        np.arange(H_out, dtype=np.float64),
    )
    ones = np.ones_like(c_grid)
    pts_out = np.stack([c_grid, r_grid, ones], axis=0).reshape(3, -1)

    # Source intrinsic coords for every output pixel.
    pts_in = A_full @ pts_out
    x_in = pts_in[0].reshape(H_out, W_out)
    y_in = pts_in[1].reshape(H_out, W_out)

    # Inside = pixel-centre range (MATLAB interp2d rule, see Notes). Using the
    # half-pixel-extended [-0.5, N-0.5] range here produced a one-pixel band
    # of fill-blended values that MATLAB does not have.
    inside = (
        (x_in >= 0.0)
        & (x_in <= (W_in - 1))
        & (y_in >= 0.0)
        & (y_in <= (H_in - 1))
    )

    # Bilinear: floor + ceil neighbours in intrinsic coords.
    x0 = np.floor(x_in).astype(np.int64)
    y0 = np.floor(y_in).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x_in - x0).astype(np.float64)
    wy = (y_in - y0).astype(np.float64)

    # Pad source with fill_value so any OOB neighbour samples the fill.
    if src_arr.ndim == 2:
        padded = np.full((H_in + 2, W_in + 2), float(fill_value), dtype=np.float64)
        padded[1:-1, 1:-1] = src_arr
    else:
        C = src_arr.shape[2]
        padded = np.full(
            (H_in + 2, W_in + 2, C), float(fill_value), dtype=np.float64
        )
        padded[1:-1, 1:-1, :] = src_arr

    # Pad index shift of +1; clip to padded bounds (all fill outside padded).
    x0p = np.clip(x0 + 1, 0, W_in + 1)
    y0p = np.clip(y0 + 1, 0, H_in + 1)
    x1p = np.clip(x1 + 1, 0, W_in + 1)
    y1p = np.clip(y1 + 1, 0, H_in + 1)

    if src_arr.ndim == 2:
        v00 = padded[y0p, x0p]
        v01 = padded[y0p, x1p]
        v10 = padded[y1p, x0p]
        v11 = padded[y1p, x1p]
        interp = (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v01
            + (1.0 - wx) * wy * v10
            + wx * wy * v11
        )
        out = np.where(inside, interp, float(fill_value))
        return out.astype(np.float64)

    out_channels = []
    for k in range(src_arr.shape[2]):
        v00 = padded[y0p, x0p, k]
        v01 = padded[y0p, x1p, k]
        v10 = padded[y1p, x0p, k]
        v11 = padded[y1p, x1p, k]
        interp = (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v01
            + (1.0 - wx) * wy * v10
            + wx * wy * v11
        )
        ch_out = np.where(inside, interp, float(fill_value))
        out_channels.append(ch_out)
    return np.stack(out_channels, axis=-1).astype(np.float64)


def gaussian_filter_matlab_like(
    image: np.ndarray,
    sigma: float,
    kernel_size: int,
    *,
    boundary: str = "zero",
) -> np.ndarray:
    """Gaussian filter matching ``fspecial`` plus ``imfilter``.

    Parameters
    ----------
    image : numpy.ndarray
        Image to blur.
    sigma : float
        Gaussian standard deviation passed to ``fspecial``.
    kernel_size : int
        Square kernel side length. MATLAB always passes this explicitly
        (``floor(ppm)`` for nuclei, ``101`` for the HE2 mask).
    boundary : {'zero', 'replicate'}, optional
        Passed to :func:`matlab_imfilter`. Registration uses ``'zero'``.
        The HE2 mask uses ``'replicate'``. Default is ``'zero'``.

    Returns
    -------
    numpy.ndarray
        Filtered float64 image.
    """
    sz = max(int(kernel_size), 1)
    h = matlab_fspecial_gaussian(sz, float(sigma))
    return matlab_imfilter(image.astype(np.float64), h, boundary=boundary)

# ---------------------------------------------------------------------------
# imadjust, decorrstretch, sRGB to Lab, strel, and k-means.
# ---------------------------------------------------------------------------

def matlab_im2uint8(x: np.ndarray) -> np.ndarray:
    """Convert a double image to uint8 the way MATLAB ``im2uint8`` does.

    Parameters
    ----------
    x : numpy.ndarray
        Double image, typically in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        uint8 image. Values are ``round(x * 255)`` and clipped to 0..255.
        Halves round away from zero.
    """
    v = np.asarray(x, dtype=np.float64) * 255.0
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def matlab_rgb2gray_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to uint8 grayscale with MATLAB's ``rgb2gray`` weights.

    Parameters
    ----------
    rgb_u8 : numpy.ndarray
        uint8 RGB image.

    Returns
    -------
    numpy.ndarray
        uint8 grayscale. The weighted sum is rounded half away from zero.
    """
    a = np.asarray(rgb_u8, dtype=np.float64)
    lin = _RGB2GRAY_COEF[0] * a[..., 0] + _RGB2GRAY_COEF[1] * a[..., 1] + _RGB2GRAY_COEF[2] * a[..., 2]
    return np.clip(np.floor(lin + 0.5), 0, 255).astype(np.uint8)


def matlab_imhist_counts(plane: np.ndarray, nbins: int) -> np.ndarray:
    """Histogram counts matching ``imhist(I, nbins)``.

    Integer images are binned by their stored value. Double images use
    ``floor(x * (nbins - 1) + 0.5)``, and values outside ``[0, 1]`` are
    clamped to the end bins. That matters for ``decorrstretch``, whose
    double output can leave ``[0, 1]``.

    Parameters
    ----------
    plane : numpy.ndarray
        One image channel.
    nbins : int
        Number of bins. Callers pass 256 for uint8 and 65536 otherwise.
        The length is used for the bin index and for ``minlength``.

    Returns
    -------
    numpy.ndarray
        Float counts of length ``nbins``.
    """
    p = np.asarray(plane)
    if np.issubdtype(p.dtype, np.integer):
        idx = p.astype(np.int64)
    else:
        idx = np.floor(p.astype(np.float64) * (nbins - 1) + 0.5)
        idx = np.nan_to_num(idx, nan=0.0).astype(np.int64)
    idx = np.clip(idx, 0, nbins - 1)
    return np.bincount(idx.ravel(), minlength=nbins).astype(np.float64)


def matlab_stretchlim(img: np.ndarray, tol: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """Find the low and high limits MATLAB ``stretchlim`` would use.

    The histogram has 256 bins for uint8 and 65536 otherwise. ``ilow`` is the
    first bin whose CDF exceeds the low tolerance. ``ihigh`` is the first
    whose CDF reaches the high tolerance. A flat plane maps to ``[0, 1]``.

    Parameters
    ----------
    img : numpy.ndarray
        2-D or multi-channel image.
    tol : tuple of float, optional
        ``(low, high)`` CDF cutoffs. Default is ``(0.01, 0.99)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(2, nplanes)``. Row 0 is the low limit, row 1 the high limit,
        both in ``[0, 1]``.
    """
    a = np.asarray(img)
    planes = a[..., None] if a.ndim == 2 else a
    nbins = 256 if planes.dtype == np.uint8 else 65536
    tol_low, tol_high = float(tol[0]), float(tol[1])
    out = np.zeros((2, planes.shape[-1]), dtype=np.float64)
    if not tol_low < tol_high:
        out[1, :] = 1.0
        return out
    for i in range(planes.shape[-1]):
        counts = matlab_imhist_counts(planes[..., i], nbins)
        cdf = np.cumsum(counts) / counts.sum()
        ilow = int(np.argmax(cdf > tol_low)) + 1          # 1-based like MATLAB find
        ihigh = int(np.argmax(cdf >= tol_high)) + 1
        if ilow == ihigh:
            ilow, ihigh = 1, nbins
        out[0, i] = (ilow - 1) / (nbins - 1)
        out[1, i] = (ihigh - 1) / (nbins - 1)
    return out


def _adjust_array(x: np.ndarray, low_in: float, high_in: float, low_out: float = 0.0, high_out: float = 1.0) -> np.ndarray:
    """Linear ``imadjust`` stretch (gamma 1).

    Parameters
    ----------
    x : numpy.ndarray
        Values to stretch.
    low_in, high_in : float
        Input range. Values outside it are clipped.
    low_out, high_out : float, optional
        Output range. Default is 0 to 1.

    Returns
    -------
    numpy.ndarray
        Stretched values.
    """
    xc = np.maximum(low_in, np.minimum(high_in, np.asarray(x, dtype=np.float64)))
    out = (xc - low_in) / (high_in - low_in)
    return out * (high_out - low_out) + low_out


def matlab_imadjust_auto_integer(img: np.ndarray) -> np.ndarray:
    """``imadjust(I)`` with no extra arguments, on a 2-D uint8 or uint16 image.

    Limits come from ``stretchlim``. The stretch is linear and the result
    keeps the input integer class.

    Parameters
    ----------
    img : numpy.ndarray
        2-D uint8 or uint16 image.

    Returns
    -------
    numpy.ndarray
        Contrast-stretched image, same dtype as ``img``.

    Raises
    ------
    ValueError
        If ``img`` is not 2-D.
    TypeError
        If ``img`` is not uint8 or uint16.
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"imadjust(I) with one argument needs a 2-D image, got {a.shape}")
    if a.dtype == np.uint8:
        scale = 255.0
    elif a.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"expected uint8/uint16 SHG, got {a.dtype}")
    low, high = matlab_stretchlim(a)[:, 0]
    lut_in = np.arange(int(scale) + 1, dtype=np.float64) / scale
    lut = np.clip(np.floor(_adjust_array(lut_in, low, high) * scale + 0.5), 0, scale)
    return lut.astype(a.dtype)[a]


def matlab_imadjust_double_rgb(img: np.ndarray, low_in: float, high_in: float) -> np.ndarray:
    """``imadjust`` on a double RGB image with one input range for every channel.

    Parameters
    ----------
    img : numpy.ndarray
        Double RGB image.
    low_in, high_in : float
        Shared input limits. Output limits are 0 and 1.

    Returns
    -------
    numpy.ndarray
        Stretched image.
    """
    return _adjust_array(img, float(low_in), float(high_in))


def _matlab_pinv_diag(D: np.ndarray) -> np.ndarray:
    """Pseudoinverse of a diagonal matrix, matching ``fitdecorrtrans``.

    Singular values at or below ``n * max(d) * sqrt(eps)`` become zero.

    Parameters
    ----------
    D : numpy.ndarray
        Square diagonal matrix.

    Returns
    -------
    numpy.ndarray
        Diagonal pseudoinverse.
    """
    d = np.diag(D).astype(np.float64)
    tol = len(d) * d.max() * np.sqrt(np.finfo(np.float64).eps)
    s = np.zeros_like(d)
    keep = d > tol
    s[keep] = 1.0 / d[keep]
    return np.diag(s)


def matlab_decorrstretch_uint8(rgb_u8: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Decorrelation stretch of a uint8 RGB image.

    Follows ``decorrstretch.m`` / ``fitdecorrtrans.m`` with correlation mode:
    sample covariance, whitening, stretch back to the original sigma, then
    ``stretchlim`` and ``im2uint8``.

    Parameters
    ----------
    rgb_u8 : numpy.ndarray
        uint8 RGB image, shape ``(H, W, 3)``.
    tol : float, optional
        Fraction trimmed from each end of the histogram. Default is 0.01.

    Returns
    -------
    numpy.ndarray
        uint8 RGB image with the same shape.

    Raises
    ------
    TypeError
        If ``rgb_u8`` is not uint8 RGB.
    """
    a = np.asarray(rgb_u8)
    if a.dtype != np.uint8 or a.ndim != 3:
        raise TypeError(f"expected uint8 (H, W, 3), got {a.dtype} {a.shape}")
    h, w, nb = a.shape
    A = a.reshape(-1, nb).astype(np.float64) / 255.0
    n = A.shape[0]
    mean_b = A.mean(axis=0)
    cov = (A.T @ A - (n * mean_b[:, None]) * mean_b[None, :]) / (n - 1)

    S = np.diag(np.sqrt(np.diag(cov)))
    target_sigma = S
    pS = _matlab_pinv_diag(S)
    corr = pS @ cov @ pS
    np.fill_diagonal(corr, 1.0)
    evals, V = np.linalg.eigh(corr)
    D = np.diag(np.maximum(evals, 0.0))
    W = np.sqrt(_matlab_pinv_diag(D))
    T = pS @ V @ W @ V.T @ target_sigma
    T = T @ _matlab_pinv_diag(np.diag(np.sqrt(np.diag(T.T @ cov @ T)))) @ target_sigma
    offset = mean_b - mean_b @ T
    Sd = (A @ T + offset).reshape(h, w, nb)

    tol_lh = (float(tol), 1.0 - float(tol))
    low_high = matlab_stretchlim(Sd, tol_lh)
    out = np.empty_like(Sd)
    for p in range(nb):
        out[..., p] = _adjust_array(Sd[..., p], low_high[0, p], low_high[1, p])
    out = np.clip(out, 0.0, 1.0)
    return matlab_im2uint8(out)


def _load_srgb2lab_components() -> dict[str, np.ndarray]:
    """Load the sRGB-to-Lab tables shipped in ``data/matlab_srgb2lab_components.npz``.

    Returns
    -------
    dict
        Arrays exported from MATLAB's ICC pipeline (TRC, colorants, adaptation).
    """
    with resources.files("pycurvelets").joinpath("data/matlab_srgb2lab_components.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: np.array(z[k]) for k in z.files}


_SRGB2LAB: dict[str, np.ndarray] | None = None


def matlab_srgb2lab_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to uint8 Lab the way MATLAB ``applycform`` does.

    Uses the sRGB ICC tables exported by
    ``archive/matlab_parity/dump_srgb2lab_components.m``: tone curve, colorant
    matrix, Bradford adaptation, then ``xyz2lab`` packed as
    ``round([255 L/100, a+128, b+128])``.

    Parameters
    ----------
    rgb_u8 : numpy.ndarray
        uint8 RGB image. The last axis must have length 3.

    Returns
    -------
    numpy.ndarray
        uint8 Lab image with the same shape.

    Raises
    ------
    TypeError
        If the array is not uint8 RGB.
    """
    global _SRGB2LAB
    if _SRGB2LAB is None:
        _SRGB2LAB = _load_srgb2lab_components()
    c = _SRGB2LAB
    a = np.asarray(rgb_u8)
    if a.dtype != np.uint8 or a.shape[-1] != 3:
        raise TypeError(f"expected uint8 (..., 3), got {a.dtype} {a.shape}")
    flat = a.reshape(-1, 3).astype(np.int64)
    lin = np.empty(flat.shape, dtype=np.float64)
    for i in range(3):
        lin[:, i] = c["trc256"][flat[:, i], i]
    xyz = (lin @ c["colorants"]) @ c["adapter"].T
    xyzn = xyz / c["whitepoint"][None, :]
    f = np.power(xyzn, 1.0 / 3.0)
    small = xyzn <= 216.0 / 24389.0
    f[small] = (841.0 / 108.0) * xyzn[small] + 16.0 / 116.0
    L = 116.0 * f[:, 1] - 16.0
    A = 500.0 * (f[:, 0] - f[:, 1])
    B = 200.0 * (f[:, 1] - f[:, 2])
    lab = np.stack([255.0 * L / 100.0, A + 128.0, B + 128.0], axis=1)
    lab = np.clip(np.floor(lab + 0.5), 0, 255)
    return lab.astype(np.uint8).reshape(a.shape)


def matlab_strel_disk(r: int, n: int = 4) -> np.ndarray:
    """Neighborhood of ``strel('disk', r, n)``.

    Radii below 3, and ``n == 0``, use the exact Euclidean disk. Larger radii
    use Adams' periodic-line approximation. That is not a Euclidean disk:
    ``r=3`` is a flat 5x5 square and ``r=4`` is a 7x7 octagon.

    Parameters
    ----------
    r : int
        Disk radius in pixels.
    n : int, optional
        Approximation order: 0, 4, 6, or 8. Default is 4, matching
        ``BDcreation_reg``.

    Returns
    -------
    numpy.ndarray
        Boolean neighborhood.

    Raises
    ------
    ValueError
        If ``n`` is not 0, 4, 6, or 8.
    """
    r = int(r)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    if r < 3 or n == 0:
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        return (xx**2 + yy**2) <= r**2
    if n == 4:
        v = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]], dtype=np.float64)
    elif n == 6:
        v = np.array([[1, 0], [1, 2], [2, 1], [0, 1], [-1, 2], [-2, 1]], dtype=np.float64)
    elif n == 8:
        v = np.array([[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1]], dtype=np.float64)
    else:
        raise ValueError("n must be 0, 4, 6 or 8")
    theta = np.pi / (2 * n)
    k = 2 * r / (1.0 / np.tan(theta) + 1.0 / np.sin(theta))

    def minkowski(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Minkowski sum of two sets of integer offsets.

        Parameters
        ----------
        a, b : set of tuple of int
            Offset sets.

        Returns
        -------
        set of tuple of int
            All pairwise sums.
        """
        return {(p[0] + q[0], p[1] + q[1]) for p in a for q in b}

    offsets: set[tuple[int, int]] = {(0, 0)}
    for q in range(n):
        rp = int(np.floor(k / np.linalg.norm(v[q])))
        line = {(int(p * v[q, 0]), int(p * v[q, 1])) for p in range(-rp, rp + 1)}
        offsets = minkowski(offsets, line)
    max_radius = max(abs(dr) for dr, _ in offsets)
    radial_difference = r - max_radius
    length = 2 * (radial_difference - 1) + 1
    if length >= 3:
        half = (length - 1) // 2
        offsets = minkowski(offsets, {(0, c) for c in range(-half, half + 1)})
        offsets = minkowski(offsets, {(rr, 0) for rr in range(-half, half + 1)})
    ext_r = max(abs(dr) for dr, _ in offsets)
    ext_c = max(abs(dc) for _, dc in offsets)
    nhood = np.zeros((2 * ext_r + 1, 2 * ext_c + 1), dtype=bool)
    for dr, dc in offsets:
        nhood[dr + ext_r, dc + ext_c] = True
    return nhood


# ---------------------------------------------------------------------------
# kmeans replay
# ---------------------------------------------------------------------------


def _sqdist(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row to one centre.

    Differences are explicit, matching ``pdist2mex`` mode ``'sqe'``.

    Parameters
    ----------
    X : numpy.ndarray
        Samples, shape ``(n, d)``.
    c : numpy.ndarray
        One centre, shape ``(d,)``.

    Returns
    -------
    numpy.ndarray
        Squared distances, shape ``(n,)``.
    """
    d = X - c[None, :]
    return np.einsum("ij,ij->i", d, d)


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Pick k-means++ seeds the way MATLAB ``'Start','plus'`` does.

    The first seed is ``floor(rand * n)``. Later seeds are weighted by
    squared distance. One uniform draw is inverted through the cumulative
    weights with ``histcounts`` bin edges.

    Parameters
    ----------
    X : numpy.ndarray
        Samples, shape ``(n, d)``.
    k : int
        Number of centres.
    rng : numpy.random.RandomState
        MATLAB-compatible generator.

    Returns
    -------
    numpy.ndarray
        Initial centres, shape ``(k, d)``.

    Raises
    ------
    RuntimeError
        If every point is the same point. MATLAB would then use ``randperm``,
        which this replay does not implement.
    """
    n = X.shape[0]
    C = np.empty((k, X.shape[1]), dtype=np.float64)
    C[0] = X[int(np.floor(rng.random_sample() * n))]
    min_dist = np.full(n, np.inf)
    for ii in range(1, k):
        min_dist = np.minimum(min_dist, _sqdist(X, C[ii - 1]))
        denominator = float(np.sum(min_dist))
        if denominator == 0.0 or not np.isfinite(denominator):
            raise RuntimeError(
                "kmeans++ degenerate data (all points coincide); MATLAB would fall "
                "back to randperm sampling, which this replay does not implement."
            )
        p = min_dist / denominator
        edges = np.minimum(np.concatenate(([0.0], np.cumsum(p))), 1.0)
        edges[-1] = 1.0
        r = rng.random_sample()
        idx = int(np.searchsorted(edges, r, side="right")) - 1
        idx = min(max(idx, 0), n - 1)
        C[ii] = X[idx]
    return C


def _lloyd_batch_matlab(
    X: np.ndarray, C: np.ndarray, max_iter: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    """One MATLAB ``kmeans`` batch phase.

    Empty clusters are refilled from the farthest point (``'singleton'``).
    A step that does not lower the objective is undone. Ties keep the old label.

    Parameters
    ----------
    X : numpy.ndarray
        Samples, shape ``(n, d)``.
    C : numpy.ndarray
        Starting centres, shape ``(k, d)``. Updated in place.
    max_iter : int
        Iteration cap.

    Returns
    -------
    idx : numpy.ndarray
        0-based labels, shape ``(n,)``.
    C : numpy.ndarray
        Final centres.
    totsumD : float
        Total within-cluster sum of squares.
    sumD : numpy.ndarray
        Per-cluster sum of squares.
    it : int
        Iterations actually used.
    """
    n, k = X.shape[0], C.shape[0]
    D = np.column_stack([_sqdist(X, C[j]) for j in range(k)])
    idx = np.argmin(D, axis=1)
    m = np.bincount(idx, minlength=k)
    changed = np.arange(k)
    previdx = np.zeros(n, dtype=np.int64)
    prevtotsumD = np.inf
    rows = np.arange(n)
    it = 0
    converged = False

    def centroids(clusters: np.ndarray) -> None:
        """Recompute centres, counts, and distances for the given clusters.

        Parameters
        ----------
        clusters : numpy.ndarray
            0-based cluster ids that changed.
        """
        for j in clusters:
            members = idx == j
            m[j] = int(members.sum())
            C[j] = X[members].sum(axis=0) / m[j] if m[j] > 0 else np.nan
            D[:, j] = _sqdist(X, C[j]) if m[j] > 0 else np.nan

    while True:
        it += 1
        centroids(changed)
        empties = [j for j in changed if m[j] == 0]
        for j in empties:
            d = D[rows, idx]
            lonely = int(np.argmax(d))
            src = int(idx[lonely])
            if m[src] < 2:
                src = int(np.argmax(m > 1))
                lonely = int(np.argmax(idx == src))
            C[j] = X[lonely]
            m[j] = 1
            idx[lonely] = j
            D[:, j] = _sqdist(X, C[j])
            centroids(np.array([src]))
            changed = np.unique(np.concatenate((changed, [src])))
        totsumD = float(np.sum(D[rows, idx]))
        if prevtotsumD <= totsumD:
            idx = previdx
            centroids(changed)
            it -= 1
            break
        if it >= max_iter:
            break
        previdx = idx.copy()
        prevtotsumD = totsumD
        nidx = np.argmin(D, axis=1)
        d = D[rows, nidx]
        moved = np.flatnonzero(nidx != previdx)
        if moved.size:
            moved = moved[D[moved, previdx[moved]] > d[moved]]
        if moved.size == 0:
            converged = True
            break
        idx = idx.copy()
        idx[moved] = nidx[moved]
        changed = np.unique(np.concatenate((idx[moved], previdx[moved])))

    nonempty = np.flatnonzero(m > 0)
    for j in nonempty:
        D[:, j] = _sqdist(X, C[j])
    d = D[rows, idx]
    sumD = np.bincount(idx, weights=d, minlength=k)
    totsumD = float(np.sum(sumD[nonempty]))
    if not converged and it >= max_iter:
        import warnings

        warnings.warn(f"kmeans replay failed to converge in {max_iter} iterations", RuntimeWarning, stacklevel=3)
    return idx, C, totsumD, sumD, it


def matlab_kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int,
    replicates: int = 3,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay MATLAB ``kmeans`` with squared Euclidean distance.

    Rows of ``X`` must already be in MATLAB observation order (column-major
    flatten of the a/b image).

    Parameters
    ----------
    X : numpy.ndarray
        Samples, shape ``(n, p)``.
    k : int
        Number of clusters.
    seed : int
        Seed for the MATLAB twister replay.
    replicates : int, optional
        Independent k-means++ starts. Default is 3.
    max_iter : int, optional
        Iteration cap per replicate. Default is 100.

    Returns
    -------
    idx : numpy.ndarray
        0-based labels of the best replicate.
    C : numpy.ndarray
        Centres, shape ``(k, p)``.
    sumd : numpy.ndarray
        Per-cluster sum of squares.
    debug : dict
        Seed and every replicate's objective.
    """
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    best: tuple | None = None
    reps: list[dict[str, Any]] = []
    for rep in range(int(replicates)):
        C0 = _kmeanspp_init(Xd, k, rng)
        idx, C, totsumD, sumD, iters = _lloyd_batch_matlab(Xd, C0.copy(), max_iter)
        reps.append({"replicate": rep + 1, "totsumD": totsumD, "iterations": iters, "init": C0.tolist()})
        if best is None or totsumD < best[2]:
            best = (idx, C, totsumD, sumD)
    assert best is not None
    debug = {"seed": int(seed), "replicates": reps, "best_totsumD": best[2]}
    return best[0], best[1], best[3], debug
