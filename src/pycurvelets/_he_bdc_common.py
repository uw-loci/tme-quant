"""Shared MATLAB image primitives for H&E registration and tumor annotation.

These helpers reproduce the Image Processing Toolbox calls used by
``BDcreation_reg.m``, ``BDcreation_reg2.m``, ``BDcreationHE.m``, and
``BDcreationHE2.m``: color conversion, Otsu thresholding, filtering, and
the bilinear warp. Annotation-only morphology lives in
:mod:`pycurvelets._he_bdc_annotation`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color, morphology

from ._matlab_imresize import matlab_imresize

# MATLAB-derived HSV threshold defaults from:
# - CurveAlign_CT-FIRE/BDcreation_reg2.m
# - CurveAlign_CT-FIRE/BDcreationHE2.m
NUCLEI_HUE_MIN = 0.500
NUCLEI_HUE_MAX = 0.790
COLLAGEN_HUE_MIN = 0.837
COLLAGEN_HUE_MAX = 0.066  # wrapped hue range (>= min OR <= max)
NUCLEI_MIN_AREA = 150
COLLAGEN_MIN_AREA = 100

# MATLAB ``graythresh`` histograms a uint8 image: one bin per level 0..255.
_GRAYTHRESH_BINS = 256

# MATLAB-derived RGB cuts from BDcreation_reg.m on decorrstretched uint8 (S).
# Exclusive inequalities, as in the MATLAB script.
RGB_NUCLEI_R_MAX = 120  # R < 120
RGB_NUCLEI_G_MIN = 150  # G > 150
RGB_NUCLEI_B_MAX = 120  # B < 120
RGB_EOSIN_R_MIN = 200  # R > 200
RGB_EOSIN_G_MAX = 100  # G < 100
RGB_EOSIN_B_MIN = 100  # B > 100


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
    return 0.298936021293775 * r + 0.587043074451121 * g + 0.114020904255103 * b


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


def ensure_rgb(image: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return a channel-last float64 RGB image.

    A grayscale image is stacked into three identical channels. Extra channels
    beyond the first three are dropped.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or multichannel image.
    axis : int, optional
        Channel axis of a multichannel input. Default is ``2``.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` float64.

    Raises
    ------
    ValueError
        If ``image`` is neither 2-D nor RGB.
    """
    arr = image
    if arr.ndim == 2:
        return np.dstack([arr, arr, arr]).astype(np.float64)
    if arr.ndim == 3:
        arr_ch_last = np.moveaxis(arr, axis, -1)
        if arr_ch_last.shape[-1] >= 3:
            return arr_ch_last[..., :3].astype(np.float64)
    raise ValueError(f"Expected 2D grayscale or RGB image, but got {arr.shape}.")


def ensure_grayscale(image: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return a 2-D float64 image.

    RGB input is converted with ``skimage.color.rgb2gray``. A 2-D input is
    only cast.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale or RGB image.
    axis : int, optional
        Channel axis when ``image`` is RGB. Default is ``2``.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` float64.

    Raises
    ------
    ValueError
        If ``image`` is neither 2-D nor RGB.
    """
    # TODO: remove unnecessary casts
    arr = image
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        arr_ch_last = np.moveaxis(arr, axis, -1)
        if arr_ch_last.shape[-1] >= 3:
            return color.rgb2gray(arr_ch_last[..., :3])
    raise ValueError(f"Expected 2D grayscale or RGB image, but got {arr.shape}.")


def resize_like(
    image: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Resize ``image`` to ``(rows, cols)`` with MATLAB bicubic ``imresize``.

    An image that is already the requested shape is copied and returned.

    Parameters
    ----------
    image : numpy.ndarray
        ``(H, W)`` or ``(H, W, C)`` array.
    out_shape : tuple of int
        Target ``(rows, cols)``.

    Returns
    -------
    numpy.ndarray
        Resized image. The intensity range is not renormalized.
    """
    if image.shape[:2] == out_shape:
        return image.copy()
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return matlab_imresize(arr, output_shape=(out_shape[0], out_shape[1]), method="bicubic")
    return matlab_imresize(arr, output_shape=(out_shape[0], out_shape[1]), method="bicubic")


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


def adjust_rgb_mean_std(rgb: np.ndarray, axis: int = 2) -> np.ndarray:
    """Stretch each RGB channel the way CurveAlign's ``imadjust`` call does.

    The high input for a channel is ``mean + 2 * std(std(channel))``, capped
    at 1. That matches ``imadjust(RGB, [0 0 0; HIGH_IN], [0 0 0; 1 1 1])``.

    Parameters
    ----------
    rgb : numpy.ndarray
        Float RGB image.
    axis : int, optional
        Channel axis. Default is ``2``.

    Returns
    -------
    numpy.ndarray
        Channel-last float64 RGB in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``rgb`` does not have three channels.
    """
    rgb_arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] < 3:
        # TODO: check dimension length and num of channel separately
        raise ValueError(f"Expected RGB image with at least 3 channels, got {rgb.shape}.")

    out = np.zeros_like(rgb_arr, dtype=np.float64)
    for ch in range(3):
        channel = rgb_arr[..., ch]
        std_std = matlab_std_std_2d(channel)
        high = float(np.mean(channel) + 2.0 * std_std)
        high = min(max(high, 1e-8), 1.0)
        out[..., ch] = np.clip(channel / high, 0.0, 1.0)
    return out


def prepare_registration_pair(
    he_rgb: np.ndarray,
    shg_gray: np.ndarray,
    pixel_per_micron: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Match ``BDcreation_reg2.m`` image sizing (not ``BDcreationHE2.m``).

    MATLAB::

        if ppm > 2: fixedSHG = imresize(SHG, 2/ppm); ppm = 2
        else: fixedSHG = SHG
        RGB = imresize(HE, size(fixedSHG))

    The working resolution is therefore defined by SHG after the ppm cap,
    then H&E is forced onto that grid. Tumor annotation resamples H&E on its
    own in :func:`pycurvelets._he_bdc_annotation.prepare_he2_image`.

    Parameters
    ----------
    he_rgb : numpy.ndarray
        H&E image, grayscale or RGB.
    shg_gray : numpy.ndarray
        SHG image. A 3-channel array is converted to grayscale first.
    pixel_per_micron : float
        Native resolution of the pair.

    Returns
    -------
    he_scaled : numpy.ndarray
        H&E resized to the working SHG grid.
    fixed_shg : numpy.ndarray
        SHG on that grid.
    pix : float
        Working pixels per micron (``2.0`` when the input was coarser).

    Raises
    ------
    ValueError
        If SHG is not a 2-D image after grayscale conversion.
    """
    pix = float(pixel_per_micron)
    he_rgb = ensure_rgb(he_rgb)
    g = np.asarray(shg_gray, dtype=np.float64)
    if g.ndim == 3:
        g = ensure_grayscale(g)
    if g.ndim != 2:
        raise ValueError(f"Expected 2-D SHG image, got shape {g.shape}.")

    if pix > 2.0:
        # MATLAB ``imresize(SHG, 2/ppm)``: output size is ceil(scale*size) and
        # the *same* scalar scale is used for the kernel on both axes. Using
        # round() + output_shape here produced a 1-px smaller grid for e.g.
        # ppm=3 (341 vs 342), which put the whole registration on a different
        # image than MATLAB's.
        fixed_shg = matlab_imresize(g, scalar_scale=2.0 / pix, method="bicubic")
        pix = 2.0
    else:
        fixed_shg = g

    he_scaled = resize_like(he_rgb, fixed_shg.shape[:2])
    return he_scaled, fixed_shg, pix


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 8-connected components smaller than ``min_size``.

    This is the boolean part of MATLAB ``bwareaopen``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean or numeric mask. Nonzero pixels are foreground.
    min_size : int
        Minimum component area in pixels. Values below 1 are treated as 1.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the components that were kept.

    Raises
    ------
    ValueError
        If ``mask`` is empty.
    """
    min_size = max(int(min_size), 1)
    if mask.size == 0:
        raise ValueError("Mask is empty; cannot remove small components.")
    labels, _ = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(labels.ravel())
    keep = areas >= min_size
    keep[0] = False  # background
    return keep[labels]


def disk_se(radius: float) -> np.ndarray:
    """Disk structuring element with MATLAB ``round`` on the radius.

    Parameters
    ----------
    radius : float
        Disk radius in pixels. A non-positive radius becomes a 1x1 element.

    Returns
    -------
    numpy.ndarray
        Boolean neighborhood.
    """
    r = max(int(matlab_round(radius)), 0)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    return morphology.disk(r)


def rgb_threshold_masks_uint8(s_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the ``BDcreation_reg.m`` RGB cuts to a decorrstretched uint8 image.

    Nuclei are ``R < 120``, ``G > 150``, ``B < 120``. Eosin is ``R > 200``,
    ``G < 100``, ``B > 100``. The comparisons are exclusive, as in the script.

    Parameters
    ----------
    s_u8 : numpy.ndarray
        ``(H, W, 3)`` uint8 RGB.

    Returns
    -------
    nuclei : numpy.ndarray
        Boolean nuclei mask.
    eosin : numpy.ndarray
        Boolean eosin mask.
    """
    arr = np.asarray(s_u8)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    nuclei = (r < RGB_NUCLEI_R_MAX) & (g > RGB_NUCLEI_G_MIN) & (b < RGB_NUCLEI_B_MAX)
    eosin = (r > RGB_EOSIN_R_MIN) & (g < RGB_EOSIN_G_MAX) & (b > RGB_EOSIN_B_MIN)
    return nuclei.astype(bool), eosin.astype(bool)


def make_nuclei_mask_rgb(
    he_decorr_rgb: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nuclei mask from the ``BDcreation_reg`` RGB cuts.

    Parameters
    ----------
    he_decorr_rgb : numpy.ndarray
        Decorrelation-stretched H&E in ``[0, 1]``.
    pix_per_mic : float
        Working pixels per micron. The opening radius is ``ceil(ppm / 2)``.

    Returns
    -------
    nuclei_opened : numpy.ndarray
        Boolean mask after area opening and morphological opening.
    masked : numpy.ndarray
        ``he_decorr_rgb`` with non-nuclei pixels set to 0.
    """
    u8 = np.round(np.clip(he_decorr_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    nuclei, _ = rgb_threshold_masks_uint8(u8)
    nuclei = remove_small_components(nuclei, NUCLEI_MIN_AREA)
    nuclei_opened = morphology.opening(nuclei, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = np.asarray(he_decorr_rgb, dtype=np.float64).copy()
    masked[~nuclei_opened] = 0.0
    return nuclei_opened, masked


def make_ecm_mask_rgb(
    he_decorr_rgb: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale ECM image from the ``BDcreation_reg`` eosin RGB cuts.

    Parameters
    ----------
    he_decorr_rgb : numpy.ndarray
        Decorrelation-stretched H&E in ``[0, 1]``.
    pix_per_mic : float
        Working pixels per micron. When positive, the eosin mask is opened
        with a disk of radius ``ceil(ppm / 2)``.

    Returns
    -------
    ecm_gray : numpy.ndarray
        Float grayscale of the eosin pixels; other pixels are 0.
    eosin : numpy.ndarray
        Boolean eosin mask.
    """
    u8 = np.round(np.clip(he_decorr_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    _, eosin = rgb_threshold_masks_uint8(u8)
    eosin = remove_small_components(eosin, COLLAGEN_MIN_AREA)
    if pix_per_mic > 0:
        eosin = morphology.opening(eosin, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = np.zeros_like(he_decorr_rgb, dtype=np.float64)
    masked[eosin] = np.asarray(he_decorr_rgb, dtype=np.float64)[eosin]
    ecm_gray = matlab_rgb2gray(masked)
    return ecm_gray.astype(np.float64), eosin.astype(bool)


def make_nuclei_mask(
    he_rgb_adjusted: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the HSV nuclei mask used by ``BDcreation_reg2``.

    Pixels in the blue-purple hue band whose saturation is at least the Otsu
    threshold are kept, small objects are removed, and the mask is opened
    with a disk of radius ``ceil(ppm / 2)``.

    Parameters
    ----------
    he_rgb_adjusted : numpy.ndarray
        Contrast-adjusted H&E, float RGB in ``[0, 1]``.
    pix_per_mic : float
        Working pixels per micron.

    Returns
    -------
    nuclei_opened : numpy.ndarray
        Boolean mask after area opening and morphological opening.
    masked : numpy.ndarray
        ``he_rgb_adjusted`` with non-nuclei pixels set to 0.
    """
    hsv = matlab_rgb2hsv(he_rgb_adjusted)
    saturation = hsv[..., 1]
    sat_thresh = matlab_graythresh(saturation)

    nuclei_raw = (
        (hsv[..., 0] >= NUCLEI_HUE_MIN)
        & (hsv[..., 0] <= NUCLEI_HUE_MAX)
        & (saturation >= sat_thresh)
    )
    # 150 px minimum area is copied from the original MATLAB scripts.
    nuclei_raw = remove_small_components(nuclei_raw, NUCLEI_MIN_AREA)
    nuclei_opened = morphology.opening(nuclei_raw, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = he_rgb_adjusted.copy()
    masked[~nuclei_opened] = 0.0
    return nuclei_opened, masked


def make_collagen_mask(
    he_rgb_adjusted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the HSV collagen mask used by ``BDcreation_reg2``.

    Collagen is the wrapped hue band (pink/red) at or above the Otsu
    saturation threshold, after dropping components smaller than 100 pixels.
    The extra dilation used by tumor annotation is not part of registration.

    Parameters
    ----------
    he_rgb_adjusted : numpy.ndarray
        Contrast-adjusted H&E, float RGB in ``[0, 1]``.

    Returns
    -------
    collagen : numpy.ndarray
        Boolean collagen mask.
    no_background : numpy.ndarray
        Boolean mask of pixels at or above the saturation threshold.
    sat_thresh : float
        Otsu threshold applied to the saturation channel.
    """
    hsv = matlab_rgb2hsv(he_rgb_adjusted)
    saturation = hsv[..., 1]
    sat_thresh = matlab_graythresh(saturation)

    collagen = (
        ((hsv[..., 0] >= COLLAGEN_HUE_MIN) | (hsv[..., 0] <= COLLAGEN_HUE_MAX))
        & (saturation >= sat_thresh)
    )
    # 100 px minimum area and the wrapped hue band come from MATLAB defaults.
    collagen = remove_small_components(collagen, COLLAGEN_MIN_AREA)

    no_background = saturation >= sat_thresh
    return collagen.astype(bool), no_background.astype(bool), sat_thresh


def decorrelation_stretch(rgb: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Approximate MATLAB ``decorrstretch`` for RGB in ``[0, 1]``.

    Channels are whitened with PCA, then each channel is stretched between
    the ``tol`` and ``1 - tol`` percentiles. The bit-exact uint8 port used by
    reg1 and ``BDcreationHE`` is
    :func:`pycurvelets._he_bdc_reg1.matlab_decorrstretch_uint8`.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB image.
    tol : float, optional
        Fraction of pixels clipped at each end of a channel. Default is 0.01.

    Returns
    -------
    numpy.ndarray
        Channel-last float64 RGB in ``[0, 1]``.
    """
    arr = ensure_rgb(np.asarray(rgb, dtype=np.float64))
    arr = np.clip(arr, 0.0, 1.0)
    flat = arr.reshape(-1, 3)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    whiten = evecs @ np.diag(1.0 / np.sqrt(evals))
    decor = centered @ whiten

    out = np.zeros_like(decor)
    lo_q = float(np.clip(tol, 0.0, 0.49))
    hi_q = 1.0 - lo_q
    for c in range(3):
        ch = decor[:, c]
        lo = float(np.quantile(ch, lo_q))
        hi = float(np.quantile(ch, hi_q))
        if hi <= lo:
            out[:, c] = 0.0
        else:
            out[:, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out.reshape(arr.shape)


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
