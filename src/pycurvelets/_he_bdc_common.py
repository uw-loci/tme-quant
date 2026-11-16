"""Shared H&E mask steps for registration and tumor annotation.

MATLAB Image Processing Toolbox ports (color, filtering, resize, warp, and
the k-means replay) live in :mod:`pycurvelets._matlab_util`. This module owns
the CurveAlign cuts and masks those scripts share: HSV bands, RGB cuts, the
mean/std ``imadjust``, and the nuclei and collagen masks. HE2-only morphology
stays in :mod:`pycurvelets._he_bdc_annotation`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color, morphology

from ._matlab_util import (
    matlab_graythresh,
    matlab_imresize,
    matlab_rgb2gray,
    matlab_rgb2hsv,
    matlab_round,
    matlab_std_std_2d,
)

# MATLAB-derived HSV threshold defaults from:
# - CurveAlign_CT-FIRE/BDcreation_reg2.m
# - CurveAlign_CT-FIRE/BDcreationHE2.m
NUCLEI_HUE_MIN = 0.500
NUCLEI_HUE_MAX = 0.790
COLLAGEN_HUE_MIN = 0.837
COLLAGEN_HUE_MAX = 0.066  # wrapped hue range (>= min OR <= max)
NUCLEI_MIN_AREA = 150
COLLAGEN_MIN_AREA = 100

# MATLAB-derived RGB cuts from BDcreation_reg.m on decorrstretched uint8 (S).
# Exclusive inequalities, as in the MATLAB script.
RGB_NUCLEI_R_MAX = 120  # R < 120
RGB_NUCLEI_G_MIN = 150  # G > 150
RGB_NUCLEI_B_MAX = 120  # B < 120
RGB_EOSIN_R_MIN = 200  # R > 200
RGB_EOSIN_G_MAX = 100  # G < 100
RGB_EOSIN_B_MIN = 100  # B > 100


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
    :func:`pycurvelets._matlab_util.matlab_decorrstretch_uint8`.

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
