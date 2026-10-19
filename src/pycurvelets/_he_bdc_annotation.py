"""
MATLAB-exact primitives and morphology for ``BDcreationHE2.m`` / ``BDcreationHE.m``.

Registration helpers stay in :mod:`pycurvelets._he_bdc_common` and
:mod:`pycurvelets._he_bdc_reg1` (PR #63). This module is annotation-only:
histogram equalization, ``fspecial('disk')``, ``padarray``, ``im2bw``, HE
cluster ranking, and the HE2/HE pipelines that use Adams ``strel('disk')``
rather than a Euclidean disk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from ._he_bdc_common import (
    COLLAGEN_HUE_MAX,
    COLLAGEN_HUE_MIN,
    COLLAGEN_MIN_AREA,
    NUCLEI_HUE_MAX,
    NUCLEI_HUE_MIN,
    NUCLEI_MIN_AREA,
    adjust_rgb_mean_std,
    ensure_rgb,
    gaussian_filter_matlab_like,
    matlab_graythresh,
    matlab_imfilter,
    matlab_rgb2gray,
    matlab_rgb2hsv,
    matlab_round,
    remove_small_components,
)
from ._he_bdc_reg1 import (
    DEFAULT_KMEANS_SEED,
    matlab_decorrstretch_uint8,
    matlab_imhist_counts,
    matlab_kmeans,
    matlab_rgb2gray_uint8,
    matlab_strel_disk,
)
from ._matlab_imresize import matlab_imresize

# BDcreationHE2.m morphology heuristics (working ppm).
EPITH_BINARY_THRESHOLD = 0.001
EPITH_DILATION_RADIUS_MULTIPLIER = 5.0
BACKGROUND_HOLE_MIN_AREA_MULTIPLIER = 60.0
TUMOR_MASK_MIN_AREA_MULTIPLIER = 35.0
FINAL_MASK_DILATION_RADIUS_MULTIPLIER = 4.0
FINAL_MASK_GAUSSIAN_SIGMA = 25.0
FINAL_MASK_GAUSSIAN_KERNEL_SIZE = 101

# BDcreationHE.m RGB / k-means path (native ppm, no cap).
HE_RGB_DISK_RADIUS_MULTIPLIER = 7.0
HE_RGB_PAD = 70
HE_RGB_N_COLORS = 4
HE_RGB_EPITH_DILATION_MULTIPLIER = 4.0

# Working-resolution cap in BDcreationHE2.m (not used by BDcreationHE.m).
HE2_PPM_CAP = 2.0


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


def matlab_padarray(image: np.ndarray, pad: int | tuple[int, int]) -> np.ndarray:
    """Pad both sides the way ``padarray(A, [pr pc], 'symmetric')`` does.

    MATLAB's symmetric reflection duplicates the edge pixel, which is
    ``numpy.pad(..., mode='symmetric')``. That is the only padding
    ``BDcreationHE`` uses, so there is no mode argument.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array to pad.
    pad : int or tuple of int
        ``(row_pad, col_pad)``, or one integer used for both axes.

    Returns
    -------
    numpy.ndarray
        Padded array.
    """
    a = np.asarray(image)
    if isinstance(pad, tuple):
        pr, pc = int(pad[0]), int(pad[1])
    else:
        pr = pc = int(pad)
    return np.pad(a, ((pr, pr), (pc, pc)), mode="symmetric")


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


def _disk_nhood(radius: float, *, how: str) -> np.ndarray:
    """Adams ``strel('disk')`` neighborhood used by the HE2 morphology.

    Parameters
    ----------
    radius : float
        Radius before it is converted to an integer.
    how : {'ceil', 'round'}
        ``'ceil'`` matches ``ceil(ppm)``. ``'round'`` matches MATLAB ``round``.

    Returns
    -------
    numpy.ndarray
        Boolean structuring element.

    Raises
    ------
    ValueError
        If ``how`` is not ``'ceil'`` or ``'round'``.
    """
    if how == "ceil":
        r = int(np.ceil(radius))
    elif how == "round":
        r = int(matlab_round(radius))
    else:
        raise ValueError(f"how must be 'ceil' or 'round', got {how!r}")
    return matlab_strel_disk(max(r, 0))


def _imdilate(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    """Binary dilation matching ``imdilate``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.
    nhood : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray
        Dilated boolean mask.
    """
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def _imopen(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    """Binary opening matching ``imopen``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.
    nhood : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray
        Opened boolean mask.
    """
    return ndimage.binary_opening(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def _imclose(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    """Binary closing matching ``imclose``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.
    nhood : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray
        Closed boolean mask.
    """
    return ndimage.binary_closing(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def prepare_he2_image(he: np.ndarray, pixel_per_micron: float) -> tuple[np.ndarray, float]:
    """
    BDcreationHE2 sizing: ``if ppm > 2: HEdata = imresize(HE, 2/ppm); ppm = 2``.

    Uses MATLAB's scalar-scale ``imresize`` (``ceil`` output size). Registration
    resamples H&E onto the SHG grid instead; this helper only resamples H&E.

    Parameters
    ----------
    he : numpy.ndarray
        H&E image in ``[0, 1]``.
    pixel_per_micron : float
        Native resolution.

    Returns
    -------
    he_rgb : numpy.ndarray
        Image on the working grid.
    pix : float
        Working pixels per micron, capped at 2.
    """
    he_rgb = ensure_rgb(he)
    pix = float(pixel_per_micron)
    if pix > HE2_PPM_CAP:
        he_rgb = matlab_imresize(he_rgb, scalar_scale=HE2_PPM_CAP / pix, method="bicubic")
        pix = HE2_PPM_CAP
    return he_rgb, pix


def he2_nuclei_mask(he_adjusted: np.ndarray, pix_per_mic: float) -> tuple[np.ndarray, np.ndarray]:
    """HSV nuclei mask from ``BDcreationHE2.m``.

    Parameters
    ----------
    he_adjusted : numpy.ndarray
        Contrast-adjusted H&E.
    pix_per_mic : float
        Working pixels per micron. The opening radius is ``ceil(ppm / 2)``.

    Returns
    -------
    bw_nuclei : numpy.ndarray
        Boolean nuclei mask.
    masked : numpy.ndarray
        H&E with non-nuclei pixels set to 0.
    """
    hsv = matlab_rgb2hsv(he_adjusted)
    sat_thresh = matlab_graythresh(hsv[..., 1])
    bw = (
        (hsv[..., 0] >= NUCLEI_HUE_MIN)
        & (hsv[..., 0] <= NUCLEI_HUE_MAX)
        & (hsv[..., 1] >= sat_thresh)
        & (hsv[..., 1] <= 1.0)
    )
    bw = remove_small_components(bw, NUCLEI_MIN_AREA)
    bw_nuclei = _imopen(bw, _disk_nhood(pix_per_mic / 2.0, how="ceil"))
    masked = np.asarray(he_adjusted, dtype=np.float64).copy()
    masked[~bw_nuclei] = 0.0
    return bw_nuclei, masked


def he2_collagen_mask(he_adjusted: np.ndarray, pix_per_mic: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Wrapped-hue collagen mask from ``BDcreationHE2.m``.

    The mask is dilated by ``ceil(ppm)`` and closed by ``round(3 * ppm)``.

    Parameters
    ----------
    he_adjusted : numpy.ndarray
        Contrast-adjusted H&E.
    pix_per_mic : float
        Working pixels per micron.

    Returns
    -------
    bw_collagen : numpy.ndarray
        Boolean collagen mask after dilation and closing.
    bw_nobackground : numpy.ndarray
        Pixels at or above the saturation threshold.
    sat_thresh : float
        Otsu threshold on the saturation channel.
    """
    hsv = matlab_rgb2hsv(he_adjusted)
    sat_thresh = matlab_graythresh(hsv[..., 1])
    collagen = (
        ((hsv[..., 0] >= COLLAGEN_HUE_MIN) | (hsv[..., 0] <= COLLAGEN_HUE_MAX))
        & (hsv[..., 1] >= sat_thresh)
        & (hsv[..., 1] <= 1.0)
    )
    collagen = remove_small_components(collagen, COLLAGEN_MIN_AREA)
    collagen = _imdilate(collagen, _disk_nhood(pix_per_mic, how="ceil"))
    bw_collagen1 = _imclose(collagen, _disk_nhood(3.0 * pix_per_mic, how="round"))
    bw_nobackground = hsv[..., 1] >= sat_thresh
    return bw_collagen1, bw_nobackground, sat_thresh


def _he2_epithelial_morphology(
    epith_cell_bw: np.ndarray,
    bw_collagen1: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow the epithelial mask and punch out collagen.

    Dilate, fill holes, drop small holes and small objects, then dilate
    again and remove collagen.

    Parameters
    ----------
    epith_cell_bw : numpy.ndarray
        Boolean epithelial pixels.
    bw_collagen1 : numpy.ndarray
        Boolean collagen mask. Those pixels are removed from the final mask.
    pix_per_mic : float
        Working pixels per micron. Sets the dilation radii and area cutoffs.

    Returns
    -------
    bwx : numpy.ndarray
        Filled mask.
    mask_image1 : numpy.ndarray
        Final mask with collagen removed.
    """
    opened = _imdilate(epith_cell_bw, _disk_nhood(EPITH_DILATION_RADIUS_MULTIPLIER * pix_per_mic, how="round"))
    bwx = ndimage.binary_fill_holes(opened)
    hole_area = int(matlab_round((BACKGROUND_HOLE_MIN_AREA_MULTIPLIER * pix_per_mic) ** 2))
    tumor_area = int(matlab_round((TUMOR_MASK_MIN_AREA_MULTIPLIER * pix_per_mic) ** 2))
    bwy = remove_small_components(~bwx, hole_area)
    mask_image = remove_small_components(~bwy, tumor_area)
    mask_image1 = _imdilate(
        mask_image,
        _disk_nhood(FINAL_MASK_DILATION_RADIUS_MULTIPLIER * pix_per_mic, how="round"),
    ) & (~bw_collagen1)
    return bwx, mask_image1


def annotate_he2(he_raw: np.ndarray, pixelpermicron: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ``BDcreationHE2.m`` on one registered H&E image.

    Parameters
    ----------
    he_raw : numpy.ndarray
        Registered H&E. Integer images are scaled with ``im2double``.
    pixelpermicron : float
        Native resolution. Values above 2 are capped by resizing the image.

    Returns
    -------
    bd_mask : numpy.ndarray
        Boolean tumor mask at the original image size.
    debug : dict
        Named intermediates (adjusted image, nuclei, collagen, smoothed mask).
    """
    orig_row, orig_col = int(he_raw.shape[0]), int(he_raw.shape[1])
    he_data, pix = prepare_he2_image(matlab_im2double(he_raw), float(pixelpermicron))
    he_adjusted = adjust_rgb_mean_std(he_data)
    bw_nuclei, maskednuclei = he2_nuclei_mask(he_adjusted, pix)
    bw_collagen1, bw_nobackground, sat_thresh = he2_collagen_mask(he_adjusted, pix)
    epith_cell_bw = (
        matlab_im2bw(matlab_rgb2gray(maskednuclei), EPITH_BINARY_THRESHOLD)
        & (~bw_collagen1)
        & bw_nobackground
    )
    bwx, mask_image1 = _he2_epithelial_morphology(epith_cell_bw, bw_collagen1, pix)
    smoothed = gaussian_filter_matlab_like(
        mask_image1.astype(np.float64),
        sigma=FINAL_MASK_GAUSSIAN_SIGMA,
        kernel_size=FINAL_MASK_GAUSSIAN_KERNEL_SIZE,
        boundary="replicate",
    )
    mask_temp = matlab_imresize(smoothed, output_shape=(orig_row, orig_col), method="bicubic")
    mask_thresh = matlab_graythresh(mask_temp)
    bd_mask = matlab_im2bw(mask_temp, mask_thresh)
    debug: dict[str, Any] = {
        "annotation_method": np.array(["hsv"]),
        "he_adjusted": he_adjusted,
        "BW_nuclei": bw_nuclei,
        "maskednucleiImage": maskednuclei,
        "BW_collagen1": bw_collagen1,
        "BW_nobackground": bw_nobackground,
        "sat_thresh": np.array([sat_thresh], dtype=np.float64),
        "epith_cell_BW": epith_cell_bw,
        "BWx": bwx,
        "mask_image1": mask_image1.astype(np.uint8),
        "B": smoothed,
        "mask_temp": mask_temp,
        "mask_thresh": np.array([mask_thresh], dtype=np.float64),
        "BDmask": bd_mask,
        "pixpermic_working": pix,
    }
    return bd_mask.astype(bool), debug


def _matlab_argsort_asc(x: np.ndarray) -> np.ndarray:
    """Stable ascending argsort with NaNs last, matching MATLAB ``sort``.

    Parameters
    ----------
    x : numpy.ndarray
        Values to order.

    Returns
    -------
    numpy.ndarray
        Indices that would sort ``x``.
    """
    v = np.asarray(x, dtype=np.float64)
    keys = np.where(np.isnan(v), np.inf, v)
    return np.argsort(keys, kind="stable")


def pick_he_epithelial_cluster(
    cluster_center: np.ndarray,
    mean_cluster_intensity: np.ndarray,
) -> int:
    """Choose the epithelial cluster the way ``BDcreationHE.m`` does.

    Each cluster gets a score equal to the product of two 1-based ranks:
    rank of the mean Lab centre, and rank of the mean nonzero grayscale
    intensity. The lowest score wins.

    Parameters
    ----------
    cluster_center : numpy.ndarray
        Centres, shape ``(k, d)``.
    mean_cluster_intensity : numpy.ndarray
        One mean intensity per cluster.

    Returns
    -------
    int
        0-based index of the chosen cluster.
    """
    n = int(cluster_center.shape[0])
    mean_cluster_value = np.mean(np.asarray(cluster_center, dtype=np.float64), axis=1)
    idx = _matlab_argsort_asc(mean_cluster_value)
    idx1 = _matlab_argsort_asc(mean_cluster_intensity)
    cluster_val = np.empty(n, dtype=np.float64)
    for k in range(n):
        rank_c = int(np.flatnonzero(idx == k)[0]) + 1
        rank_i = int(np.flatnonzero(idx1 == k)[0]) + 1
        cluster_val[k] = float(rank_c * rank_i)
    idx2 = _matlab_argsort_asc(cluster_val)
    return int(idx2[0])


def _mean_nonzero_gray(segmented: np.ndarray) -> float:
    """Mean of the nonzero grayscale pixels in one cluster image.

    Parameters
    ----------
    segmented : numpy.ndarray
        RGB crop of one cluster. uint8 uses ``rgb2gray`` on uint8.

    Returns
    -------
    float
        Mean of nonzero gray values, or NaN when every pixel is zero.
    """
    if segmented.dtype == np.uint8:
        gray = matlab_rgb2gray_uint8(segmented)
        nz = gray[gray > 0]
    else:
        gray = matlab_rgb2gray(segmented)
        nz = gray[gray != 0]
    if nz.size == 0:
        return float("nan")
    return float(nz.mean())


def annotate_he(
    he_u8: np.ndarray,
    pixelpermicron: float,
    *,
    kmeans_seed: int = DEFAULT_KMEANS_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Pixel-exact ``BDcreationHE.m``.

    Returns a boolean mask (``BDmask > 0``) plus debug arrays. MATLAB's
    saved mask is ``uint8(255 * mask)`` and is stored as ``debug['BDmask']``.

    Parameters
    ----------
    he_u8 : numpy.ndarray
        Registered uint8 RGB H&E. Other dtypes are rounded to uint8.
    pixelpermicron : float
        Native resolution. This path does not cap ppm at 2.
    kmeans_seed : int, optional
        Seed for ``rng(seed, 'twister')`` before ``kmeans``. Default is the
        seed that matches the committed goldens.

    Returns
    -------
    mask : numpy.ndarray
        Boolean tumor mask.
    debug : dict
        Decorrelated image, cluster labels, and the uint8 ``BDmask``.

    Raises
    ------
    ValueError
        If ``he_u8`` is not a 3-channel image.
    """
    he = np.asarray(he_u8)
    if he.ndim != 3 or he.shape[-1] < 3:
        raise ValueError(f"BDcreationHE expects uint8 RGB, got shape {he.shape}")
    he = he[..., :3]
    if he.dtype != np.uint8:
        he = np.clip(np.floor(matlab_im2double(he) * 255.0 + 0.5), 0, 255).astype(np.uint8)
    pix = float(pixelpermicron)
    s = matlab_decorrstretch_uint8(he, tol=0.01)
    m, n, _ = s.shape
    disk = matlab_fspecial_disk(float(matlab_round(HE_RGB_DISK_RADIUS_MULTIPLIER * pix)))
    k3 = np.empty((m, n, 3), dtype=s.dtype)
    for j in range(3):
        padded = matlab_padarray(s[..., j], HE_RGB_PAD)
        k2 = matlab_histeq(padded)
        k1 = matlab_imfilter_keep_class(k2, disk, boundary="zero")
        k1 = matlab_imfilter_keep_class(k1, disk, boundary="zero")
        k3[..., j] = k1[HE_RGB_PAD : HE_RGB_PAD + m, HE_RGB_PAD : HE_RGB_PAD + n]
    ab = np.asarray(k3, dtype=np.float64)
    ab_flat = ab.reshape((m * n, 3), order="F")
    labels0, centers, _sumd, km_debug = matlab_kmeans(
        ab_flat, HE_RGB_N_COLORS, seed=int(kmeans_seed), replicates=3
    )
    pixel_labels = labels0.reshape((m, n), order="F")  # 0-based
    mean_intensity = np.empty(HE_RGB_N_COLORS, dtype=np.float64)
    segmented: list[np.ndarray] = []
    for k in range(HE_RGB_N_COLORS):
        color = k3.copy()
        color[pixel_labels != k] = 0
        segmented.append(color)
        mean_intensity[k] = _mean_nonzero_gray(color)
    blue = pick_he_epithelial_cluster(centers, mean_intensity)
    epith = matlab_im2double(segmented[blue])
    epith_cell_bw = matlab_im2bw(matlab_rgb2gray(epith), EPITH_BINARY_THRESHOLD)
    opened = _imdilate(epith_cell_bw, _disk_nhood(HE_RGB_EPITH_DILATION_MULTIPLIER * pix, how="round"))
    bwx = ndimage.binary_fill_holes(opened)
    hole_area = int(matlab_round((BACKGROUND_HOLE_MIN_AREA_MULTIPLIER * pix) ** 2))
    tumor_area = int(matlab_round((TUMOR_MASK_MIN_AREA_MULTIPLIER * pix) ** 2))
    bwy = remove_small_components(~bwx, hole_area)
    mask_image = remove_small_components(~bwy, tumor_area)
    bd_u8 = (np.asarray(mask_image, dtype=np.uint8) * 255)
    debug: dict[str, Any] = {
        "annotation_method": np.array(["rgb_kmeans"]),
        "S": s,
        "k3": k3,
        "pixel_labels": pixel_labels.astype(np.int32),
        "cluster_center": centers,
        "blue_cluster_num": np.array([blue], dtype=np.int32),
        "epith_cell_BW": epith_cell_bw,
        "BWx": bwx,
        "mask_image": mask_image.astype(np.uint8),
        "BDmask": bd_u8,
        "kmeans_seed": np.array([int(kmeans_seed)], dtype=np.int32),
        "kmeans_debug": km_debug,
    }
    return mask_image.astype(bool), debug
