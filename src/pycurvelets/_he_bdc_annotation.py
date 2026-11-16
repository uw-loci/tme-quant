"""HSV and RGB k-means tumor annotation (``BDcreationHE2.m`` / ``BDcreationHE.m``).

Registration helpers stay in :mod:`pycurvelets._he_bdc_common` and
:mod:`pycurvelets._he_bdc_reg1`. MATLAB Image Processing Toolbox ports live in
:mod:`pycurvelets._matlab_util`. This module is the annotation pipeline only.

``annotate_he_hsv`` is the HSV path (``BDcreationHE2.m``). It caps the working
resolution at 2 px/um with scalar ``imresize`` (``ceil``), builds nuclei and
collagen masks, and smooths the tumor mask before thresholding it back onto
the original grid.

``annotate_he_kmeans`` is the RGB k-means path (``BDcreationHE.m``). It runs
``decorrstretch``, ``histeq``, and ``fspecial('disk')`` at native resolution,
then picks the epithelial cluster. MATLAB ``kmeans`` is unseeded;
``kmeans_seed`` replays ``rng(seed, 'twister')``.

Morphology uses Adams ``strel('disk')``
(:func:`pycurvelets._matlab_util.matlab_strel_disk`), not a Euclidean disk.
The RGB path pads with ``np.pad(..., mode="symmetric")``.
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
    remove_small_components,
)
from ._he_bdc_reg1 import DEFAULT_KMEANS_SEED
from ._matlab_util import (
    gaussian_filter_matlab_like,
    matlab_decorrstretch_uint8,
    matlab_fspecial_disk,
    matlab_graythresh,
    matlab_histeq,
    matlab_im2bw,
    matlab_im2double,
    matlab_imfilter_keep_class,
    matlab_imresize,
    matlab_kmeans,
    matlab_rgb2gray,
    matlab_rgb2gray_uint8,
    matlab_rgb2hsv,
    matlab_round,
    matlab_strel_disk,
)

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


def annotate_he_hsv(he_raw: np.ndarray, pixelpermicron: float) -> tuple[np.ndarray, dict[str, Any]]:
    """HSV tumor mask from ``BDcreationHE2.m`` on one registered H&E image.

    This is the hue path. It caps the working resolution at 2 px/um, builds
    nuclei and collagen masks, and returns a boolean mask at the original
    size. It does not cluster colors. The RGB k-means path is
    :func:`annotate_he_kmeans` (``BDcreationHE.m``), which stays at native
    resolution and stores a uint8 0/255 mask.

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
        MATLAB-named intermediates: ``he_adjusted``, ``BW_nuclei``,
        ``maskednucleiImage``, ``BW_collagen1``, ``BW_nobackground``,
        ``sat_thresh``, ``epith_cell_BW``, ``BWx``, ``mask_image1``, ``B``
        (smoothed mask), ``mask_temp``, ``mask_thresh``, ``BDmask``, and
        ``pixpermic_working`` (the capped resolution).
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


def annotate_he_kmeans(
    he_u8: np.ndarray,
    pixelpermicron: float,
    *,
    kmeans_seed: int = DEFAULT_KMEANS_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """RGB k-means tumor mask from ``BDcreationHE.m`` on one registered H&E image.

    This path stays at native resolution. It runs ``decorrstretch``, picks
    the epithelial cluster, and stores the saved mask as uint8 0/255 in
    ``debug['BDmask']``. The returned mask is that array as a boolean
    (``BDmask > 0``). The HSV path is :func:`annotate_he_hsv`
    (``BDcreationHE2.m``), which caps resolution at 2 px/um and does not
    cluster colors.

    Parameters
    ----------
    he_u8 : numpy.ndarray
        Registered uint8 RGB H&E. Other dtypes are rounded to uint8.
    pixelpermicron : float
        Native resolution. This path does not cap ppm at 2.
    kmeans_seed : int, optional
        Seed for ``rng(seed, 'twister')`` before ``kmeans``. MATLAB's
        ``kmeans`` is unseeded, so a reference mask matches only this seed.
        The default, 28, matches ``dumps/he_testN`` and the RGB
        ``CA_Boundary`` TIFFs. The HSV ``CA_Boundary`` TIFFs do not use it.

    Returns
    -------
    mask : numpy.ndarray
        Boolean tumor mask.
    debug : dict
        ``S`` (decorrstretched image), ``k3``, 0-based ``pixel_labels``,
        ``cluster_center``, 0-based ``blue_cluster_num``, ``epith_cell_BW``,
        ``BWx``, ``mask_image``, the uint8 ``BDmask`` MATLAB saves (0 or 255),
        ``kmeans_seed``, and ``kmeans_debug`` (seed and each replicate's
        objective).

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
        # padarray(S(:,:,j), [70 70], 'symmetric'): a scalar width pads every side.
        padded = np.pad(s[..., j], HE_RGB_PAD, mode="symmetric")
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
