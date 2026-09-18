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
    """``im2double``: uint8/uint16 scale to ``[0, 1]``; float left as-is."""
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float64) / 65535.0
    if a.dtype == np.bool_:
        return a.astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def matlab_im2bw(image: np.ndarray, thresh: float) -> np.ndarray:
    """``im2bw(I, level)``: ``im2double(I) > level`` (logical)."""
    a = np.asarray(image)
    if a.dtype == np.bool_:
        return a.copy()
    return matlab_im2double(a) > float(thresh)


def matlab_padarray(image: np.ndarray, pad: int | tuple[int, int], mode: str = "symmetric") -> np.ndarray:
    """
    ``padarray(A, [p p], 'symmetric')`` (both sides). MATLAB's symmetric
    reflection duplicates the edge pixel, which is ``np.pad(..., mode='symmetric')``.
    """
    a = np.asarray(image)
    if isinstance(pad, tuple):
        pr, pc = int(pad[0]), int(pad[1])
    else:
        pr = pc = int(pad)
    if mode != "symmetric":
        raise ValueError(f"only 'symmetric' padarray is implemented, got {mode!r}")
    return np.pad(a, ((pr, pr), (pc, pc)), mode="symmetric")


def matlab_fspecial_disk(radius: float) -> np.ndarray:
    """``fspecial('disk', radius)``: normalized circular averaging kernel."""
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
    """``createTransformationToIntensityImage`` from MATLAB ``histeq.m``."""
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
    """
    ``images.internal.builtins.grayxform(I, T)``.

    ``T`` is a double LUT of length ``N`` with values in ``[0, 1]``. uint8 with
    ``N == 256`` maps ``uint8(floor(255*T(I+1)+0.5))``. Double maps
    ``T(floor(I*(N-1)+0.5)+1)`` with values outside ``[0, 1]`` clamped to the
    end bins.
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


def matlab_histeq(image: np.ndarray, n_levels: int | None = None) -> np.ndarray:
    """
    ``histeq(I)`` / ``histeq(I, N)``.

    Default ``N`` is 64 for every class (including uint8). The *input*
    histogram always uses 256 bins for uint8 and double. Same class out.
    """
    a = np.asarray(image)
    npts = 256
    m = 64 if n_levels is None else int(n_levels)
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
    """``imfilter`` returning the same integer class as ``image`` (round + saturate)."""
    a = np.asarray(image)
    out = matlab_imfilter(a.astype(np.float64), kernel, boundary=boundary)
    if a.dtype == np.uint8:
        return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)
    if a.dtype == np.uint16:
        return np.clip(np.floor(out + 0.5), 0, 65535).astype(np.uint16)
    return out.astype(np.float64)


def _disk_nhood(radius: float, *, how: str) -> np.ndarray:
    if how == "ceil":
        r = int(np.ceil(radius))
    elif how == "round":
        r = int(matlab_round(radius))
    else:
        raise ValueError(f"how must be 'ceil' or 'round', got {how!r}")
    return matlab_strel_disk(max(r, 0))


def _imdilate(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def _imopen(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    return ndimage.binary_opening(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def _imclose(mask: np.ndarray, nhood: np.ndarray) -> np.ndarray:
    return ndimage.binary_closing(np.asarray(mask, dtype=bool), structure=np.asarray(nhood, dtype=bool))


def prepare_he2_image(he: np.ndarray, pixel_per_micron: float) -> tuple[np.ndarray, float]:
    """
    BDcreationHE2 sizing: ``if ppm > 2: HEdata = imresize(HE, 2/ppm); ppm = 2``.

    Uses MATLAB's scalar-scale ``imresize`` (``ceil`` output size), not
    :func:`prepare_he_image` (that helper ``round``s and is shared with registration).
    """
    he_rgb = ensure_rgb(he)
    pix = float(pixel_per_micron)
    if pix > HE2_PPM_CAP:
        he_rgb = matlab_imresize(he_rgb, scalar_scale=HE2_PPM_CAP / pix, method="bicubic")
        pix = HE2_PPM_CAP
    return he_rgb, pix


def he2_nuclei_mask(he_adjusted: np.ndarray, pix_per_mic: float) -> tuple[np.ndarray, np.ndarray, float]:
    """HSV nuclei band + ``bwareaopen(150)`` + ``imopen(strel('disk', ceil(ppm/2)))``."""
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
    return bw_nuclei, masked, sat_thresh


def he2_collagen_mask(he_adjusted: np.ndarray, pix_per_mic: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Wrapped-hue collagen + dilate ``ceil(ppm)`` + close ``round(3*ppm)``."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dilate / fill / dual ``bwareaopen`` / collagen-masked dilate (HE2 lines 143-149)."""
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
    return opened, bwx, mask_image, mask_image1


def annotate_he2(he_raw: np.ndarray, pixelpermicron: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Pixel-exact ``BDcreationHE2.m``. Returns logical ``BDmask`` and named intermediates."""
    orig_row, orig_col = int(he_raw.shape[0]), int(he_raw.shape[1])
    he_data, pix = prepare_he2_image(matlab_im2double(he_raw), float(pixelpermicron))
    he_adjusted = adjust_rgb_mean_std(he_data)
    bw_nuclei, maskednuclei, _sat_n = he2_nuclei_mask(he_adjusted, pix)
    bw_collagen1, bw_nobackground, sat_thresh = he2_collagen_mask(he_adjusted, pix)
    epith_cell_bw = (
        matlab_im2bw(matlab_rgb2gray(maskednuclei), EPITH_BINARY_THRESHOLD)
        & (~bw_collagen1)
        & bw_nobackground
    )
    _opened, bwx, _mask_image, mask_image1 = _he2_epithelial_morphology(
        epith_cell_bw, bw_collagen1, pix
    )
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
        "nuclei_opened": bw_nuclei.astype(np.uint8),
        "maskednucleiImage": maskednuclei,
        "masked_nuclei": maskednuclei,
        "BW_collagen1": bw_collagen1,
        "bw_collagen1": bw_collagen1.astype(np.uint8),
        "BW_nobackground": bw_nobackground,
        "bw_no_background": bw_nobackground.astype(np.uint8),
        "sat_thresh": np.array([sat_thresh], dtype=np.float64),
        "epith_cell_BW": epith_cell_bw,
        "epith_cell_bw": epith_cell_bw.astype(np.uint8),
        "BWx": bwx,
        "bwx": bwx.astype(np.uint8),
        "mask_image1": mask_image1.astype(np.uint8),
        "B": smoothed,
        "smoothed": smoothed,
        "mask_temp": mask_temp,
        "mask_thresh": np.array([mask_thresh], dtype=np.float64),
        "BDmask": bd_mask,
        "bd_mask": bd_mask.astype(np.uint8),
        "pixpermic_working": pix,
    }
    return bd_mask.astype(bool), debug


def _matlab_argsort_asc(x: np.ndarray) -> np.ndarray:
    """Stable ascending argsort with NaNs last, matching MATLAB ``sort``."""
    v = np.asarray(x, dtype=np.float64)
    keys = np.where(np.isnan(v), np.inf, v)
    return np.argsort(keys, kind="stable")


def pick_he_epithelial_cluster(
    cluster_center: np.ndarray,
    mean_cluster_intensity: np.ndarray,
) -> int:
    """
    Dual-sort cluster pick from ``BDcreationHE.m`` lines 56-65.

    ``cluster_val(k) = find(idx==k)*find(idx1==k)`` on 1-based ranks of
    ``mean(cluster_center, 2)`` and ``mean(nonzeros(rgb2gray(segmented)))``.
    Returns the **0-based** cluster index of ``idx2(1)``.
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
    """``mean(nonzeros(rgb2gray(segmented)))``; empty -> NaN."""
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

    Returns a **boolean** mask (``BDmask > 0``) plus debug arrays. MATLAB's
    saved mask is ``uint8(255 * mask)`` and is stored as ``debug['BDmask']``.
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
        padded = matlab_padarray(s[..., j], HE_RGB_PAD, mode="symmetric")
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
        "labels": pixel_labels.astype(np.int32),
        "cluster_center": centers,
        "blue_cluster_num": np.array([blue], dtype=np.int32),
        "epith_cell_BW": epith_cell_bw,
        "epith_cell_bw": epith_cell_bw.astype(np.uint8),
        "BWx": bwx,
        "bwx": bwx.astype(np.uint8),
        "mask_image": mask_image.astype(np.uint8),
        "BDmask": bd_u8,
        "bd_mask": (bd_u8 > 0).astype(np.uint8),
        "kmeans_seed": np.array([int(kmeans_seed)], dtype=np.int32),
        "kmeans_debug": km_debug,
    }
    return mask_image.astype(bool), debug
