from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import morphology
from sklearn.cluster import KMeans

from ._he_bdc_common import (
    adjust_rgb_mean_std,
    decorrelation_stretch,
    disk_se,
    ensure_rgb,
    gaussian_filter_matlab_like,
    load_image_as_float,
    make_collagen_mask,
    make_nuclei_mask,
    matlab_area_open,
    matlab_rgb2gray,
    matlab_round,
    prepare_he_image,
    remove_small_components,
    resize_like,
    safe_otsu,
    save_image_uint8,
)

# Morphology/filter constants copied from MATLAB BDcreationHE2 heuristics.
EPITH_BINARY_THRESHOLD = 0.001
EPITH_DILATION_RADIUS_MULTIPLIER = 5.0
BACKGROUND_HOLE_MIN_AREA_MULTIPLIER = 60.0
TUMOR_MASK_MIN_AREA_MULTIPLIER = 35.0
FINAL_MASK_DILATION_RADIUS_MULTIPLIER = 4.0
FINAL_MASK_GAUSSIAN_SIGMA = 25.0
FINAL_MASK_GAUSSIAN_KERNEL_SIZE = 101

# BDcreationHE.m RGB/k-means path.
HE_RGB_DISK_RADIUS_MULTIPLIER = 7.0
HE_RGB_PAD = 70
HE_RGB_N_COLORS = 4
HE_RGB_EPITH_DILATION_MULTIPLIER = 4.0


@dataclass
class TumorAnnotationFromHEParameters:
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
    areaThreshold: float
    SHGfilepath: str
    # "hsv" (default): BDcreationHE2.m HSV path.
    # "rgb_kmeans": BDcreationHE.m decorrstretch + LAB/RGB k-means path.
    annotation_method: str = "hsv"


def _to_params(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
) -> TumorAnnotationFromHEParameters:
    if isinstance(params, TumorAnnotationFromHEParameters):
        return params
    return TumorAnnotationFromHEParameters(**params)


def _resolve_save_dir(params: TumorAnnotationFromHEParameters) -> Path:
    if params.SHGfilepath:
        return Path(params.SHGfilepath) / "CA_Boundary"
    return Path(params.HEfilepath) / "CA_Boundary"


def _resolve_mask_name(he_filename: str) -> str:
    return f"mask for {he_filename.replace('HE', 'SHG')}.tif"


def _annotate_hsv(
    he_raw: np.ndarray,
    pixelpermicron: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """BDcreationHE2.m HSV + morphology path with full intermediate dump (F1)."""
    he_rgb, pix_per_mic = prepare_he_image(he_raw, pixelpermicron)
    orig_shape = he_raw.shape[:2]

    he_adjusted = adjust_rgb_mean_std(he_rgb)
    nuclei_opened, masked_nuclei_image = make_nuclei_mask(he_adjusted, pix_per_mic)
    bw_collagen1, bw_no_background, sat_thresh = make_collagen_mask(
        he_adjusted,
        pix_per_mic,
        enhanced_postprocessing=True,
    )

    epith_cell_bw = (
        (matlab_rgb2gray(masked_nuclei_image) > EPITH_BINARY_THRESHOLD)
        & (~bw_collagen1)
        & bw_no_background
    )
    epith_cell_bw_open = morphology.dilation(
        epith_cell_bw,
        disk_se(EPITH_DILATION_RADIUS_MULTIPLIER * pix_per_mic),
    )
    bwx = ndimage.binary_fill_holes(epith_cell_bw_open)
    bwy = matlab_area_open(
        ~bwx,
        int(matlab_round((BACKGROUND_HOLE_MIN_AREA_MULTIPLIER * pix_per_mic) ** 2)),
    )
    mask_image = matlab_area_open(
        ~bwy,
        int(matlab_round((TUMOR_MASK_MIN_AREA_MULTIPLIER * pix_per_mic) ** 2)),
    )
    mask_image1 = morphology.dilation(
        mask_image,
        disk_se(FINAL_MASK_DILATION_RADIUS_MULTIPLIER * pix_per_mic),
    ) & (~bw_collagen1)

    smoothed = gaussian_filter_matlab_like(
        mask_image1.astype(np.float64),
        sigma=FINAL_MASK_GAUSSIAN_SIGMA,
        kernel_size=FINAL_MASK_GAUSSIAN_KERNEL_SIZE,
        boundary="replicate",
    )
    mask_temp = resize_like(smoothed, orig_shape)
    mask_thresh = safe_otsu(mask_temp)
    bd_mask = mask_temp > mask_thresh

    debug = {
        "annotation_method": np.array(["hsv"]),
        "he_adjusted": he_adjusted,
        "nuclei_opened": nuclei_opened.astype(np.uint8),
        "masked_nuclei": masked_nuclei_image,
        "bw_collagen1": bw_collagen1.astype(np.uint8),
        "bw_no_background": bw_no_background.astype(np.uint8),
        "sat_thresh": np.array([sat_thresh], dtype=np.float64),
        "epith_cell_bw": epith_cell_bw.astype(np.uint8),
        "epith_cell_bw_open": epith_cell_bw_open.astype(np.uint8),
        "bwx": bwx.astype(np.uint8),
        "bwy": bwy.astype(np.uint8),
        "mask_image": mask_image.astype(np.uint8),
        "mask_image1": mask_image1.astype(np.uint8),
        "smoothed": smoothed,
        "mask_temp": mask_temp,
        "mask_thresh": np.array([mask_thresh], dtype=np.float64),
        "bd_mask": bd_mask.astype(np.uint8),
    }
    return bd_mask.astype(bool), debug


def _annotate_rgb_kmeans(
    he_raw: np.ndarray,
    pixelpermicron: float,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Port of MATLAB ``BDcreationHE.m`` (F4): decorrstretch + dual histeq/disk
    blur + 4-cluster k-means; pick darkest/blue epithelial cluster.
    """
    from skimage import exposure

    he_rgb = ensure_rgb(he_raw)
    pix = float(pixelpermicron)
    # BDcreationHE does not downsample by ppm cap the same way HE2 does;
    # operate at native resolution like the MATLAB script.
    decorr = decorrelation_stretch(he_rgb, tol=0.01)
    h, w = decorr.shape[:2]
    radius = int(matlab_round(HE_RGB_DISK_RADIUS_MULTIPLIER * pix))
    disk = morphology.disk(max(radius, 1)).astype(np.float64)
    disk /= float(disk.sum())

    # Pad, hist-eq, double disk filter per channel (matches BDcreationHE.m).
    filtered = np.zeros_like(decorr, dtype=np.float64)
    pad = HE_RGB_PAD
    for j in range(3):
        k = np.pad(decorr[..., j], pad, mode="symmetric")
        k2 = exposure.equalize_hist(k)
        k1 = ndimage.correlate(k2, disk, mode="nearest")
        k1 = ndimage.correlate(k1, disk, mode="nearest")
        filtered[..., j] = k1[pad : pad + h, pad : pad + w]

    ab = filtered.reshape(-1, 3)
    km = KMeans(
        n_clusters=HE_RGB_N_COLORS,
        n_init=3,
        random_state=random_state,
    )
    labels = km.fit_predict(ab).reshape(h, w)
    centers = km.cluster_centers_

    mean_cluster_value = centers.mean(axis=1)
    idx_by_center = np.argsort(mean_cluster_value)  # ascending

    mean_cluster_intensity = np.zeros(HE_RGB_N_COLORS, dtype=np.float64)
    segmented = []
    for k in range(HE_RGB_N_COLORS):
        color = filtered.copy()
        color[labels != k] = 0.0
        segmented.append(color)
        nz = matlab_rgb2gray(color)
        nz_vals = nz[nz > 0]
        mean_cluster_intensity[k] = float(nz_vals.mean()) if nz_vals.size else 0.0
    idx_by_intensity = np.argsort(mean_cluster_intensity)

    # MATLAB: cluster_val(k) = find(idx==k)*find(idx1==k); pick argmin.
    cluster_val = np.zeros(HE_RGB_N_COLORS, dtype=np.float64)
    for k in range(HE_RGB_N_COLORS):
        rank_c = int(np.where(idx_by_center == k)[0][0]) + 1
        rank_i = int(np.where(idx_by_intensity == k)[0][0]) + 1
        cluster_val[k] = float(rank_c * rank_i)
    blue_cluster_num = int(np.argmin(cluster_val))

    epith_cell = segmented[blue_cluster_num]
    epith_cell_bw = matlab_rgb2gray(epith_cell) > EPITH_BINARY_THRESHOLD
    epith_cell_bw_open = morphology.dilation(
        epith_cell_bw,
        disk_se(HE_RGB_EPITH_DILATION_MULTIPLIER * pix),
    )
    bwx = ndimage.binary_fill_holes(epith_cell_bw_open)
    bwy = remove_small_components(
        ~bwx,
        int(matlab_round((BACKGROUND_HOLE_MIN_AREA_MULTIPLIER * pix) ** 2)),
    )
    mask_image = remove_small_components(
        ~bwy,
        int(matlab_round((TUMOR_MASK_MIN_AREA_MULTIPLIER * pix) ** 2)),
    )
    bd_mask = mask_image.astype(bool)

    debug = {
        "annotation_method": np.array(["rgb_kmeans"]),
        "decorr": decorr,
        "filtered": filtered,
        "labels": labels.astype(np.int32),
        "blue_cluster_num": np.array([blue_cluster_num], dtype=np.int32),
        "epith_cell_bw": epith_cell_bw.astype(np.uint8),
        "epith_cell_bw_open": epith_cell_bw_open.astype(np.uint8),
        "bwx": bwx.astype(np.uint8),
        "bwy": bwy.astype(np.uint8),
        "mask_image": mask_image.astype(np.uint8),
        "bd_mask": bd_mask.astype(np.uint8),
    }
    return bd_mask, debug


def tumor_annotation_from_he(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Python conversion of MATLAB BDcreationHE2.m (default) / BDcreationHE.m.

    Generates a tumor boundary mask from an HE image using nuclei/collagen
    segmentation and morphology heuristics.
    """
    p = _to_params(params)
    he_path = Path(p.HEfilepath) / p.HEfilename
    he_raw = load_image_as_float(he_path)

    method = (p.annotation_method or "hsv").lower()
    if method in ("hsv", "he2"):
        bd_mask, debug = _annotate_hsv(he_raw, p.pixelpermicron)
    elif method in ("rgb_kmeans", "rgb", "he"):
        bd_mask, debug = _annotate_rgb_kmeans(he_raw, p.pixelpermicron)
    else:
        raise ValueError(
            f"Unknown annotation_method={p.annotation_method!r}; "
            f"expected 'hsv' or 'rgb_kmeans'."
        )

    if save_output:
        save_dir = _resolve_save_dir(p)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            save_dir = Path(p.HEfilepath) / "CA_Boundary"
            save_dir.mkdir(parents=True, exist_ok=True)

        mask_name = _resolve_mask_name(p.HEfilename)
        save_image_uint8(save_dir / mask_name, bd_mask)

    if not return_debug:
        return bd_mask.astype(bool)

    return bd_mask.astype(bool), debug


def BDcreationHE2(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name (HSV path)."""
    p = replace(_to_params(BDCparameters), annotation_method="hsv")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)


def BDcreationHE(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper for MATLAB ``BDcreationHE.m`` (RGB k-means path)."""
    p = replace(_to_params(BDCparameters), annotation_method="rgb_kmeans")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)
