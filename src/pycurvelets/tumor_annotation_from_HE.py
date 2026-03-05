from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import morphology

from ._he_bdc_common import (
    adjust_rgb_mean_std,
    disk_se,
    gaussian_filter_matlab_like,
    load_image_as_float,
    make_collagen_mask,
    make_nuclei_mask,
    matlab_area_open,
    prepare_he_image,
    resize_like,
    safe_otsu,
    save_image_uint8,
    to_grayscale,
)


@dataclass
class TumorAnnotationFromHEParameters:
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
    areaThreshold: float
    SHGfilepath: str


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


def tumor_annotation_from_he(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Python conversion of MATLAB BDcreationHE2.m.

    Generates a tumor boundary mask from an HE image using nuclei/collagen
    segmentation and morphology heuristics.
    """
    p = _to_params(params)
    he_path = Path(p.HEfilepath) / p.HEfilename
    he_raw = load_image_as_float(he_path)

    he_rgb, pix_per_mic = prepare_he_image(he_raw, p.pixelpermicron)
    orig_shape = he_raw.shape[:2]

    he_adjusted = adjust_rgb_mean_std(he_rgb)
    _, masked_nuclei_image = make_nuclei_mask(he_adjusted, pix_per_mic)
    bw_collagen1, bw_no_background, _ = make_collagen_mask(
        he_adjusted,
        pix_per_mic,
        enhanced_postprocessing=True,
    )

    epith_cell_bw = (
        (to_grayscale(masked_nuclei_image) > 0.001)
        & (~bw_collagen1)
        & bw_no_background
    )
    epith_cell_bw_open = morphology.dilation(epith_cell_bw, disk_se(np.round(5.0 * pix_per_mic)))
    bwx = ndimage.binary_fill_holes(epith_cell_bw_open)
    bwy = matlab_area_open(~bwx, int(np.round((60.0 * pix_per_mic) ** 2)))
    mask_image = matlab_area_open(~bwy, int(np.round((35.0 * pix_per_mic) ** 2)))
    mask_image1 = morphology.dilation(mask_image, disk_se(np.round(4.0 * pix_per_mic))) & (~bw_collagen1)

    smoothed = gaussian_filter_matlab_like(mask_image1.astype(np.float64), sigma=25.0, kernel_size=101)
    mask_temp = resize_like(smoothed, orig_shape)
    mask_thresh = safe_otsu(mask_temp)
    bd_mask = mask_temp > mask_thresh

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

    debug = {
        "he_adjusted": he_adjusted,
        "masked_nuclei": masked_nuclei_image,
        "bw_collagen1": bw_collagen1.astype(np.uint8),
        "mask_temp": mask_temp,
    }
    return bd_mask.astype(bool), debug


def BDcreationHE2(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name."""
    return tumor_annotation_from_he(BDCparameters, save_output=True, return_debug=False)
