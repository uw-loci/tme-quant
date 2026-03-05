from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.registration import phase_cross_correlation

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
    save_image_uint8,
    to_grayscale,
)


@dataclass
class SHGHERegistrationParameters:
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
    SHGfilepath: str
    areaThreshold: float | None = None


def _to_params(
    params: SHGHERegistrationParameters | dict[str, Any],
) -> SHGHERegistrationParameters:
    if isinstance(params, SHGHERegistrationParameters):
        return params
    return SHGHERegistrationParameters(**params)


def _normalize_for_registration(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def _shift_rgb(image: np.ndarray, shift_rc: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(image, dtype=np.float64)
    for ch in range(image.shape[2]):
        shifted[:, :, ch] = ndimage.shift(
            image[:, :, ch],
            shift=shift_rc,
            order=1,
            mode="constant",
            cval=1.0,
            prefilter=False,
        )
    return shifted


def _warp_rgb_affine(image: np.ndarray, warp_matrix: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    h, w = out_shape
    warped = np.zeros((h, w, image.shape[2]), dtype=np.float64)
    for ch in range(image.shape[2]):
        warped[:, :, ch] = cv2.warpAffine(
            image[:, :, ch].astype(np.float32),
            warp_matrix,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
    return np.clip(warped, 0.0, 1.0)


def _estimate_affine_refinement(
    moving: np.ndarray,
    fixed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate deterministic coarse+fine registration transform.
    1) Phase cross correlation for coarse translational alignment.
    2) ECC affine refinement to mimic MATLAB multimodal registration strategy.
    """
    moving_n = _normalize_for_registration(moving)
    fixed_n = _normalize_for_registration(fixed)

    if np.allclose(moving_n, 0.0) or np.allclose(fixed_n, 0.0):
        return np.zeros(2, dtype=np.float64), np.eye(2, 3, dtype=np.float32)

    shift_rc, _, _ = phase_cross_correlation(
        fixed_n,
        moving_n,
        upsample_factor=10,
        normalization=None,
    )

    moving_shifted = ndimage.shift(
        moving_n,
        shift=shift_rc,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        700,
        1e-7,
    )

    try:
        cv2.findTransformECC(
            fixed_n.astype(np.float32),
            moving_shifted,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        # Keep identity affine when ECC does not converge on low-information images.
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    return shift_rc.astype(np.float64), warp_matrix


def shg_he_registration(
    params: SHGHERegistrationParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Python conversion of MATLAB BDcreation_reg2.m.

    Registers an H&E bright-field image to its corresponding SHG image and writes
    the aligned H&E image to HE_registered/<HEfilename> for downstream segmentation.
    """
    p = _to_params(params)

    he_path = Path(p.HEfilepath) / p.HEfilename
    shg_path = Path(p.SHGfilepath) / p.HEfilename

    he = load_image_as_float(he_path)
    shg = load_image_as_float(shg_path)

    shg_gray = to_grayscale(shg)
    original_shg_shape = shg_gray.shape[:2]

    he_scaled, pix_per_mic = prepare_he_image(he, p.pixelpermicron)
    if float(p.pixelpermicron) > 2.0:
        fixed_shg = resize_like(shg_gray, he_scaled.shape[:2])
    else:
        fixed_shg = shg_gray
        he_scaled = resize_like(he_scaled, fixed_shg.shape[:2])

    he_adjusted = adjust_rgb_mean_std(he_scaled)

    _, masked_nuclei_image = make_nuclei_mask(he_adjusted, pix_per_mic)
    bw_collagen, _, _ = make_collagen_mask(
        he_adjusted,
        pix_per_mic,
        enhanced_postprocessing=False,
    )

    gray_nuclei = to_grayscale(masked_nuclei_image)
    nuclei_filtered = gaussian_filter_matlab_like(
        gray_nuclei,
        sigma=0.5,
        kernel_size=max(int(np.floor(pix_per_mic)), 1),
    )
    bw_nuclei = nuclei_filtered > 0.001
    bw_nuclei_discard = matlab_area_open(bw_nuclei, int(np.ceil(50.0 * pix_per_mic**2)))
    bw_nuclei_dilated = morphology.dilation(
        bw_nuclei_discard, disk_se(np.floor(pix_per_mic))
    )
    bw_nuclei_filled = ndimage.binary_fill_holes(bw_nuclei_dilated)

    he_collagen_bw = bw_collagen & (~bw_nuclei_filled)
    bw_discard = matlab_area_open(he_collagen_bw, int(np.ceil(max(pix_per_mic**2, 1.0))))
    he_collagen_exclude = he_collagen_bw & bw_discard

    shift_rc, affine_warp = _estimate_affine_refinement(
        moving=he_collagen_exclude.astype(np.float64),
        fixed=fixed_shg.astype(np.float64),
    )

    he_shifted = _shift_rgb(he_scaled, shift_rc=shift_rc)
    registered_on_fixed = _warp_rgb_affine(
        he_shifted,
        warp_matrix=affine_warp,
        out_shape=fixed_shg.shape[:2],
    )

    registered_img = resize_like(registered_on_fixed, original_shg_shape)
    registered_img = np.clip(registered_img, 0.0, 1.0)

    if save_output:
        save_dir = Path(p.HEfilepath) / "HE_registered"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image_uint8(save_dir / p.HEfilename, registered_img)

    if not return_debug:
        return registered_img

    debug = {
        "he_adjusted": he_adjusted,
        "he_collagen_exclude": he_collagen_exclude.astype(np.uint8),
        "fixed_shg": fixed_shg,
        "shift_rc": shift_rc,
        "affine_warp": affine_warp,
    }
    return registered_img, debug


def BDcreation_reg2(
    BDCparameters: SHGHERegistrationParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name."""
    return shg_he_registration(BDCparameters, save_output=True, return_debug=False)
