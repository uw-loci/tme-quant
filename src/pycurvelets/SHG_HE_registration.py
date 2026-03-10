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
    normalize_array_to_unit_interval,
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


def _shift_channels(
    image: np.ndarray,
    shift_rc: np.ndarray,
    channel_axis: int = -1,
) -> np.ndarray:
    image_ch_last = np.moveaxis(image, channel_axis, -1)
    shifted = np.zeros_like(image_ch_last, dtype=np.float64)
    for ch in range(image_ch_last.shape[-1]):
        shifted[:, :, ch] = ndimage.shift(
            image_ch_last[:, :, ch],
            shift=shift_rc,
            order=1,
            mode="constant",
            cval=1.0,
            prefilter=False,
        )
    return np.moveaxis(shifted, -1, channel_axis)


def _warp_channels_affine(
    image: np.ndarray,
    warp_matrix: np.ndarray,
    out_shape: tuple[int, int],
    channel_axis: int = -1,
) -> np.ndarray:
    image_ch_last = np.moveaxis(image, channel_axis, -1)
    h, w = out_shape
    warped = np.zeros((h, w, image_ch_last.shape[-1]), dtype=np.float64)
    for ch in range(image_ch_last.shape[-1]):
        warped[:, :, ch] = cv2.warpAffine(
            image_ch_last[:, :, ch].astype(np.float32),
            warp_matrix,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
    warped = np.clip(warped, 0.0, 1.0)
    return np.moveaxis(warped, -1, channel_axis)


def _estimate_affine_refinement(
    moving: np.ndarray,
    fixed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate deterministic coarse+fine registration transform.
    1) Phase cross correlation for coarse translational alignment.
    2) ECC affine refinement to mimic MATLAB multimodal registration strategy.
    """
    moving_n = normalize_array_to_unit_interval(
        moving,
        normalization_epsilon=1e-12,
        raise_on_homogeneous=True,
    )
    fixed_n = normalize_array_to_unit_interval(
        fixed,
        normalization_epsilon=1e-12,
        raise_on_homogeneous=True,
    )

    if moving_n.size == 0 or fixed_n.size == 0:
        raise ValueError("Registration input image is empty.")

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
    include_debug_images: bool = True,
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

    he_shifted = _shift_channels(he_scaled, shift_rc=shift_rc, channel_axis=-1)
    registered_on_fixed = _warp_channels_affine(
        he_shifted,
        warp_matrix=affine_warp,
        out_shape=fixed_shg.shape[:2],
        channel_axis=-1,
    )

    registered_img = resize_like(registered_on_fixed, original_shg_shape)
    registered_img = np.clip(registered_img, 0.0, 1.0)

    if save_output:
        save_dir = Path(p.HEfilepath) / "HE_registered"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image_uint8(save_dir / p.HEfilename, registered_img)

    if not return_debug:
        return registered_img

    debug: dict[str, np.ndarray] = {
        "shift_rc": shift_rc,
        "affine_warp": affine_warp,
    }
    if include_debug_images:
        debug.update(
            {
                "he_adjusted": he_adjusted,
                "he_collagen_exclude": he_collagen_exclude.astype(np.uint8),
                "fixed_shg": fixed_shg,
            }
        )
    return registered_img, debug


def BDcreation_reg2(
    BDCparameters: SHGHERegistrationParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name."""
    return shg_he_registration(BDCparameters, save_output=True, return_debug=False)
