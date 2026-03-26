"""SHG↔H&E registration (``BDcreation_reg2.m``).

Preprocessing matches the MATLAB script. **Registration** defaults to
**SimpleITK** Mattes mutual information with a similarity stage plus affine
refinement (same moving/fixed pair as MATLAB: collagen mask vs SHG), which
tracks MATLAB ``imregconfig('multimodal')`` (OnePlusOneEvolutionary) / ``imregtform`` much more closely
than phase correlation + ECC. If SimpleITK is missing or registration fails,
the code falls back to phase cross-correlation + OpenCV ``findTransformECC``.
"""

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
    matlab_rgb2gray,
    normalize_array_to_unit_interval,
    prepare_registration_pair,
    resize_like,
    save_image_uint8,
    to_grayscale,
)
from ._shg_he_registration_sitk import (
    build_fixed_sitk_image,
    has_simpleitk,
    register_collagen_to_shg_sitk,
    warn_sitk_fallback,
    warp_rgb_with_sitk_transform,
)

# Registration constants derived from MATLAB BDcreation_reg2 defaults/behavior.
REG_NORMALIZATION_EPSILON = 1e-12
PHASE_XCORR_UPSAMPLE = 10
ECC_MAX_ITERATIONS = 700
ECC_EPSILON = 1e-7
ECC_GAUSSIAN_FILTER_SIZE = 5

# Segmentation/morphology constants derived from MATLAB script thresholds.
NUCLEI_FILTER_SIGMA = 0.5
NUCLEI_BINARY_THRESHOLD = 0.001
NUCLEI_MIN_AREA_MULTIPLIER = 50.0
COLLAGEN_MIN_AREA_MULTIPLIER = 1.0

# Fill values chosen to match MATLAB-style bright background handling.
SHIFT_FILL_VALUE = 1.0
WARP_BORDER_VALUE = 1.0


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
            cval=SHIFT_FILL_VALUE,
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
            borderValue=WARP_BORDER_VALUE,
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
        normalization_epsilon=REG_NORMALIZATION_EPSILON,
        raise_on_homogeneous=True,
    )
    fixed_n = normalize_array_to_unit_interval(
        fixed,
        normalization_epsilon=REG_NORMALIZATION_EPSILON,
        raise_on_homogeneous=True,
    )

    if moving_n.size == 0 or fixed_n.size == 0:
        raise ValueError("Registration input image is empty.")

    shift_rc, _, _ = phase_cross_correlation(
        fixed_n,
        moving_n,
        upsample_factor=PHASE_XCORR_UPSAMPLE,
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
        ECC_MAX_ITERATIONS,
        ECC_EPSILON,
    )

    try:
        cv2.findTransformECC(
            fixed_n.astype(np.float32),
            moving_shifted,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            ECC_GAUSSIAN_FILTER_SIZE,
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

    he_scaled, fixed_shg, pix_per_mic = prepare_registration_pair(
        he, shg_gray, p.pixelpermicron
    )

    he_adjusted = adjust_rgb_mean_std(he_scaled)

    _, masked_nuclei_image = make_nuclei_mask(he_adjusted, pix_per_mic)
    bw_collagen, _, _ = make_collagen_mask(
        he_adjusted,
        pix_per_mic,
        enhanced_postprocessing=False,
    )

    gray_nuclei = matlab_rgb2gray(masked_nuclei_image)
    nuclei_filtered = gaussian_filter_matlab_like(
        gray_nuclei,
        sigma=NUCLEI_FILTER_SIGMA,
        kernel_size=max(int(np.floor(pix_per_mic)), 1),
        boundary="zero",
    )
    bw_nuclei = nuclei_filtered > NUCLEI_BINARY_THRESHOLD
    bw_nuclei_discard = matlab_area_open(
        bw_nuclei,
        int(np.ceil(NUCLEI_MIN_AREA_MULTIPLIER * pix_per_mic**2)),
    )
    bw_nuclei_dilated = morphology.dilation(
        bw_nuclei_discard, disk_se(np.floor(pix_per_mic))
    )
    bw_nuclei_filled = ndimage.binary_fill_holes(bw_nuclei_dilated)

    he_collagen_bw = bw_collagen & (~bw_nuclei_filled)
    bw_discard = matlab_area_open(
        he_collagen_bw,
        int(np.ceil(max(COLLAGEN_MIN_AREA_MULTIPLIER * pix_per_mic**2, 1.0))),
    )
    he_collagen_exclude = he_collagen_bw & bw_discard

    moving_f = he_collagen_exclude.astype(np.float64)
    fixed_f = fixed_shg.astype(np.float64)

    # Diagnostic-only shift (matches legacy ECC path behavior; useful for tests/debug).
    shift_rc, affine_warp = _estimate_affine_refinement(moving=moving_f, fixed=fixed_f)

    registration_backend = "opencv_ecc"
    registered_on_fixed: np.ndarray

    if has_simpleitk():
        try:
            sitk_tx = register_collagen_to_shg_sitk(moving_f, fixed_f)
            fixed_sitk = build_fixed_sitk_image(fixed_shg)
            registered_on_fixed = warp_rgb_with_sitk_transform(
                he_scaled.astype(np.float64),
                fixed_sitk,
                sitk_tx,
                fill_value=WARP_BORDER_VALUE,
            )
            registration_backend = "simpleitk_mattes"
        except Exception as exc:
            warn_sitk_fallback(exc)
            he_shifted = _shift_channels(he_scaled, shift_rc=shift_rc, channel_axis=-1)
            registered_on_fixed = _warp_channels_affine(
                he_shifted,
                warp_matrix=affine_warp,
                out_shape=fixed_shg.shape[:2],
                channel_axis=-1,
            )
    else:
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

    debug: dict[str, Any] = {
        "shift_rc": shift_rc,
        "affine_warp": affine_warp,
        "registration_backend": registration_backend,
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
