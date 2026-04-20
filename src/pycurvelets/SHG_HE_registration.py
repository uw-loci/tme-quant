"""H&E ↔ SHG registration — Python port of MATLAB ``BDcreation_reg2.m``.

Algorithm: skimage preprocessing, SimpleITK Mattes MI with exhaustive
grid search (angle, scale, translation) followed by Nelder-Mead
refinement of similarity and affine parameters.  The grid search
replaces MATLAB's stochastic 1+1-ES initial exploration, while
Nelder-Mead provides deterministic sub-pixel convergence for all
transform parameters simultaneously.

Public entry points: :func:`shg_he_registration`, :func:`BDcreation_reg2` (MATLAB
name), :class:`SHGHERegistrationParameters`, :func:`has_simpleitk`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.optimize import minimize as _scipy_minimize
from skimage import io, morphology, registration

from ._he_bdc_common import (
    adjust_rgb_mean_std,
    disk_se,
    gaussian_filter_matlab_like,
    make_collagen_mask,
    make_nuclei_mask,
    matlab_rgb2gray,
    prepare_registration_pair,
    remove_small_components,
    resize_like,
)

try:
    import SimpleITK as sitk  # type: ignore[import-untyped]

    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False
    sitk = None  # type: ignore[assignment]


def has_simpleitk() -> bool:
    return _HAS_SITK


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


def _shg_he_registration_core(
    he_filepath: str,
    he_filename: str,
    shg_filepath: str,
    pixelpermicron: float,
) -> tuple[np.ndarray, str]:
    """
    Core registration (same algorithm as ``tests/debug_reg_2.bdcreation_reg2``).

    Returns
    -------
    registered_img
        Float RGB in ``[0, 1]``, shape matching original SHG (H, W, 3).
    backend
        ``"simpleitk_mattes"`` or ``"skimage_ecc_fallback"``.
    """
    he_path = os.path.join(he_filepath, he_filename)
    shg_path = os.path.join(shg_filepath, he_filename)

    he_img = io.imread(he_path).astype(np.float64) / 255.0
    shg_img = io.imread(shg_path).astype(np.float64) / 255.0
    if shg_img.ndim == 3:
        shg_img = matlab_rgb2gray(shg_img)

    original_shg_shape = shg_img.shape[:2]
    # MATLAB-parity sizing: fixed grid is (possibly downsampled) SHG, HE resized to match.
    he_scaled, fixed_shg, pixpermic = prepare_registration_pair(
        he_img, shg_img, float(pixelpermicron)
    )

    # MATLAB-parity intensity adjust + nuclei/collagen masks (HSV thresholds use matlab_graythresh).
    he_adjusted = adjust_rgb_mean_std(he_scaled)
    _bw_nuclei_opened, masked_nuclei_image = make_nuclei_mask(he_adjusted, pixpermic)
    bw_collagen, _bw_no_background, _sat_thresh = make_collagen_mask(
        he_adjusted, pixpermic, enhanced_postprocessing=False
    )

    # Reproduce BDcreation_reg2 nuclei suppression + collagen exclusion.
    # MATLAB:
    #   gray_nuclei=rgb2gray(maskednucleiImage);
    #   h = fspecial('gaussian', floor(ppm), 0.5);
    #   nuclei_filtered = imfilter(gray_nuclei, h);   % zero-pad
    #   BW_nuclei = im2bw(nuclei_filtered, 0.001);
    #   BW_nuclei_discard = bwareaopen(BW_nuclei, ceil(50*ppm^2));  % 8-connected
    #   se = strel('disk', floor(ppm));
    #   BW_nuclei_dilated = imdilate(BW_nuclei_discard, se);
    #   BW_nuclei_filled = imfill(BW_nuclei_dilated,'holes');
    #   HE_collagen_BW = BW_collagen .* (~BW_nuclei_filled);
    #   BW_discard = bwareaopen(HE_collagen_BW, ceil(ppm^2));
    #   HEmoving = HE_collagen_BW .* BW_discard;
    gray_nuclei = matlab_rgb2gray(masked_nuclei_image)
    ksize = max(1, int(np.floor(pixpermic)))
    nuclei_filtered = gaussian_filter_matlab_like(
        gray_nuclei, sigma=0.5, kernel_size=ksize, boundary="zero"
    )
    bw_nuclei = nuclei_filtered > 0.001
    bw_nuclei_discard = remove_small_components(
        bw_nuclei, int(np.ceil(50.0 * pixpermic**2))
    )
    bw_nuclei_dilated = morphology.dilation(
        bw_nuclei_discard, disk_se(np.floor(pixpermic))
    )
    bw_nuclei_filled = binary_fill_holes(bw_nuclei_dilated)

    he_collagen_bw = bw_collagen & (~bw_nuclei_filled)
    he_collagen_bw = remove_small_components(
        he_collagen_bw, int(np.ceil(pixpermic**2))
    )
    he_moving = he_collagen_bw.astype(np.float64)

    fixed = fixed_shg.astype(np.float64)
    if fixed.ndim == 3:
        fixed = matlab_rgb2gray(fixed)

    backend: str
    if _HAS_SITK:
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(he_moving)
        fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat64)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat64)

        geom_init = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk,
            sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        center = list(geom_init.GetFixedParameters())

        # ----------------------------------------------------------
        # Stage 1a: Grid search over (angle, scale, translation).
        # 3-level refinement for angle/scale, then 2-level for
        # translation at the best (angle, scale).
        # ----------------------------------------------------------
        best_angle = 0.0
        best_scale = 1.0
        best_tx = 0.0
        best_ty = 0.0
        best_metric = float("inf")
        eval_method = sitk.ImageRegistrationMethod()
        eval_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        eval_method.SetMetricSamplingStrategy(eval_method.NONE)
        eval_method.SetInterpolator(sitk.sitkLinear)

        def _eval_similarity(angle_deg, scale_val, tx, ty):
            probe = sitk.Similarity2DTransform()
            probe.SetAngle(float(np.radians(angle_deg)))
            probe.SetScale(float(scale_val))
            probe.SetCenter(center)
            probe.SetTranslation([float(tx), float(ty)])
            eval_method.SetInitialTransform(probe)
            return eval_method.MetricEvaluate(fixed_sitk, moving_sitk)

        def _probe_angle_scale(angle_range, scale_range):
            nonlocal best_angle, best_scale, best_metric
            for scale_val in scale_range:
                for angle_deg in angle_range:
                    val = _eval_similarity(angle_deg, scale_val, best_tx, best_ty)
                    if val < best_metric:
                        best_metric = val
                        best_angle = float(angle_deg)
                        best_scale = float(scale_val)

        # Coarse angle/scale: 2° angle, 0.02 scale
        _probe_angle_scale(range(-45, 46, 2), np.arange(0.80, 1.25, 0.02))
        # Fine angle/scale: 0.5° angle, 0.005 scale
        _probe_angle_scale(
            np.arange(best_angle - 3, best_angle + 3.01, 0.5),
            np.arange(best_scale - 0.04, best_scale + 0.041, 0.005),
        )
        # Ultra-fine angle/scale: 0.1° angle, 0.001 scale
        _probe_angle_scale(
            np.arange(best_angle - 0.5, best_angle + 0.51, 0.1),
            np.arange(best_scale - 0.005, best_scale + 0.0051, 0.001),
        )

        # Coarse translation: 5-pixel steps in [-50, 50]
        for tx in np.arange(-50, 51, 5):
            for ty in np.arange(-50, 51, 5):
                val = _eval_similarity(best_angle, best_scale, tx, ty)
                if val < best_metric:
                    best_metric = val
                    best_tx = float(tx)
                    best_ty = float(ty)

        # Fine translation: 1-pixel steps around coarse optimum
        for tx in np.arange(best_tx - 5, best_tx + 5.01, 1):
            for ty in np.arange(best_ty - 5, best_ty + 5.01, 1):
                val = _eval_similarity(best_angle, best_scale, tx, ty)
                if val < best_metric:
                    best_metric = val
                    best_tx = float(tx)
                    best_ty = float(ty)

        # Joint ultra-fine: 0.1° angle, 0.001 scale, 0.5px translation
        for scale_val in np.arange(best_scale - 0.003, best_scale + 0.0031, 0.001):
            for angle_deg in np.arange(best_angle - 0.3, best_angle + 0.31, 0.1):
                for tx in np.arange(best_tx - 1.5, best_tx + 1.51, 0.5):
                    for ty in np.arange(best_ty - 1.5, best_ty + 1.51, 0.5):
                        val = _eval_similarity(angle_deg, scale_val, tx, ty)
                        if val < best_metric:
                            best_metric = val
                            best_angle = float(angle_deg)
                            best_scale = float(scale_val)
                            best_tx = float(tx)
                            best_ty = float(ty)

        # ----------------------------------------------------------
        # Stage 1b: Nelder-Mead similarity refinement.
        # The 1+1-ES cannot refine translation (perturbation ~0.001px
        # with InitialRadius=0.001786), leaving the result at the grid
        # search's 0.5px precision.  Nelder-Mead uses per-dimension
        # step sizes via its simplex, achieving sub-pixel refinement
        # in all 4 parameters simultaneously.
        # ----------------------------------------------------------
        _sim_eval = sitk.ImageRegistrationMethod()
        _sim_eval.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        _sim_eval.SetMetricSamplingStrategy(_sim_eval.NONE)
        _sim_eval.SetInterpolator(sitk.sitkLinear)
        _sim_penalty = 0.0

        def _similarity_cost(params):
            nonlocal _sim_penalty
            probe = sitk.Similarity2DTransform()
            probe.SetAngle(float(params[0]))
            probe.SetScale(float(params[1]))
            probe.SetCenter(center)
            probe.SetTranslation([float(params[2]), float(params[3])])
            _sim_eval.SetInitialTransform(probe)
            try:
                return _sim_eval.MetricEvaluate(fixed_sitk, moving_sitk)
            except RuntimeError:
                _sim_penalty += 1.0
                return _sim_penalty

        x0_sim = np.array([
            np.radians(best_angle), best_scale, best_tx, best_ty
        ], dtype=np.float64)
        sim_simplex = np.vstack([
            x0_sim,
            x0_sim + [0.005, 0, 0, 0],
            x0_sim + [0, 0.002, 0, 0],
            x0_sim + [0, 0, 2.0, 0],
            x0_sim + [0, 0, 0, 2.0],
        ])
        sim_opt = _scipy_minimize(
            _similarity_cost, x0_sim, method="Nelder-Mead",
            options={
                "maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12,
                "adaptive": True, "initial_simplex": sim_simplex,
            },
        )

        opt_angle = float(sim_opt.x[0])
        opt_scale = float(sim_opt.x[1])
        opt_tx = float(sim_opt.x[2])
        opt_ty = float(sim_opt.x[3])

        # ----------------------------------------------------------
        # Stage 2: Nelder-Mead affine refinement.
        # Convert similarity → zero-centered affine, then let
        # Nelder-Mead discover shear and anisotropic scaling that
        # the similarity model cannot represent.
        # ----------------------------------------------------------
        cos_t, sin_t = float(np.cos(opt_angle)), float(np.sin(opt_angle))
        sim_matrix = [
            opt_scale * cos_t, -opt_scale * sin_t,
            opt_scale * sin_t,  opt_scale * cos_t,
        ]
        A_mat = np.asarray(sim_matrix, dtype=np.float64).reshape(2, 2)
        sim_center = np.asarray(center, dtype=np.float64)
        sim_trans = np.array([opt_tx, opt_ty], dtype=np.float64)
        zero_center_trans = (np.eye(2) - A_mat) @ sim_center + sim_trans

        _aff_eval = sitk.ImageRegistrationMethod()
        _aff_eval.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        _aff_eval.SetMetricSamplingStrategy(_aff_eval.NONE)
        _aff_eval.SetInterpolator(sitk.sitkLinear)
        _aff_penalty = 0.0

        def _affine_cost(params):
            nonlocal _aff_penalty
            aff = sitk.AffineTransform(2)
            aff.SetMatrix(params[:4].tolist())
            aff.SetCenter([0.0, 0.0])
            aff.SetTranslation(params[4:].tolist())
            _aff_eval.SetInitialTransform(aff)
            try:
                return _aff_eval.MetricEvaluate(fixed_sitk, moving_sitk)
            except RuntimeError:
                _aff_penalty += 1.0
                return _aff_penalty

        x0_aff = np.array(
            sim_matrix + [float(zero_center_trans[0]), float(zero_center_trans[1])],
            dtype=np.float64,
        )
        aff_simplex = np.vstack([
            x0_aff,
            x0_aff + [0.01, 0, 0, 0, 0, 0],
            x0_aff + [0, 0.01, 0, 0, 0, 0],
            x0_aff + [0, 0, 0.01, 0, 0, 0],
            x0_aff + [0, 0, 0, 0.01, 0, 0],
            x0_aff + [0, 0, 0, 0, 2.0, 0],
            x0_aff + [0, 0, 0, 0, 0, 2.0],
        ])
        aff_opt = _scipy_minimize(
            _affine_cost, x0_aff, method="Nelder-Mead",
            options={
                "maxiter": 10000, "xatol": 1e-10, "fatol": 1e-12,
                "adaptive": True, "initial_simplex": aff_simplex,
            },
        )

        final_affine = sitk.AffineTransform(2)
        final_affine.SetMatrix(aff_opt.x[:4].tolist())
        final_affine.SetCenter([0.0, 0.0])
        final_affine.SetTranslation(aff_opt.x[4:].tolist())
        final_transform = final_affine

        # ----------------------------------------------------------
        # Warp RGB using the final affine transform.
        # MATLAB uses FillValues=255 on im2double ([0,1]) images, which
        # causes all boundary-blended pixels to clip to 1.0 (white).
        # ----------------------------------------------------------
        rgb_for_warp = resize_like(he_img, fixed_shg.shape[:2])
        registered_channels = []
        for c in range(3):
            ch_sitk = sitk.GetImageFromArray(rgb_for_warp[:, :, c].astype(np.float64))
            ch_sitk = sitk.Cast(ch_sitk, sitk.sitkFloat64)
            warped = sitk.Resample(
                ch_sitk,
                fixed_sitk,
                final_transform,
                sitk.sitkLinear,
                255.0,
            )
            registered_channels.append(sitk.GetArrayFromImage(warped))
        registered = np.stack(registered_channels, axis=-1)
        backend = "simpleitk_mattes"
    else:
        shift, _, _ = registration.phase_cross_correlation(fixed, he_moving)
        from skimage.transform import AffineTransform, warp

        tform_fallback = AffineTransform(translation=(-shift[1], -shift[0]))
        rgb_for_warp = resize_like(he_img, fixed_shg.shape[:2])
        registered = warp(
            rgb_for_warp,
            tform_fallback.inverse,
            output_shape=fixed_shg.shape[:2],
            cval=1.0,
            channel_axis=-1,
        )
        backend = "skimage_ecc_fallback"

    registered_img = resize_like(registered, original_shg_shape)
    registered_img = np.clip(registered_img, 0.0, 1.0)
    return registered_img, backend


def shg_he_registration(
    params: SHGHERegistrationParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
    include_debug_images: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Register H&E to SHG (Python port of MATLAB ``BDcreation_reg2.m``).

    Writes ``HE_registered/<HEfilename>`` under ``HEfilepath`` when
    ``save_output`` is True, and prints the absolute output path to stdout.
    """
    del include_debug_images  # reserved for API compatibility; unused here
    p = _to_params(params)
    registered_img, backend = _shg_he_registration_core(
        p.HEfilepath,
        p.HEfilename,
        p.SHGfilepath,
        p.pixelpermicron,
    )

    if save_output:
        save_path = os.path.join(p.HEfilepath, "HE_registered")
        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, p.HEfilename)
        registered_uint8 = (np.clip(registered_img, 0, 1) * 255).astype(np.uint8)
        io.imsave(output_path, registered_uint8, check_contrast=False)
        print(f"Registered image {p.HEfilename} was saved at {save_path}")

    if not return_debug:
        return registered_img

    debug: dict[str, Any] = {"registration_backend": backend}
    return registered_img, debug


def BDcreation_reg2(
    BDCparameters: SHGHERegistrationParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name."""
    out = shg_he_registration(
        BDCparameters, save_output=True, return_debug=False
    )
    assert isinstance(out, np.ndarray)
    return out
