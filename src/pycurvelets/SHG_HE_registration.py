"""H&E ↔ SHG registration — Python port of MATLAB ``BDcreation_reg2.m``.

Algorithm: skimage preprocessing, SimpleITK Mattes MI with
OnePlusOneEvolutionary (similarity then affine refinement).  Optimizer
parameters match MATLAB's ``imregconfig('multimodal')`` defaults:
InitialRadius 6.25e-3 (divided by 3.5), GrowthFactor 1.05, Epsilon 1.5e-6,
MaxIterations 700, all-pixel sampling.

Public entry points: :func:`shg_he_registration`, :func:`BDcreation_reg2` (MATLAB
name), :class:`SHGHERegistrationParameters`, :func:`has_simpleitk`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import color, filters, io, morphology, registration, transform
from skimage.filters import threshold_otsu

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
        shg_img = color.rgb2gray(shg_img)

    original_shg_shape = shg_img.shape[:2]
    pixpermic = float(pixelpermicron)

    if pixpermic > 2:
        scale = 2.0 / pixpermic
        fixed_shg = transform.resize(
            shg_img,
            (int(shg_img.shape[0] * scale), int(shg_img.shape[1] * scale)),
            anti_aliasing=True,
        )
        pixpermic = 2.0
    else:
        fixed_shg = shg_img.copy()

    target_h, target_w = fixed_shg.shape[:2]
    rgb = transform.resize(he_img, (target_h, target_w, 3), anti_aliasing=True)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    stats: dict[str, tuple[float, float, float]] = {}
    for ch, name in [(r, "r"), (g, "g"), (b, "b")]:
        mu = float(ch.mean())
        sigma = float(ch.std())
        high_in = min(mu + 2.0 * sigma, 1.0)
        stats[name] = (mu, sigma, high_in)

    he_data = np.zeros_like(rgb)
    for i, name in enumerate(["r", "g", "b"]):
        _, _, high_in = stats[name]
        channel = rgb[:, :, i]
        stretched = np.clip(channel / high_in, 0.0, 1.0) if high_in > 0 else channel
        he_data[:, :, i] = stretched

    he_gray = color.rgb2gray(he_data)
    _ = threshold_otsu(he_gray)

    he_hsv = color.rgb2hsv(he_data)
    h_ch, s_ch, _v_ch = he_hsv[:, :, 0], he_hsv[:, :, 1], he_hsv[:, :, 2]

    nuclei_hue_mask = (h_ch >= 0.500) & (h_ch <= 0.790)
    sat_thresh_nuclei = threshold_otsu(s_ch)
    nuclei_sat_mask = s_ch >= sat_thresh_nuclei
    bw_nuclei_raw = nuclei_hue_mask & nuclei_sat_mask
    bw_nuclei_raw = morphology.remove_small_objects(bw_nuclei_raw, min_size=150)

    disk_radius_half = max(1, int(np.ceil(pixpermic / 2)))
    se_open = morphology.disk(disk_radius_half)
    bw_nuclei = morphology.opening(bw_nuclei_raw, se_open)

    masked_nuclei = he_data.copy()
    for c in range(3):
        masked_nuclei[:, :, c][~bw_nuclei] = 0.0

    collagen_hue_mask = (h_ch >= 0.837) | (h_ch <= 0.066)
    sat_thresh_collagen = threshold_otsu(s_ch)
    collagen_sat_mask = s_ch >= sat_thresh_collagen
    bw_collagen = collagen_hue_mask & collagen_sat_mask
    bw_collagen = morphology.remove_small_objects(bw_collagen, min_size=100)

    gray_nuclei = color.rgb2gray(masked_nuclei)
    kernel_size = max(1, int(np.floor(pixpermic)))
    nuclei_filtered = filters.gaussian(gray_nuclei, sigma=0.5, truncate=kernel_size)

    bw_nuclei2 = nuclei_filtered > 0.001
    min_nucleus_area = int(np.ceil(50 * pixpermic**2))
    bw_nuclei2 = morphology.remove_small_objects(bw_nuclei2, min_size=min_nucleus_area)
    se_dilate = morphology.disk(max(1, int(np.floor(pixpermic))))
    bw_nuclei_dilated = morphology.dilation(bw_nuclei2, se_dilate)
    bw_nuclei_filled = binary_fill_holes(bw_nuclei_dilated)

    he_collagen_bw = bw_collagen & (~bw_nuclei_filled)
    min_collagen_area = int(np.ceil(pixpermic**2))
    he_collagen_bw = morphology.remove_small_objects(
        he_collagen_bw, min_size=min_collagen_area
    )
    he_moving = he_collagen_bw.astype(np.float64)

    fixed = fixed_shg.astype(np.float64)
    if fixed.ndim == 3:
        fixed = color.rgb2gray(fixed)

    backend: str
    if _HAS_SITK:
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(he_moving)
        fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat64)
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat64)

        # MATLAB: [optimizer,metric] = imregconfig('multimodal');
        #   optimizer.InitialRadius  = 6.25e-3 (then /3.5)
        #   optimizer.GrowthFactor   = 1.05
        #   optimizer.Epsilon        = 1.5e-6
        #   optimizer.MaximumIterations = 700
        #   metric.UseAllPixels      = true
        #   imregtform default PyramidLevels = 3
        _INITIAL_RADIUS = 6.25e-3 / 3.5

        # Stage 1a: Exhaustive coarse 2D search over angle AND scale.
        # MATLAB's multi-resolution pyramid explores both rotation and
        # scaling at coarse levels.  SimpleITK's 1+1-ES with small
        # initialRadius cannot explore far enough on the sparse collagen
        # mask, so we grid-search (angle, scale) explicitly before
        # handing off to the optimizer.
        geom_init = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk,
            sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        center = list(geom_init.GetFixedParameters())

        best_angle = 0.0
        best_scale = 1.0
        best_metric = float("inf")
        eval_method = sitk.ImageRegistrationMethod()
        eval_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        eval_method.SetMetricSamplingStrategy(eval_method.NONE)
        eval_method.SetInterpolator(sitk.sitkLinear)

        # Coarse pass: 2° angle steps, 0.02 scale steps
        for scale_val in np.arange(0.80, 1.25, 0.02):
            for angle_deg in range(-45, 46, 2):
                probe = sitk.Similarity2DTransform()
                probe.SetAngle(float(np.radians(angle_deg)))
                probe.SetScale(float(scale_val))
                probe.SetCenter(center)
                eval_method.SetInitialTransform(probe)
                val = eval_method.MetricEvaluate(fixed_sitk, moving_sitk)
                if val < best_metric:
                    best_metric = val
                    best_angle = float(angle_deg)
                    best_scale = float(scale_val)

        # Fine pass: 0.5° angle, 0.005 scale around coarse optimum
        for scale_val in np.arange(best_scale - 0.04, best_scale + 0.05, 0.005):
            for angle_deg_f in np.arange(best_angle - 3, best_angle + 4, 0.5):
                probe = sitk.Similarity2DTransform()
                probe.SetAngle(float(np.radians(angle_deg_f)))
                probe.SetScale(float(scale_val))
                probe.SetCenter(center)
                eval_method.SetInitialTransform(probe)
                val = eval_method.MetricEvaluate(fixed_sitk, moving_sitk)
                if val < best_metric:
                    best_metric = val
                    best_angle = float(angle_deg_f)
                    best_scale = float(scale_val)

        # Stage 1b: Refine similarity from best coarse (angle, scale).
        sim_init = sitk.Similarity2DTransform()
        sim_init.SetAngle(float(np.radians(best_angle)))
        sim_init.SetScale(best_scale)
        sim_init.SetCenter(center)

        reg_method = sitk.ImageRegistrationMethod()
        reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg_method.SetMetricSamplingStrategy(reg_method.NONE)
        reg_method.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=700,
            epsilon=1.5e-6,
            initialRadius=_INITIAL_RADIUS,
            growthFactor=1.05,
        )
        reg_method.SetOptimizerScalesFromPhysicalShift()
        reg_method.SetInitialTransform(sim_init, inPlace=False)
        reg_method.SetInterpolator(sitk.sitkLinear)
        reg_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        similarity_transform = reg_method.Execute(fixed_sitk, moving_sitk)

        # Stage 2: Affine refinement from similarity result.
        sim_result = sitk.Similarity2DTransform(
            similarity_transform.GetNthTransform(0)
        )
        affine_init = sitk.AffineTransform(2)
        s = sim_result.GetScale()
        theta = sim_result.GetAngle()
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        affine_init.SetMatrix([
            s * cos_t, -s * sin_t,
            s * sin_t,  s * cos_t,
        ])
        affine_init.SetTranslation(list(sim_result.GetTranslation()))
        affine_init.SetCenter(list(sim_result.GetCenter()))

        reg_method2 = sitk.ImageRegistrationMethod()
        reg_method2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg_method2.SetMetricSamplingStrategy(reg_method2.NONE)
        reg_method2.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=700,
            epsilon=1.5e-6,
            initialRadius=_INITIAL_RADIUS,
            growthFactor=1.05,
        )
        reg_method2.SetOptimizerScalesFromPhysicalShift()
        reg_method2.SetInitialTransform(affine_init, inPlace=False)
        reg_method2.SetInterpolator(sitk.sitkLinear)
        reg_method2.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg_method2.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        reg_method2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        final_transform = reg_method2.Execute(fixed_sitk, moving_sitk)

        rgb_for_warp = transform.resize(
            he_img, (target_h, target_w, 3), anti_aliasing=True
        )
        registered_channels = []
        for c in range(3):
            ch_sitk = sitk.GetImageFromArray(rgb_for_warp[:, :, c].astype(np.float64))
            ch_sitk = sitk.Cast(ch_sitk, sitk.sitkFloat64)
            warped = sitk.Resample(
                ch_sitk,
                fixed_sitk,
                final_transform,
                sitk.sitkLinear,
                1.0,
            )
            registered_channels.append(sitk.GetArrayFromImage(warped))
        registered = np.stack(registered_channels, axis=-1)
        backend = "simpleitk_mattes"
    else:
        shift, _, _ = registration.phase_cross_correlation(fixed, he_moving)
        from skimage.transform import AffineTransform, warp

        tform_fallback = AffineTransform(translation=(-shift[1], -shift[0]))
        rgb_for_warp = transform.resize(
            he_img, (target_h, target_w, 3), anti_aliasing=True
        )
        registered = warp(
            rgb_for_warp,
            tform_fallback.inverse,
            output_shape=(target_h, target_w),
            cval=1.0,
            channel_axis=-1,
        )
        backend = "skimage_ecc_fallback"

    registered_img = transform.resize(
        registered,
        (original_shg_shape[0], original_shg_shape[1], 3),
        anti_aliasing=True,
    )
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
