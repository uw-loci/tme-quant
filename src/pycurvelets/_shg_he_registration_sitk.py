"""
MATLAB-style multimodal registration using SimpleITK (Mattes MI + multi-resolution).

Mirrors ``BDcreation_reg2.m``: similarity init, then affine refinement with the same
moving/fixed images (collagen mask vs SHG). Intended to sit much closer to MATLAB's
``imregconfig('multimodal')`` / ``imregtform`` than phase correlation + ECC.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import SimpleITK as sitk  # type: ignore[import-untyped]

    # Single-thread registration so repeated calls with identical inputs match (regression tests).
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False
    sitk = None  # type: ignore[assignment]

# MATLAB ``imregconfig('multimodal')`` + ``BDcreation_reg2.m`` adjustments:
# MattesMutualInformation: NumberOfHistogramBins=50, UseAllPixels=1
# Optimizer: OnePlusOneEvolutionary (not RegularStepGradientDescent) with
# InitialRadius = default/3.5, GrowthFactor=1.05, Epsilon=1.5e-6, MaximumIterations=700.
SITK_HISTOGRAM_BINS = 50
# MATLAB sets optimizer.MaximumIterations = 700 before similarity + affine ``imregtform``.
SITK_SIMILARITY_ITERATIONS = 700
SITK_AFFINE_ITERATIONS = 700
# Default multimodal InitialRadius is 6.25e-3; BDcreation_reg2 divides by 3.5.
SITK_INITIAL_RADIUS = 6.25e-3 / 3.5
SITK_GROWTH_FACTOR = 1.05
SITK_EPSILON = 1.5e-6
# Fixed seed so regression tests and repeated runs are deterministic.
SITK_ONEPLUSONE_SEED = 42


def has_simpleitk() -> bool:
    return _HAS_SITK


def register_collagen_to_shg_sitk(
    moving: np.ndarray,
    fixed: np.ndarray,
) -> sitk.Transform:
    """
    Estimate transform mapping **fixed** image indices to **moving** image sampling
    (SimpleITK ``Resample`` convention used with ``Execute`` output).

    Parameters
    ----------
    moving, fixed
        2-D arrays, same shape (e.g. collagen mask and SHG at registration resolution).
    """
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is not installed.")

    moving = np.asarray(moving, dtype=np.float32)
    fixed = np.asarray(fixed, dtype=np.float32)
    if moving.shape != fixed.shape:
        raise ValueError(f"Shape mismatch: moving {moving.shape} vs fixed {fixed.shape}")

    fixed_img = sitk.GetImageFromArray(fixed)
    moving_img = sitk.GetImageFromArray(moving)
    for im in (fixed_img, moving_img):
        im.SetSpacing((1.0, 1.0))
        im.SetOrigin((0.0, 0.0))

    def _set_one_plus_one_evolutionary(reg: sitk.ImageRegistrationMethod, n_iter: int) -> None:
        """Match MATLAB ``imregconfig('multimodal')`` OnePlusOneEvolutionary defaults."""
        reg.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=n_iter,
            epsilon=SITK_EPSILON,
            initialRadius=SITK_INITIAL_RADIUS,
            growthFactor=SITK_GROWTH_FACTOR,
            shrinkFactor=-1.0,
            seed=SITK_ONEPLUSONE_SEED,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

    # --- Stage 1: similarity (MATLAB ``imregtform(..., 'similarity', ...)``) ---
    R1 = sitk.ImageRegistrationMethod()
    R1.SetMetricAsMattesMutualInformation(numberOfHistogramBins=SITK_HISTOGRAM_BINS)
    R1.SetMetricSamplingStrategy(R1.NONE)
    R1.SetMetricSamplingPercentage(1.0)
    R1.SetInterpolator(sitk.sitkLinear)
    _set_one_plus_one_evolutionary(R1, SITK_SIMILARITY_ITERATIONS)

    init_sim = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R1.SetInitialTransform(init_sim)
    R1.SetShrinkFactorsPerLevel([4, 2, 1])
    R1.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R1.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    tx_similarity = R1.Execute(fixed_img, moving_img)

    # --- Stage 2: affine with initial similarity (MATLAB ``imreg_new3`` affine + InitialTransformation) ---
    R2 = sitk.ImageRegistrationMethod()
    R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=SITK_HISTOGRAM_BINS)
    R2.SetMetricSamplingStrategy(R2.NONE)
    R2.SetMetricSamplingPercentage(1.0)
    R2.SetInterpolator(sitk.sitkLinear)
    _set_one_plus_one_evolutionary(R2, SITK_AFFINE_ITERATIONS)
    R2.SetMovingInitialTransform(tx_similarity)
    affine = sitk.AffineTransform(2)
    R2.SetInitialTransform(affine, inPlace=False)
    R2.SetShrinkFactorsPerLevel([2, 1])
    R2.SetSmoothingSigmasPerLevel([1.0, 0.0])
    R2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    return R2.Execute(fixed_img, moving_img)


def warp_rgb_with_sitk_transform(
    rgb: np.ndarray,
    fixed_sitk_template: sitk.Image,
    transform: sitk.Transform,
    fill_value: float,
) -> np.ndarray:
    """Apply ``transform`` to each RGB channel; output matches ``fixed_sitk_template`` geometry."""
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is not installed.")

    h, w, nc = rgb.shape
    if nc != 3:
        raise ValueError(f"Expected RGB with 3 channels, got {nc}.")

    out = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        moving_ch = sitk.GetImageFromArray(rgb[:, :, c].astype(np.float32))
        moving_ch.SetSpacing((1.0, 1.0))
        moving_ch.SetOrigin((0.0, 0.0))
        # Interpolator must be the enum (e.g. ``sitk.sitkLinear``), not ``sitkLinear()`` — the latter raises
        # ``TypeError: 'int' object is not callable`` because ``sitkLinear`` is an integer constant.
        resampled = sitk.Resample(
            moving_ch,
            fixed_sitk_template,
            transform,
            sitk.sitkLinear,
            float(fill_value),
            sitk.sitkFloat32,
        )
        out[:, :, c] = sitk.GetArrayFromImage(resampled).astype(np.float64)
    return out


def build_fixed_sitk_image(fixed_2d: np.ndarray) -> sitk.Image:
    """Build reference ``SimpleITK`` image for registration and resampling (spacing 1, origin 0)."""
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is not installed.")
    fixed_img = sitk.GetImageFromArray(np.asarray(fixed_2d, dtype=np.float32))
    fixed_img.SetSpacing((1.0, 1.0))
    fixed_img.SetOrigin((0.0, 0.0))
    return fixed_img


def warn_sitk_fallback(exc: BaseException) -> None:
    warnings.warn(
        f"SimpleITK multimodal registration failed ({exc!r}); "
        "falling back to phase-correlation + ECC (larger deviation from MATLAB).",
        RuntimeWarning,
        stacklevel=3,
    )
