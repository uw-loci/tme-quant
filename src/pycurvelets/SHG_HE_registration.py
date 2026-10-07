"""H&E <-> SHG registration - Python port of MATLAB ``BDcreation_reg2.m`` and
``BDcreation_reg.m``.

Two MATLAB pipelines are available through ``pipeline``:

* ``"reg2"`` (default) - ``BDcreation_reg2.m``: SHG capped at 2 px/um. Default
  ``ecm_method="hsv"`` uses the HSV collagen mask; ``ecm_method="rgb"`` uses
  the BDcreation_reg RGB cuts (sometimes better isolation on a given slide).
* ``"reg1"`` - ``BDcreation_reg.m``: SHG ``imadjust``-ed at native resolution,
  ``decorrstretch`` + fixed RGB cuts + CIELAB k-means for the eosin/collagen
  moving image (continuous grey levels), output warped on the SHG grid. Ported
  exactly in :mod:`pycurvelets._he_bdc_reg1`, including a bit-exact replay of
  MATLAB's ``kmeans`` (which ``BDcreation_reg.m`` leaves *unseeded*, so MATLAB
  itself is not deterministic here; ``kmeans_seed`` selects the k-means optimum,
  and the default reproduces the reference goldens).

Registration is the ITK v3 (1+1)-ES port of MATLAB ``imregtform``:

1. Build the collagen moving image from the H&E exactly as ``BDcreation_reg2``
   does (``imresize`` / ``imadjust`` / ``rgb2hsv`` / ``graythresh`` /
   ``bwareaopen`` / ``strel`` / ``imfilter`` / ``imfill`` ports).
2. Register with a bit-for-bit port of MATLAB ``imregtform``: ITK v3
   multiresolution Mattes MI + (1+1) evolutionary optimizer with MATLAB's
   scales, centre, seed (12345) and per-level radius/epsilon refiner, run
   through the ``itk`` package (no MATLAB involved). Stage 1 similarity, stage
   2 affine initialised from it. See :mod:`pycurvelets._itk_v3_matlab_engine`.
3. Warp the raw HE RGB onto the SHG grid via :func:`matlab_imwarp_bilinear`,
   which matches MATLAB ``imref2d`` + ``imwarp`` conventions (pixel-centre
   sampling, pixel-centre inside test, no fill blending, ``FillValues=255``).

This reproduces the ``BDcreation_reg2`` golden TIFFs pixel-for-pixel on all
seven reference cases (``tests/test_shg_he_registration_matlab_parity.py``,
dev-only). It faithfully reproduces MATLAB's *result*, including cases where
MATLAB itself lands in a poor local optimum (patient_02 test5 is ~69 px from
ground truth in both). Requires the ``itk`` package.

The pipeline requires no MATLAB licence at runtime.

Public entry points: :func:`shg_he_registration`, :func:`BDcreation_reg2` and
:func:`BDcreation_reg` (MATLAB-compatible names),
:class:`SHGHERegistrationParameters`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import io, morphology

from ._he_bdc_common import (
    adjust_rgb_mean_std,
    decorrelation_stretch,
    disk_se,
    gaussian_filter_matlab_like,
    make_collagen_mask,
    make_ecm_mask_rgb,
    make_nuclei_mask,
    make_nuclei_mask_rgb,
    matlab_imwarp_bilinear,
    matlab_rgb2gray,
    prepare_registration_pair,
    remove_small_components,
    resize_like,
)
from ._he_bdc_reg1 import DEFAULT_KMEANS_SEED, bdcreation_reg1_preprocess
from ._matlab_imresize import matlab_imresize
from ._registration_quality import compute_shg_alignment_metrics


def _require_matlab_method(method: str | None) -> str:
    resolved = (method or "matlab").lower()
    if resolved != "matlab":
        raise ValueError(
            f"registration_method={method!r} is not supported; only the ITK v3 "
            "(1+1)-ES port ('matlab') remains."
        )
    return resolved


def _require_ecm_method(ecm_method: str | None) -> str:
    resolved = (ecm_method or "hsv").lower()
    if resolved not in ("hsv", "rgb"):
        raise ValueError(
            f"ecm_method={ecm_method!r} is not supported; use 'hsv' "
            "(BDcreation_reg2) or 'rgb' (BDcreation_reg RGB cuts)."
        )
    return resolved


# Kept so existing test imports still resolve.
_require_hsv_ecm = _require_ecm_method


@dataclass
class SHGHERegistrationParameters:
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
    SHGfilepath: str
    areaThreshold: float | None = None
    # Only the ITK v3 (1+1)-ES port of MATLAB imregtform is implemented.
    # The field is kept so existing call sites that pass "matlab" still work.
    registration_method: str = "matlab"
    # reg2 only: "hsv" (BDcreation_reg2, default) or "rgb" (BDcreation_reg
    # RGB cuts on decorrstretched HE). RGB sometimes isolates collagen
    # better; the registrar is still ITK v3. Ignored when pipeline="reg1".
    ecm_method: str = "hsv"
    random_state: int = 0
    # "reg2" (default): BDcreation_reg2.m.
    # "reg1"          : BDcreation_reg.m (decorrstretch + RGB cuts + LAB
    #                   k-means; ecm_method is ignored).
    pipeline: str = "reg2"
    # reg1 only: seed for the exact replay of MATLAB's unseeded kmeans.
    kmeans_seed: int = DEFAULT_KMEANS_SEED


def _to_params(
    params: SHGHERegistrationParameters | dict[str, Any],
) -> SHGHERegistrationParameters:
    if isinstance(params, SHGHERegistrationParameters):
        return params
    return SHGHERegistrationParameters(**params)


def _affine_fixed_to_moving_from_forward(
    forward_2x3: np.ndarray,
) -> np.ndarray:
    """
    Invert a 2x3 forward affine ``p_out = M @ p_in + t`` into the 3x3
    fixed->moving matrix accepted by the warp helper.
    """
    M = np.asarray(forward_2x3[:, :2], dtype=np.float64)
    t = np.asarray(forward_2x3[:, 2], dtype=np.float64)
    M_inv = np.linalg.inv(M)
    t_inv = -M_inv @ t
    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = M_inv
    A[:2, 2] = t_inv
    return A


def _refine_nuclei_filled(
    masked_nuclei_image: np.ndarray,
    pixpermic: float,
) -> np.ndarray:
    """Shared nuclei cleanup used before collagen/ECM isolation."""
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
    return binary_fill_holes(bw_nuclei_dilated)


def _build_he_moving_hsv(
    he_adjusted: np.ndarray,
    pixpermic: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """BDcreation_reg2 HSV collagen mask (binary moving image)."""
    extras: dict[str, Any] = {}
    _MIN_MASK_COVERAGE = 0.02

    _bw_nuclei_opened, masked_nuclei_image = make_nuclei_mask(he_adjusted, pixpermic)
    bw_collagen, _bw_no_background, _sat_thresh = make_collagen_mask(
        he_adjusted, pixpermic, enhanced_postprocessing=False
    )
    bw_nuclei_filled = _refine_nuclei_filled(masked_nuclei_image, pixpermic)
    he_collagen_bw = bw_collagen & (~bw_nuclei_filled)
    he_collagen_bw = remove_small_components(
        he_collagen_bw, int(np.ceil(pixpermic**2))
    )
    he_moving = he_collagen_bw.astype(np.float64)
    mask_coverage = float(he_moving.mean())
    extras["mask_coverage"] = mask_coverage
    mode = "hsv"
    if mask_coverage < _MIN_MASK_COVERAGE:
        he_gray_inv = 1.0 - matlab_rgb2gray(he_adjusted)
        tissue_mask = (~bw_nuclei_filled).astype(np.float64)
        he_moving_fb = he_gray_inv * tissue_mask
        fb_coverage = float((he_moving_fb > 0.01).mean())
        if fb_coverage > mask_coverage:
            he_moving = he_moving_fb
            mode = "hsv->gray_fallback"
            extras["mask_coverage"] = fb_coverage
    return he_moving, mode, extras


def _build_he_moving_rgb(
    he_adjusted: np.ndarray,
    pixpermic: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """BDcreation_reg RGB cuts on decorrstretched HE (continuous moving image)."""
    he_decorr = decorrelation_stretch(he_adjusted, tol=0.01)
    _bw_nuclei_opened, masked_nuclei_image = make_nuclei_mask_rgb(he_decorr, pixpermic)
    he_collagen_gray, _bw_collagen = make_ecm_mask_rgb(he_decorr, pixpermic)
    bw_nuclei_filled = _refine_nuclei_filled(masked_nuclei_image, pixpermic)
    he_collagen_exclude = he_collagen_gray * (~bw_nuclei_filled).astype(np.float64)
    he_collagen_bw = he_collagen_exclude > 0.01
    he_collagen_bw = remove_small_components(
        he_collagen_bw, int(np.ceil(pixpermic**2))
    )
    he_moving = he_collagen_exclude * he_collagen_bw.astype(np.float64)
    extras = {"mask_coverage": float((he_moving > 0.01).mean())}
    return he_moving, "rgb", extras


def _build_he_moving(
    he_scaled: np.ndarray,
    he_adjusted: np.ndarray,
    he_decorr: np.ndarray,
    pixpermic: float,
    ecm_mode: str,
    random_state: int = 0,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Dispatch HSV / RGB moving-image builders. Extra args kept for callers."""
    del he_scaled, he_decorr, random_state
    mode = _require_ecm_method(ecm_mode)
    if mode == "rgb":
        return _build_he_moving_rgb(he_adjusted, pixpermic)
    return _build_he_moving_hsv(he_adjusted, pixpermic)


def _register_matlab_itk_v3(
    he_moving: np.ndarray,
    fixed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    MATLAB-parity registration: ITK v3 framework configured exactly like
    ``imregtform`` (see :mod:`pycurvelets._itk_v3_matlab_engine`).
    """
    from ._itk_v3_matlab_engine import (
        DEFAULT_SEED,
        has_itk,
        register_bdcreation_reg2_matlab,
    )

    if not has_itk():
        raise RuntimeError(
            "SHG–HE registration requires the 'itk' package "
            "(pip install itk; it is a declared dependency of tme-quant)."
        )
    if not np.any(he_moving > 0):
        forward_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return forward_2x3, {"matlab_degenerate_moving": True}

    forward_2x3, ml_debug = register_bdcreation_reg2_matlab(
        he_moving.astype(np.float64), fixed.astype(np.float64), seed=DEFAULT_SEED
    )
    return forward_2x3, ml_debug


def _score_forward_vs_shg(
    he_moving: np.ndarray,
    fixed: np.ndarray,
    forward_2x3: np.ndarray,
) -> dict[str, Any]:
    """Warp moving image with ``forward_2x3`` and score against SHG."""
    A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)
    warped = matlab_imwarp_bilinear(
        he_moving.astype(np.float64),
        fixed.shape[:2],
        A_inv,
        fill_value=0.0,
    )
    return compute_shg_alignment_metrics(
        warped, fixed, forward_2x3=forward_2x3
    )


def _reg1_core(
    he_path: str,
    shg_path: str,
    pixelpermicron: float,
    kmeans_seed: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """
    ``BDcreation_reg.m`` (reg1). Preprocessing in :func:`bdcreation_reg1_preprocess`;
    registration is the ITK v3 port, fed ``double(fixedSHG)`` in 0..255 exactly
    as ``imregtform`` is.
    """
    he_u8 = io.imread(he_path)
    shg_raw = io.imread(shg_path)
    if he_u8.dtype != np.uint8:
        raise TypeError(f"reg1 expects a uint8 H&E TIFF, got {he_u8.dtype} ({he_path})")

    pre = bdcreation_reg1_preprocess(he_u8, shg_raw, float(pixelpermicron), kmeans_seed=kmeans_seed)
    he_moving = pre["HEmoving"]
    fixed_double = pre["fixedSHG_double"]
    fixed_shape = tuple(int(x) for x in fixed_double.shape)

    debug: dict[str, Any] = {
        "pipeline": "reg1",
        "registration_method_requested": "matlab",
        "kmeans_seed": int(kmeans_seed),
        "kmeans": pre["kmeans_debug"],
        "collagen_cluster": pre["collagen_cluster"],
        "mask_coverage": pre["mask_coverage"],
        "pixpermic_working": float(pixelpermicron),
        "fixed_shape": fixed_shape,
        "fixed_dtype": str(pre["fixedSHG"].dtype),
    }
    forward_2x3, opt_debug = _register_matlab_itk_v3(he_moving, fixed_double)
    debug.update(opt_debug)
    debug["forward_2x3"] = forward_2x3.tolist()
    backend = "itk_v3_matlab_parity"

    fixed_unit = fixed_double / float(fixed_double.max() or 1.0)
    debug["shg_alignment"] = _score_forward_vs_shg(he_moving, fixed_unit, forward_2x3)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    debug["shg_alignment_identity_mi"] = float(_score_forward_vs_shg(he_moving, fixed_unit, identity)["shg_mi"])

    A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)
    he_data_registered = matlab_imresize(pre["HEdata"], output_shape=fixed_shape, method="bicubic")
    registered = matlab_imwarp_bilinear(he_data_registered.astype(np.float64), fixed_shape, A_inv, fill_value=255.0)
    registered_img = np.clip(registered, 0.0, 1.0)

    reg_luma = matlab_rgb2gray(registered_img)
    debug["shg_alignment_fullres"] = compute_shg_alignment_metrics(reg_luma, fixed_unit, forward_2x3=None)
    return registered_img, backend, debug


def _shg_he_registration_core(
    he_filepath: str,
    he_filename: str,
    shg_filepath: str,
    pixelpermicron: float,
    registration_method: str = "matlab",
    ecm_method: str = "hsv",
    pipeline: str = "reg2",
    kmeans_seed: int = DEFAULT_KMEANS_SEED,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """
    Core registration (same algorithm description as the module docstring).

    Returns
    -------
    registered_img
        Float RGB in ``[0, 1]``, shape matching original SHG (H, W, 3).
    backend
        Optimizer backend label.
    debug
        Intermediate registration parameters (transform matrices, SHG scores).
    """
    he_path = os.path.join(he_filepath, he_filename)
    shg_path = os.path.join(shg_filepath, he_filename)

    _require_matlab_method(registration_method)
    pipe = (pipeline or "reg2").lower()
    if pipe == "reg1":
        return _reg1_core(he_path, shg_path, pixelpermicron, kmeans_seed)
    if pipe != "reg2":
        raise ValueError(f"pipeline must be 'reg2' or 'reg1', got {pipeline!r}")
    ecm_requested = _require_ecm_method(ecm_method)

    he_img = io.imread(he_path).astype(np.float64) / 255.0
    shg_img = io.imread(shg_path).astype(np.float64) / 255.0
    if shg_img.ndim == 3:
        shg_img = matlab_rgb2gray(shg_img)

    original_shg_shape = shg_img.shape[:2]
    he_scaled, fixed_shg, pixpermic = prepare_registration_pair(
        he_img, shg_img, float(pixelpermicron)
    )

    he_adjusted = adjust_rgb_mean_std(he_scaled)

    fixed = fixed_shg.astype(np.float64)
    if fixed.ndim == 3:
        fixed = matlab_rgb2gray(fixed)

    debug: dict[str, Any] = {
        "registration_method_requested": "matlab",
        "ecm_method_requested": ecm_requested,
        "pixpermic_working": float(pixpermic),
        "fixed_shape": tuple(int(x) for x in fixed.shape),
    }

    if ecm_requested == "rgb":
        he_moving, ecm_mode, extras = _build_he_moving_rgb(he_adjusted, pixpermic)
    else:
        he_moving, ecm_mode, extras = _build_he_moving_hsv(he_adjusted, pixpermic)
    debug.update(extras)
    debug["ecm_method_selected"] = ecm_mode
    forward_2x3, opt_debug = _register_matlab_itk_v3(he_moving, fixed)
    debug.update(opt_debug)
    backend = "itk_v3_matlab_parity"

    debug["forward_2x3"] = forward_2x3.tolist()

    shg_metrics = _score_forward_vs_shg(he_moving, fixed, forward_2x3)
    debug["shg_alignment"] = shg_metrics
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    debug["shg_alignment_identity_mi"] = float(
        _score_forward_vs_shg(he_moving, fixed, identity)["shg_mi"]
    )

    A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)

    rgb_for_warp = resize_like(he_img, fixed.shape[:2])
    # MATLAB: imwarp(RGB, ..., 'FillValues', 255) on a *double* image, then
    # imresize(B, size(SHG)) and imwrite (clip to [0, 1]).
    registered = matlab_imwarp_bilinear(
        rgb_for_warp.astype(np.float64),
        fixed.shape[:2],
        A_inv,
        fill_value=255.0,
    )

    registered_img = resize_like(registered, original_shg_shape)
    registered_img = np.clip(registered_img, 0.0, 1.0)

    reg_luma = matlab_rgb2gray(registered_img)
    shg_full = shg_img.astype(np.float64)
    if shg_full.ndim == 3:
        shg_full = matlab_rgb2gray(shg_full)
    debug["shg_alignment_fullres"] = compute_shg_alignment_metrics(
        reg_luma, shg_full, forward_2x3=None
    )
    return registered_img, backend, debug


def shg_he_registration(
    params: SHGHERegistrationParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
    include_debug_images: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Register H&E to SHG (Python port of MATLAB ``BDcreation_reg2.m`` /
    ``BDcreation_reg.m``, selected by ``params.pipeline``).

    Writes ``HE_registered/<HEfilename>`` under ``HEfilepath`` when
    ``save_output`` is True, and prints the absolute output path to stdout.
    """
    del include_debug_images  # reserved for API compatibility; unused here
    p = _to_params(params)
    registered_img, backend, debug = _shg_he_registration_core(
        p.HEfilepath,
        p.HEfilename,
        p.SHGfilepath,
        p.pixelpermicron,
        registration_method=p.registration_method,
        ecm_method=p.ecm_method,
        pipeline=p.pipeline,
        kmeans_seed=p.kmeans_seed,
    )

    if save_output:
        save_path = os.path.join(p.HEfilepath, "HE_registered")
        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, p.HEfilename)
        # MATLAB imwrite(double) == im2uint8: round(x*255), not truncate.
        registered_uint8 = np.round(np.clip(registered_img, 0, 1) * 255).astype(np.uint8)
        io.imsave(output_path, registered_uint8, check_contrast=False)
        print(f"Registered image {p.HEfilename} was saved at {save_path}")

    if not return_debug:
        return registered_img

    debug_out: dict[str, Any] = {"registration_backend": backend}
    debug_out.update(debug)
    return registered_img, debug_out


def BDcreation_reg2(
    BDCparameters: SHGHERegistrationParameters | dict[str, Any],
) -> np.ndarray:
    """MATLAB-named wrapper: ``BDcreation_reg2.m`` (forces ``pipeline="reg2"``)."""
    p = replace(_to_params(BDCparameters), pipeline="reg2")
    out = shg_he_registration(p, save_output=True, return_debug=False)
    assert isinstance(out, np.ndarray)
    return out


def BDcreation_reg(
    BDCparameters: SHGHERegistrationParameters | dict[str, Any],
) -> np.ndarray:
    """MATLAB-named wrapper: ``BDcreation_reg.m`` (forces ``pipeline="reg1"``)."""
    p = replace(_to_params(BDCparameters), pipeline="reg1")
    out = shg_he_registration(p, save_output=True, return_debug=False)
    assert isinstance(out, np.ndarray)
    return out
