"""H&E <-> SHG registration - Python port of MATLAB ``BDcreation_reg2.m``.

Default algorithm (``registration_method="mi_ncc"``):

1. Build an ECM moving image from the H&E (controlled by ``ecm_method``).
2. Use SimpleITK's Mattes MI with a deterministic grid search over
   angle/scale/translation, then Nelder-Mead similarity and affine
   refinement, followed by a bounded NCC trust-region sub-pixel polish.
3. Warp the raw HE RGB onto the SHG grid via :func:`matlab_imwarp_bilinear`,
   which matches MATLAB ``imref2d`` + ``imwarp`` conventions (pixel-centre
   sampling, pixel-centre inside test, no fill blending, ``FillValues=255``).

Other ``registration_method`` values:

* ``"matlab"`` - Bit-for-bit port of MATLAB ``imregtform`` as used by
  ``BDcreation_reg2.m``: ITK v3 multiresolution Mattes MI + (1+1)-ES with
  MATLAB's scales, centre, seed (12345) and per-level radius/epsilon refiner,
  driven through the ``itk`` package (no MATLAB involved). Together with the
  MATLAB-exact preprocessing in :mod:`pycurvelets._he_bdc_common`
  (``imresize``, ``graythresh``, ``imfilter``, ``rgb2gray``, ``imwarp``) it
  reproduces the ``BDcreation_reg2`` golden TIFFs pixel-for-pixel on all
  seven reference cases (``tests/test_shg_he_registration_matlab_parity.py``).
  See :mod:`pycurvelets._itk_v3_matlab_engine`. Requires ``itk``
  (``pip install "pycurvelets[matlab-parity]"``). Note this faithfully
  reproduces MATLAB's *result*, including cases where MATLAB itself lands in a
  poor local optimum (patient_02 test5 is ~69 px from ground truth in both).
* ``"oneplusone"`` - Multiresolution Mattes MI with a stochastic (1+1)
  evolutionary optimizer (paper/MATLAB-inspired). Available as an option.
* ``"mi"`` - Mattes MI alone (skip NCC polish). Slightly worse on average
  but useful when the polish would be untrusted.
* ``"ncc"`` (alias ``"dice"``) - Fully deterministic NCC-on-blurred-masks
  TRF pipeline. No SimpleITK dependency, but mask<->grayscale NCC is
  non-convex without MI-style histogram matching, so it tends to land in
  worse local minima. Used automatically if SimpleITK is missing.

ECM extraction is controlled separately by ``ecm_method``:

* ``"hsv"`` - default BDcreation_reg2.m HSV thresholding.
* ``"rgb"`` - BDcreation_reg.m-style decorrelation stretch + fixed RGB cuts.
* ``"lab"`` - BDcreation_reg.m-style RGB eosin gate + LAB k-means refinement.
* ``"gray"`` - inverted HE luma (smoother MI surface than binary collagen).
* ``"auto"`` - try hsv/rgb/lab/gray; pick the ECM with highest unregistered
  SHG MI, then run the full optimizer once on that moving image.

The pipeline requires no MATLAB licence at runtime.

Public entry points: :func:`shg_he_registration`, :func:`BDcreation_reg2`
(MATLAB-compatible name), :class:`SHGHERegistrationParameters`,
:func:`has_simpleitk`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.optimize import least_squares
from scipy.optimize import minimize as _scipy_minimize
from skimage import io, morphology, registration

from ._he_bdc_common import (
    adjust_rgb_mean_std,
    decorrelation_stretch,
    disk_se,
    gaussian_filter_matlab_like,
    make_ecm_mask_lab,
    make_ecm_mask_rgb,
    make_collagen_mask,
    make_nuclei_mask,
    make_nuclei_mask_rgb,
    matlab_imwarp_bilinear,
    matlab_rgb2gray,
    prepare_registration_pair,
    remove_small_components,
    resize_like,
)
from ._registration_quality import compute_shg_alignment_metrics

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
    # "mi_ncc" (default): SITK Mattes MI grid+Nelder-Mead basin finder +
    #                     bounded NCC TRF sub-pixel polish. Deterministic;
    #                     approximates MATLAB output.
    # "matlab"          : Exact port of MATLAB imregtform (ITK v3 engine via
    #                     the `itk` package): same metric, optimizer, scales,
    #                     seed and per-level refiner as BDcreation_reg2.m.
    #                     Reproduces the MATLAB golden TIFFs pixel-for-pixel
    #                     (tests/test_shg_he_registration_matlab_parity.py).
    #                     Ignores random_state (seed=12345). Needs `itk`.
    # "oneplusone"      : Multiresolution Mattes MI with a stochastic
    #                     (1+1)-evolutionary optimizer. Closest to the
    #                     paper/MATLAB ``imregtform('multimodal')`` workflow.
    #                     Available as an option per Yuming's request.
    # "mi"              : MI only (no polish). Slightly worse than ``mi_ncc``
    #                     but faster; kept for debugging.
    # "ncc"             : Deterministic NCC on blurred masks + bounded TRF.
    #                     No SimpleITK dependency. Less reliable because
    #                     mask<->grayscale NCC is non-convex without MI-style
    #                     histogram matching; used as a fallback when SITK
    #                     is unavailable.
    registration_method: str = "mi_ncc"
    # ECM extraction method:
    # "hsv" (default): BDcreation_reg2.m-style HSV thresholding.
    # "rgb"          : BDcreation_reg.m-style decorrelation stretch + fixed
    #                  RGB thresholds for nuclei/eosin.
    # "lab"          : BDcreation_reg.m-style RGB eosin gate then LAB k-means.
    # "gray"         : inverted HE luma * (~nuclei); continuous moving image.
    # "auto"         : try hsv/rgb/lab/gray; keep best SHG histogram-MI.
    ecm_method: str = "hsv"
    # Used by stochastic methods (e.g. oneplusone optimizer / LAB k-means).
    random_state: int = 0


def _to_params(
    params: SHGHERegistrationParameters | dict[str, Any],
) -> SHGHERegistrationParameters:
    if isinstance(params, SHGHERegistrationParameters):
        return params
    return SHGHERegistrationParameters(**params)


# ---------------------------------------------------------------------------
# Affine helpers (column-vector form: [x'; y'; 1] = A @ [x; y; 1]).
# A maps *fixed (output) intrinsic* -> *moving (input) intrinsic*, matching
# the convention accepted by :func:`matlab_imwarp_bilinear`.
# ---------------------------------------------------------------------------


def _similarity_fixed_to_moving(
    angle_rad: float,
    scale: float,
    tx: float,
    ty: float,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Build the inverse-similarity matrix: given a *forward* similarity
    (moving -> fixed) defined as "rotate+scale around ``center``, then
    translate by ``(tx, ty)``", return the matrix that maps fixed pixel
    centres back to moving pixel centres (for use with the warp helper).

    Debug aid: at ``angle_rad=0, scale=1, tx=ty=0`` this returns the identity,
    so a zero-parameter warp should be a no-op.
    """
    cx, cy = float(center[0]), float(center[1])
    s = float(scale)
    if s == 0:
        s = 1e-8
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    # Forward similarity F: p_out = R*s*(p_in - c) + c + t.
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    M_forward = s * R
    t_forward = (np.array([cx, cy]) + np.array([tx, ty]) - M_forward @ np.array([cx, cy]))
    # Inverse: p_in = M_forward^{-1} @ (p_out - t_forward).
    M_inv = np.linalg.inv(M_forward)
    t_inv = -M_inv @ t_forward
    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = M_inv
    A[:2, 2] = t_inv
    return A


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


def _forward_from_fixed_to_moving(A_inv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_affine_fixed_to_moving_from_forward`."""
    A_inv = np.asarray(A_inv, dtype=np.float64)
    if A_inv.shape == (3, 3):
        M = A_inv[:2, :2]
        t = A_inv[:2, 2]
    else:
        M = A_inv[:, :2]
        t = A_inv[:, 2]
    M_f = np.linalg.inv(M)
    t_f = -M_f @ t
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = M_f
    out[:, 2] = t_f
    return out


# ---------------------------------------------------------------------------
# Dice-based deterministic registration.
# ---------------------------------------------------------------------------


def _phase_correlation_translation(
    moving_float: np.ndarray,
    target_float: np.ndarray,
) -> tuple[float, float]:
    """
    Return ``(tx, ty)`` such that warping ``moving_float`` by that translation
    aligns it with ``target_float``. Wraps
    :func:`skimage.registration.phase_cross_correlation`; returns 0 on failure
    so the caller can keep iterating.
    """
    try:
        shift, _, _ = registration.phase_cross_correlation(
            target_float, moving_float, upsample_factor=1, normalization=None
        )
    except Exception:  # pragma: no cover - skimage versions / degenerate masks
        return 0.0, 0.0
    # skimage returns (dy, dx) such that moving + shift ~ target.
    ty = float(shift[0])
    tx = float(shift[1])
    return tx, ty


def _smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-blur a binary mask (or any float image) to create a smooth target
    with well-behaved gradients for the optimization residual.
    """
    arr = np.asarray(mask, dtype=np.float64)
    if sigma <= 0:
        return arr
    return gaussian_filter(arr, sigma=float(sigma), mode="constant", cval=0.0)


def _normalize_image(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Zero-mean, unit-std normalization used for NCC residuals.

    Returns an array of the same shape; if the input std is below ``eps`` the
    output is all zeros (avoids a degenerate ``inf`` residual).
    """
    arr = np.asarray(img, dtype=np.float64)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd < eps:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _ncc_residual(
    warped: np.ndarray, target_normalized: np.ndarray
) -> np.ndarray:
    """
    Per-pixel residual vector for NCC minimization.

    ``sum(resid**2)`` equals ``2*N*(1 - NCC(warped, target))`` up to the
    normalization constants, so minimizing ``sum(resid**2)`` is equivalent to
    maximizing NCC. The ``_normalize_image`` applied to ``warped`` here
    (inside the residual) handles the "push out of frame" degenerate case by
    mapping a constant-valued warp to all-zero residuals, which LM never
    prefers over a partial match (since 2*N*(1-NCC) > 0 still beats the
    ``mean(target_normalized^2)`` floor of a true zero image).
    """
    w_norm = _normalize_image(warped)
    return (w_norm - target_normalized).ravel()


def _ncc_score(warped: np.ndarray, target_normalized: np.ndarray) -> float:
    """NCC coefficient in ``[-1, 1]``; higher is better alignment."""
    w_norm = _normalize_image(warped)
    return float(np.mean(w_norm * target_normalized))


def _register_similarity_seed(
    moving_smooth: np.ndarray,
    target_smooth: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Grid search on NCC-equivalent SSD of zero-mean/unit-std images.

    For every ``(angle, scale)`` candidate: warp the moving mask by rotation +
    scale, estimate translation via phase correlation of the rotated mask
    against the target, then evaluate the NCC residual at the full
    similarity. Returns ``(angle_rad, scale, tx, ty, final_mean_sse_ncc)``.

    NCC (unlike plain SSE on masks) cannot be minimized by pushing the warped
    image out of frame: a constant warp has zero variance, so its normalized
    form is the zero vector, giving a fixed residual of ``mean(target^2)``
    that any partial alignment beats.
    """
    H, W = target_smooth.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    target_norm = _normalize_image(target_smooth)

    best_angle_deg = 0.0
    best_scale = 1.0
    best_tx = 0.0
    best_ty = 0.0
    best_sse = float("inf")

    def _eval(angle_deg: float, scale: float) -> tuple[float, float, float]:
        A0 = _similarity_fixed_to_moving(
            float(np.radians(angle_deg)), float(scale), 0.0, 0.0, (cx, cy)
        )
        rotated = matlab_imwarp_bilinear(moving_smooth, (H, W), A0, fill_value=0.0)
        tx, ty = _phase_correlation_translation(rotated, target_smooth)
        A = _similarity_fixed_to_moving(
            float(np.radians(angle_deg)), float(scale), float(tx), float(ty), (cx, cy)
        )
        warped = matlab_imwarp_bilinear(moving_smooth, (H, W), A, fill_value=0.0)
        resid = _ncc_residual(warped, target_norm)
        return float(np.mean(resid ** 2)), tx, ty

    # Coarse: angle +/-45 deg @ 2, scale 0.80..1.25 @ 0.02.
    for scale in np.arange(0.80, 1.2501, 0.02):
        for angle_deg in range(-45, 46, 2):
            sse, tx, ty = _eval(float(angle_deg), float(scale))
            if sse < best_sse:
                best_sse = sse
                best_angle_deg = float(angle_deg)
                best_scale = float(scale)
                best_tx = float(tx)
                best_ty = float(ty)

    # Fine: +/-3 deg @ 0.5, +/-0.04 scale @ 0.005.
    for scale in np.arange(best_scale - 0.04, best_scale + 0.0401, 0.005):
        for angle_deg in np.arange(best_angle_deg - 3.0, best_angle_deg + 3.01, 0.5):
            sse, tx, ty = _eval(float(angle_deg), float(scale))
            if sse < best_sse:
                best_sse = sse
                best_angle_deg = float(angle_deg)
                best_scale = float(scale)
                best_tx = float(tx)
                best_ty = float(ty)

    # Ultra-fine: +/-0.5 deg @ 0.1, +/-0.005 scale @ 0.001.
    for scale in np.arange(best_scale - 0.005, best_scale + 0.0051, 0.001):
        for angle_deg in np.arange(best_angle_deg - 0.5, best_angle_deg + 0.51, 0.1):
            sse, tx, ty = _eval(float(angle_deg), float(scale))
            if sse < best_sse:
                best_sse = sse
                best_angle_deg = float(angle_deg)
                best_scale = float(scale)
                best_tx = float(tx)
                best_ty = float(ty)

    return (
        float(np.radians(best_angle_deg)),
        float(best_scale),
        float(best_tx),
        float(best_ty),
        float(best_sse),
    )


def _lm_refine_similarity(
    moving_f: np.ndarray,
    target_f: np.ndarray,
    seed: tuple[float, float, float, float],
    *,
    max_nfev: int = 300,
) -> tuple[tuple[float, float, float, float], float]:
    """
    Bounded trust-region refinement of 4-dof similarity on the NCC residual.

    Returns ``(params, final_mean_sse)`` so multi-start callers can pick the best.
    """
    H, W = target_f.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    target_norm = _normalize_image(target_f)

    def _resid(p: np.ndarray) -> np.ndarray:
        A = _similarity_fixed_to_moving(
            float(p[0]), float(p[1]), float(p[2]), float(p[3]), (cx, cy)
        )
        warped = matlab_imwarp_bilinear(moving_f, (H, W), A, fill_value=0.0)
        return _ncc_residual(warped, target_norm)

    x0 = np.array(seed, dtype=np.float64)
    x0_clamped = np.array(
        [
            float(np.clip(x0[0], -np.pi, np.pi)),
            float(np.clip(x0[1], 0.5, 2.0)),
            float(np.clip(x0[2], -W, W)),
            float(np.clip(x0[3], -H, H)),
        ],
        dtype=np.float64,
    )
    lb = np.array([-np.pi, 0.5, -float(W), -float(H)], dtype=np.float64)
    ub = np.array([ np.pi, 2.0,  float(W),  float(H)], dtype=np.float64)
    try:
        res = least_squares(
            _resid,
            x0_clamped,
            method="trf",
            bounds=(lb, ub),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        params = (float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3]))
        final_sse = float(np.mean(_resid(res.x) ** 2))
        return params, final_sse
    except Exception:  # pragma: no cover
        return seed, float(np.mean(_resid(x0_clamped) ** 2))


def _lm_refine_affine(
    moving_f: np.ndarray,
    target_f: np.ndarray,
    seed_similarity: tuple[float, float, float, float],
    *,
    max_nfev: int = 500,
) -> tuple[np.ndarray, float]:
    """
    Bounded trust-region refinement of a 6-dof affine on the NCC residual,
    seeded from the similarity result.

    Returns ``(forward_2x3, final_mean_sse)``.
    """
    H, W = target_f.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    target_norm = _normalize_image(target_f)

    angle_rad, scale, tx, ty = seed_similarity
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    M_forward = scale * np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    t_forward = (
        np.array([cx, cy]) + np.array([tx, ty]) - M_forward @ np.array([cx, cy])
    )
    x0 = np.array(
        [
            M_forward[0, 0], M_forward[0, 1],
            M_forward[1, 0], M_forward[1, 1],
            t_forward[0], t_forward[1],
        ],
        dtype=np.float64,
    )
    lb = np.array([-3.0, -3.0, -3.0, -3.0, -2.0 * W, -2.0 * H], dtype=np.float64)
    ub = np.array([ 3.0,  3.0,  3.0,  3.0,  2.0 * W,  2.0 * H], dtype=np.float64)
    x0_clamped = np.clip(x0, lb, ub)

    def _resid(p: np.ndarray) -> np.ndarray:
        forward_2x3 = np.array(
            [[p[0], p[1], p[4]], [p[2], p[3], p[5]]], dtype=np.float64
        )
        try:
            A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)
        except np.linalg.LinAlgError:
            return np.full(H * W, 1e3, dtype=np.float64)
        warped = matlab_imwarp_bilinear(moving_f, (H, W), A_inv, fill_value=0.0)
        return _ncc_residual(warped, target_norm)

    try:
        res = least_squares(
            _resid,
            x0_clamped,
            method="trf",
            bounds=(lb, ub),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        p = res.x
    except Exception:  # pragma: no cover
        p = x0_clamped

    forward_2x3 = np.array(
        [[p[0], p[1], p[4]], [p[2], p[3], p[5]]], dtype=np.float64
    )
    final_sse = float(np.mean(_resid(p) ** 2))
    return forward_2x3, final_sse


def _refine_fwd_with_ncc(
    he_moving: np.ndarray,
    fixed_shg: np.ndarray,
    seed_forward_2x3: np.ndarray,
    *,
    matrix_delta: float = 0.05,
    translation_delta_px: float = 10.0,
    max_nfev: int = 200,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Polish a forward affine with a bounded NCC trust-region step.

    Called after the MI optimizer has found the correct basin; the small
    bounds prevent NCC from wandering off to spurious local minima that
    trip up NCC-from-scratch on this cross-modality data.

    Parameters
    ----------
    matrix_delta
        Allowed per-entry deviation of the 2x2 linear matrix from the seed.
    translation_delta_px
        Allowed translation deviation (pixels).
    """
    H, W = fixed_shg.shape[:2]
    moving_f = he_moving.astype(np.float64)
    target_f = fixed_shg.astype(np.float64)

    blur_sigma = 3.0
    moving_smooth = _smooth_mask(moving_f, blur_sigma)
    target_smooth = _smooth_mask(target_f, blur_sigma)
    target_norm_smooth = _normalize_image(target_smooth)
    # Gate metric: NCC on the *unsmoothed* mask vs raw SHG. Smoothed NCC is
    # what we optimize (smooth landscape -> well-conditioned LM), but we
    # accept the refined transform only if it also improves NCC on the
    # original images. Smoothed-NCC alone can improve while edge alignment
    # (and therefore RGB pixel match) silently degrades; the raw-NCC gate
    # catches that case.
    target_norm_raw = _normalize_image(target_f)

    seed = np.asarray(seed_forward_2x3, dtype=np.float64).reshape(2, 3)
    x0 = np.array(
        [seed[0, 0], seed[0, 1], seed[1, 0], seed[1, 1], seed[0, 2], seed[1, 2]],
        dtype=np.float64,
    )
    lb = np.array(
        [
            x0[0] - matrix_delta, x0[1] - matrix_delta,
            x0[2] - matrix_delta, x0[3] - matrix_delta,
            x0[4] - translation_delta_px, x0[5] - translation_delta_px,
        ],
        dtype=np.float64,
    )
    ub = np.array(
        [
            x0[0] + matrix_delta, x0[1] + matrix_delta,
            x0[2] + matrix_delta, x0[3] + matrix_delta,
            x0[4] + translation_delta_px, x0[5] + translation_delta_px,
        ],
        dtype=np.float64,
    )

    def _resid(p: np.ndarray) -> np.ndarray:
        forward_2x3 = np.array(
            [[p[0], p[1], p[4]], [p[2], p[3], p[5]]], dtype=np.float64
        )
        try:
            A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)
        except np.linalg.LinAlgError:
            return np.full(H * W, 1e3, dtype=np.float64)
        warped = matlab_imwarp_bilinear(moving_smooth, (H, W), A_inv, fill_value=0.0)
        return _ncc_residual(warped, target_norm_smooth)

    def _ncc_raw_at(p: np.ndarray) -> float:
        forward_2x3 = np.array(
            [[p[0], p[1], p[4]], [p[2], p[3], p[5]]], dtype=np.float64
        )
        try:
            A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)
        except np.linalg.LinAlgError:
            return -1.0
        warped_raw = matlab_imwarp_bilinear(moving_f, (H, W), A_inv, fill_value=0.0)
        return _ncc_score(warped_raw, target_norm_raw)

    before_sse = float(np.mean(_resid(x0) ** 2))
    before_ncc_raw = _ncc_raw_at(x0)
    try:
        res = least_squares(
            _resid, x0, method="trf", bounds=(lb, ub),
            max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        p = res.x
    except Exception:  # pragma: no cover
        p = x0
    after_sse = float(np.mean(_resid(p) ** 2))
    after_ncc_raw = _ncc_raw_at(p)

    # Two-gate accept rule:
    #   1. Smoothed-NCC SSE must have decreased (the LM objective).
    #   2. Raw-NCC must have *not* decreased (proxy for RGB pixel match).
    # If either gate fails, revert to the seed. This empirically prevents
    # the failure mode where smoothed-NCC improves but the RGB MAE
    # regresses (observed on test3 of the BDcreation_reg2 fixtures).
    accepted = (after_sse < before_sse) and (after_ncc_raw >= before_ncc_raw - 1e-6)
    if not accepted:
        p = x0

    forward_2x3 = np.array(
        [[p[0], p[1], p[4]], [p[2], p[3], p[5]]], dtype=np.float64
    )
    debug = {
        "refine_before_sse": before_sse,
        "refine_after_sse": after_sse,
        "refine_before_ncc_raw": before_ncc_raw,
        "refine_after_ncc_raw": after_ncc_raw,
        "refine_accepted": accepted,
        "refine_matrix_delta": matrix_delta,
        "refine_translation_delta_px": translation_delta_px,
    }
    return forward_2x3, debug


def _register_dice(
    he_moving: np.ndarray, fixed_shg: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Deterministic registration pipeline (NCC objective).

    The objective is the mean-squared residual on zero-mean/unit-std images,
    which is equivalent to maximizing NCC. This mirrors what MATLAB does
    logically (Mattes MI on the same HE collagen mask vs grayscale SHG) but
    is deterministic and differentiable-friendly. The name
    ``_register_dice`` is kept for backwards compatibility with call sites.

    Steps
    -----
    1. Gaussian-blur the HE collagen mask (already binary) and the grayscale
       SHG so the NCC residual has a smooth landscape for LM.
    2. Coarse-to-fine NCC grid search over (angle, scale), with translation
       per candidate taken from phase correlation of the rotated mask.
    3. Multi-start similarity LM (seeds: grid-best and identity); keep the
       result with the lower NCC residual.
    4. Affine LM refinement seeded from the best similarity.

    Returns
    -------
    forward_2x3 : np.ndarray
        Forward affine ``p_fixed = M @ p_moving + t``.
    debug : dict
        Intermediate parameters for inspection.
    """
    moving_f = he_moving.astype(np.float64)
    target_f = fixed_shg.astype(np.float64)

    # Blur sigma chosen so the gradient field is wider than the LM step size
    # (~1 px) but narrower than typical misregistration (~10 px).
    blur_sigma = 3.0
    moving_smooth = _smooth_mask(moving_f, blur_sigma)
    target_smooth = _smooth_mask(target_f, blur_sigma)

    seed = _register_similarity_seed(moving_smooth, target_smooth)

    sim_grid_params, sim_grid_sse = _lm_refine_similarity(
        moving_smooth, target_smooth, seed[:4]
    )
    sim_id_params, sim_id_sse = _lm_refine_similarity(
        moving_smooth, target_smooth, (0.0, 1.0, 0.0, 0.0)
    )
    if sim_id_sse < sim_grid_sse:
        sim_params = sim_id_params
        sim_best_sse = sim_id_sse
        sim_best_origin = "identity"
    else:
        sim_params = sim_grid_params
        sim_best_sse = sim_grid_sse
        sim_best_origin = "grid"

    forward_2x3, aff_sse = _lm_refine_affine(
        moving_smooth, target_smooth, sim_params
    )

    # NCC score at the final affine (for debug / confidence checks).
    H, W = target_smooth.shape
    A_inv_final = _affine_fixed_to_moving_from_forward(forward_2x3)
    warped_final = matlab_imwarp_bilinear(
        moving_smooth, (H, W), A_inv_final, fill_value=0.0
    )
    target_norm = _normalize_image(target_smooth)
    final_ncc = _ncc_score(warped_final, target_norm)

    debug = {
        "seed_angle_rad": seed[0],
        "seed_scale": seed[1],
        "seed_tx": seed[2],
        "seed_ty": seed[3],
        "seed_sse": seed[4],
        "sim_grid_sse": sim_grid_sse,
        "sim_id_sse": sim_id_sse,
        "sim_best_origin": sim_best_origin,
        "sim_angle_rad": sim_params[0],
        "sim_scale": sim_params[1],
        "sim_tx": sim_params[2],
        "sim_ty": sim_params[3],
        "sim_best_sse": sim_best_sse,
        "aff_final_sse": aff_sse,
        "final_ncc": final_ncc,
        "blur_sigma": blur_sigma,
    }
    return forward_2x3, debug


# ---------------------------------------------------------------------------
# Legacy Mattes MI path (kept behind registration_method="mi" for debugging).
# ---------------------------------------------------------------------------


def _register_mi(
    he_moving: np.ndarray,
    fixed_shg: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Mattes MI + grid search + Nelder-Mead registration.

    Search ranges match the basin that agrees with MATLAB ``BDcreation_reg2``
    goldens on patient_001 (angle ±45°, scale 0.80–1.25, translation ±50 px).
    A wider grid was tried and left that basin on the ppm=3 case. Affine is
    kept only when it improves Mattes MI over similarity (C6).
    """
    if not _HAS_SITK:  # pragma: no cover
        raise RuntimeError(
            "registration_method='mi' requires SimpleITK; install it or "
            "switch to the default 'dice' path."
        )

    fixed = fixed_shg.astype(np.float64)
    moving = he_moving.astype(np.float64)
    if float(moving.max()) - float(moving.min()) < 1e-12:
        forward_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return forward_2x3, {"mi_degenerate_moving": True}

    fixed_sitk = sitk.GetImageFromArray(fixed)
    moving_sitk = sitk.GetImageFromArray(moving)
    fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat64)
    moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat64)

    geom_init = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    center = list(geom_init.GetFixedParameters())

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
        try:
            return eval_method.MetricEvaluate(fixed_sitk, moving_sitk)
        except RuntimeError:
            return float("inf")

    def _probe_angle_scale(angle_range, scale_range):
        nonlocal best_angle, best_scale, best_metric
        for scale_val in scale_range:
            for angle_deg in angle_range:
                val = _eval_similarity(angle_deg, scale_val, best_tx, best_ty)
                if val < best_metric:
                    best_metric = val
                    best_angle = float(angle_deg)
                    best_scale = float(scale_val)

    # MATLAB-parity grid (do not widen: ppm=3 left the golden basin).
    _probe_angle_scale(range(-45, 46, 2), np.arange(0.80, 1.25, 0.02))
    _probe_angle_scale(
        np.arange(best_angle - 3, best_angle + 3.01, 0.5),
        np.arange(best_scale - 0.04, best_scale + 0.041, 0.005),
    )
    _probe_angle_scale(
        np.arange(best_angle - 0.5, best_angle + 0.51, 0.1),
        np.arange(best_scale - 0.005, best_scale + 0.0051, 0.001),
    )

    for tx in np.arange(-50, 51, 5):
        for ty in np.arange(-50, 51, 5):
            val = _eval_similarity(best_angle, best_scale, tx, ty)
            if val < best_metric:
                best_metric = val
                best_tx = float(tx)
                best_ty = float(ty)
    for tx in np.arange(best_tx - 5, best_tx + 5.01, 1):
        for ty in np.arange(best_ty - 5, best_ty + 5.01, 1):
            val = _eval_similarity(best_angle, best_scale, tx, ty)
            if val < best_metric:
                best_metric = val
                best_tx = float(tx)
                best_ty = float(ty)

    # Joint fine sweep.
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

    x0_sim = np.array(
        [np.radians(best_angle), best_scale, best_tx, best_ty], dtype=np.float64
    )
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
    sim_metric = float(_similarity_cost(sim_opt.x))

    # Similarity forward 2x3 (moving -> fixed) for optional C6 keep.
    A_sim_inv = _similarity_fixed_to_moving(
        opt_angle, opt_scale, opt_tx, opt_ty, (float(center[0]), float(center[1]))
    )
    sim_forward = _forward_from_fixed_to_moving(A_sim_inv)

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
    p = aff_opt.x
    aff_metric = float(_affine_cost(p))
    A_inv = np.eye(3, dtype=np.float64)
    A_inv[:2, :2] = np.array([[p[0], p[1]], [p[2], p[3]]], dtype=np.float64)
    A_inv[:2, 2] = np.array([p[4], p[5]], dtype=np.float64)
    aff_forward = _forward_from_fixed_to_moving(A_inv)

    # C6: keep affine only if it improves (lower) Mattes MI cost vs similarity.
    if aff_metric <= sim_metric:
        forward_2x3 = aff_forward
        kept_stage = "affine"
        final_metric = aff_metric
    else:
        forward_2x3 = sim_forward
        kept_stage = "similarity"
        final_metric = sim_metric

    grid_metric = float(best_metric)
    debug = {
        "mi_sim_params": (opt_angle, opt_scale, opt_tx, opt_ty),
        "mi_aff_params": tuple(p.tolist()),
        "mi_grid_metric": grid_metric,
        "mi_sim_metric": float(sim_metric),
        "mi_aff_metric": float(aff_metric),
        "mi_kept_stage": kept_stage,
        "mi_final_metric": float(final_metric),
    }
    return forward_2x3, debug


def _affine_m_t0_from_transform(tx: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed->moving affine ``x' = Mx + t0`` from any SITK transform,
    including CompositeTransform, by probing transformed basis points.
    """
    p00 = np.array(tx.TransformPoint((0.0, 0.0)), dtype=np.float64)
    p10 = np.array(tx.TransformPoint((1.0, 0.0)), dtype=np.float64)
    p01 = np.array(tx.TransformPoint((0.0, 1.0)), dtype=np.float64)
    m = np.column_stack((p10 - p00, p01 - p00))
    return m, p00


def _register_oneplusone_mi(
    he_moving: np.ndarray,
    fixed_shg: np.ndarray,
    *,
    random_state: int = 0,
    init: str = "mi_seed",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Two-stage multiresolution (1+1)-ES with Mattes MI via SimpleITK (ITK v4/v5).

    MATLAB-*like* configuration of ``BDcreation_reg2.m``:
      1. ``imregconfig('multimodal')`` -> OnePlusOneEvolutionary + MattesMI(50 bins)
      2. ``optimizer.InitialRadius = 6.25e-3 / 3.5`` (the optimizer is a value
         object, so the reduced radius applies to *both* imregtform calls)
      3. ``optimizer.MaximumIterations = 700`` (per pyramid level)
      4. Stage 1: ``imregtform(..., 'similarity')`` from identity (+centre offset)
      5. Stage 2: ``imregtform(..., 'affine', 'InitialTransformation', sim)``
    with MATLAB's optimizer scales (``[1,1,(1,1,)1/maxT,1/maxT]``), shrink
    factor ``GrowthFactor^-0.25``, 3-level pyramid [4,2,1] and a fixed seed.

    This is **not** bit-exact with MATLAB (v4 Mattes MI, no per-level
    radius/epsilon refiner, different RNG); use ``registration_method="matlab"``
    for that. ``init="identity"`` is MATLAB's start point, but with the v4 ES
    it frequently misses the golden basin (patient_001 ppm=2: MAE 32 vs 5.3
    with ``init="mi_seed"``), so the default seeds from the deterministic MI
    grid search and uses the ES for stochastic refinement only.
    """
    if not _HAS_SITK:  # pragma: no cover
        raise RuntimeError(
            "registration_method='oneplusone' requires SimpleITK; "
            "install it or choose 'ncc'."
        )

    fixed = fixed_shg.astype(np.float64)
    moving = he_moving.astype(np.float64)
    if float(moving.max()) - float(moving.min()) < 1e-12:
        forward_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return forward_2x3, {"oneplusone_degenerate_moving": True}

    fixed_sitk = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat64)
    moving_sitk = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat64)

    seed = max(1, int(random_state))
    # MATLAB: imregconfig('multimodal') InitialRadius 6.25e-3, then /3.5 in
    # BDcreation_reg2.m before both imregtform calls; GrowthFactor 1.05,
    # ShrinkFactor = GrowthFactor^-0.25, Epsilon 1.5e-6, 700 iterations/level.
    growth_factor = 1.05
    shrink_factor = float(growth_factor) ** -0.25
    radius = 6.25e-3 / 3.5
    epsilon = 1.5e-6
    n_iter_per_level = 700
    shrink_factors = [4, 2, 1]
    smoothing_sigmas = [2, 1, 0]
    # computeDefaultRegmexSettings.m: translation scale = 1 / hypot(W, H).
    fh, fw = fixed.shape[:2]
    translation_scale = 1.0 / float(np.hypot(fw, fh))

    def _make_reg(transform: Any, seed_val: int, scales: list[float]) -> Any:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.NONE)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=n_iter_per_level,
            epsilon=epsilon,
            initialRadius=radius,
            growthFactor=growth_factor,
            shrinkFactor=shrink_factor,
            seed=seed_val,
        )
        reg.SetOptimizerScales(scales)
        reg.SetShrinkFactorsPerLevel(shrink_factors)
        reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        reg.SetInitialTransform(transform, inPlace=False)
        return reg

    center = [(fixed.shape[1] - 1) / 2.0, (fixed.shape[0] - 1) / 2.0]
    center_arr = np.array(center, dtype=np.float64)
    mi_seed_debug: dict[str, Any] | None = None
    if init == "mi_seed":
        mi_seed_forward, mi_seed_debug = _register_mi(he_moving, fixed_shg)
        mi_inv = _affine_fixed_to_moving_from_forward(mi_seed_forward)
        init_mat = mi_inv[:2, :2]
        init_t = mi_inv[:2, 2]
    elif init == "identity":
        # MATLAB initial: identity linear part, translation = offset between
        # the moving and fixed image centres (zero when the grids match).
        mh, mw = moving.shape[:2]
        init_mat = np.eye(2, dtype=np.float64)
        init_t = np.array([(mw - fw) / 2.0, (mh - fh) / 2.0], dtype=np.float64)
    else:
        raise ValueError(f"init must be 'identity' or 'mi_seed', got {init!r}")

    # Decompose into Similarity2DTransform parameters (scale, angle, t).
    sx = float(np.sqrt(init_mat[0, 0] ** 2 + init_mat[1, 0] ** 2))
    angle = float(np.arctan2(init_mat[1, 0], init_mat[0, 0]))
    t_from_center = init_t + init_mat @ center_arr - center_arr

    sim_init = sitk.Similarity2DTransform()
    sim_init.SetScale(float(sx))
    sim_init.SetAngle(float(angle))
    sim_init.SetCenter(center)
    sim_init.SetTranslation(t_from_center.tolist())

    sim_scales = [1.0, 1.0, translation_scale, translation_scale]
    sim_reg = _make_reg(sim_init, seed, sim_scales)
    sim_result = sim_reg.Execute(fixed_sitk, moving_sitk)
    sim_metric = float(sim_reg.GetMetricValue())
    sim_stop = str(sim_reg.GetOptimizerStopConditionDescription())

    # Stage 2: affine about the same centre, initialised from the similarity.
    sim_mat, sim_t0 = _affine_m_t0_from_transform(sim_result)
    aff_init = sitk.AffineTransform(2)
    aff_init.SetCenter(center)
    aff_init.SetMatrix(sim_mat.ravel().tolist())
    aff_init.SetTranslation((sim_t0 + sim_mat @ center_arr - center_arr).tolist())

    aff_scales = [1.0, 1.0, 1.0, 1.0, translation_scale, translation_scale]
    aff_reg = _make_reg(aff_init, seed, aff_scales)
    aff_result = aff_reg.Execute(fixed_sitk, moving_sitk)
    aff_metric = float(aff_reg.GetMetricValue())
    aff_stop = str(aff_reg.GetOptimizerStopConditionDescription())

    aff_matrix, aff_t0 = _affine_m_t0_from_transform(aff_result)

    A_inv = np.eye(3, dtype=np.float64)
    A_inv[:2, :2] = aff_matrix
    A_inv[:2, 2] = aff_t0
    forward_2x3 = _forward_from_fixed_to_moving(A_inv)

    debug = {
        "oneplusone_random_state": int(random_state),
        "oneplusone_seed": int(seed),
        "oneplusone_init": init,
        "oneplusone_radius": float(radius),
        "oneplusone_growth_factor": float(growth_factor),
        "oneplusone_shrink_factor": float(shrink_factor),
        "oneplusone_epsilon": float(epsilon),
        "oneplusone_iterations_per_level": int(n_iter_per_level),
        "oneplusone_translation_scale": float(translation_scale),
        "oneplusone_shrink_factors": tuple(shrink_factors),
        "oneplusone_smoothing_sigmas": tuple(smoothing_sigmas),
        "oneplusone_sim_metric": float(sim_metric),
        "oneplusone_sim_stop": sim_stop,
        "oneplusone_aff_metric": float(aff_metric),
        "oneplusone_aff_stop": aff_stop,
        "oneplusone_mi_seed_debug": mi_seed_debug,
        "oneplusone_affine_matrix": aff_matrix.tolist(),
        "oneplusone_affine_t0": aff_t0.tolist(),
    }
    return forward_2x3, debug


# ---------------------------------------------------------------------------
# Core pipeline.
# ---------------------------------------------------------------------------


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


def _build_he_moving(
    he_scaled: np.ndarray,
    he_adjusted: np.ndarray,
    he_decorr: np.ndarray,
    pixpermic: float,
    ecm_mode: str,
    random_state: int = 0,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """
    Build the moving image for registration from one ECM method.

    Returns ``(he_moving, resolved_mode, extras)``.
    """
    mode = (ecm_mode or "hsv").lower()
    extras: dict[str, Any] = {}
    _MIN_MASK_COVERAGE = 0.02

    if mode == "hsv":
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

    if mode == "gray":
        # B5: continuous inverted luma (optionally nuclei-suppressed).
        _, masked_nuclei_image = make_nuclei_mask(he_adjusted, pixpermic)
        bw_nuclei_filled = _refine_nuclei_filled(masked_nuclei_image, pixpermic)
        he_gray_inv = 1.0 - matlab_rgb2gray(he_adjusted)
        he_moving = he_gray_inv * (~bw_nuclei_filled).astype(np.float64)
        extras["mask_coverage"] = float((he_moving > 0.01).mean())
        return he_moving, mode, extras

    if mode == "rgb":
        _bw_nuclei_opened, masked_nuclei_image = make_nuclei_mask_rgb(he_decorr, pixpermic)
        he_collagen_gray, _bw_collagen = make_ecm_mask_rgb(he_decorr, pixpermic)
    elif mode == "lab":
        _bw_nuclei_opened, masked_nuclei_image = make_nuclei_mask_rgb(he_decorr, pixpermic)
        he_collagen_gray, _bw_collagen = make_ecm_mask_lab(
            he_scaled, he_decorr, pixpermic, random_state=random_state
        )
    else:
        raise ValueError(
            f"Unknown ecm_method={ecm_mode!r}; expected one of "
            f"'hsv', 'rgb', 'lab', 'gray', 'auto'."
        )

    bw_nuclei_filled = _refine_nuclei_filled(masked_nuclei_image, pixpermic)
    he_collagen_exclude = he_collagen_gray * (~bw_nuclei_filled).astype(np.float64)
    he_collagen_bw = he_collagen_exclude > 0.01
    he_collagen_bw = remove_small_components(
        he_collagen_bw, int(np.ceil(pixpermic**2))
    )
    he_moving = he_collagen_exclude * he_collagen_bw.astype(np.float64)
    extras["mask_coverage"] = float((he_moving > 0.01).mean())
    return he_moving, mode, extras


def _register_matlab_itk_v3(
    he_moving: np.ndarray,
    fixed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    MATLAB-parity registration: ITK v3 framework configured exactly like
    ``imregtform`` (see :mod:`pycurvelets._itk_v3_matlab_engine`).

    Given the same ``he_moving``/``fixed`` arrays MATLAB used, this reproduces
    ``BDcreation_reg2.m``'s ``tformSimilarity`` and ``tform`` to floating-point
    precision (validated iteration-by-iteration against MATLAB's
    ``DisplayOptimization`` trace in ``tests/matlab_parity``). It requires the
    ``itk`` package; SimpleITK cannot expose the v3 classes.
    """
    from ._itk_v3_matlab_engine import (
        DEFAULT_SEED,
        has_itk,
        register_bdcreation_reg2_matlab,
    )

    if not has_itk():
        raise RuntimeError(
            "registration_method='matlab' requires the 'itk' package "
            "(pip install itk, or the 'matlab-parity' optional dependency group)."
        )
    if not np.any(he_moving > 0):
        forward_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return forward_2x3, {"matlab_degenerate_moving": True}

    forward_2x3, ml_debug = register_bdcreation_reg2_matlab(
        he_moving.astype(np.float64), fixed.astype(np.float64), seed=DEFAULT_SEED
    )
    return forward_2x3, ml_debug


def _run_optimizer(
    he_moving: np.ndarray,
    fixed: np.ndarray,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Dispatch one registration optimizer; returns forward_2x3, backend, debug."""
    debug: dict[str, Any] = {}
    forward_2x3: np.ndarray | None = None
    backend: str

    if method == "matlab":
        forward_2x3, ml_debug = _register_matlab_itk_v3(he_moving, fixed)
        debug.update(ml_debug)
        backend = "itk_v3_matlab_parity"
    elif method == "mi" and _HAS_SITK:
        forward_2x3, mi_debug = _register_mi(he_moving, fixed)
        debug.update(mi_debug)
        backend = "simpleitk_mattes"
    elif method == "oneplusone" and _HAS_SITK:
        forward_2x3, one_debug = _register_oneplusone_mi(
            he_moving, fixed, random_state=random_state
        )
        debug.update(one_debug)
        backend = "oneplusone_mattes"
    elif method in ("ncc", "dice"):
        forward_2x3, ncc_debug = _register_dice(he_moving, fixed)
        debug.update(ncc_debug)
        backend = "ncc_trf"
    elif method == "mi_ncc" and _HAS_SITK:
        mi_fwd, mi_debug = _register_mi(he_moving, fixed)
        debug.update(mi_debug)
        forward_2x3, refine_debug = _refine_fwd_with_ncc(
            he_moving, fixed, mi_fwd,
            matrix_delta=0.05, translation_delta_px=10.0,
        )
        debug.update(refine_debug)
        backend = "mi_then_ncc"
    else:
        forward_2x3 = None
        backend = "unset"

    if forward_2x3 is None and method in ("mi", "oneplusone", "mi_ncc") and not _HAS_SITK:
        forward_2x3, ncc_debug = _register_dice(he_moving, fixed)
        debug.update(ncc_debug)
        backend = "ncc_trf"

    if forward_2x3 is None:
        shift, _, _ = registration.phase_cross_correlation(fixed, he_moving)
        forward_2x3 = np.array(
            [[1.0, 0.0, float(-shift[1])], [0.0, 1.0, float(-shift[0])]],
            dtype=np.float64,
        )
        backend = "skimage_ecc_fallback"

    return forward_2x3, backend, debug


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


def _shg_he_registration_core(
    he_filepath: str,
    he_filename: str,
    shg_filepath: str,
    pixelpermicron: float,
    registration_method: str = "mi_ncc",
    ecm_method: str = "hsv",
    random_state: int = 0,
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

    he_img = io.imread(he_path).astype(np.float64) / 255.0
    shg_img = io.imread(shg_path).astype(np.float64) / 255.0
    if shg_img.ndim == 3:
        shg_img = matlab_rgb2gray(shg_img)

    original_shg_shape = shg_img.shape[:2]
    he_scaled, fixed_shg, pixpermic = prepare_registration_pair(
        he_img, shg_img, float(pixelpermicron)
    )

    he_adjusted = adjust_rgb_mean_std(he_scaled)
    he_decorr = decorrelation_stretch(he_adjusted, tol=0.01)

    fixed = fixed_shg.astype(np.float64)
    if fixed.ndim == 3:
        fixed = matlab_rgb2gray(fixed)

    method = (registration_method or "mi_ncc").lower()
    ecm_requested = (ecm_method or "hsv").lower()

    debug: dict[str, Any] = {
        "registration_method_requested": method,
        "ecm_method_requested": ecm_requested,
        "pixpermic_working": float(pixpermic),
        "fixed_shape": tuple(int(x) for x in fixed.shape),
    }

    # B3: auto-pick ECM by SHG histogram MI of the *unregistered* moving
    # image (cheap), then run the full optimizer once on the winner.
    if ecm_requested == "auto":
        candidates = ("hsv", "rgb", "lab", "gray")
        best_mi = -np.inf
        best_moving: np.ndarray | None = None
        best_mode = "hsv"
        best_extras: dict[str, Any] = {}
        auto_scores: dict[str, float] = {}
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        for cand in candidates:
            try:
                moving_c, mode_c, extras_c = _build_he_moving(
                    he_scaled, he_adjusted, he_decorr, pixpermic, cand, random_state
                )
            except Exception as exc:
                auto_scores[cand] = float("nan")
                debug.setdefault("ecm_auto_errors", {})[cand] = str(exc)
                continue
            score_c = _score_forward_vs_shg(moving_c, fixed, identity)
            auto_scores[mode_c] = float(score_c["shg_mi"])
            if float(score_c["shg_mi"]) > best_mi:
                best_mi = float(score_c["shg_mi"])
                best_moving = moving_c
                best_mode = mode_c
                best_extras = extras_c
        debug["ecm_auto_shg_mi"] = auto_scores
        if best_moving is None:
            raise RuntimeError("ecm_method='auto' failed for all ECM candidates.")
        he_moving = best_moving
        ecm_mode = best_mode
        debug.update(best_extras)
        debug["ecm_method_selected"] = ecm_mode
        forward_2x3, backend, opt_debug = _run_optimizer(
            he_moving, fixed, method, random_state
        )
        debug.update(opt_debug)
    else:
        he_moving, ecm_mode, extras = _build_he_moving(
            he_scaled, he_adjusted, he_decorr, pixpermic, ecm_requested, random_state
        )
        debug.update(extras)
        debug["ecm_method_selected"] = ecm_mode
        forward_2x3, backend, opt_debug = _run_optimizer(
            he_moving, fixed, method, random_state
        )
        debug.update(opt_debug)

    debug["forward_2x3"] = forward_2x3.tolist()

    # A1: always report SHG alignment of the moving image under the chosen transform.
    shg_metrics = _score_forward_vs_shg(he_moving, fixed, forward_2x3)
    debug["shg_alignment"] = shg_metrics
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    debug["shg_alignment_identity_mi"] = float(
        _score_forward_vs_shg(he_moving, fixed, identity)["shg_mi"]
    )

    A_inv = _affine_fixed_to_moving_from_forward(forward_2x3)

    rgb_for_warp = resize_like(he_img, fixed.shape[:2])
    # MATLAB: imwarp(RGB, ..., 'FillValues', 255) on a *double* image, then
    # imresize(B, size(SHG)) and imwrite (clip to [0, 1]). Blending 255 rather
    # than 1.0 at the border saturates the halo band to white exactly as
    # MATLAB does; clipping happens after the final resize, like imwrite.
    registered = matlab_imwarp_bilinear(
        rgb_for_warp.astype(np.float64),
        fixed.shape[:2],
        A_inv,
        fill_value=255.0,
    )

    registered_img = resize_like(registered, original_shg_shape)
    registered_img = np.clip(registered_img, 0.0, 1.0)

    # Also score registered HE luma vs original SHG (full-res).
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
    Register H&E to SHG (Python port of MATLAB ``BDcreation_reg2.m``).

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
        random_state=p.random_state,
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
    """Compatibility wrapper retaining MATLAB function name."""
    out = shg_he_registration(
        BDCparameters, save_output=True, return_debug=False
    )
    assert isinstance(out, np.ndarray)
    return out
