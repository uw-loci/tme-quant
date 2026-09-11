"""
Ground-truth evaluation helpers for SHG<->HE registration.

Developer / test-only (not installed in the wheel). The MATLAB
``BDcreation_reg2`` goldens are *reference outputs*, not ground truth: on the
synthetic patient_02 cases MATLAB itself lands 5-70 px from the true
alignment. These helpers score a registration against the true transform.

All affines are 0-based pixel-centre ``forward`` maps (moving -> fixed),
row-vector-free ``[x'; y'; 1] = F @ [x; y; 1]`` with ``F`` 2x3 or 3x3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def _as_3x3(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if F.shape == (2, 3):
        return np.vstack([F, [0.0, 0.0, 1.0]])
    if F.shape == (3, 3):
        return F
    raise ValueError(f"affine must be 2x3 or 3x3, got {F.shape}")


def decompose_affine(F: np.ndarray) -> dict[str, float]:
    """Return ``angle_deg`` (CCW, image coords), ``scale_x``, ``scale_y``, ``shear``, ``tx``, ``ty``."""
    M = _as_3x3(F)
    a, c = M[0, 0], M[1, 0]  # first column = image of the x unit vector
    scale_x = float(np.hypot(a, c))
    angle = float(np.degrees(np.arctan2(c, a)))
    # remove rotation, read off shear/scale_y from the remaining upper-triangular part
    cos_t, sin_t = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    U = R_inv @ M[:2, :2]
    return {
        "angle_deg": angle,
        "scale_x": scale_x,
        "scale_y": float(U[1, 1]),
        "shear": float(U[0, 1] / U[0, 0]) if U[0, 0] != 0 else float("nan"),
        "tx": float(M[0, 2]),
        "ty": float(M[1, 2]),
    }


def mean_corner_displacement_px(F_a: np.ndarray, F_b: np.ndarray, shape: tuple[int, int]) -> float:
    """
    Mean |F_a(p) - F_b(p)| over the 4 corners and centre of a ``shape``=(H, W)
    moving grid. A single-number transform error that weights rotation/scale
    by how much they actually move pixels.
    """
    A, B = _as_3x3(F_a), _as_3x3(F_b)
    H, W = int(shape[0]), int(shape[1])
    pts = np.array(
        [[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1], [(W - 1) / 2, (H - 1) / 2, 1]],
        dtype=np.float64,
    ).T
    d = (A @ pts)[:2] - (B @ pts)[:2]
    return float(np.mean(np.hypot(d[0], d[1])))


def _resize_map(src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
    """0-based pixel-centre map for an imresize-style resize (x' = (x+0.5)*s - 0.5)."""
    sy = dst_shape[0] / src_shape[0]
    sx = dst_shape[1] / src_shape[1]
    S = np.eye(3)
    S[0, 0], S[0, 2] = sx, 0.5 * sx - 0.5
    S[1, 1], S[1, 2] = sy, 0.5 * sy - 0.5
    return S


def gt_forward_to_working_grid(
    gt_forward_input_grids: np.ndarray,
    he_in_shape: tuple[int, int],
    shg_in_shape: tuple[int, int],
    work_shape: tuple[int, int],
) -> np.ndarray:
    """
    GT affine recovered between the *input* HE grid and the *input* SHG grid
    -> the equivalent forward affine on the working grid, where both images
    were resized to ``work_shape`` (as ``prepare_registration_pair`` does).
    """
    G = _as_3x3(gt_forward_input_grids)
    S_he = _resize_map(he_in_shape, work_shape)
    S_shg = _resize_map(shg_in_shape, work_shape)
    return (S_shg @ G @ np.linalg.inv(S_he))[:2, :]


def registration_transform_to_grid(
    forward_working: np.ndarray,
    work_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    fixed_shape: tuple[int, int],
) -> np.ndarray:
    """
    Inverse of :func:`gt_forward_to_working_grid`: express a working-grid
    forward affine between arbitrary moving/fixed grids (e.g. the full-res HE
    and SHG), so transforms obtained at different ppm can be compared.
    """
    F = _as_3x3(forward_working)
    S_mov = _resize_map(moving_shape, work_shape)
    S_fix = _resize_map(fixed_shape, work_shape)
    return (np.linalg.inv(S_fix) @ F @ S_mov)[:2, :]


def ssim_vs_reference_rgb(registered_uint8: np.ndarray, reference_uint8: np.ndarray) -> float:
    """SSIM between two uint8 RGB (or gray) images of the same shape."""
    a = np.asarray(registered_uint8)
    b = np.asarray(reference_uint8)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    kwargs: dict[str, Any] = {"data_range": 255}
    if a.ndim == 3:
        kwargs["channel_axis"] = -1
    return float(structural_similarity(a, b, **kwargs))


def gt_report(
    forward_working: np.ndarray,
    gt_forward_working: np.ndarray,
    work_shape: tuple[int, int],
) -> dict[str, float]:
    """Transform-error summary of a registration vs GT on the working grid."""
    d_reg = decompose_affine(forward_working)
    d_gt = decompose_affine(gt_forward_working)
    return {
        "corner_disp_px": mean_corner_displacement_px(forward_working, gt_forward_working, work_shape),
        "identity_corner_disp_px": mean_corner_displacement_px(np.eye(3), gt_forward_working, work_shape),
        "d_angle_deg": float(d_reg["angle_deg"] - d_gt["angle_deg"]),
        "d_scale": float(d_reg["scale_x"] - d_gt["scale_x"]),
        "reg_angle_deg": d_reg["angle_deg"],
        "gt_angle_deg": d_gt["angle_deg"],
        "reg_scale": d_reg["scale_x"],
        "gt_scale": d_gt["scale_x"],
    }
