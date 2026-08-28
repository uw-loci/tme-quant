from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    if out.dtype == np.uint8:
        return out
    if np.issubdtype(out.dtype, np.floating):
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _to_float01_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    if np.issubdtype(out.dtype, np.floating):
        return np.clip(out.astype(np.float64), 0.0, 1.0)
    return np.clip(out.astype(np.float64) / 255.0, 0.0, 1.0)


def _to_float01_gray(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 3:
        out = (
            0.2989 * out[..., 0]
            + 0.5870 * out[..., 1]
            + 0.1140 * out[..., 2]
        )
    # Only rescale integer-like images; float data already in ~[0,1] stays.
    if np.issubdtype(np.asarray(arr).dtype, np.integer) or float(np.nanmax(out)) > 1.5:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def compute_registration_quality_metrics(
    python_img: np.ndarray,
    matlab_golden_img: np.ndarray,
) -> dict[str, Any]:
    """
    Compute registration quality metrics against MATLAB golden output.

    Metrics are reported in mixed scales:
    - MAE/RMSE are on uint8 differences (0..255).
    - PSNR/SSIM are on float images in [0, 1].
    - exact/withinN are fractions in [0, 1].
    """
    py_u8 = _to_uint8_rgb(python_img)
    gt_u8 = _to_uint8_rgb(matlab_golden_img)
    if py_u8.shape != gt_u8.shape:
        raise ValueError(
            f"Shape mismatch: python image {py_u8.shape} vs golden image {gt_u8.shape}"
        )

    py_f = _to_float01_rgb(py_u8)
    gt_f = _to_float01_rgb(gt_u8)

    diff_i16 = py_u8.astype(np.int16) - gt_u8.astype(np.int16)
    abs_diff = np.abs(diff_i16).astype(np.float64)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff_i16.astype(np.float64) ** 2)))
    exact_frac = float(np.mean(py_u8 == gt_u8))
    within5 = float(np.mean(abs_diff <= 5.0))
    within10 = float(np.mean(abs_diff <= 10.0))
    within20 = float(np.mean(abs_diff <= 20.0))

    psnr = float(peak_signal_noise_ratio(gt_f, py_f, data_range=1.0))
    ssim = float(
        structural_similarity(gt_f, py_f, data_range=1.0, channel_axis=-1)
    )

    return {
        "shape": tuple(int(x) for x in py_u8.shape),
        "mae_uint8": mae,
        "rmse_uint8": rmse,
        "exact_frac": exact_frac,
        "within5_frac": within5,
        "within10_frac": within10,
        "within20_frac": within20,
        "psnr": psnr,
        "ssim": ssim,
    }


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean NCC in ``[-1, 1]`` for two same-shaped images."""
    x = _to_float01_gray(a).ravel()
    y = _to_float01_gray(b).ravel()
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def histogram_mutual_information(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bins: int = 50,
) -> float:
    """
    Histogram mutual information (nats) between two images.

    Higher is better. Independent of SimpleITK; used as the primary
    SHG-alignment score for tests and ECM auto-selection.
    """
    x = _to_float01_gray(a).ravel()
    y = _to_float01_gray(b).ravel()
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    pxy = hist_2d.astype(np.float64)
    total = float(pxy.sum())
    if total <= 0:
        return 0.0
    pxy /= total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(
            (pxy > 0) & (px > 0) & (py > 0),
            np.log(pxy / (px * py)),
            0.0,
        )
    return float(np.sum(pxy * log_term))


def decompose_similarity_from_forward(
    forward_2x3: np.ndarray,
    center: tuple[float, float] | None = None,
) -> dict[str, float]:
    """
    Approximate similarity parameters from a 2x3 forward affine.

    Returns angle (rad), scale, tx, ty relative to ``center`` (image centre
    if omitted). Shear is ignored; this is a diagnostic summary only.
    """
    M = np.asarray(forward_2x3, dtype=np.float64)
    A = M[:, :2]
    t = M[:, 2]
    sx = float(np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2))
    angle = float(np.arctan2(A[1, 0], A[0, 0]))
    if center is None:
        return {
            "angle_rad": angle,
            "scale": sx if sx > 1e-12 else 1.0,
            "tx": float(t[0]),
            "ty": float(t[1]),
        }
    c = np.asarray(center, dtype=np.float64)
    # Forward: p' = s*R*(p-c) + c + t_sim  =>  t = (I - sR)c + t_sim + ...
    # Recover translation-as-shift of centre.
    t_sim = t - (np.eye(2) - A) @ c
    return {
        "angle_rad": angle,
        "scale": sx if sx > 1e-12 else 1.0,
        "tx": float(t_sim[0]),
        "ty": float(t_sim[1]),
    }


def compute_shg_alignment_metrics(
    warped_moving: np.ndarray,
    fixed_shg: np.ndarray,
    *,
    forward_2x3: np.ndarray | None = None,
    bins: int = 50,
) -> dict[str, Any]:
    """
    Primary registration quality vs SHG (not vs MATLAB golden RGB).

    Returns histogram MI (higher better), NCC in [-1,1], and optional
    transform summary when ``forward_2x3`` is provided.
    """
    mi = histogram_mutual_information(warped_moving, fixed_shg, bins=bins)
    ncc = normalized_cross_correlation(warped_moving, fixed_shg)
    out: dict[str, Any] = {
        "shg_mi": float(mi),
        "shg_ncc": float(ncc),
        "fixed_shape": tuple(int(x) for x in np.asarray(fixed_shg).shape[:2]),
        "moving_shape": tuple(int(x) for x in np.asarray(warped_moving).shape[:2]),
    }
    if forward_2x3 is not None:
        H, W = np.asarray(fixed_shg).shape[:2]
        center = ((W - 1) / 2.0, (H - 1) / 2.0)
        params = decompose_similarity_from_forward(forward_2x3, center=center)
        out.update(params)
        out["forward_2x3"] = np.asarray(forward_2x3, dtype=np.float64).tolist()
    return out


def make_checkerboard(
    a: np.ndarray,
    b: np.ndarray,
    block: int = 64,
) -> np.ndarray:
    """
    Interleave two images in a checkerboard for visual SHG/HE alignment checks.

    Both inputs are converted to float RGB in ``[0, 1]``. Grayscale is
    replicated to 3 channels.
    """
    aa = _to_float01_rgb(a)
    bb = _to_float01_rgb(b)
    if aa.shape[:2] != bb.shape[:2]:
        raise ValueError(f"Checkerboard shape mismatch: {aa.shape} vs {bb.shape}")
    h, w = aa.shape[:2]
    out = np.empty_like(aa)
    for y in range(0, h, block):
        for x in range(0, w, block):
            iy, ix = y // block, x // block
            src = aa if (iy + ix) % 2 == 0 else bb
            out[y : y + block, x : x + block] = src[y : y + block, x : x + block]
    return out


def compute_mask_boundary_metrics(
    pred: np.ndarray,
    golden: np.ndarray,
) -> dict[str, float]:
    """
    Boundary-focused mask metrics: IoU, Dice, pixel accuracy, Hausdorff, boundary F1.

    Hausdorff uses a coarse distance-transform approximation (pixels).
    """
    from scipy import ndimage

    p = np.asarray(pred, dtype=bool)
    g = np.asarray(golden, dtype=bool)
    if p.shape != g.shape:
        raise ValueError(f"Mask shape mismatch: {p.shape} vs {g.shape}")

    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    iou = float(inter / union) if union else 1.0
    denom = int(p.sum()) + int(g.sum())
    dice = float(2.0 * inter / denom) if denom else 1.0
    acc = float((p == g).mean())

    # Boundary pixels via XOR with eroded mask.
    struct = np.ones((3, 3), dtype=bool)
    p_bound = p & ~ndimage.binary_erosion(p, structure=struct, border_value=0)
    g_bound = g & ~ndimage.binary_erosion(g, structure=struct, border_value=0)
    if not p_bound.any() and not g_bound.any():
        hausdorff = 0.0
        boundary_f1 = 1.0
    elif not p_bound.any() or not g_bound.any():
        hausdorff = float(max(p.shape))
        boundary_f1 = 0.0
    else:
        dist_g = ndimage.distance_transform_edt(~g_bound)
        dist_p = ndimage.distance_transform_edt(~p_bound)
        d_pg = float(dist_g[p_bound].max())
        d_gp = float(dist_p[g_bound].max())
        hausdorff = max(d_pg, d_gp)
        # Boundary F1: predicted boundary pixels within 2 px of golden boundary.
        tol = 2.0
        tp = int((dist_g[p_bound] <= tol).sum())
        fp = int(p_bound.sum()) - tp
        fn = int((dist_p[g_bound] > tol).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        boundary_f1 = (
            float(2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        )

    return {
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": acc,
        "hausdorff_px": float(hausdorff),
        "boundary_f1": float(boundary_f1),
    }
