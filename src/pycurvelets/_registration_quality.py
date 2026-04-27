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
