from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from skimage import io

from ._he_bdc_annotation import annotate_he, annotate_he2
from ._he_bdc_common import save_image_uint8
from ._he_bdc_reg1 import DEFAULT_KMEANS_SEED

# Re-export morphology constants so existing imports keep working.
from ._he_bdc_annotation import (  # noqa: F401
    BACKGROUND_HOLE_MIN_AREA_MULTIPLIER,
    EPITH_BINARY_THRESHOLD,
    EPITH_DILATION_RADIUS_MULTIPLIER,
    FINAL_MASK_DILATION_RADIUS_MULTIPLIER,
    FINAL_MASK_GAUSSIAN_KERNEL_SIZE,
    FINAL_MASK_GAUSSIAN_SIGMA,
    HE_RGB_DISK_RADIUS_MULTIPLIER,
    HE_RGB_N_COLORS,
    HE_RGB_PAD,
    TUMOR_MASK_MIN_AREA_MULTIPLIER,
)


@dataclass
class TumorAnnotationFromHEParameters:
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
    areaThreshold: float
    SHGfilepath: str
    # "hsv" (default): BDcreationHE2.m HSV path.
    # "rgb_kmeans": BDcreationHE.m decorrstretch + RGB k-means path.
    annotation_method: str = "hsv"
    # BDcreationHE.m never seeds kmeans. This seed replays
    # ``rng(kmeans_seed,'twister')`` immediately before kmeans, matching the
    # dump harness (same policy as registration's DEFAULT_KMEANS_SEED).
    kmeans_seed: int = DEFAULT_KMEANS_SEED


def _to_params(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
) -> TumorAnnotationFromHEParameters:
    if isinstance(params, TumorAnnotationFromHEParameters):
        return params
    return TumorAnnotationFromHEParameters(**params)


def _resolve_save_dir(params: TumorAnnotationFromHEParameters, *, he2_fallback: bool) -> Path:
    primary = Path(params.SHGfilepath) / "CA_Boundary" if params.SHGfilepath else Path(params.HEfilepath) / "CA_Boundary"
    if not he2_fallback:
        return primary
    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    except Exception:
        fallback = Path(params.HEfilepath) / "CA_Boundary"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _resolve_mask_name(he_filename: str) -> str:
    return f"mask for {he_filename.replace('HE', 'SHG')}.tif"


def _load_he(path: Path) -> np.ndarray:
    img = io.imread(str(path))
    if img.ndim == 2:
        return np.dstack([img, img, img])
    return img[..., :3]


def tumor_annotation_from_he(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Python conversion of MATLAB BDcreationHE2.m (default) / BDcreationHE.m.

    Generates a tumor boundary mask from an already-registered HE image.
    """
    p = _to_params(params)
    he_path = Path(p.HEfilepath) / p.HEfilename
    he_raw = _load_he(he_path)

    method = (p.annotation_method or "hsv").lower()
    if method in ("hsv", "he2"):
        bd_mask, debug = annotate_he2(he_raw, p.pixelpermicron)
        save_arr: np.ndarray = bd_mask
        he2_fallback = True
    elif method in ("rgb_kmeans", "rgb", "he"):
        bd_mask, debug = annotate_he(he_raw, p.pixelpermicron, kmeans_seed=int(p.kmeans_seed))
        save_arr = np.asarray(debug["BDmask"])
        he2_fallback = False
    else:
        raise ValueError(
            f"Unknown annotation_method={p.annotation_method!r}; "
            f"expected 'hsv' or 'rgb_kmeans'."
        )

    if save_output:
        save_dir = _resolve_save_dir(p, he2_fallback=he2_fallback)
        if not he2_fallback:
            save_dir.mkdir(parents=True, exist_ok=True)
        save_image_uint8(save_dir / _resolve_mask_name(p.HEfilename), save_arr)

    if not return_debug:
        return bd_mask.astype(bool)
    return bd_mask.astype(bool), debug


def BDcreationHE2(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper retaining MATLAB function name (HSV path)."""
    p = replace(_to_params(BDCparameters), annotation_method="hsv")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)


def BDcreationHE(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Compatibility wrapper for MATLAB ``BDcreationHE.m`` (RGB k-means path)."""
    p = replace(_to_params(BDCparameters), annotation_method="rgb_kmeans")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)
