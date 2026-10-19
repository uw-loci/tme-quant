"""Tumor-boundary masks from an already-registered H&E image.

``tumor_annotation_from_he`` reproduces ``BDcreationHE2.m`` (HSV, the default)
and ``BDcreationHE.m`` (RGB k-means). Registration is a separate step; this
module does not move the image.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from skimage import io

from ._he_bdc_annotation import annotate_he, annotate_he2
from ._he_bdc_reg1 import DEFAULT_KMEANS_SEED


@dataclass
class TumorAnnotationFromHEParameters:
    """Inputs for :func:`tumor_annotation_from_he`.

    Attributes
    ----------
    HEfilepath : str
        Directory that contains the already-registered H&E TIFF.
    HEfilename : str
        File name of that TIFF.
    pixelpermicron : float
        Resolution of the registered image. Mask areas and structuring
        elements are computed from this value.
    SHGfilepath : str
        Directory whose ``CA_Boundary`` folder receives the saved mask.
        When it is empty, the mask is written under ``HEfilepath`` instead.
    annotation_method : {'hsv', 'rgb_kmeans'}, optional
        ``'hsv'`` is ``BDcreationHE2.m``. ``'rgb_kmeans'`` is ``BDcreationHE.m``.
        Default is ``'hsv'``.
    kmeans_seed : int, optional
        Seed for the RGB k-means path. MATLAB's ``kmeans`` is unseeded; this
        seed replays ``rng(seed, 'twister')``. Default matches the committed
        goldens.
    """
    HEfilepath: str
    HEfilename: str
    pixelpermicron: float
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
    """Accept a parameter object or a dict of the same fields.

    Parameters
    ----------
    params : TumorAnnotationFromHEParameters or dict
        Annotation inputs.

    Returns
    -------
    TumorAnnotationFromHEParameters
    """
    if isinstance(params, TumorAnnotationFromHEParameters):
        return params
    return TumorAnnotationFromHEParameters(**params)


def _resolve_save_dir(params: TumorAnnotationFromHEParameters, *, he2_fallback: bool) -> Path:
    """Choose the ``CA_Boundary`` directory that receives the mask.

    The HE2 path tries the SHG directory and falls back to the H&E directory
    if that folder cannot be created. The RGB path uses the SHG directory only.

    Parameters
    ----------
    params : TumorAnnotationFromHEParameters
        Annotation inputs.
    he2_fallback : bool
        True for the HSV path, which may fall back to ``HEfilepath``.

    Returns
    -------
    pathlib.Path
        Directory the mask will be written into.
    """
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
    """MATLAB mask file name: ``mask for <name with HE replaced by SHG>.tif``.

    Parameters
    ----------
    he_filename : str
        Registered H&E file name.

    Returns
    -------
    str
        Mask file name.
    """
    return f"mask for {he_filename.replace('HE', 'SHG')}.tif"


def _load_he(path: Path) -> np.ndarray:
    """Load an H&E TIFF as a 3-channel image.

    A grayscale file is stacked into three identical channels. Extra channels
    beyond RGB are dropped.

    Parameters
    ----------
    path : pathlib.Path
        TIFF path.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` image in the file's dtype.
    """
    img = io.imread(str(path))
    if img.ndim == 2:
        return np.dstack([img, img, img])
    return img[..., :3]


def _save_image_uint8(path: str | Path, image: np.ndarray) -> None:
    """Write ``image`` as uint8, matching MATLAB ``imwrite`` on a double image.

    Boolean images become 0 or 255. Integer images are scaled by their own
    maximum. Floating images are clipped to ``[0, 1]`` and rounded to uint8.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path.
    image : numpy.ndarray
        Mask or image to save.
    """
    arr = image
    if arr.dtype == np.bool_:
        out = arr.astype(np.uint8) * 255
    else:
        if np.issubdtype(arr.dtype, np.integer):
            arr_float = arr.astype(np.float64)
            arr_float = arr_float / max(float(arr_float.max()), 1.0)
        else:
            arr_float = arr.astype(np.float64)
        out = np.clip(arr_float, 0.0, 1.0)
        out = np.round(out * 255.0).astype(np.uint8)
    io.imsave(str(path), out, check_contrast=False)


def tumor_annotation_from_he(
    params: TumorAnnotationFromHEParameters | dict[str, Any],
    save_output: bool = True,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a tumor-boundary mask from an already-registered H&E image.

    ``params.annotation_method`` selects the script. ``'hsv'`` follows
    ``BDcreationHE2.m``. ``'rgb_kmeans'`` follows ``BDcreationHE.m``.

    When ``save_output`` is true, the mask is written under
    ``<SHGfilepath>/CA_Boundary/``. The HSV path writes a boolean mask. The
    RGB path writes the uint8 mask MATLAB saves (0 or 255).

    Parameters
    ----------
    params : TumorAnnotationFromHEParameters or dict
        Paths, resolution, method, and k-means seed.
    save_output : bool, optional
        Write the mask TIFF. Default is true.
    return_debug : bool, optional
        When true, also return the intermediate arrays (adjusted image,
        masks, cluster labels). Default is false.

    Returns
    -------
    numpy.ndarray or tuple
        Boolean mask. If ``return_debug`` is true, ``(mask, debug)``.

    Raises
    ------
    ValueError
        If ``annotation_method`` is not ``'hsv'`` or ``'rgb_kmeans'``.
    """
    p = _to_params(params)
    he_path = Path(p.HEfilepath) / p.HEfilename
    he_raw = _load_he(he_path)

    method = (p.annotation_method or "hsv").lower()
    if method == "hsv":
        bd_mask, debug = annotate_he2(he_raw, p.pixelpermicron)
        save_arr: np.ndarray = bd_mask
        he2_fallback = True
    elif method == "rgb_kmeans":
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
        _save_image_uint8(save_dir / _resolve_mask_name(p.HEfilename), save_arr)

    if not return_debug:
        return bd_mask.astype(bool)
    return bd_mask.astype(bool), debug


def BDcreationHE2(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Run ``BDcreationHE2.m`` and save the HSV tumor mask.

    Parameters
    ----------
    BDCparameters : TumorAnnotationFromHEParameters or dict
        Annotation inputs. ``annotation_method`` is forced to ``'hsv'``.

    Returns
    -------
    numpy.ndarray
        Boolean tumor mask.
    """
    p = replace(_to_params(BDCparameters), annotation_method="hsv")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)


def BDcreationHE(
    BDCparameters: TumorAnnotationFromHEParameters | dict[str, Any],
) -> np.ndarray:
    """Run ``BDcreationHE.m`` and save the RGB k-means tumor mask.

    Parameters
    ----------
    BDCparameters : TumorAnnotationFromHEParameters or dict
        Annotation inputs. ``annotation_method`` is forced to ``'rgb_kmeans'``.

    Returns
    -------
    numpy.ndarray
        Boolean tumor mask. The file on disk is the uint8 0/255 mask.
    """
    p = replace(_to_params(BDCparameters), annotation_method="rgb_kmeans")
    return tumor_annotation_from_he(p, save_output=True, return_debug=False)
