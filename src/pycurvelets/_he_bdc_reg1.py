"""
MATLAB-exact port of the *preprocessing* half of ``BDcreation_reg.m`` ("reg1"),
the RGB/decorrstretch + LAB k-means variant of the H&E->SHG registration.

``BDcreation_reg2.m`` (HSV masks) is ported in :mod:`pycurvelets.SHG_HE_registration`
/ :mod:`pycurvelets._he_bdc_common`. reg1 differs structurally:

* the SHG is **not** resized by ``2/ppm``; it is ``imadjust``-ed in place and
  keeps its native integer type, and ``imregtform`` sees ``double(uint8)`` (0..255);
* the H&E is globally rescaled by its max, quantised to uint8, ``decorrstretch``-ed,
  then thresholded with fixed RGB cuts; the eosin/collagen pixels are clustered in
  CIELAB ``a*b*`` with ``kmeans(..., 3, 'Replicates', 3)`` and the reddest cluster
  becomes the moving image (continuous grey levels, not a binary mask);
* the moving image is ``imresize``-d to the SHG grid; the output is the H&E resized to
  the SHG grid and warped with ``FillValues=255`` (no final resize).

Determinism
-----------
``BDcreation_reg.m`` never seeds MATLAB's RNG, and ``kmeans`` (k-means++ init) has
several local optima on real H&E data, so MATLAB's own output depends on the
session's RNG state (3 optima with ~82/13/5 % frequency on ``patient_02_roi4``;
the committed goldens correspond to the *5 %* one). This module calls
:func:`pycurvelets._matlab_util.kmeans`, which replays
MATLAB's ``kmeans`` exactly - k-means++ via
``datasample`` on the Mersenne-Twister stream (``numpy.random.RandomState`` is
bit-identical to ``rng(seed,'twister')``), Lloyd batch updates with MATLAB's tie
rules - so a given ``kmeans_seed`` gives the same labels MATLAB gives for
``rng(kmeans_seed,'twister')``. :data:`DEFAULT_KMEANS_SEED` is a seed that lands in
the goldens' optimum.

Every intermediate is returned under its MATLAB variable name so a parity mismatch
can be attributed to a single step (``archive/matlab_parity/dump_bdc_reg1.m`` dumps the
same names from MATLAB).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from ._he_bdc_common import (
    remove_small_components,
    rgb_threshold_masks_uint8,
)
from ._matlab_util import (
    decorrstretch_uint8,
    fspecial_gaussian,
    im2uint8,
    imadjust_auto_integer,
    imadjust_double_rgb,
    imfilter,
    imresize,
    kmeans,
    rgb2gray,
    rgb2gray_uint8,
    srgb2lab_uint8,
    strel_disk,
)

# Seed for the exact kmeans replay. Chosen so that the k-means lands in the same
# local optimum as the reference outputs shipped with the tests
# (HE_registered_for_reg1_test6b_ppm3 / _test9_ppm2p6); see module docstring.
DEFAULT_KMEANS_SEED = 28


# ---------------------------------------------------------------------------
# BDcreation_reg.m preprocessing
# ---------------------------------------------------------------------------


def bdcreation_reg1_preprocess(
    he_u8: np.ndarray,
    shg_int: np.ndarray,
    pixelpermicron: float,
    *,
    kmeans_seed: int = DEFAULT_KMEANS_SEED,
) -> dict[str, Any]:
    """
    ``BDcreation_reg.m`` from ``imread`` up to ``HEmoving`` / ``fixedSHG``.

    Parameters
    ----------
    he_u8
        H&E RGB as read from disk (uint8, ``(H, W, 3)``).
    shg_int
        SHG as read from disk (uint8 or uint16, 2-D; an RGB SHG is converted with
        ``rgb2gray`` first - MATLAB's ``imadjust(I)`` would reject it).
    pixelpermicron
        Only drives kernel/area sizes here (no ppm-based resizing in reg1).
    kmeans_seed
        RNG seed for the exact ``kmeans`` replay (see module docstring).

    Returns
    -------
    dict
        Intermediates keyed by MATLAB variable name, plus ``HEmoving`` (double, SHG
        grid) and ``fixedSHG_double`` (``double(fixedSHG)``, 0..255 for uint8).
    """
    he = np.asarray(he_u8)
    if he.dtype != np.uint8 or he.ndim != 3:
        raise TypeError(f"reg1 expects a uint8 RGB H&E, got {he.dtype} {he.shape}")
    he = he[..., :3]
    shg = np.asarray(shg_int)
    if shg.ndim == 3:
        shg = rgb2gray_uint8(shg[..., :3]) if shg.dtype == np.uint8 else shg[..., 0]
    ppm = float(pixelpermicron)
    out: dict[str, Any] = {"pixelpermicron": ppm, "kmeans_seed": int(kmeans_seed)}

    fixedSHG = imadjust_auto_integer(shg)
    out["fixedSHG"] = fixedSHG

    HEdata = he.astype(np.float64) / 255.0
    max_HEdata = float(HEdata.max())
    HEdata_adj0 = im2uint8(imadjust_double_rgb(HEdata, 0.0, max_HEdata))
    out["HEdata"], out["max_HEdata"], out["HEdata_adj0"] = HEdata, max_HEdata, HEdata_adj0

    S = decorrstretch_uint8(HEdata_adj0, tol=0.01)
    out["S"] = S
    nuclei_cond, red_cond = rgb_threshold_masks_uint8(S)
    HEdata_nuclei = np.where(nuclei_cond[..., None], S, np.uint8(0))
    # Quirk preserved from the MATLAB loop: only the red channel is taken from the
    # decorrelated image; green/blue keep the pre-decorrstretch values.
    HEdata_red = np.where(red_cond[..., None], np.dstack([S[..., 0], HEdata_adj0[..., 1], HEdata_adj0[..., 2]]), np.uint8(0))
    out["HEdata_nuclei"], out["HEdata_red"] = HEdata_nuclei, HEdata_red

    lab_HEdata = srgb2lab_uint8(HEdata_red)
    out["lab_HEdata"] = lab_HEdata
    nrows, ncols = lab_HEdata.shape[:2]
    ab = np.column_stack(
        [lab_HEdata[..., 1].astype(np.float64).ravel(order="F"), lab_HEdata[..., 2].astype(np.float64).ravel(order="F")]
    )
    labels0, centers, sumd, km_debug = kmeans(ab, 3, seed=kmeans_seed, replicates=3)
    pixel_labels = (labels0 + 1).reshape((nrows, ncols), order="F")
    order = np.argsort(centers.mean(axis=1), kind="stable")
    collagen_cluster = int(order[-1]) + 1
    out["pixel_labels"], out["cluster_center"], out["collagen_cluster"] = pixel_labels, centers, collagen_cluster
    out["kmeans_debug"] = {**km_debug, "sumd": sumd.tolist()}

    collagen_rgb = np.where((pixel_labels == collagen_cluster)[..., None], HEdata, 0.0)
    HE_collagen = rgb2gray(collagen_rgb)
    gray_nuclei = rgb2gray_uint8(HEdata_nuclei).astype(np.float64) / 255.0
    out["HE_collagen"], out["gray_nuclei"] = HE_collagen, gray_nuclei

    h_nuclei = fspecial_gaussian(max(1, int(np.floor(ppm))), 0.5)
    nuclei_filtered = imfilter(gray_nuclei, h_nuclei, boundary="zero")
    BW_nuclei = nuclei_filtered > 0.001
    BW_nuclei_discard = remove_small_components(BW_nuclei, int(np.ceil(50.0 * ppm**2)))
    se = strel_disk(int(np.floor(ppm)))
    BW_nuclei_dilated = ndimage.binary_dilation(BW_nuclei_discard, structure=se)
    BW_nuclei_filled = ndimage.binary_fill_holes(BW_nuclei_dilated)
    out.update(
        h_nuclei=h_nuclei, nuclei_filtered=nuclei_filtered, BW_nuclei=BW_nuclei,
        BW_nuclei_discard=BW_nuclei_discard, BW_nuclei_dilated=BW_nuclei_dilated,
        BW_nuclei_filled=BW_nuclei_filled,
    )

    HE_collagen_exclude0 = HE_collagen * (~BW_nuclei_filled)
    HE_collagen_BW = HE_collagen_exclude0 > 0.01
    BW_discard = remove_small_components(HE_collagen_BW, int(np.ceil(ppm**2)))
    HE_collagen_exclude = HE_collagen_exclude0 * BW_discard
    out.update(
        HE_collagen_exclude0=HE_collagen_exclude0, HE_collagen_BW=HE_collagen_BW,
        BW_discard=BW_discard, HE_collagen_exclude=HE_collagen_exclude,
    )

    HEmoving = imresize(HE_collagen_exclude, output_shape=fixedSHG.shape, method="bicubic")
    out["HEmoving"] = HEmoving
    out["fixedSHG_double"] = fixedSHG.astype(np.float64)
    out["mask_coverage"] = float(np.mean(HEmoving > 0))
    return out
