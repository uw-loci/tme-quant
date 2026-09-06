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
session's RNG state (see ``tests/matlab_parity/probe_reg1_kmeans.m``: 3 optima with
~82/13/5 % frequency on ``patient_02_roi4``; the committed goldens correspond to
the *5 %* one). This module replays MATLAB's ``kmeans`` exactly - k-means++ via
``datasample`` on the Mersenne-Twister stream (``numpy.random.RandomState`` is
bit-identical to ``rng(seed,'twister')``), Lloyd batch updates with MATLAB's tie
rules - so a given ``kmeans_seed`` gives the same labels MATLAB gives for
``rng(kmeans_seed,'twister')``. :data:`DEFAULT_KMEANS_SEED` is a seed that lands in
the goldens' optimum.

Every intermediate is returned under its MATLAB variable name so a parity mismatch
can be attributed to a single step (``tests/matlab_parity/dump_bdc_reg1.m`` dumps the
same names from MATLAB).
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import numpy as np
from scipy import ndimage

from ._he_bdc_common import (
    matlab_fspecial_gaussian,
    matlab_imfilter,
    matlab_rgb2gray,
    remove_small_components,
)
from ._matlab_imresize import matlab_imresize

# Seed for the exact kmeans replay. Chosen so that the k-means lands in the same
# local optimum as the reference outputs shipped with the tests
# (HE_registered_for_reg1_test6b_ppm3 / _test9_ppm2p6); see module docstring.
DEFAULT_KMEANS_SEED = 28

# Full-precision Rec.601 weights MATLAB's rgb2gray uses (not 0.2989/0.5870/0.1140).
_RGB2GRAY_COEF = (0.298936021293775, 0.587043074451121, 0.114020904255103)


# ---------------------------------------------------------------------------
# Elementary MATLAB primitives
# ---------------------------------------------------------------------------


def matlab_im2uint8(x: np.ndarray) -> np.ndarray:
    """``im2uint8`` on a double image: ``uint8(round(x*255))`` with saturation (half-up)."""
    v = np.asarray(x, dtype=np.float64) * 255.0
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def matlab_rgb2gray_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    """``rgb2gray`` on a uint8 RGB image: weighted sum rounded half-up to uint8."""
    a = np.asarray(rgb_u8, dtype=np.float64)
    lin = _RGB2GRAY_COEF[0] * a[..., 0] + _RGB2GRAY_COEF[1] * a[..., 1] + _RGB2GRAY_COEF[2] * a[..., 2]
    return np.clip(np.floor(lin + 0.5), 0, 255).astype(np.uint8)


def matlab_imhist_counts(plane: np.ndarray, nbins: int) -> np.ndarray:
    """
    ``imhist(I, nbins)`` counts. Integer images bin by value; double images use
    ``imhistc`` binning ``floor(x*(nbins-1)+0.5)`` with out-of-range values clamped to
    the end bins (relevant for ``decorrstretch``, whose double output exceeds [0,1]).
    """
    p = np.asarray(plane)
    if np.issubdtype(p.dtype, np.integer):
        idx = p.astype(np.int64)
    else:
        idx = np.floor(p.astype(np.float64) * (nbins - 1) + 0.5)
        idx = np.nan_to_num(idx, nan=0.0).astype(np.int64)
    idx = np.clip(idx, 0, nbins - 1)
    return np.bincount(idx.ravel(), minlength=nbins).astype(np.float64)


def matlab_stretchlim(img: np.ndarray, tol: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """
    ``stretchlim(img, tol)`` -> ``(2, nplanes)`` ``[low; high]`` in [0, 1].

    ``nbins`` is 256 for uint8 and 65536 otherwise (MATLAB uses 65536 for uint16
    *and* double). ``ilow`` is the first bin whose CDF exceeds ``tol_low``; ``ihigh``
    the first whose CDF reaches ``tol_high``; a flat plane maps to ``[0; 1]``.
    """
    a = np.asarray(img)
    planes = a[..., None] if a.ndim == 2 else a
    nbins = 256 if planes.dtype == np.uint8 else 65536
    tol_low, tol_high = float(tol[0]), float(tol[1])
    out = np.zeros((2, planes.shape[-1]), dtype=np.float64)
    if not tol_low < tol_high:
        out[1, :] = 1.0
        return out
    for i in range(planes.shape[-1]):
        counts = matlab_imhist_counts(planes[..., i], nbins)
        cdf = np.cumsum(counts) / counts.sum()
        ilow = int(np.argmax(cdf > tol_low)) + 1          # 1-based like MATLAB find
        ihigh = int(np.argmax(cdf >= tol_high)) + 1
        if ilow == ihigh:
            ilow, ihigh = 1, nbins
        out[0, i] = (ilow - 1) / (nbins - 1)
        out[1, i] = (ihigh - 1) / (nbins - 1)
    return out


def _adjust_array(x: np.ndarray, low_in: float, high_in: float, low_out: float = 0.0, high_out: float = 1.0) -> np.ndarray:
    """``imadjust``'s ``adjustArray`` with gamma 1: clip to ``[low_in, high_in]`` then rescale."""
    xc = np.maximum(low_in, np.minimum(high_in, np.asarray(x, dtype=np.float64)))
    out = (xc - low_in) / (high_in - low_in)
    return out * (high_out - low_out) + low_out


def matlab_imadjust_auto_integer(img: np.ndarray) -> np.ndarray:
    """
    ``imadjust(I)`` (single argument) on a 2-D uint8/uint16 image: ``stretchlim``
    limits, linear stretch, same integer class out. Implemented value-wise, which is
    what MATLAB's LUT path (``numel > 65536``) and direct path both reduce to.
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"imadjust(I) with one argument needs a 2-D image, got {a.shape}")
    if a.dtype == np.uint8:
        scale = 255.0
    elif a.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"expected uint8/uint16 SHG, got {a.dtype}")
    low, high = matlab_stretchlim(a)[:, 0]
    lut_in = np.arange(int(scale) + 1, dtype=np.float64) / scale
    lut = np.clip(np.floor(_adjust_array(lut_in, low, high) * scale + 0.5), 0, scale)
    return lut.astype(a.dtype)[a]


def matlab_imadjust_double_rgb(img: np.ndarray, low_in: float, high_in: float) -> np.ndarray:
    """``imadjust(RGB_double, [low_in high_in], [0 1])`` - same limits on every channel."""
    return _adjust_array(img, float(low_in), float(high_in))


def _matlab_pinv_diag(D: np.ndarray) -> np.ndarray:
    """``fitdecorrtrans``'s local ``pinv`` for a diagonal matrix (tolerance ``n*max*sqrt(eps)``)."""
    d = np.diag(D).astype(np.float64)
    tol = len(d) * d.max() * np.sqrt(np.finfo(np.float64).eps)
    s = np.zeros_like(d)
    keep = d > tol
    s[keep] = 1.0 / d[keep]
    return np.diag(s)


def matlab_decorrstretch_uint8(rgb_u8: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    ``decorrstretch(uint8 RGB, 'tol', tol)`` (default ``Mode='correlation'``,
    target mean/sigma taken from the data). Steps as in ``decorrstretch.m`` /
    ``private/fitdecorrtrans.m``: sample covariance, correlation whitening
    ``T = pinv(S) V W V' S`` renormalised to the target sigma, ``S = A*T + offset``,
    then ``stretchlim`` (65536 bins on the double result) + linear stretch, clip,
    ``im2uint8``.
    """
    a = np.asarray(rgb_u8)
    if a.dtype != np.uint8 or a.ndim != 3:
        raise TypeError(f"expected uint8 (H, W, 3), got {a.dtype} {a.shape}")
    h, w, nb = a.shape
    A = a.reshape(-1, nb).astype(np.float64) / 255.0
    n = A.shape[0]
    mean_b = A.mean(axis=0)
    cov = (A.T @ A - (n * mean_b[:, None]) * mean_b[None, :]) / (n - 1)

    S = np.diag(np.sqrt(np.diag(cov)))
    target_sigma = S
    pS = _matlab_pinv_diag(S)
    corr = pS @ cov @ pS
    np.fill_diagonal(corr, 1.0)
    evals, V = np.linalg.eigh(corr)
    D = np.diag(np.maximum(evals, 0.0))
    W = np.sqrt(_matlab_pinv_diag(D))
    T = pS @ V @ W @ V.T @ target_sigma
    T = T @ _matlab_pinv_diag(np.diag(np.sqrt(np.diag(T.T @ cov @ T)))) @ target_sigma
    offset = mean_b - mean_b @ T
    Sd = (A @ T + offset).reshape(h, w, nb)

    tol_lh = (float(tol), 1.0 - float(tol))
    low_high = matlab_stretchlim(Sd, tol_lh)
    out = np.empty_like(Sd)
    for p in range(nb):
        out[..., p] = _adjust_array(Sd[..., p], low_high[0, p], low_high[1, p])
    out = np.clip(out, 0.0, 1.0)
    return matlab_im2uint8(out)


def _load_srgb2lab_components() -> dict[str, np.ndarray]:
    with resources.files("pycurvelets").joinpath("data/matlab_srgb2lab_components.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: np.array(z[k]) for k in z.files}


_SRGB2LAB: dict[str, np.ndarray] | None = None


def matlab_srgb2lab_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    """
    ``applycform(uint8 RGB, makecform('srgb2lab'))`` -> uint8 Lab.

    Closed-form evaluation of MATLAB's ICC pipeline with constants exported from
    MATLAB (``tests/matlab_parity/dump_srgb2lab_components.m``): per-channel sRGB.icm
    TRC (spline-interpolated 1024-entry curve tabulated at the 256 uint8 levels) ->
    colorant matrix -> Bradford adaptation D50 -> ICC white -> ``xyz2lab`` -> uint8
    encoding ``round([255 L/100, a+128, b+128])``. Validated exactly on 200k random
    triplets and the fixture images.
    """
    global _SRGB2LAB
    if _SRGB2LAB is None:
        _SRGB2LAB = _load_srgb2lab_components()
    c = _SRGB2LAB
    a = np.asarray(rgb_u8)
    if a.dtype != np.uint8 or a.shape[-1] != 3:
        raise TypeError(f"expected uint8 (..., 3), got {a.dtype} {a.shape}")
    flat = a.reshape(-1, 3).astype(np.int64)
    lin = np.empty(flat.shape, dtype=np.float64)
    for i in range(3):
        lin[:, i] = c["trc256"][flat[:, i], i]
    xyz = (lin @ c["colorants"]) @ c["adapter"].T
    xyzn = xyz / c["whitepoint"][None, :]
    f = np.power(xyzn, 1.0 / 3.0)
    small = xyzn <= 216.0 / 24389.0
    f[small] = (841.0 / 108.0) * xyzn[small] + 16.0 / 116.0
    L = 116.0 * f[:, 1] - 16.0
    A = 500.0 * (f[:, 0] - f[:, 1])
    B = 200.0 * (f[:, 1] - f[:, 2])
    lab = np.stack([255.0 * L / 100.0, A + 128.0, B + 128.0], axis=1)
    lab = np.clip(np.floor(lab + 0.5), 0, 255)
    return lab.astype(np.uint8).reshape(a.shape)


def matlab_strel_disk(r: int, n: int = 4) -> np.ndarray:
    """
    ``strel('disk', r, n).Neighborhood`` (default ``n=4``).

    For ``r < 3`` MATLAB uses the exact Euclidean disk. Otherwise it is Adams'
    radial decomposition into ``n`` periodic lines (``floor``-ed lengths), dilated
    together, then padded with horizontal/vertical lines to recover the radius. This
    is *not* the Euclidean disk: ``r=3`` is a flat 5x5 square, ``r=4`` a 7x7 octagon.
    """
    r = int(r)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    if r < 3 or n == 0:
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        return (xx**2 + yy**2) <= r**2
    if n == 4:
        v = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]], dtype=np.float64)
    elif n == 6:
        v = np.array([[1, 0], [1, 2], [2, 1], [0, 1], [-1, 2], [-2, 1]], dtype=np.float64)
    elif n == 8:
        v = np.array([[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1]], dtype=np.float64)
    else:
        raise ValueError("n must be 0, 4, 6 or 8")
    theta = np.pi / (2 * n)
    k = 2 * r / (1.0 / np.tan(theta) + 1.0 / np.sin(theta))

    def minkowski(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(p[0] + q[0], p[1] + q[1]) for p in a for q in b}

    offsets: set[tuple[int, int]] = {(0, 0)}
    for q in range(n):
        rp = int(np.floor(k / np.linalg.norm(v[q])))
        line = {(int(p * v[q, 0]), int(p * v[q, 1])) for p in range(-rp, rp + 1)}
        offsets = minkowski(offsets, line)
    max_radius = max(abs(dr) for dr, _ in offsets)
    radial_difference = r - max_radius
    length = 2 * (radial_difference - 1) + 1
    if length >= 3:
        half = (length - 1) // 2
        offsets = minkowski(offsets, {(0, c) for c in range(-half, half + 1)})
        offsets = minkowski(offsets, {(rr, 0) for rr in range(-half, half + 1)})
    ext_r = max(abs(dr) for dr, _ in offsets)
    ext_c = max(abs(dc) for _, dc in offsets)
    nhood = np.zeros((2 * ext_r + 1, 2 * ext_c + 1), dtype=bool)
    for dr, dc in offsets:
        nhood[dr + ext_r, dc + ext_c] = True
    return nhood


# ---------------------------------------------------------------------------
# kmeans replay
# ---------------------------------------------------------------------------


def _sqdist(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance of every row of ``X`` to one centre (explicit differences, as pdist2mex 'sqe')."""
    d = X - c[None, :]
    return np.einsum("ij,ij->i", d, d)


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    MATLAB ``'Start','plus'``: first seed uniform via ``datasample`` (=``randi``,
    i.e. ``floor(rand*n)``), then D^2-weighted draws via ``datasample(...,'Replace',
    false,'Weights',p)`` which (for one sample) inverts the cumulative weights with a
    single ``rand`` (``histcounts`` semantics: bin ``i`` with ``edges[i] <= r < edges[i+1]``).
    """
    n = X.shape[0]
    C = np.empty((k, X.shape[1]), dtype=np.float64)
    C[0] = X[int(np.floor(rng.random_sample() * n))]
    min_dist = np.full(n, np.inf)
    for ii in range(1, k):
        min_dist = np.minimum(min_dist, _sqdist(X, C[ii - 1]))
        denominator = float(np.sum(min_dist))
        if denominator == 0.0 or not np.isfinite(denominator):
            raise RuntimeError(
                "kmeans++ degenerate data (all points coincide); MATLAB would fall "
                "back to randperm sampling, which this replay does not implement."
            )
        p = min_dist / denominator
        edges = np.minimum(np.concatenate(([0.0], np.cumsum(p))), 1.0)
        edges[-1] = 1.0
        r = rng.random_sample()
        idx = int(np.searchsorted(edges, r, side="right")) - 1
        idx = min(max(idx, 0), n - 1)
        C[ii] = X[idx]
    return C


def _lloyd_batch_matlab(
    X: np.ndarray, C: np.ndarray, max_iter: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    """
    ``kmeansHelper``'s batch phase (``OnlinePhase='off'``, ``EmptyAction='singleton'``):
    recompute centroids of changed clusters, back out if the objective did not
    decrease, reassign with ties resolved in favour of not moving.
    """
    n, k = X.shape[0], C.shape[0]
    D = np.column_stack([_sqdist(X, C[j]) for j in range(k)])
    idx = np.argmin(D, axis=1)
    m = np.bincount(idx, minlength=k)
    changed = np.arange(k)
    previdx = np.zeros(n, dtype=np.int64)
    prevtotsumD = np.inf
    rows = np.arange(n)
    it = 0
    converged = False

    def centroids(clusters: np.ndarray) -> None:
        for j in clusters:
            members = idx == j
            m[j] = int(members.sum())
            C[j] = X[members].sum(axis=0) / m[j] if m[j] > 0 else np.nan
            D[:, j] = _sqdist(X, C[j]) if m[j] > 0 else np.nan

    while True:
        it += 1
        centroids(changed)
        empties = [j for j in changed if m[j] == 0]
        for j in empties:
            d = D[rows, idx]
            lonely = int(np.argmax(d))
            src = int(idx[lonely])
            if m[src] < 2:
                src = int(np.argmax(m > 1))
                lonely = int(np.argmax(idx == src))
            C[j] = X[lonely]
            m[j] = 1
            idx[lonely] = j
            D[:, j] = _sqdist(X, C[j])
            centroids(np.array([src]))
            changed = np.unique(np.concatenate((changed, [src])))
        totsumD = float(np.sum(D[rows, idx]))
        if prevtotsumD <= totsumD:
            idx = previdx
            centroids(changed)
            it -= 1
            break
        if it >= max_iter:
            break
        previdx = idx.copy()
        prevtotsumD = totsumD
        nidx = np.argmin(D, axis=1)
        d = D[rows, nidx]
        moved = np.flatnonzero(nidx != previdx)
        if moved.size:
            moved = moved[D[moved, previdx[moved]] > d[moved]]
        if moved.size == 0:
            converged = True
            break
        idx = idx.copy()
        idx[moved] = nidx[moved]
        changed = np.unique(np.concatenate((idx[moved], previdx[moved])))

    nonempty = np.flatnonzero(m > 0)
    for j in nonempty:
        D[:, j] = _sqdist(X, C[j])
    d = D[rows, idx]
    sumD = np.bincount(idx, weights=d, minlength=k)
    totsumD = float(np.sum(sumD[nonempty]))
    if not converged and it >= max_iter:
        import warnings

        warnings.warn(f"kmeans replay failed to converge in {max_iter} iterations", RuntimeWarning, stacklevel=3)
    return idx, C, totsumD, sumD, it


def matlab_kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int,
    replicates: int = 3,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Replay ``rng(seed,'twister'); [idx, C, sumd] = kmeans(X, k, 'distance',
    'sqEuclidean', 'Replicates', replicates)`` bit-for-bit for well-separated data.

    ``X`` is ``(n, p)`` with rows in MATLAB's observation order (column-major image
    flatten for ``reshape(ab, nrows*ncols, 2)``). Returns 0-based ``idx``, ``C`` (k, p),
    ``sumd`` (k,), and a debug dict with every replicate's objective.
    """
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    best: tuple | None = None
    reps: list[dict[str, Any]] = []
    for rep in range(int(replicates)):
        C0 = _kmeanspp_init(Xd, k, rng)
        idx, C, totsumD, sumD, iters = _lloyd_batch_matlab(Xd, C0.copy(), max_iter)
        reps.append({"replicate": rep + 1, "totsumD": totsumD, "iterations": iters, "init": C0.tolist()})
        if best is None or totsumD < best[2]:
            best = (idx, C, totsumD, sumD)
    assert best is not None
    debug = {"seed": int(seed), "replicates": reps, "best_totsumD": best[2]}
    return best[0], best[1], best[3], debug


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
        shg = matlab_rgb2gray_uint8(shg[..., :3]) if shg.dtype == np.uint8 else shg[..., 0]
    ppm = float(pixelpermicron)
    out: dict[str, Any] = {"pixelpermicron": ppm, "kmeans_seed": int(kmeans_seed)}

    fixedSHG = matlab_imadjust_auto_integer(shg)
    out["fixedSHG"] = fixedSHG

    HEdata = he.astype(np.float64) / 255.0
    max_HEdata = float(HEdata.max())
    HEdata_adj0 = matlab_im2uint8(matlab_imadjust_double_rgb(HEdata, 0.0, max_HEdata))
    out["HEdata"], out["max_HEdata"], out["HEdata_adj0"] = HEdata, max_HEdata, HEdata_adj0

    S = matlab_decorrstretch_uint8(HEdata_adj0, tol=0.01)
    out["S"] = S
    Sr, Sg, Sb = (S[..., i].astype(np.int64) for i in range(3))
    nuclei_cond = (Sr < 120) & (Sg > 150) & (Sb < 120)
    HEdata_nuclei = np.where(nuclei_cond[..., None], S, np.uint8(0))
    # Quirk preserved from the MATLAB loop: only the red channel is taken from the
    # decorrelated image; green/blue keep the pre-decorrstretch values.
    red_cond = (Sr > 200) & (Sg < 100) & (Sb > 100)
    HEdata_red = np.where(red_cond[..., None], np.dstack([S[..., 0], HEdata_adj0[..., 1], HEdata_adj0[..., 2]]), np.uint8(0))
    out["HEdata_nuclei"], out["HEdata_red"] = HEdata_nuclei, HEdata_red

    lab_HEdata = matlab_srgb2lab_uint8(HEdata_red)
    out["lab_HEdata"] = lab_HEdata
    nrows, ncols = lab_HEdata.shape[:2]
    ab = np.column_stack(
        [lab_HEdata[..., 1].astype(np.float64).ravel(order="F"), lab_HEdata[..., 2].astype(np.float64).ravel(order="F")]
    )
    labels0, centers, sumd, km_debug = matlab_kmeans(ab, 3, seed=kmeans_seed, replicates=3)
    pixel_labels = (labels0 + 1).reshape((nrows, ncols), order="F")
    order = np.argsort(centers.mean(axis=1), kind="stable")
    collagen_cluster = int(order[-1]) + 1
    out["pixel_labels"], out["cluster_center"], out["collagen_cluster"] = pixel_labels, centers, collagen_cluster
    out["kmeans_debug"] = {**km_debug, "sumd": sumd.tolist()}

    collagen_rgb = np.where((pixel_labels == collagen_cluster)[..., None], HEdata, 0.0)
    HE_collagen = matlab_rgb2gray(collagen_rgb)
    gray_nuclei = matlab_rgb2gray_uint8(HEdata_nuclei).astype(np.float64) / 255.0
    out["HE_collagen"], out["gray_nuclei"] = HE_collagen, gray_nuclei

    h_nuclei = matlab_fspecial_gaussian(max(1, int(np.floor(ppm))), 0.5)
    nuclei_filtered = matlab_imfilter(gray_nuclei, h_nuclei, boundary="zero")
    BW_nuclei = nuclei_filtered > 0.001
    BW_nuclei_discard = remove_small_components(BW_nuclei, int(np.ceil(50.0 * ppm**2)))
    se = matlab_strel_disk(int(np.floor(ppm)))
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

    HEmoving = matlab_imresize(HE_collagen_exclude, output_shape=fixedSHG.shape, method="bicubic")
    out["HEmoving"] = HEmoving
    out["fixedSHG_double"] = fixedSHG.astype(np.float64)
    out["mask_coverage"] = float(np.mean(HEmoving > 0))
    return out
