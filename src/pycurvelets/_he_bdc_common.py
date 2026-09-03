from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import color, filters, io, morphology
from sklearn.cluster import KMeans

from ._matlab_imresize import matlab_imresize

# MATLAB-derived HSV threshold defaults from:
# - CurveAlign_CT-FIRE/BDcreation_reg2.m
# - CurveAlign_CT-FIRE/BDcreationHE2.m
NUCLEI_HUE_MIN = 0.500
NUCLEI_HUE_MAX = 0.790
COLLAGEN_HUE_MIN = 0.837
COLLAGEN_HUE_MAX = 0.066  # wrapped hue range (>= min OR <= max)
NUCLEI_MIN_AREA = 150
COLLAGEN_MIN_AREA = 100


def matlab_rgb2hsv(rgb: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    MATLAB-compatible ``rgb2hsv`` for float RGB in ``[0, 1]``.

    Matches MathWorks' channel ordering (H, S, V) and hue wrap to ``[0, 1]``.
    Used instead of ``skimage.color.rgb2hsv`` for CurveAlign parity (F2/D1).
    """
    arr = np.moveaxis(np.asarray(rgb, dtype=np.float64), axis, -1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB image, got shape {rgb.shape}.")
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    m = np.minimum(np.minimum(r, g), b)
    delta = v - m
    s = np.zeros_like(v)
    nonzero_v = v > 0
    s[nonzero_v] = delta[nonzero_v] / v[nonzero_v]

    h = np.zeros_like(v)
    mask = delta > 0
    # Avoid division by zero; only fill where delta > 0.
    r_eq = mask & (v == r)
    g_eq = mask & (v == g) & ~r_eq
    b_eq = mask & (v == b) & ~r_eq & ~g_eq
    h[r_eq] = np.mod((g[r_eq] - b[r_eq]) / delta[r_eq], 6.0) / 6.0
    h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq] + 2.0) / 6.0
    h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq] + 4.0) / 6.0
    h = np.clip(h, 0.0, 1.0)
    out = np.stack([h, s, v], axis=-1)
    if axis != -1:
        out = np.moveaxis(out, -1, axis)
    return out


def matlab_round(x: float | np.ndarray) -> float | np.ndarray:
    """MATLAB ``round``: half-integers round away from zero (unlike ``numpy.round``)."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    if np.ndim(x) == 0:
        return float(out)
    return out


def matlab_rgb2gray(rgb: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    MATLAB ``rgb2gray`` for double input. MATLAB uses the full-precision
    Rec.601 coefficients below (``rgb2gray.m``: ``T = inv([1 0.956 0.621; 1
    -0.272 -0.647; 1 -1.106 1.703])``, first row), not the rounded
    ``[0.2989, 0.5870, 0.1140]``; the rounded set differs by up to ~1e-4.
    """
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB with >=3 channels, got shape {rgb.shape}.")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return 0.298936021293775 * r + 0.587043074451121 * g + 0.114020904255103 * b


def matlab_graythresh(image: np.ndarray, nbins: int = 256) -> float:
    """
    Exact port of MATLAB ``graythresh`` / ``otsuthresh`` for ``double`` images.

    MATLAB does ``counts = imhist(im2uint8(I(:)), 256)`` - i.e. one bin per
    uint8 level after ``round(I * 255)`` - then Otsu on bin indices ``1..256``
    with ties resolved by averaging the arg-max indices, and returns
    ``(idx - 1) / 255``. (An earlier version used 256 equal-width bins on
    ``[0, 1]`` and returned the bin *centre*, which shifted thresholds by up to
    ~0.002 and changed a few hundred mask pixels.)
    """
    if nbins != 256:
        raise ValueError("MATLAB graythresh always uses 256 bins.")
    arr = np.asarray(image, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot compute graythresh for empty image.")
    # im2uint8 for double input: clip to [0, 1], scale by 255, round half away
    # from zero (values are non-negative here so np.floor(x + 0.5) matches).
    levels = np.floor(np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.int64)
    counts = np.bincount(levels, minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot compute graythresh for empty histogram.")
    p = counts / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257, dtype=np.float64))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    # MATLAB max() ignores NaN (0/0 at omega==0 or 1) but propagates Inf.
    not_nan = ~np.isnan(sigma_b_squared)
    if not not_nan.any():
        return 0.0
    maxval = np.max(sigma_b_squared[not_nan])
    if not np.isfinite(maxval):
        return 0.0
    idx = np.mean(np.flatnonzero(sigma_b_squared == maxval)) + 1.0  # 1-based
    return float((idx - 1.0) / 255.0)


def matlab_fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """MATLAB ``fspecial('gaussian', size, sigma)`` for scalar ``size`` (square kernel)."""
    if size < 1:
        raise ValueError(f"Kernel size must be >= 1, got {size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(x, x)
    h = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return h / np.sum(h)


def matlab_imfilter(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    boundary: str = "zero",
) -> np.ndarray:
    """
    MATLAB ``imfilter(..., 'corr')`` with ``'symmetric'``-equivalent boundaries.

    ``boundary='zero'``: pad with 0 (default ``imfilter`` padding).
    ``boundary='replicate'``: ``'replicate'`` edge padding.
    """
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != img.ndim:
        raise ValueError(f"kernel ndim {k.ndim} != image ndim {img.ndim}")
    # MATLAB places the kernel centre at floor((size+1)/2) (1-based), i.e. the
    # top-left of the two middle elements for even sizes, whereas scipy uses
    # size//2 (bottom-right). origin=-1 on even axes realigns them; this
    # matters for fspecial('gaussian', floor(ppm)) with ppm = 2 (2x2 kernel).
    origin = tuple(-1 if (s % 2 == 0) else 0 for s in k.shape)
    if boundary == "zero":
        return ndimage.correlate(
            img, k, mode="constant", cval=0.0, origin=origin
        ).astype(np.float64)
    if boundary == "replicate":
        return ndimage.correlate(img, k, mode="nearest", origin=origin).astype(np.float64)
    raise ValueError(f"boundary must be 'zero' or 'replicate', got {boundary!r}")


def normalize_array_to_unit_interval(
    image: np.ndarray,
    normalization_epsilon: float = 1e-12,
    raise_on_homogeneous: bool = False,
) -> np.ndarray:
    """
    Normalize an in-memory numeric array to [0, 1] after NaN/Inf cleanup.

    Parameters
    ----------
    image : np.ndarray
        Input array.
    normalization_epsilon : float, default 1e-12
        Lower bound for denominator stability.
    raise_on_homogeneous : bool, default False
        If True, raise for arrays with no dynamic range.
    """
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        raise ValueError("Cannot normalize an empty array.")

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    dynamic = arr_max - arr_min
    if dynamic <= float(normalization_epsilon):
        if raise_on_homogeneous:
            raise ValueError("Cannot normalize homogeneous array.")
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - arr_min) / max(dynamic, float(normalization_epsilon))


def load_and_normalize_image(
    path: str | Path,
    normalization_epsilon: float = 1e-12,
) -> np.ndarray:
    """Load image from disk and normalize intensities to [0, 1]."""
    image = io.imread(str(path))
    original_dtype = image.dtype

    if image.dtype == np.bool_:
        return image.astype(np.float64)

    image = image.astype(np.float64)
    if image.size == 0:
        raise ValueError(f"Opened image at {path} but it is empty.")

    max_val = float(np.nanmax(image))
    min_val = float(np.nanmin(image))
    if max_val > 1.0 or min_val < 0.0:
        if np.issubdtype(original_dtype, np.integer):
            dtype_max = np.iinfo(original_dtype).max
            image = image / float(dtype_max)
        else:
            dynamic = max(max_val - min_val, float(normalization_epsilon))
            image = (image - min_val) / dynamic

    return np.clip(image, 0.0, 1.0)


def save_as_uint8_image(path: str | Path, image: np.ndarray) -> None:
    """Save image to disk as uint8, normalizing/clipping numeric arrays to [0, 1]."""
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


def ensure_rgb(image: np.ndarray, axis: int = 2) -> np.ndarray:
    """Ensure image is RGB with channel-last layout and float64 dtype."""
    arr = image
    if arr.ndim == 2:
        return np.dstack([arr, arr, arr]).astype(np.float64)
    if arr.ndim == 3:
        arr_ch_last = np.moveaxis(arr, axis, -1)
        if arr_ch_last.shape[-1] >= 3:
            return arr_ch_last[..., :3].astype(np.float64)
    raise ValueError(f"Expected 2D grayscale or RGB image, but got {arr.shape}.")


def ensure_grayscale(image: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return a grayscale float64 image; RGB input is converted, 2D input is cast."""
    # TODO: remove unnecessary casts
    arr = image
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        arr_ch_last = np.moveaxis(arr, axis, -1)
        if arr_ch_last.shape[-1] >= 3:
            return color.rgb2gray(arr_ch_last[..., :3])
    raise ValueError(f"Expected 2D grayscale or RGB image, but got {arr.shape}.")


def resize_like(
    image: np.ndarray,
    out_shape: tuple[int, int],
    *,
    order: int = 3,
    anti_aliasing: bool | None = None,
) -> np.ndarray:
    """
    Resize image to target (rows, cols) while preserving range.

    Uses MATLAB-compatible bicubic (keys cubic) via :func:`matlab_imresize` for
    parity with ``BDcreation_reg2.m`` / ``BDcreationHE2.m``. ``order`` and
    ``anti_aliasing`` are accepted for API compatibility but ignored.
    """
    del order, anti_aliasing  # MATLAB path is fixed; kept for call-site compatibility
    if image.shape[:2] == out_shape:
        return image.copy()
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return matlab_imresize(arr, output_shape=(out_shape[0], out_shape[1]), method="bicubic")
    return matlab_imresize(arr, output_shape=(out_shape[0], out_shape[1]), method="bicubic")


def prepare_he_image(he: np.ndarray, pixel_per_micron: float) -> tuple[np.ndarray, float]:
    """MATLAB-compatible HE scaling cap: if ppm > 2, resize by 2/ppm and set ppm=2."""
    pix = float(pixel_per_micron)
    he_rgb = ensure_rgb(he)
    if pix > 2.0:
        target_rows = int(round(he_rgb.shape[0] * 2.0 / pix))
        target_cols = int(round(he_rgb.shape[1] * 2.0 / pix))
        he_rgb = resize_like(he_rgb, (target_rows, target_cols))
        pix = 2.0
    return he_rgb, pix


def matlab_std_std_2d(channel: np.ndarray) -> float:
    """
    MATLAB ``std(std(r))`` for a 2-D channel image.

    In MATLAB, ``std(r)`` for a matrix is the standard deviation along rows
    (one value per column); the outer ``std`` reduces that vector to a scalar.
    This differs from ``std(r(:))`` (global standard deviation) and matches
    ``BDcreation_reg2.m`` / ``BDcreationHE2.m``.
    """
    if channel.ndim != 2:
        raise ValueError(f"Expected 2-D channel, got shape {channel.shape}.")
    if channel.size == 0:
        return 0.0
    # Sample std (ddof=1) matches MATLAB ``std`` for vectors/matrices.
    col_std = np.std(channel, axis=0, ddof=1)
    col_std = col_std[~np.isnan(col_std)]
    if col_std.size <= 1:
        flat = channel.ravel()
        return float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
    return float(np.std(col_std, ddof=1))


def adjust_rgb_mean_std(rgb: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Reproduce MATLAB ``imadjust`` with per-channel high input at ``mean + 2*std(std(r))``.

    Matches ``imadjust(RGB,[0 0 0; HIGH_IN],[0 0 0; 1 1 1])`` in the CurveAlign
    scripts (per-channel ``HIGH_IN`` capped at 1.0).
    """
    rgb_arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] < 3:
        # TODO: check dimension length and num of channel separately
        raise ValueError(f"Expected RGB image with at least 3 channels, got {rgb.shape}.")

    out = np.zeros_like(rgb_arr, dtype=np.float64)
    for ch in range(3):
        channel = rgb_arr[..., ch]
        std_std = matlab_std_std_2d(channel)
        high = float(np.mean(channel) + 2.0 * std_std)
        high = min(max(high, 1e-8), 1.0)
        out[..., ch] = np.clip(channel / high, 0.0, 1.0)
    return out


def prepare_registration_pair(
    he_rgb: np.ndarray,
    shg_gray: np.ndarray,
    pixel_per_micron: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Match ``BDcreation_reg2.m`` image sizing (not ``BDcreationHE2.m``).

    MATLAB::

        if ppm > 2: fixedSHG = imresize(SHG, 2/ppm); ppm = 2
        else: fixedSHG = SHG
        RGB = imresize(HE, size(fixedSHG))

    The working resolution is therefore **defined by SHG** after the ppm cap,
    then H&E is forced to that grid. This differs from ``prepare_he_image``,
    which only resamples HE (tumor pipeline).
    """
    pix = float(pixel_per_micron)
    he_rgb = ensure_rgb(he_rgb)
    g = np.asarray(shg_gray, dtype=np.float64)
    if g.ndim == 3:
        g = ensure_grayscale(g)
    if g.ndim != 2:
        raise ValueError(f"Expected 2-D SHG image, got shape {g.shape}.")

    if pix > 2.0:
        # MATLAB ``imresize(SHG, 2/ppm)``: output size is ceil(scale*size) and
        # the *same* scalar scale is used for the kernel on both axes. Using
        # round() + output_shape here produced a 1-px smaller grid for e.g.
        # ppm=3 (341 vs 342), which put the whole registration on a different
        # image than MATLAB's.
        fixed_shg = matlab_imresize(g, scalar_scale=2.0 / pix, method="bicubic")
        pix = 2.0
    else:
        fixed_shg = g

    he_scaled = resize_like(he_rgb, fixed_shg.shape[:2])
    return he_scaled, fixed_shg, pix


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size using 8-connectivity."""
    min_size = max(int(min_size), 1)
    if mask.size == 0:
        raise ValueError("Mask is empty; cannot remove small components.")
    labels, _ = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(labels.ravel())
    keep = areas >= min_size
    keep[0] = False  # background
    return keep[labels]


def disk_se(radius: float) -> np.ndarray:
    """Create a disk structuring element; radius uses :func:`matlab_round` like MATLAB ``strel``."""
    r = max(int(matlab_round(radius)), 0)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    return morphology.disk(r)


def compute_otsu_threshold(image: np.ndarray) -> float:
    """scikit-image Otsu (non-MATLAB). Prefer :func:`matlab_graythresh` for CurveAlign parity."""
    arr = image.astype(np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute Otsu threshold for empty image.")
    if np.allclose(arr, arr.flat[0]):
        raise ValueError(
            "Cannot compute Otsu threshold for homogeneous image "
            f"(single value={arr.flat[0]})."
        )
    return float(filters.threshold_otsu(arr))


def make_nuclei_mask(
    he_rgb_adjusted: np.ndarray,
    pix_per_mic: float,
    saturation_channel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate nuclei mask and masked nuclei RGB image from adjusted HE."""
    hsv = matlab_rgb2hsv(he_rgb_adjusted)
    sat_thresh = matlab_graythresh(hsv[..., saturation_channel])

    nuclei_raw = (
        (hsv[..., 0] >= NUCLEI_HUE_MIN)
        & (hsv[..., 0] <= NUCLEI_HUE_MAX)
        & (hsv[..., saturation_channel] >= sat_thresh)
    )
    # 150 px minimum area is copied from the original MATLAB scripts.
    nuclei_raw = remove_small_components(nuclei_raw, NUCLEI_MIN_AREA)
    nuclei_opened = morphology.opening(nuclei_raw, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = he_rgb_adjusted.copy()
    masked[~nuclei_opened] = 0.0
    return nuclei_opened, masked


def make_collagen_mask(
    he_rgb_adjusted: np.ndarray,
    pix_per_mic: float,
    enhanced_postprocessing: bool = False,
    saturation_channel: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate collagen mask and non-background mask from adjusted HE."""
    hsv = matlab_rgb2hsv(he_rgb_adjusted)
    sat_thresh = matlab_graythresh(hsv[..., saturation_channel])

    collagen = (
        ((hsv[..., 0] >= COLLAGEN_HUE_MIN) | (hsv[..., 0] <= COLLAGEN_HUE_MAX))
        & (hsv[..., saturation_channel] >= sat_thresh)
    )
    # 100 px minimum area and the wrapped hue band come from MATLAB defaults.
    collagen = remove_small_components(collagen, COLLAGEN_MIN_AREA)

    if enhanced_postprocessing:
        # Morphology radii are MATLAB-derived heuristics in units of pixel/micron.
        # BDcreationHE2.m: strel('disk', ceil(ppm)) then strel('disk', round(3*ppm)).
        collagen = morphology.dilation(collagen, disk_se(np.ceil(pix_per_mic)))
        collagen = morphology.closing(
            collagen, disk_se(float(matlab_round(3.0 * pix_per_mic)))
        )

    no_background = hsv[..., saturation_channel] >= sat_thresh
    return collagen.astype(bool), no_background.astype(bool), sat_thresh


def decorrelation_stretch(rgb: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    Approximate MATLAB ``decorrstretch`` for RGB images in ``[0, 1]``.

    A PCA whitening transform decorrelates channels, then each channel is
    contrast-stretched using percentile clipping controlled by ``tol``.
    """
    arr = ensure_rgb(np.asarray(rgb, dtype=np.float64))
    arr = np.clip(arr, 0.0, 1.0)
    flat = arr.reshape(-1, 3)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    whiten = evecs @ np.diag(1.0 / np.sqrt(evals))
    decor = centered @ whiten

    out = np.zeros_like(decor)
    lo_q = float(np.clip(tol, 0.0, 0.49))
    hi_q = 1.0 - lo_q
    for c in range(3):
        ch = decor[:, c]
        lo = float(np.quantile(ch, lo_q))
        hi = float(np.quantile(ch, hi_q))
        if hi <= lo:
            out[:, c] = 0.0
        else:
            out[:, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out.reshape(arr.shape)


def _rgb_threshold_masks(he_decorr_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (nuclei_mask, eosin_mask) from BDcreation_reg RGB thresholds."""
    u8 = np.clip(he_decorr_rgb, 0.0, 1.0)
    u8 = np.round(u8 * 255.0).astype(np.uint8)
    r = u8[..., 0]
    g = u8[..., 1]
    b = u8[..., 2]

    nuclei = (r < 120) & (g > 150) & (b < 120)
    eosin = (r > 200) & (g < 100) & (b > 100)
    return nuclei.astype(bool), eosin.astype(bool)


def make_nuclei_mask_rgb(
    he_decorr_rgb: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nuclei mask from BDcreation_reg-style RGB thresholds on decorrelated RGB.

    Returns ``(nuclei_opened, masked_nuclei_rgb)`` analogous to
    :func:`make_nuclei_mask`.
    """
    nuclei, _ = _rgb_threshold_masks(he_decorr_rgb)
    nuclei = remove_small_components(nuclei, NUCLEI_MIN_AREA)
    nuclei_opened = morphology.opening(nuclei, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = np.asarray(he_decorr_rgb, dtype=np.float64).copy()
    masked[~nuclei_opened] = 0.0
    return nuclei_opened, masked


def make_ecm_mask_rgb(
    he_decorr_rgb: np.ndarray,
    pix_per_mic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ECM candidate image from BDcreation_reg-style RGB eosin threshold.

    Returns ``(ecm_gray, eosin_mask)`` where ``ecm_gray`` is continuous in
    ``[0, 1]`` and suitable as a moving image for MI registration.
    """
    _, eosin = _rgb_threshold_masks(he_decorr_rgb)
    eosin = remove_small_components(eosin, COLLAGEN_MIN_AREA)
    if pix_per_mic > 0:
        eosin = morphology.opening(eosin, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = np.zeros_like(he_decorr_rgb, dtype=np.float64)
    masked[eosin] = np.asarray(he_decorr_rgb, dtype=np.float64)[eosin]
    ecm_gray = matlab_rgb2gray(masked)
    return ecm_gray.astype(np.float64), eosin.astype(bool)


def make_ecm_mask_lab(
    he_source_rgb: np.ndarray,
    he_decorr_rgb: np.ndarray,
    pix_per_mic: float,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ECM candidate from BDcreation_reg LAB k-means refinement.

    1) RGB eosin threshold gate in decorrelated RGB.
    2) K-means on LAB (a*, b*) for eosin pixels.
    3) Select cluster with maximum mean(a*, b*).
    4) Return grayscale ECM from the selected cluster on source RGB.
    """
    _, eosin = _rgb_threshold_masks(he_decorr_rgb)
    eosin = remove_small_components(eosin, COLLAGEN_MIN_AREA)
    if int(np.count_nonzero(eosin)) < 3:
        return make_ecm_mask_rgb(he_decorr_rgb, pix_per_mic)

    lab = color.rgb2lab(np.clip(he_decorr_rgb, 0.0, 1.0))
    ab = lab[..., 1:3][eosin]
    if ab.shape[0] < 3:
        return make_ecm_mask_rgb(he_decorr_rgb, pix_per_mic)

    kmeans = KMeans(n_clusters=3, n_init=3, random_state=int(random_state))
    labels = kmeans.fit_predict(ab)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    target_cluster = int(np.argmax(np.mean(centers, axis=1)))

    selected = np.zeros(eosin.shape, dtype=bool)
    selected_indices = np.flatnonzero(eosin)
    selected.flat[selected_indices[labels == target_cluster]] = True
    selected = remove_small_components(selected, COLLAGEN_MIN_AREA)
    if pix_per_mic > 0:
        selected = morphology.opening(selected, disk_se(np.ceil(pix_per_mic / 2.0)))

    source = np.asarray(he_source_rgb, dtype=np.float64)
    masked = np.zeros_like(source, dtype=np.float64)
    masked[selected] = source[selected]
    ecm_gray = matlab_rgb2gray(masked)
    return ecm_gray.astype(np.float64), selected.astype(bool)


def matlab_imwarp_bilinear(
    src: np.ndarray,
    out_shape: tuple[int, int],
    A: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    MATLAB-parity bilinear warp replicating ``imwarp(moving, Rmoving, tform, 'OutputView',
    Rfixed, 'FillValues', fill_value)`` with ``imref2d(size)`` defaults.

    Parameters
    ----------
    src : np.ndarray
        Source image ``(H_in, W_in)`` or ``(H_in, W_in, C)``, float.
    out_shape : (int, int)
        Output ``(H_out, W_out)``.
    A : np.ndarray, shape (2, 3) or (3, 3)
        Affine mapping **fixed (output) 0-based intrinsic coords -> moving (input)
        0-based intrinsic coords**, in column-vector form:
        ``[x_moving; y_moving; 1] = A @ [x_fixed; y_fixed; 1]``.
        Equivalent to applying the inverse of a forward (moving->fixed) affine.
    fill_value : float
        Value used for output pixels whose source sample lies outside the
        MATLAB-parity half-pixel-extended input domain, and for out-of-bound
        neighbours encountered during bilinear blending.

    Returns
    -------
    np.ndarray
        Warped image with the same number of channels as ``src`` and shape
        ``(H_out, W_out[, C])``, dtype float64.

    Notes
    -----
    Matches the ``imref2d``/``imwarp`` conventions critical for pixel-exact
    parity with MATLAB's ``BDcreation_reg2.m``:

    - Output pixel ``(r, c)`` center is at world ``(c, r)`` (0-based world
      coords, which differ from MATLAB's 1-based world only by a constant
      offset that cancels out when both domains share the same convention).
    - A source sample ``(x, y)`` is "inside" iff ``-0.5 <= x <= W_in - 0.5``
      and ``-0.5 <= y <= H_in - 0.5`` (half-pixel extension of the intrinsic
      grid, matching MATLAB's ``[0.5, N+0.5]`` world limits).
    - Bilinear interpolation blends ``fill_value`` for any of the four
      neighbours that fall outside the intrinsic grid, which reproduces
      MATLAB's boundary halo behaviour.

    Implemented with vectorised NumPy rather than
    ``scipy.ndimage.map_coordinates`` because the latter treats the grid as
    ``[0, N-1]`` with no half-pixel extension, diverging from ``imwarp`` at
    the outermost band.
    """
    src_arr = np.asarray(src, dtype=np.float64)
    if src_arr.ndim not in (2, 3):
        raise ValueError(f"src must be 2D or 3D, got shape {src_arr.shape}")

    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.shape == (2, 3):
        A_full = np.vstack([A_arr, [0.0, 0.0, 1.0]])
    elif A_arr.shape == (3, 3):
        A_full = A_arr
    else:
        raise ValueError(f"A must be 2x3 or 3x3, got shape {A_arr.shape}")

    H_in, W_in = src_arr.shape[:2]
    H_out, W_out = int(out_shape[0]), int(out_shape[1])

    # Build output intrinsic grid (0-based pixel indices).
    c_grid, r_grid = np.meshgrid(
        np.arange(W_out, dtype=np.float64),
        np.arange(H_out, dtype=np.float64),
    )
    ones = np.ones_like(c_grid)
    pts_out = np.stack([c_grid, r_grid, ones], axis=0).reshape(3, -1)

    # Source intrinsic coords for every output pixel.
    pts_in = A_full @ pts_out
    x_in = pts_in[0].reshape(H_out, W_out)
    y_in = pts_in[1].reshape(H_out, W_out)

    # Inside = half-pixel-extended input domain (MATLAB imref2d limits).
    inside = (
        (x_in >= -0.5)
        & (x_in <= (W_in - 1) + 0.5)
        & (y_in >= -0.5)
        & (y_in <= (H_in - 1) + 0.5)
    )

    # Bilinear: floor + ceil neighbours in intrinsic coords.
    x0 = np.floor(x_in).astype(np.int64)
    y0 = np.floor(y_in).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x_in - x0).astype(np.float64)
    wy = (y_in - y0).astype(np.float64)

    # Pad source with fill_value so any OOB neighbour samples the fill.
    if src_arr.ndim == 2:
        padded = np.full((H_in + 2, W_in + 2), float(fill_value), dtype=np.float64)
        padded[1:-1, 1:-1] = src_arr
    else:
        C = src_arr.shape[2]
        padded = np.full(
            (H_in + 2, W_in + 2, C), float(fill_value), dtype=np.float64
        )
        padded[1:-1, 1:-1, :] = src_arr

    # Pad index shift of +1; clip to padded bounds (all fill outside padded).
    x0p = np.clip(x0 + 1, 0, W_in + 1)
    y0p = np.clip(y0 + 1, 0, H_in + 1)
    x1p = np.clip(x1 + 1, 0, W_in + 1)
    y1p = np.clip(y1 + 1, 0, H_in + 1)

    if src_arr.ndim == 2:
        v00 = padded[y0p, x0p]
        v01 = padded[y0p, x1p]
        v10 = padded[y1p, x0p]
        v11 = padded[y1p, x1p]
        interp = (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v01
            + (1.0 - wx) * wy * v10
            + wx * wy * v11
        )
        out = np.where(inside, interp, float(fill_value))
        return out.astype(np.float64)

    out_channels = []
    for k in range(src_arr.shape[2]):
        v00 = padded[y0p, x0p, k]
        v01 = padded[y0p, x1p, k]
        v10 = padded[y1p, x0p, k]
        v11 = padded[y1p, x1p, k]
        interp = (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v01
            + (1.0 - wx) * wy * v10
            + wx * wy * v11
        )
        ch_out = np.where(inside, interp, float(fill_value))
        out_channels.append(ch_out)
    return np.stack(out_channels, axis=-1).astype(np.float64)


def gaussian_filter_with_size_hint(
    image: np.ndarray,
    sigma: float,
    kernel_size: int | None = None,
    *,
    boundary: str = "zero",
) -> np.ndarray:
    """
    MATLAB ``fspecial('gaussian', kernel_size, sigma)`` + ``imfilter`` (correlation).

    ``boundary='zero'`` for ``BDcreation_reg2`` nuclei filtering; ``'replicate'`` for
    ``BDcreationHE2`` final Gaussian blur.
    """
    if kernel_size is None:
        raise ValueError("kernel_size is required for MATLAB-compatible Gaussian filtering.")
    sz = max(int(kernel_size), 1)
    h = matlab_fspecial_gaussian(sz, float(sigma))
    return matlab_imfilter(image.astype(np.float64), h, boundary=boundary)


# Backward-compatible aliases for current call sites.
load_image_as_float = load_and_normalize_image
save_image_uint8 = save_as_uint8_image
to_grayscale = ensure_grayscale
matlab_area_open = remove_small_components
safe_otsu = matlab_graythresh
gaussian_filter_matlab_like = gaussian_filter_with_size_hint
