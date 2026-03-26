from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import color, filters, io, morphology

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


def matlab_round(x: float | np.ndarray) -> float | np.ndarray:
    """MATLAB ``round``: half-integers round away from zero (unlike ``numpy.round``)."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    if np.ndim(x) == 0:
        return float(out)
    return out


def matlab_rgb2gray(rgb: np.ndarray, axis: int = -1) -> np.ndarray:
    """MATLAB ``rgb2gray``: Rec.601 luma ``[0.2989, 0.5870, 0.1140]``."""
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB with >=3 channels, got shape {rgb.shape}.")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def matlab_graythresh(image: np.ndarray, nbins: int = 256) -> float:
    """
    MATLAB ``graythresh`` for ``double`` images in ``[0, 1]``: 256-bin histogram on
    ``[0, 1]``, then Otsu threshold (bin center of optimal split).
    """
    arr = np.asarray(image, dtype=np.float64).ravel()
    arr = np.clip(arr, 0.0, 1.0)
    if arr.size == 0:
        raise ValueError("Cannot compute graythresh for empty image.")
    hist, bin_edges = np.histogram(arr, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        raise ValueError("Cannot compute graythresh for empty histogram.")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_b = np.cumsum(hist * bin_centers)
    mu_t = sum_b[-1]
    between = np.zeros(nbins, dtype=np.float64)
    for t in range(nbins):
        w0_t = w0[t]
        w1_t = w1[t]
        if w0_t <= 0 or w1_t <= 0:
            continue
        m0 = sum_b[t] / w0_t
        m1 = (mu_t - sum_b[t]) / w1_t
        between[t] = w0_t * w1_t * (m0 - m1) ** 2
    idx = int(np.argmax(between))
    return float(bin_centers[idx])


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
    if boundary == "zero":
        return ndimage.correlate(img, k, mode="constant", cval=0.0).astype(np.float64)
    if boundary == "replicate":
        return ndimage.correlate(img, k, mode="nearest").astype(np.float64)
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
        scale = 2.0 / pix
        new_h = int(round(g.shape[0] * scale))
        new_w = int(round(g.shape[1] * scale))
        fixed_shg = resize_like(g, (new_h, new_w))
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
    hsv = color.rgb2hsv(he_rgb_adjusted)
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
    hsv = color.rgb2hsv(he_rgb_adjusted)
    sat_thresh = matlab_graythresh(hsv[..., saturation_channel])

    collagen = (
        ((hsv[..., 0] >= COLLAGEN_HUE_MIN) | (hsv[..., 0] <= COLLAGEN_HUE_MAX))
        & (hsv[..., saturation_channel] >= sat_thresh)
    )
    # 100 px minimum area and the wrapped hue band come from MATLAB defaults.
    collagen = remove_small_components(collagen, COLLAGEN_MIN_AREA)

    if enhanced_postprocessing:
        # Morphology radii are MATLAB-derived heuristics in units of pixel/micron.
        collagen = morphology.dilation(collagen, disk_se(np.ceil(pix_per_mic)))
        collagen = morphology.closing(collagen, disk_se(3.0 * pix_per_mic))

    no_background = hsv[..., saturation_channel] >= sat_thresh
    return collagen.astype(bool), no_background.astype(bool), sat_thresh


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
