from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import color, filters, io, morphology, transform

# MATLAB-derived HSV threshold defaults from:
# - CurveAlign_CT-FIRE/BDcreation_reg2.m
# - CurveAlign_CT-FIRE/BDcreationHE2.m
NUCLEI_HUE_MIN = 0.500
NUCLEI_HUE_MAX = 0.790
COLLAGEN_HUE_MIN = 0.837
COLLAGEN_HUE_MAX = 0.066  # wrapped hue range (>= min OR <= max)
NUCLEI_MIN_AREA = 150
COLLAGEN_MIN_AREA = 100

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
    arr = image
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        arr_ch_last = np.moveaxis(arr, axis, -1)
        if arr_ch_last.shape[-1] >= 3:
            return color.rgb2gray(arr_ch_last[..., :3])
    raise ValueError(f"Expected 2D grayscale or RGB image, but got {arr.shape}.")


def resize_like(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resize image to target (rows, cols) while preserving range."""
    if image.shape[:2] == out_shape:
        return image.copy()
    return transform.resize(
        image,
        output_shape=out_shape if image.ndim == 2 else (*out_shape, image.shape[2]),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float64)


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


def adjust_rgb_mean_std(rgb: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Reproduce MATLAB imadjust with per-channel high input at mean + 2*std.
    Equivalent to imadjust(RGB,[0;high],[0;1]) channel-wise.
    """
    rgb_arr = np.moveaxis(rgb, axis, -1).astype(np.float64)
    if rgb_arr.ndim != 3 or rgb_arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with at least 3 channels, got {rgb.shape}.")

    out = np.zeros_like(rgb_arr, dtype=np.float64)
    for ch in range(3):
        channel = rgb_arr[..., ch]
        high = float(np.mean(channel) + 2.0 * np.std(channel))
        high = min(max(high, 1e-8), 1.0)
        out[..., ch] = np.clip(channel / high, 0.0, 1.0)
    return out


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
    """Create a disk structuring element with MATLAB-like minimum size behavior."""
    r = max(int(round(radius)), 0)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    return morphology.disk(r)


def compute_otsu_threshold(image: np.ndarray) -> float:
    """Compute Otsu threshold and raise if input is empty or homogeneous."""
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
    sat_thresh = compute_otsu_threshold(hsv[..., saturation_channel])

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
    sat_thresh = compute_otsu_threshold(hsv[..., saturation_channel])

    collagen = (
        ((hsv[..., 0] >= COLLAGEN_HUE_MIN) | (hsv[..., 0] <= COLLAGEN_HUE_MAX))
        & (hsv[..., saturation_channel] >= sat_thresh)
    )
    # 100 px minimum area and the wrapped hue band come from MATLAB defaults.
    collagen = remove_small_components(collagen, COLLAGEN_MIN_AREA)

    if enhanced_postprocessing:
        # Morphology radii are MATLAB-derived heuristics in units of pixel/micron.
        collagen = morphology.dilation(collagen, disk_se(np.ceil(pix_per_mic)))
        collagen = morphology.closing(collagen, disk_se(np.round(3.0 * pix_per_mic)))

    no_background = hsv[..., saturation_channel] >= sat_thresh
    return collagen.astype(bool), no_background.astype(bool), sat_thresh


def gaussian_filter_with_size_hint(
    image: np.ndarray, sigma: float, kernel_size: int | None = None
) -> np.ndarray:
    """
    Gaussian filter with optional kernel-size hint to constrain effective radius.
    """
    truncate = 4.0
    if kernel_size is not None and sigma > 0:
        radius = (max(int(kernel_size), 1) - 1) / 2.0
        truncate = max(radius / float(sigma), 0.5)
    return ndimage.gaussian_filter(
        image.astype(np.float64),
        sigma=float(sigma),
        mode="nearest",
        truncate=truncate,
    )


# Backward-compatible aliases for current call sites.
load_image_as_float = load_and_normalize_image
save_image_uint8 = save_as_uint8_image
to_grayscale = ensure_grayscale
matlab_area_open = remove_small_components
safe_otsu = compute_otsu_threshold
gaussian_filter_matlab_like = gaussian_filter_with_size_hint
