from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import color, filters, io, morphology, transform


def load_image_as_float(path: str | Path) -> np.ndarray:
    """Load image and normalize intensity to [0, 1]."""
    image = np.asarray(io.imread(str(path)))
    original_dtype = image.dtype

    if image.dtype == np.bool_:
        return image.astype(np.float64)

    image = image.astype(np.float64)
    if image.size == 0:
        return image

    max_val = float(np.nanmax(image))
    min_val = float(np.nanmin(image))
    if max_val > 1.0 or min_val < 0.0:
        if np.issubdtype(original_dtype, np.integer):
            dtype_max = np.iinfo(original_dtype).max
            image = image / float(dtype_max)
        else:
            dynamic = max(max_val - min_val, 1e-12)
            image = (image - min_val) / dynamic

    return np.clip(image, 0.0, 1.0)


def save_image_uint8(path: str | Path, image: np.ndarray) -> None:
    """Save image as uint8, preserving logical masks and RGB images."""
    arr = np.asarray(image)

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


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB in float [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.dstack([arr, arr, arr]).astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.float64)
    raise ValueError("Expected 2D grayscale or RGB image.")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return color.rgb2gray(arr[:, :, :3])
    raise ValueError("Expected 2D grayscale or RGB image.")


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


def adjust_rgb_mean_std(rgb: np.ndarray) -> np.ndarray:
    """
    Reproduce MATLAB imadjust with per-channel high input at mean + 2*std.
    Equivalent to imadjust(RGB,[0;high],[0;1]) channel-wise.
    """
    out = np.zeros_like(rgb, dtype=np.float64)
    for ch in range(3):
        channel = rgb[:, :, ch]
        high = float(np.mean(channel) + 2.0 * np.std(channel))
        high = min(max(high, 1e-8), 1.0)
        out[:, :, ch] = np.clip(channel / high, 0.0, 1.0)
    return out


def matlab_area_open(mask: np.ndarray, min_size: int) -> np.ndarray:
    """MATLAB bwareaopen-like operation using 8-connectivity in 2D."""
    min_size = max(int(min_size), 1)
    if mask.size == 0:
        return mask.astype(bool)
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


def safe_otsu(image: np.ndarray) -> float:
    """Otsu threshold with constant-image fallback."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.allclose(arr, arr.flat[0]):
        return float(arr.flat[0])
    return float(filters.threshold_otsu(arr))


def make_nuclei_mask(he_rgb_adjusted: np.ndarray, pix_per_mic: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate nuclei mask and masked nuclei RGB image from adjusted HE."""
    hsv = color.rgb2hsv(he_rgb_adjusted)
    sat_thresh = safe_otsu(hsv[:, :, 1])

    nuclei_raw = (
        (hsv[:, :, 0] >= 0.500)
        & (hsv[:, :, 0] <= 0.790)
        & (hsv[:, :, 1] >= sat_thresh)
    )
    nuclei_raw = matlab_area_open(nuclei_raw, 150)
    nuclei_opened = morphology.opening(nuclei_raw, disk_se(np.ceil(pix_per_mic / 2.0)))

    masked = he_rgb_adjusted.copy()
    masked[~nuclei_opened] = 0.0
    return nuclei_opened, masked


def make_collagen_mask(
    he_rgb_adjusted: np.ndarray,
    pix_per_mic: float,
    enhanced_postprocessing: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate collagen mask and non-background mask from adjusted HE."""
    hsv = color.rgb2hsv(he_rgb_adjusted)
    sat_thresh = safe_otsu(hsv[:, :, 1])

    collagen = (
        ((hsv[:, :, 0] >= 0.837) | (hsv[:, :, 0] <= 0.066))
        & (hsv[:, :, 1] >= sat_thresh)
    )
    collagen = matlab_area_open(collagen, 100)

    if enhanced_postprocessing:
        collagen = morphology.dilation(collagen, disk_se(np.ceil(pix_per_mic)))
        collagen = morphology.closing(collagen, disk_se(np.round(3.0 * pix_per_mic)))

    no_background = hsv[:, :, 1] >= sat_thresh
    return collagen.astype(bool), no_background.astype(bool), sat_thresh


def gaussian_filter_matlab_like(
    image: np.ndarray, sigma: float, kernel_size: int | None = None
) -> np.ndarray:
    """
    Approximate MATLAB fspecial('gaussian', size, sigma) + imfilter(...,'replicate','corr').
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
