"""Small, UI-independent segmentation operations.

These functions intentionally accept and return NumPy arrays.  GUI adapters
such as napari, Fiji, and scikit-ops are responsible for assigning display
semantics to those arrays.
"""

from __future__ import annotations

import numpy as np


_THRESHOLD_METHODS = ("otsu", "triangle", "isodata", "mean", "minimum")


def _remove_small(binary: np.ndarray, minimum: int, *, holes: bool) -> np.ndarray:
    """Bridge the scikit-image 0.25/0.26 size-parameter rename."""
    from inspect import signature

    from skimage import morphology

    function = (
        morphology.remove_small_holes if holes else morphology.remove_small_objects
    )
    if "max_size" in signature(function).parameters:
        return function(binary, max_size=minimum - 1)
    keyword = "area_threshold" if holes else "min_size"
    return function(binary, **{keyword: minimum})


def segment_threshold(
    image: np.ndarray,
    *,
    method: str = "otsu",
    min_area: int = 100,
    max_area: int | None = None,
    fill_holes: bool = True,
    remove_border_objects: bool = True,
) -> np.ndarray:
    """Segment a 2-D grayscale or RGB image with a global threshold.

    Returns an integer label image with zero reserved for the background.
    """
    from skimage import filters, measure
    from skimage.segmentation import clear_border

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = (
            0.2125 * image[..., 0]
            + 0.7154 * image[..., 1]
            + 0.0721 * image[..., 2]
        )
    elif image.ndim != 2:
        raise ValueError("segment_threshold expects a 2-D grayscale or RGB(A) image")

    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    if max_area is not None and max_area < 0:
        raise ValueError("max_area must be non-negative or None")

    method = method.lower()
    if method not in _THRESHOLD_METHODS:
        choices = ", ".join(_THRESHOLD_METHODS)
        raise ValueError(f"Unknown threshold method {method!r}; choose one of {choices}")

    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint16)

    low = float(np.nanmin(image))
    high = float(np.nanmax(image))
    if high == low:
        return np.zeros(image.shape, dtype=np.uint16)
    normalized = (image - low) / (high - low)
    normalized[~finite] = 0

    threshold = getattr(filters, f"threshold_{method}")(normalized)
    binary = normalized > threshold
    if min_area:
        binary = _remove_small(binary, min_area, holes=False)
    if fill_holes and min_area:
        binary = _remove_small(binary, min_area, holes=True)
    if remove_border_objects:
        binary = clear_border(binary)

    labels = measure.label(binary)
    if max_area is not None:
        for region in measure.regionprops(labels):
            if region.area > max_area:
                labels[labels == region.label] = 0
        labels = measure.label(labels > 0)

    max_label = int(labels.max(initial=0))
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    return labels.astype(dtype, copy=False)


def boundary_labels(labels: np.ndarray, *, thickness: int = 1) -> np.ndarray:
    """Return labeled inner boundaries for the objects in a label image."""
    from skimage.morphology import binary_dilation, disk
    from skimage.segmentation import find_boundaries

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("boundary_labels expects a 2-D label image")
    if not np.issubdtype(labels.dtype, np.integer) and labels.dtype != np.bool_:
        raise TypeError("labels must have an integer or boolean dtype")
    if thickness < 1:
        raise ValueError("thickness must be at least 1")

    boundaries = find_boundaries(labels, mode="inner")
    if thickness > 1:
        boundaries = binary_dilation(boundaries, footprint=disk(thickness - 1))
    return boundaries.astype(np.uint16)


def largest_boundary_points(
    labels: np.ndarray, *, simplify_tolerance: float = 1.0
) -> np.ndarray:
    """Return the longest ordered contour from a 2-D label image."""
    from skimage.measure import approximate_polygon, find_contours

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("largest_boundary_points expects a 2-D label image")
    if simplify_tolerance < 0:
        raise ValueError("simplify_tolerance must be non-negative")
    contours = find_contours(labels > 0, 0.5)
    if not contours:
        return np.empty((0, 2), dtype=float)
    contour = max(contours, key=len)
    if simplify_tolerance:
        contour = approximate_polygon(contour, tolerance=simplify_tolerance)
    return np.asarray(contour, dtype=float)


__all__ = ["boundary_labels", "largest_boundary_points", "segment_threshold"]
