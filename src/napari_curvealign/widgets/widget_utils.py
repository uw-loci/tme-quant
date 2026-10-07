"""Shared helpers for CurveAlign napari widgets (image I/O and ROI save dialogs).

This module mirrors the role of ``widget_utils`` in `napari-imagej`_—small, testable
utilities kept out of the main dock widget class.

.. _napari-imagej: https://github.com/imagej/napari-imagej/tree/main/src/napari_imagej/widgets
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ROI_SAVE_DIALOG_FILTERS",
    "rgb_to_grayscale_luma",
    "roi_save_format_kw",
]


def rgb_to_grayscale_luma(image: np.ndarray) -> np.ndarray:
    """Convert last-axis RGB or RGBA to single-channel luma (Rec. 709 coefficients)."""
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    return 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]


ROI_SAVE_DIALOG_FILTERS = (
    "JSON files (*.json);;"
    "Fiji/ImageJ ROI (*.roi *.zip);;"
    "StarDist ROI (*.roi *.zip);;"
    "Label image mask (*.npy);;"
    "QuPath annotations (*.geojson);;"
    "CSV files (*.csv);;"
    "TIFF mask (*.tif);;"
    "All files (*)"
)


def roi_save_format_kw(file_path: str, selected_filter: str) -> tuple[str, str]:
    """Infer ``roi_manager.save_rois(..., format=...)`` and a short label from dialog state."""
    sf = selected_filter or ""
    if "JSON" in sf or file_path.endswith(".json"):
        return "json", "JSON"
    if "Fiji" in sf or (
        file_path.endswith((".roi", ".zip")) and "StarDist" not in sf
    ):
        return "fiji", "Fiji/ImageJ"
    if "StarDist" in sf:
        return "stardist", "StarDist"
    if "Label image" in sf or "Cellpose" in sf or file_path.endswith(".npy"):
        return "label_image", "Label image"
    if "QuPath" in sf or file_path.endswith(".geojson"):
        return "qupath", "QuPath"
    if "CSV" in sf or file_path.endswith(".csv"):
        return "csv", "CSV"
    if "TIFF" in sf or file_path.endswith((".tif", ".tiff")):
        return "mask", "TIFF mask"
    return "auto", "auto-detected"
