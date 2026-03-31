"""
Immunofluorescence marker-based cell classification.
"""

import numpy as np
from typing import Optional

from ..config import ClassificationParams, ClassificationResult
from tme_quant.core.tme_objects.cell_objects import SegmentationResult
from .base_classification import BaseClassificationMethod


class MarkerClassifier(BaseClassificationMethod):
    """
    Classify cells based on immunofluorescence marker expression.

    Uses per-cell mean intensities in marker channels (e.g. CD3, CD8, CD68, CD20)
    against configurable thresholds.
    """

    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None,
    ) -> ClassificationResult:
        """Classify cells using marker expression."""
        if image is None:
            raise ValueError("Image required for marker-based classification")
        if not params.marker_channels:
            raise ValueError("marker_channels required for marker classification")

        result = ClassificationResult(mode=params.mode)
        marker_intensities = self._extract_marker_intensities(
            segmentation_result, image, params
        )

        for cell in segmentation_result.cells:
            cell_markers = marker_intensities.get(cell.cell_id, {})
            cell_type, confidence = self._classify_by_markers(cell_markers, params)
            result.cell_types[cell.cell_id] = cell_type
            result.confidences[cell.cell_id] = confidence

        return result

    def _extract_marker_intensities(
        self,
        segmentation_result: SegmentationResult,
        image: np.ndarray,
        params: ClassificationParams,
    ) -> dict:
        """Extract mean marker intensities per cell."""
        marker_intensities = {}
        labels = segmentation_result.label_mask

        for cell in segmentation_result.cells:
            cell_mask = labels == cell.cell_id
            cell_intensities = {}

            for marker_name, channel_idx in params.marker_channels.items():
                if image.ndim == 2:
                    if channel_idx != 0:
                        continue
                    marker_image = image
                else:
                    if channel_idx >= image.shape[-1]:
                        continue
                    marker_image = image[:, :, channel_idx]

                mean_intensity = np.mean(marker_image[cell_mask])
                if marker_image.max() > 1.0:
                    mean_intensity = mean_intensity / marker_image.max()
                cell_intensities[marker_name] = mean_intensity

            marker_intensities[cell.cell_id] = cell_intensities

        return marker_intensities

    def _classify_by_markers(
        self,
        marker_values: dict,
        params: ClassificationParams,
    ) -> tuple:
        """Return (CellType, confidence) from marker expression."""
        from tme_quant.core.tme_objects.cell_objects import CellType

        thresholds = params.marker_thresholds or {}

        cd3_pos  = marker_values.get('CD3',  0) > thresholds.get('CD3',  0.30)
        cd8_pos  = marker_values.get('CD8',  0) > thresholds.get('CD8',  0.25)
        cd68_pos = marker_values.get('CD68', 0) > thresholds.get('CD68', 0.40)
        cd20_pos = marker_values.get('CD20', 0) > thresholds.get('CD20', 0.30)

        if cd3_pos and cd8_pos:
            return CellType.T_CELL, 0.9
        if cd3_pos:
            return CellType.T_CELL, 0.8
        if cd20_pos:
            return CellType.B_CELL, 0.85
        if cd68_pos:
            return CellType.MACROPHAGE, 0.85
        return CellType.TUMOR, 0.5
