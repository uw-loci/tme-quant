"""
Morphology-based cell classification.
"""

import numpy as np
from typing import Optional

from ..config import ClassificationParams, ClassificationResult
from tme_quant.core.tme_objects.cell_objects import SegmentationResult
from .base_classification import BaseClassificationMethod


class MorphologyClassifier(BaseClassificationMethod):
    """
    Classify cells based on morphological features.

    Uses area, circularity, and eccentricity to distinguish cell types
    via simple rule-based logic:
    - Large + irregular  → TUMOR
    - Small + round      → IMMUNE
    - Elongated          → FIBROBLAST
    - Default            → STROMAL
    """

    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None,
    ) -> ClassificationResult:
        """Classify cells using morphology."""
        result = ClassificationResult(mode=params.mode)

        for cell in segmentation_result.cells:
            features = self._extract_morphology_features(cell)
            cell_type, confidence = self._classify_morphology(features, params)
            result.cell_types[cell.cell_id] = cell_type
            result.confidences[cell.cell_id] = confidence

        return result

    def _extract_morphology_features(self, cell) -> dict:
        """Extract morphological features from a cell."""
        features = {}
        for attr in ('area', 'perimeter', 'circularity', 'eccentricity',
                     'solidity', 'extent'):
            if hasattr(cell, attr):
                features[attr] = getattr(cell, attr)

        if features.get('area', 0) > 0:
            features['equivalent_diameter'] = 2 * np.sqrt(features['area'] / np.pi)

        return features

    def _classify_morphology(self, features: dict, params) -> tuple:
        """Return (CellType, confidence) from morphology rules."""
        from tme_quant.core.tme_objects.cell_objects import CellType

        area = features.get('area', 0)
        circularity = features.get('circularity', 0)
        eccentricity = features.get('eccentricity', 0)

        if area > 200 and circularity < 0.7:
            return CellType.TUMOR, 0.7
        if area < 100 and circularity > 0.8:
            return CellType.IMMUNE, 0.6
        if eccentricity > 0.7:
            return CellType.FIBROBLAST, 0.6
        return CellType.STROMAL, 0.4
