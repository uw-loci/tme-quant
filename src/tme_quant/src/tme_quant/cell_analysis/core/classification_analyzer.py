# tme_quant/cell_analysis/core/classification_analyzer.py
"""
Cell classification analyzer.

Assigns cell types based on morphology, markers, or deep learning.
"""

import numpy as np
from typing import Optional, Dict
import time
from collections import Counter

from ..methods.classification.base_classification import BaseClassificationMethod
from ..methods.registry import MethodRegistry
from ..config.classification_params import ClassificationParams, ClassificationResult
from ...core.tme_models.cell_model import ClassificationMode, CellType, SegmentationResult


class CellClassificationAnalyzer:
    """
    Coordinates cell classification using different methods.
    
    Supports:
        - Morphology-based classification
        - Marker-based classification (IF)
        - Deep learning classification
        - Ensemble methods
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize classification analyzer.
        
        Args:
            verbose: Print progress messages
        """
        self.registry = MethodRegistry()
        self.verbose = verbose
        self._register_methods()
    
    def _register_methods(self):
        """Register all available classification methods."""
        from ..methods.classification.morphology_classifier import MorphologyClassifier
        from ..methods.classification.marker_classifier import MarkerClassifier
        
        self.registry.register_classification_method(
            ClassificationMode.MORPHOLOGY, MorphologyClassifier
        )
        self.registry.register_classification_method(
            ClassificationMode.MARKER, MarkerClassifier
        )
    
    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None
    ) -> ClassificationResult:
        """
        Classify segmented cells.
        
        Args:
            segmentation_result: Segmentation result with cells
            params: Classification parameters
            image: Original image (required for some methods)
            
        Returns:
            ClassificationResult with cell type assignments
        """
        if not segmentation_result.cells:
            return ClassificationResult(mode=params.mode)
        
        # Get method
        method_class = self.registry.get_classification_method(params.mode)
        method = method_class(verbose=self.verbose)
        
        # Run classification with timing
        start_time = time.time()
        result = method.classify(segmentation_result, params, image)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.mode = params.mode
        result.processing_time = processing_time
        result.parameters = params.to_dict()
        
        # Compute type distribution
        self._compute_type_distribution(result, segmentation_result)
        
        return result
    
    def _compute_type_distribution(
        self,
        result: ClassificationResult,
        segmentation_result: SegmentationResult
    ):
        """Compute distribution of cell types."""
        # Count types
        type_counts = Counter(result.cell_types.values())
        result.type_counts = dict(type_counts)
        
        # Compute ratios
        total = len(result.cell_types)
        if total > 0:
            result.type_ratios = {
                cell_type: count / total
                for cell_type, count in type_counts.items()
            }
        
        # Update cell properties with classifications
        for cell in segmentation_result.cells:
            if cell.cell_id in result.cell_types:
                cell.cell_type = result.cell_types[cell.cell_id]
                cell.cell_type_confidence = result.confidences.get(cell.cell_id)