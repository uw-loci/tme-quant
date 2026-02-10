"""Individual fiber extraction analyzer."""

import numpy as np
from typing import Dict, Type
import time

from ..methods.extraction.base_extraction import BaseExtractionMethod
from ..methods.registry import MethodRegistry
from ..config.extraction_params import ExtractionParams, ExtractionResult


class FiberExtractionAnalyzer:
    """
    Coordinates individual fiber extraction using different methods.
    
    Supports:
        - CT-FIRE (curvelet-based, 2D/3D)
        - Ridge Detection (Fiji plugin, 2D)
        - Fiber tracing algorithms
        - Skeletonization-based methods
    """
    
    def __init__(self):
        self.registry = MethodRegistry()
        self._register_methods()
    
    def _register_methods(self):
        """Register all available extraction methods."""
        from ..methods.extraction.ctfire import CTFireExtraction
        from ..methods.extraction.ridge_detection import RidgeDetectionMethod
        
        self.registry.register_extraction_method(
            ExtractionMode.CTFIRE, CTFireExtraction
        )
        self.registry.register_extraction_method(
            ExtractionMode.RIDGE_DETECTION, RidgeDetectionMethod
        )
    
    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams
    ) -> ExtractionResult:
        """
        Extract individual fibers from 2D image.
        
        Args:
            image: 2D grayscale image
            params: Extraction parameters with mode selection
            
        Returns:
            ExtractionResult with fiber properties
        """
        # Validate input
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_extraction_method(params.mode)
        method = method_class()
        
        # Run extraction with timing
        start_time = time.time()
        result = method.extract_2d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "2D"
        result.mode = params.mode
        result.processing_time = processing_time
        result.parameters = params.__dict__
        
        # Compute summary statistics
        self._compute_summary_statistics(result)
        
        return result
    
    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams
    ) -> ExtractionResult:
        """
        Extract individual fibers from 3D image.
        
        Args:
            image: 3D grayscale image
            params: Extraction parameters with mode selection
            
        Returns:
            ExtractionResult with fiber properties
        """
        # Validate input
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_extraction_method(params.mode)
        method = method_class()
        
        # Check if method supports 3D
        if not method.supports_3d():
            raise ValueError(
                f"{params.mode.value} does not support 3D extraction"
            )
        
        # Run extraction with timing
        start_time = time.time()
        result = method.extract_3d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "3D"
        result.mode = params.mode
        result.processing_time = processing_time
        result.parameters = params.__dict__
        
        # Compute summary statistics
        self._compute_summary_statistics(result)
        
        return result
    
    def _compute_summary_statistics(self, result: ExtractionResult):
        """Compute summary statistics from extracted fibers."""
        if not result.fibers:
            result.total_fiber_count = 0
            result.mean_fiber_length = 0.0
            result.mean_fiber_width = 0.0
            result.mean_straightness = 0.0
            return
        
        result.total_fiber_count = len(result.fibers)
        result.mean_fiber_length = np.mean([f.length for f in result.fibers])
        result.mean_fiber_width = np.mean([f.width for f in result.fibers])
        result.mean_straightness = np.mean([f.straightness for f in result.fibers])