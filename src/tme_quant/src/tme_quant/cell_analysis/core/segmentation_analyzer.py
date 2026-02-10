# tme_quant/cell_analysis/core/segmentation_analyzer.py
"""
Cell segmentation analyzer coordinator.

Manages different segmentation methods and orchestrates the segmentation pipeline.
"""

import numpy as np
from typing import Dict, Type
import time

from ..methods.segmentation.base_segmentation import BaseSegmentationMethod
from ..methods.registry import MethodRegistry
from ..config.segmentation_params import SegmentationParams, SegmentationResult
from ...core.tme_models.cell_model import SegmentationMode


class CellSegmentationAnalyzer:
    """
    Coordinates cell segmentation using different methods.
    
    Supports:
        - StarDist (deep learning, 2D/3D)
        - Cellpose (deep learning, 2D/3D)
        - Thresholding (classical, 2D/3D)
        - Watershed (classical, 2D/3D)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize segmentation analyzer.
        
        Args:
            verbose: Print progress messages
        """
        self.registry = MethodRegistry()
        self.verbose = verbose
        self._register_methods()
    
    def _register_methods(self):
        """Register all available segmentation methods."""
        from ..methods.segmentation.stardist import StarDistSegmentation
        from ..methods.segmentation.cellpose import CellposeSegmentation
        from ..methods.segmentation.thresholding import ThresholdingSegmentation
        from ..methods.segmentation.watershed import WatershedSegmentation
        
        self.registry.register_segmentation_method(
            SegmentationMode.STARDIST, StarDistSegmentation
        )
        self.registry.register_segmentation_method(
            SegmentationMode.CELLPOSE, CellposeSegmentation
        )
        self.registry.register_segmentation_method(
            SegmentationMode.THRESHOLDING, ThresholdingSegmentation
        )
        self.registry.register_segmentation_method(
            SegmentationMode.WATERSHED, WatershedSegmentation
        )
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells in 2D image.
        
        Args:
            image: 2D image (H, W) or (H, W, C)
            params: Segmentation parameters with mode selection
            
        Returns:
            SegmentationResult with cell properties and masks
        """
        # Validate input
        if image.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 2D multi-channel image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_segmentation_method(params.mode)
        method = method_class(verbose=self.verbose)
        
        # Run segmentation with timing
        start_time = time.time()
        result = method.segment_2d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "2D"
        result.mode = params.mode
        result.image_modality = params.image_modality
        result.processing_time = processing_time
        result.parameters = params.to_dict()
        
        # Compute summary statistics
        self._compute_summary_statistics(result)
        
        return result
    
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells in 3D image.
        
        Args:
            image: 3D image (D, H, W) or (D, H, W, C)
            params: Segmentation parameters
            
        Returns:
            SegmentationResult with cell properties
        """
        # Validate input
        if image.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 3D multi-channel image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_segmentation_method(params.mode)
        method = method_class(verbose=self.verbose)
        
        # Check if method supports 3D
        if not method.supports_3d():
            raise ValueError(
                f"{params.mode.value} does not support 3D segmentation"
            )
        
        # Run segmentation with timing
        start_time = time.time()
        result = method.segment_3d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "3D"
        result.mode = params.mode
        result.image_modality = params.image_modality
        result.processing_time = processing_time
        result.parameters = params.to_dict()
        
        # Compute summary statistics
        self._compute_summary_statistics(result)
        
        return result
    
    def _compute_summary_statistics(self, result: SegmentationResult):
        """Compute summary statistics from segmented cells."""
        if not result.cells:
            result.total_cell_count = 0
            result.mean_cell_area = 0.0
            result.mean_circularity = 0.0
            return
        
        result.total_cell_count = len(result.cells)
        result.mean_cell_area = float(np.mean([c.area for c in result.cells]))
        result.mean_circularity = float(np.mean([c.circularity for c in result.cells]))