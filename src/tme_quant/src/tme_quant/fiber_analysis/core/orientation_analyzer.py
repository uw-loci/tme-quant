"""Fiber orientation analysis coordinator."""

import numpy as np
from typing import Dict, Type
import time

from ..methods.orientation.base_orientation import BaseOrientationMethod
from ..methods.registry import MethodRegistry
from ..config.orientation_params import OrientationParams, OrientationResult


class FiberOrientationAnalyzer:
    """
    Coordinates fiber orientation analysis using different methods.
    
    Supports:
        - CurveAlign (curvelet-based, 2D/3D)
        - OrientationJ (Fiji plugin, 2D)
        - Pixel-wise gradient (2D)
        - Voxel-wise gradient (3D)
        - Structure tensor (2D/3D)
    """
    
    def __init__(self):
        self.registry = MethodRegistry()
        self._register_methods()
    
    def _register_methods(self):
        """Register all available orientation methods."""
        from ..methods.orientation.curvealign import CurveAlignOrientation
        from ..methods.orientation.orientationj import OrientationJMethod
        # Additional methods can be registered here
        
        self.registry.register_orientation_method(
            OrientationMode.CURVEALIGN, CurveAlignOrientation
        )
        self.registry.register_orientation_method(
            OrientationMode.ORIENTATIONJ, OrientationJMethod
        )
    
    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams
    ) -> OrientationResult:
        """
        Analyze fiber orientation in 2D image.
        
        Args:
            image: 2D grayscale image
            params: Orientation parameters with mode selection
            
        Returns:
            OrientationResult with orientation maps and statistics
        """
        # Validate input
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_orientation_method(params.mode)
        method = method_class()
        
        # Run analysis with timing
        start_time = time.time()
        result = method.analyze_2d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "2D"
        result.mode = params.mode
        result.processing_time = processing_time
        result.parameters = params.__dict__
        
        return result
    
    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams
    ) -> OrientationResult:
        """
        Analyze fiber orientation in 3D image.
        
        Args:
            image: 3D grayscale image
            params: Orientation parameters with mode selection
            
        Returns:
            OrientationResult with orientation maps and statistics
        """
        # Validate input
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image.ndim}D")
        
        # Get method
        method_class = self.registry.get_orientation_method(params.mode)
        method = method_class()
        
        # Check if method supports 3D
        if not method.supports_3d():
            raise ValueError(
                f"{params.mode.value} does not support 3D analysis"
            )
        
        # Run analysis with timing
        start_time = time.time()
        result = method.analyze_3d(image, params)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.dimension = "3D"
        result.mode = params.mode
        result.processing_time = processing_time
        result.parameters = params.__dict__
        
        return result