"""Ridge Detection Fiji plugin interface."""

import numpy as np
from typing import List
from .base_extraction import BaseExtractionMethod
from ...config.extraction_params import ExtractionParams, ExtractionResult, FiberProperties
from ...io.fiji_bridge import FijiBridge
from ...utils.geometry_utils import compute_fiber_properties


class RidgeDetectionMethod(BaseExtractionMethod):
    """
    Ridge Detection method using Fiji plugin interface.
    
    Uses Fiji's Ridge Detection plugin to extract fiber-like structures.
    
    Note: Requires Fiji to be installed and accessible.
    """
    
    def __init__(self):
        super().__init__()
        self.fiji_bridge = FijiBridge()
    
    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams
    ) -> ExtractionResult:
        """
        Extract fibers using Ridge Detection plugin.
        
        Calls Fiji's Ridge Detection for fiber extraction.
        """
        # Check Fiji availability
        if not self.fiji_bridge.is_fiji_available():
            raise RuntimeError(
                "Fiji is not available. Please install Fiji and set FIJI_PATH."
            )
        
        # Prepare Ridge Detection parameters
        ridge_params = {
            'line_width': params.ridge_sigma * 2,
            'high_contrast': params.upper_threshold * 255,
            'low_contrast': params.lower_threshold * 255,
            'make_binary': False,
            'extend_line': True
        }
        
        # Call Fiji plugin
        result_dict = self.fiji_bridge.call_ridge_detection(
            image, ridge_params
        )
        
        # Parse results - get list of line coordinates
        line_coords = result_dict['lines']  # List of Nx2 arrays
        
        # Compute properties for each fiber
        fibers = []
        for fiber_id, coords in enumerate(line_coords):
            # Filter by length
            pixel_length = len(coords)
            physical_length = pixel_length * params.pixel_size
            
            if physical_length < params.min_fiber_length:
                continue
            
            # Compute properties
            props = compute_fiber_properties(
                coords,
                image,
                params.pixel_size,
                width_range=(1.0, 20.0)
            )
            
            # Create FiberProperties
            fiber = FiberProperties(
                fiber_id=fiber_id,
                length=props['length'],
                width=props['width'],
                straightness=props['straightness'],
                angle=props['angle'],
                curvature=props['curvature'],
                centerline=coords
            )
            fibers.append(fiber)
        
        # Create result
        result = ExtractionResult(
            mode=params.mode,
            dimension="2D",
            fibers=fibers,
            total_fiber_count=len(fibers),
            mean_fiber_length=0.0,
            mean_fiber_width=0.0,
            mean_straightness=0.0,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def supports_3d(self) -> bool:
        return False  # Ridge Detection is 2D only