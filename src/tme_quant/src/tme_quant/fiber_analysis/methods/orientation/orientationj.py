"""OrientationJ Fiji plugin interface."""

import numpy as np
from .base_orientation import BaseOrientationMethod
from ...config.orientation_params import OrientationParams, OrientationResult
from ...io.fiji_bridge import FijiBridge


class OrientationJMethod(BaseOrientationMethod):
    """
    OrientationJ method using Fiji plugin interface.
    
    Provides pixel-wise orientation calculation using structure tensor
    or gradient-based methods through the OrientationJ Fiji plugin.
    
    Note: Requires Fiji to be installed and accessible.
    """
    
    def __init__(self):
        super().__init__()
        self.fiji_bridge = FijiBridge()
    
    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams
    ) -> OrientationResult:
        """
        Analyze 2D fiber orientation using OrientationJ.
        
        Calls Fiji's OrientationJ plugin for analysis.
        """
        # Check Fiji availability
        if not self.fiji_bridge.is_fiji_available():
            raise RuntimeError(
                "Fiji is not available. Please install Fiji and set FIJI_PATH."
            )
        
        # Prepare OrientationJ parameters
        orientationj_params = {
            'gradient': params.gradient_method,
            'min-coherency': params.coherency_threshold,
            'min-energy': 0.0
        }
        
        # Call Fiji plugin
        result_dict = self.fiji_bridge.call_orientationj(
            image, orientationj_params
        )
        
        # Parse results
        orientation_map = result_dict['orientation']  # -90 to 90 degrees
        coherency_map = result_dict['coherency']  # 0 to 1
        energy_map = result_dict['energy']
        
        # Compute statistics
        stats = self._compute_statistics(orientation_map, coherency_map)
        
        # Create result
        result = OrientationResult(
            mode=params.mode,
            dimension="2D",
            orientation_map=orientation_map,
            coherency_map=coherency_map,
            energy_map=energy_map,
            mean_orientation=stats['mean_orientation'],
            orientation_distribution=stats['orientation_distribution'],
            alignment_score=stats['alignment_score'],
            pixel_size=params.pixel_size
        )
        
        return result
    
    def supports_3d(self) -> bool:
        return False  # OrientationJ is 2D only