"""Base class for fiber orientation methods."""

from abc import ABC, abstractmethod
import numpy as np
from ...config.orientation_params import OrientationParams, OrientationResult


class BaseOrientationMethod(ABC):
    """Abstract base class for fiber orientation analysis methods."""
    
    @abstractmethod
    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams
    ) -> OrientationResult:
        """Analyze fiber orientation in 2D image."""
        pass
    
    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams
    ) -> OrientationResult:
        """Analyze fiber orientation in 3D image."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 3D analysis"
        )
    
    def supports_3d(self) -> bool:
        """Check if method supports 3D analysis."""
        return False
    
    def _compute_statistics(
        self,
        orientation_map: np.ndarray,
        coherency_map: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute orientation statistics."""
        stats = {}
        
        # Mean orientation (circular mean)
        angles_rad = np.deg2rad(orientation_map)
        mean_x = np.mean(np.cos(2 * angles_rad))
        mean_y = np.mean(np.sin(2 * angles_rad))
        stats['mean_orientation'] = np.rad2deg(np.arctan2(mean_y, mean_x) / 2)
        
        # Alignment score (order parameter)
        stats['alignment_score'] = np.sqrt(mean_x**2 + mean_y**2)
        
        # Orientation distribution (histogram)
        hist, bin_edges = np.histogram(
            orientation_map, bins=36, range=(-90, 90)
        )
        stats['orientation_distribution'] = hist
        
        return stats