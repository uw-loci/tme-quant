"""
Configuration and parameter dataclasses for image registration.
"""

from __future__ import annotations


# ========================================================
# REGISTRATION PARAMS
# ========================================================

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum
import numpy as np


class RegistrationMethod(Enum):
    """Registration methods."""
    # Intensity-based
    MUTUAL_INFORMATION = "mutual_information"
    CROSS_CORRELATION = "cross_correlation"
    MEAN_SQUARES = "mean_squares"
    
    # Feature-based
    SIFT = "sift"
    ORB = "orb"
    
    # Landmark-based
    MANUAL_LANDMARKS = "manual_landmarks"
    AUTO_LANDMARKS = "auto_landmarks"
    
    # Deep learning
    COMIR = "comir"
    
    # Specialized
    HE_SHG = "he_shg"  # Bredfeldt method


class TransformType(Enum):
    """Geometric transformation types."""
    TRANSLATION = "translation"      # 2 DOF (tx, ty)
    RIGID = "rigid"                  # 3 DOF (tx, ty, rotation)
    SIMILARITY = "similarity"        # 4 DOF (tx, ty, rotation, scale)
    AFFINE = "affine"               # 6 DOF (full affine)
    DEFORMABLE = "deformable"       # Non-rigid (B-spline)


class MicroscopyModality(Enum):
    """Microscopy imaging modalities."""
    # Cell imaging
    HE_BRIGHTFIELD = "he_brightfield"
    FLUORESCENCE_NUCLEI = "fluorescence_nuclei"
    FLUORESCENCE_MARKERS = "fluorescence_markers"
    PHASE_CONTRAST = "phase_contrast"
    FLIM = "flim"
    MULTIPHOTON = "multiphoton"
    
    # Fiber imaging
    SHG = "shg"
    POLARIZED_LIGHT = "polarized_light"
    LC_POLSCOPE = "lc_polscope"
    QLIPP = "qlipp"
    UPTI = "upti"
    PPM = "ppm"
    TRICHROME = "trichrome"
    CONFOCAL_COLLAGEN = "confocal_collagen"
    
    # Generic
    UNKNOWN = "unknown"


@dataclass
class RegistrationParams:
    """
    Parameters for image registration.
    """
    
    # Method
    method: RegistrationMethod = RegistrationMethod.MUTUAL_INFORMATION
    transform_type: TransformType = TransformType.AFFINE
    
    # Modality information
    fixed_modality: MicroscopyModality = MicroscopyModality.UNKNOWN
    moving_modality: MicroscopyModality = MicroscopyModality.UNKNOWN
    
    # Optimization parameters
    num_iterations: int = 200
    learning_rate: float = 1.0
    convergence_threshold: float = 1e-6
    
    # Multi-resolution
    use_multiresolution: bool = True
    pyramid_levels: int = 3
    
    # Preprocessing
    normalize_intensity: bool = True
    histogram_matching: bool = False
    
    # Method-specific parameters
    mi_bins: int = 50  # Mutual information histogram bins
    feature_detector_threshold: float = 0.01
    landmark_smoothing: float = 0.1
    
    # Output options
    return_transform_matrix: bool = True
    return_registered_image: bool = True
    return_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'transform_type': self.transform_type.value,
            'fixed_modality': self.fixed_modality.value,
            'moving_modality': self.moving_modality.value,
            'num_iterations': self.num_iterations,
            'use_multiresolution': self.use_multiresolution,
        }


@dataclass
class Transform:
    """
    Geometric transformation representation.
    """
    
    transform_type: TransformType
    matrix: np.ndarray  # Transformation matrix
    parameters: Optional[np.ndarray] = None  # Parameter vector
    
    # Metadata
    source_image_shape: Optional[Tuple[int, int]] = None
    target_image_shape: Optional[Tuple[int, int]] = None
    pixel_size_source: Optional[float] = None
    pixel_size_target: Optional[float] = None
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply transformation to an image."""
        from scipy.ndimage import affine_transform
        
        if self.transform_type in [TransformType.TRANSLATION, TransformType.RIGID, 
                                     TransformType.SIMILARITY, TransformType.AFFINE]:
            # Affine transformation
            return affine_transform(image, np.linalg.inv(self.matrix))
        else:
            raise NotImplementedError(f"Transform type {self.transform_type} not yet implemented")
    
    def invert(self) -> 'Transform':
        """Return inverse transformation."""
        return Transform(
            transform_type=self.transform_type,
            matrix=np.linalg.inv(self.matrix),
            parameters=-self.parameters if self.parameters is not None else None
        )
    
    def compose(self, other: 'Transform') -> 'Transform':
        """Compose with another transformation."""
        return Transform(
            transform_type=self.transform_type,
            matrix=self.matrix @ other.matrix
        )


@dataclass
class RegistrationResult:
    """
    Result from image registration.
    """
    
    # Core results
    transform: Transform
    registered_image: Optional[np.ndarray] = None
    
    # Quality metrics
    final_metric_value: Optional[float] = None
    initial_metric_value: Optional[float] = None
    mutual_information: Optional[float] = None
    normalized_cross_correlation: Optional[float] = None
    
    # Optimization info
    num_iterations: int = 0
    converged: bool = False
    optimization_time: float = 0.0
    
    # Method info
    method: Optional[RegistrationMethod] = None
    transform_type: Optional[TransformType] = None
    
    # Additional data
    landmarks_fixed: Optional[np.ndarray] = None
    landmarks_moving: Optional[np.ndarray] = None
    target_registration_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value if self.method else None,
            'transform_type': self.transform_type.value if self.transform_type else None,
            'final_metric_value': self.final_metric_value,
            'mutual_information': self.mutual_information,
            'num_iterations': self.num_iterations,
            'converged': self.converged,
            'optimization_time': self.optimization_time,
        }