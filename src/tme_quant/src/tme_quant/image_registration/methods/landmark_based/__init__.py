"""
Landmark-based registration methods.
"""

from .manual_landmarks import ManualLandmarkRegistration
from .thin_plate_spline import ThinPlateSpline

__all__ = [
    'ManualLandmarkRegistration',
    'ThinPlateSpline',
]