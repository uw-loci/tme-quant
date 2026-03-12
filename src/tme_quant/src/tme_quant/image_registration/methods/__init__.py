"""
Registration methods.

Provides various registration algorithms organized by approach:
- Intensity-based: MI, NCC, H&E-SHG
- Feature-based: SIFT, ORB
- Landmark-based: Manual landmarks, Thin-Plate Spline
- Deep learning: CoMIR, VoxelMorph
"""

from .base_registration import BaseRegistration

# Note: specialized/ folder removed - H&E-SHG is now in intensity_based/

__all__ = ['BaseRegistration']