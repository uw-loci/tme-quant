# -*- coding: utf-8 -*-
"""
Fiber analysis utility modules.

curvelet_utils  — 2-D and 3-D curvelet transform (curvelops / MATLAB / NumPy)
ctfire_utils    — CT-FIRE FIRE algorithm (C++ wrapper + Python fallback)
geometry_utils  — fiber geometry and property measurement helpers
"""

from .curvelet_utils import curvelet_transform_2d, curvelet_transform_3d, available_backends
from .ctfire_utils   import fire_2d, fire_3d, ctfire_backend_status
from .geometry_utils import (
    find_nearest_boundary_point,
    compute_boundary_normal,
    compute_relative_angles,
    compute_fiber_to_boundary_alignment,
    compute_angle_to_boundary_normal,
    compute_angle_to_boundary_normal_simplified,
    compute_fiber_properties,
    compute_boundary_tangent_angle,
    find_nearest_boundary_index,
    compute_relative_fiber_angles,
)

__all__ = [
    'curvelet_transform_2d',
    'curvelet_transform_3d',
    'available_backends',
    'fire_2d',
    'fire_3d',
    'ctfire_backend_status',
    # Geometry / boundary-angle utilities
    'find_nearest_boundary_point',
    'compute_boundary_normal',
    'compute_relative_angles',
    'compute_fiber_to_boundary_alignment',
    'compute_angle_to_boundary_normal',
    'compute_angle_to_boundary_normal_simplified',
    'compute_fiber_properties',
    # Dense-trace tangent and unified relative-angle utilities
    'compute_boundary_tangent_angle',
    'find_nearest_boundary_index',
    'compute_relative_fiber_angles',
]