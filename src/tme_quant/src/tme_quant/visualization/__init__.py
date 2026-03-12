"""
Visualization utilities for registration assessment and TME interaction display.
"""

from .interaction_visualization import InteractionVisualizer

__all__ = [
    'InteractionVisualizer',
]

# Optional: registration visualization utilities (modules not yet created)
try:
    from .checkerboard import (
        create_checkerboard,
        create_checkerboard_rgb,
    )
    __all__ += ['create_checkerboard', 'create_checkerboard_rgb']
except ImportError:
    pass

try:
    from .overlay import (
        create_overlay,
        create_rgb_overlay,
        create_side_by_side,
    )
    __all__ += ['create_overlay', 'create_rgb_overlay', 'create_side_by_side']
except ImportError:
    pass

try:
    from .difference_map import (
        compute_difference_map,
        create_difference_overlay,
        compute_registration_quality_map,
    )
    __all__ += ['compute_difference_map', 'create_difference_overlay', 'compute_registration_quality_map']
except ImportError:
    pass