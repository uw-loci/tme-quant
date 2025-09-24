"""
Visualization backends for different frameworks.

This module provides backend implementations for various visualization
frameworks, allowing users to choose the most appropriate tool for
their workflow.
"""

# Import backends with try/except to handle optional dependencies
try:
    from .matplotlib_backend import (
        create_overlay as matplotlib_create_overlay,
        create_angle_maps_backend as matplotlib_create_angle_maps,
        plot_results as matplotlib_plot_results
    )
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from .napari_backend import (
        curvelets_to_napari_vectors,
        curvelets_to_napari_points,
        launch_napari_viewer,
        analysis_result_to_napari_layers
    )
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

try:
    from .imagej_backend import (
        analysis_result_to_imagej,
        launch_imagej_with_results,
        create_imagej_macro
    )
    IMAGEJ_AVAILABLE = True
except ImportError:
    IMAGEJ_AVAILABLE = False


def get_available_backends():
    """
    Get list of available visualization backends.
    
    Returns
    -------
    List[str]
        List of available backend names
    """
    backends = []
    if MATPLOTLIB_AVAILABLE:
        backends.append('matplotlib')
    if NAPARI_AVAILABLE:
        backends.append('napari')
    if IMAGEJ_AVAILABLE:
        backends.append('imagej')
    return backends


__all__ = [
    'get_available_backends',
    'MATPLOTLIB_AVAILABLE', 'NAPARI_AVAILABLE', 'IMAGEJ_AVAILABLE'
]

# Add backend-specific exports if available
if MATPLOTLIB_AVAILABLE:
    __all__.extend(['matplotlib_create_overlay', 'matplotlib_create_angle_maps', 'matplotlib_plot_results'])

if NAPARI_AVAILABLE:
    __all__.extend(['curvelets_to_napari_vectors', 'curvelets_to_napari_points', 
                   'launch_napari_viewer', 'analysis_result_to_napari_layers'])

if IMAGEJ_AVAILABLE:
    __all__.extend(['analysis_result_to_imagej', 'launch_imagej_with_results', 'create_imagej_macro'])
