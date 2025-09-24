"""
CurveAlign visualization system.

This module provides a pluggable visualization architecture supporting
multiple backends for different scientific workflows:

- matplotlib: Default backend for basic visualization
- napari: Interactive 3D visualization and analysis
- imagej: Integration with ImageJ/FIJI workflows

The visualization system is organized into:
- backends: Framework-specific implementations
- renderers: Low-level rendering functions
"""

# Import backend detection
from .backends import (
    get_available_backends,
    MATPLOTLIB_AVAILABLE, NAPARI_AVAILABLE, IMAGEJ_AVAILABLE
)

# Import high-level convenience functions
def create_overlay(*args, backend="matplotlib", **kwargs):
    """
    Create an overlay visualization using specified backend.
    
    Parameters
    ----------
    backend : str, default "matplotlib"
        Visualization backend to use
    *args, **kwargs
        Arguments passed to backend-specific function
        
    Returns
    -------
    np.ndarray
        Overlay visualization
    """
    if backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        from .backends.matplotlib_backend import create_overlay
        return create_overlay(*args, **kwargs)
    else:
        raise ValueError(f"Backend '{backend}' not available. Available: {get_available_backends()}")


def create_angle_maps(*args, backend="matplotlib", **kwargs):
    """
    Create angle maps using specified backend.
    
    Parameters
    ----------
    backend : str, default "matplotlib"  
        Visualization backend to use
    *args, **kwargs
        Arguments passed to backend-specific function
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw and processed angle maps
    """
    if backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        from .backends.matplotlib_backend import create_angle_maps_backend
        return create_angle_maps_backend(*args, **kwargs)
    else:
        raise ValueError(f"Backend '{backend}' not available. Available: {get_available_backends()}")


# Import backend modules for direct access
from . import backends
from . import renderers

# Legacy compatibility - import standalone as alias for matplotlib backend
if MATPLOTLIB_AVAILABLE:
    from .backends import matplotlib_backend as standalone
    
    # Create standalone module compatibility
    class _StandaloneModule:
        """Compatibility module for legacy standalone imports."""
        
        def __init__(self):
            if MATPLOTLIB_AVAILABLE:
                from .backends.matplotlib_backend import (
                    create_overlay, create_angle_maps_backend, plot_results
                )
                self.create_overlay = create_overlay
                self.create_angle_maps = create_angle_maps_backend
                self.plot_results = plot_results
            else:
                def _not_available(*args, **kwargs):
                    raise ImportError("matplotlib backend not available")
                self.create_overlay = _not_available
                self.create_angle_maps = _not_available
                self.plot_results = _not_available
    
    standalone = _StandaloneModule()

__all__ = [
    'create_overlay', 'create_angle_maps',
    'get_available_backends',
    'backends', 'renderers',
    'standalone',  # Legacy compatibility
    'MATPLOTLIB_AVAILABLE', 'NAPARI_AVAILABLE', 'IMAGEJ_AVAILABLE'
]