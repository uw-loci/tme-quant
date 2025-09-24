"""
napari-based visualization backend.

This module provides napari integration for interactive 3D visualization
and layer-based analysis of CurveAlign results.
"""

from typing import List, Tuple, Optional, Sequence, Dict, Any
import numpy as np

from ...types import Curvelet, AnalysisResult


def curvelets_to_napari_vectors(
    curvelets: Sequence[Curvelet],
    vector_length: float = 10.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert curvelets to napari vector layer format.
    
    Parameters
    ----------
    curvelets : Sequence[Curvelet]
        List of curvelets to convert
    vector_length : float, default 10.0
        Length of vector arrows in pixels
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Vector data array and properties dictionary for napari
    """
    if not curvelets:
        return np.empty((0, 2, 2)), {}
    
    vectors = []
    colors = []
    
    for curvelet in curvelets:
        # Calculate vector endpoints
        angle_rad = np.radians(curvelet.angle_deg)
        dx = vector_length * np.cos(angle_rad)
        dy = vector_length * np.sin(angle_rad)
        
        # Vector format: start point, end point
        start_point = [curvelet.center_row, curvelet.center_col]
        end_point = [
            curvelet.center_row + dy,
            curvelet.center_col + dx
        ]
        
        vectors.append([start_point, end_point])
        
        # Color based on angle
        color = _angle_to_napari_color(curvelet.angle_deg)
        colors.append(color)
    
    vector_data = np.array(vectors)
    properties = {
        'edge_color': colors,
        'edge_width': 2,
        'vector_style': 'arrow'
    }
    
    return vector_data, properties


def curvelets_to_napari_points(
    curvelets: Sequence[Curvelet]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert curvelets to napari points layer format.
    
    Parameters
    ----------
    curvelets : Sequence[Curvelet]
        List of curvelets to convert
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Points data array and properties dictionary for napari
    """
    if not curvelets:
        return np.empty((0, 2)), {}
    
    # Extract center points
    points = np.array([[c.center_row, c.center_col] for c in curvelets])
    
    # Create properties for visualization
    angles = [c.angle_deg for c in curvelets]
    weights = [c.weight or 1.0 for c in curvelets]
    colors = [_angle_to_napari_color(angle) for angle in angles]
    
    properties = {
        'face_color': colors,
        'edge_color': 'white',
        'size': np.array(weights) * 5 + 2,  # Scale weights to reasonable point sizes
        'properties': {
            'angle': angles,
            'weight': weights
        }
    }
    
    return points, properties


def launch_napari_viewer(
    result: AnalysisResult,
    image: np.ndarray,
    show_vectors: bool = True,
    show_points: bool = False
) -> 'napari.Viewer':
    """
    Launch napari viewer with CurveAlign results.
    
    Parameters
    ----------
    result : AnalysisResult
        Complete analysis results
    image : np.ndarray
        Original image
    show_vectors : bool, default True
        Whether to show curvelets as vectors
    show_points : bool, default False
        Whether to show curvelets as points
        
    Returns
    -------
    napari.Viewer
        napari viewer instance with loaded data
    """
    try:
        import napari
    except ImportError:
        raise ImportError("napari is required for this backend. Install with: pip install 'napari[all]'")
    
    viewer = napari.Viewer()
    
    # Add original image
    viewer.add_image(
        image,
        name='Original Image',
        colormap='gray'
    )
    
    # Add curvelet vectors
    if show_vectors and result.curvelets:
        vector_data, vector_props = curvelets_to_napari_vectors(result.curvelets)
        viewer.add_vectors(
            vector_data,
            name='Fiber Orientations',
            **vector_props
        )
    
    # Add curvelet points
    if show_points and result.curvelets:
        point_data, point_props = curvelets_to_napari_points(result.curvelets)
        viewer.add_points(
            point_data,
            name='Fiber Centers',
            **point_props
        )
    
    return viewer


def analysis_result_to_napari_layers(
    result: AnalysisResult,
    image: np.ndarray
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """
    Convert analysis result to napari layer specifications.
    
    Parameters
    ----------
    result : AnalysisResult
        Complete analysis results
    image : np.ndarray
        Original image
        
    Returns
    -------
    List[Tuple[str, Any, Dict[str, Any]]]
        List of (layer_type, data, properties) tuples for napari
    """
    layers = []
    
    # Image layer
    layers.append(('image', image, {'name': 'Original Image', 'colormap': 'gray'}))
    
    # Vector layer for curvelets
    if result.curvelets:
        vector_data, vector_props = curvelets_to_napari_vectors(result.curvelets)
        layers.append(('vectors', vector_data, {**vector_props, 'name': 'Fiber Orientations'}))
        
        point_data, point_props = curvelets_to_napari_points(result.curvelets)
        layers.append(('points', point_data, {**point_props, 'name': 'Fiber Centers'}))
    
    return layers


def _angle_to_napari_color(angle_deg: float) -> str:
    """
    Convert angle to napari-compatible color.
    
    Parameters
    ----------
    angle_deg : float
        Angle in degrees
        
    Returns
    -------
    str
        Color string for napari
    """
    import matplotlib.pyplot as plt
    
    # Normalize angle to [0, 1]
    normalized_angle = (angle_deg % 180) / 180.0
    
    # Get color from HSV colormap
    cmap = plt.get_cmap('hsv')
    color_rgba = cmap(normalized_angle)
    
    # Convert to hex string
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(color_rgba[0] * 255),
        int(color_rgba[1] * 255),
        int(color_rgba[2] * 255)
    )
    
    return hex_color
