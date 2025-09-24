"""
ImageJ/FIJI-based visualization backend.

This module provides ImageJ integration for leveraging existing
ImageJ workflows and plugins with CurveAlign results.
"""

from typing import Dict, Any, Optional
import numpy as np
import tempfile
import os

from ...types import AnalysisResult, Curvelet


def analysis_result_to_imagej(
    result: AnalysisResult,
    image: np.ndarray,
    image_name: str = "curvealign_analysis"
) -> Dict[str, Any]:
    """
    Convert analysis result to ImageJ-compatible format.
    
    Parameters
    ----------
    result : AnalysisResult
        Complete analysis results
    image : np.ndarray
        Original image
    image_name : str, default "curvealign_analysis"
        Name for the image in ImageJ
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing ImageJ-compatible data structures
    """
    imagej_data = {
        'image': image,
        'image_name': image_name,
        'curvelets': [],
        'roi_manager_data': [],
        'results_table': []
    }
    
    # Convert curvelets to ImageJ format
    for i, curvelet in enumerate(result.curvelets):
        imagej_curvelet = {
            'id': i,
            'x': curvelet.center_col,
            'y': curvelet.center_row,
            'angle': curvelet.angle_deg,
            'weight': curvelet.weight or 1.0
        }
        imagej_data['curvelets'].append(imagej_curvelet)
    
    # Create ROI manager entries (lines representing curvelets)
    for i, curvelet in enumerate(result.curvelets):
        roi_entry = _curvelet_to_imagej_line_roi(curvelet, i)
        imagej_data['roi_manager_data'].append(roi_entry)
    
    # Create results table
    if result.features:
        for key, values in result.features.items():
            if hasattr(values, '__len__') and len(values) == len(result.curvelets):
                for i, value in enumerate(values):
                    if i >= len(imagej_data['results_table']):
                        imagej_data['results_table'].append({})
                    imagej_data['results_table'][i][key] = value
    
    return imagej_data


def launch_imagej_with_results(
    result: AnalysisResult,
    image: np.ndarray,
    image_name: str = "curvealign_analysis"
) -> 'imagej.ImageJ':
    """
    Launch ImageJ with CurveAlign results loaded.
    
    Parameters
    ----------
    result : AnalysisResult
        Complete analysis results
    image : np.ndarray
        Original image
    image_name : str, default "curvealign_analysis"
        Name for the image in ImageJ
        
    Returns
    -------
    imagej.ImageJ
        ImageJ instance with loaded data
    """
    try:
        import imagej
        import scyjava
    except ImportError:
        raise ImportError("PyImageJ is required for this backend. Install with: pip install pyimagej")
    
    # Initialize ImageJ
    ij = imagej.init()
    
    # Convert image to ImageJ format
    ij_image = ij.py.to_java(image)
    
    # Show image
    ij.ui().show(image_name, ij_image)
    
    # Add curvelets as ROIs
    if result.curvelets:
        _add_curvelets_to_imagej_roi_manager(ij, result.curvelets)
    
    # Add results to results table
    if result.features:
        _add_features_to_imagej_results_table(ij, result.features)
    
    return ij


def create_imagej_macro(
    result: AnalysisResult,
    image_path: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Generate ImageJ macro for reproducing the analysis visualization.
    
    Parameters
    ----------
    result : AnalysisResult
        Complete analysis results
    image_path : str
        Path to the original image file
    output_dir : str, optional
        Directory to save output files
        
    Returns
    -------
    str
        ImageJ macro code as string
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    macro_lines = [
        "// CurveAlign Analysis Results - Auto-generated ImageJ Macro",
        "",
        f'open("{image_path}");',
        "roiManager('reset');",
        "run('Clear Results');",
        ""
    ]
    
    # Add curvelets as line ROIs
    for i, curvelet in enumerate(result.curvelets):
        line_coords = _curvelet_to_imagej_line_coords(curvelet)
        macro_lines.extend([
            f"makeLine({line_coords['x1']}, {line_coords['y1']}, {line_coords['x2']}, {line_coords['y2']});",
            f"roiManager('add');",
            f"roiManager('select', {i});",
            f"roiManager('rename', 'Curvelet_{i}');",
        ])
    
    # Add measurements
    if result.features:
        macro_lines.extend([
            "",
            "// Add measurements to Results table",
            "for (i = 0; i < roiManager('count'); i++) {",
            "    roiManager('select', i);",
        ])
        
        # Add feature measurements
        for key in result.features.keys():
            macro_lines.append(f"    // {key} measurements would be added here")
        
        macro_lines.extend([
            "}",
            "roiManager('show all');",
        ])
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    macro_lines.extend([
        "",
        f"// Save outputs",
        f'roiManager("save", "{output_dir}/{base_name}_ROIs.zip");',
        f'saveAs("Results", "{output_dir}/{base_name}_Results.csv");',
        ""
    ])
    
    return "\n".join(macro_lines)


def _curvelet_to_imagej_line_roi(curvelet: Curvelet, roi_id: int) -> Dict[str, Any]:
    """Convert curvelet to ImageJ line ROI format."""
    line_coords = _curvelet_to_imagej_line_coords(curvelet)
    
    return {
        'type': 'line',
        'id': roi_id,
        'name': f'Curvelet_{roi_id}',
        'x1': line_coords['x1'],
        'y1': line_coords['y1'],
        'x2': line_coords['x2'],
        'y2': line_coords['y2'],
        'properties': {
            'angle': curvelet.angle_deg,
            'weight': curvelet.weight or 1.0
        }
    }


def _curvelet_to_imagej_line_coords(curvelet: Curvelet, line_length: float = 20.0) -> Dict[str, float]:
    """Calculate line coordinates for curvelet representation in ImageJ."""
    angle_rad = np.radians(curvelet.angle_deg)
    half_length = line_length / 2
    
    dx = half_length * np.cos(angle_rad)
    dy = half_length * np.sin(angle_rad)
    
    return {
        'x1': curvelet.center_col - dx,
        'y1': curvelet.center_row - dy,
        'x2': curvelet.center_col + dx,
        'y2': curvelet.center_row + dy
    }


def _add_curvelets_to_imagej_roi_manager(ij, curvelets):
    """Add curvelets as line ROIs to ImageJ ROI Manager."""
    # This would require more detailed PyImageJ integration
    # Implementation depends on specific PyImageJ API
    pass


def _add_features_to_imagej_results_table(ij, features):
    """Add feature data to ImageJ Results table."""
    # This would require more detailed PyImageJ integration
    # Implementation depends on specific PyImageJ API
    pass
