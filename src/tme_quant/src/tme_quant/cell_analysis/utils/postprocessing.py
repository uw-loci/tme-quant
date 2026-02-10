"""
Post-processing utilities for cell segmentation results.
"""

import numpy as np
from typing import List
from ...core.tme_models.cell_model import CellProperties


def filter_cells(
    cells: List[CellProperties],
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    min_circularity: Optional[float] = None,
    max_eccentricity: Optional[float] = None
) -> List[CellProperties]:
    """
    Filter cells based on criteria.
    
    Args:
        cells: List of cells
        min_area: Minimum area
        max_area: Maximum area
        min_circularity: Minimum circularity
        max_eccentricity: Maximum eccentricity
        
    Returns:
        Filtered list of cells
    """
    filtered = []
    
    for cell in cells:
        # Area filter
        if min_area is not None and cell.area < min_area:
            continue
        if max_area is not None and cell.area > max_area:
            continue
        
        # Circularity filter
        if min_circularity is not None and cell.circularity < min_circularity:
            continue
        
        # Eccentricity filter
        if max_eccentricity is not None and cell.eccentricity > max_eccentricity:
            continue
        
        filtered.append(cell)
    
    return filtered


def remove_border_cells(
    cells: List[CellProperties],
    image_shape: Tuple[int, int],
    border_width: int = 5
) -> List[CellProperties]:
    """
    Remove cells touching image border.
    
    Args:
        cells: List of cells
        image_shape: Image shape (H, W)
        border_width: Border width in pixels
        
    Returns:
        Cells not touching border
    """
    h, w = image_shape
    
    filtered = []
    for cell in cells:
        x, y = cell.centroid
        
        # Check if near border
        if (x < border_width or x > w - border_width or
            y < border_width or y > h - border_width):
            continue
        
        filtered.append(cell)
    
    return filtered