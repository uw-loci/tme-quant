"""
Cell-specific utility functions.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree
from ...core.tme_models.cell_model import CellObject


def compute_cell_neighbors(
    cells: List[CellObject],
    max_distance: float = 50.0
) -> None:
    """
    Compute neighbors for all cells (modifies cells in-place).
    
    Args:
        cells: List of cells
        max_distance: Maximum distance for neighbors
    """
    if len(cells) < 2:
        return
    
    # Build KD-tree
    centroids = np.array([c.centroid for c in cells])
    tree = cKDTree(centroids)
    
    # Find neighbors for each cell
    for i, cell in enumerate(cells):
        # Query neighbors within max_distance
        indices = tree.query_ball_point(centroids[i], max_distance)
        
        # Remove self
        indices = [j for j in indices if j != i]
        
        # Update cell
        cell.neighbor_cell_ids = [cells[j].object_id for j in indices]
        
        # Compute distances
        if indices:
            distances = [
                np.linalg.norm(centroids[i] - centroids[j])
                for j in indices
            ]
            cell.neighbor_distances = distances
            cell.nearest_neighbor_distance = min(distances)


def compute_cell_density(
    cells: List[CellObject],
    region_area: float
) -> float:
    """
    Compute cell density (cells per square mm).
    
    Args:
        cells: List of cells
        region_area: Region area in square microns
        
    Returns:
        Cell density (cells per mm²)
    """
    if region_area <= 0:
        return 0.0
    
    # Convert to cells per mm²
    density = len(cells) / (region_area / 1e6)
    
    return density


def merge_touching_cells(
    cells: List[CellObject],
    distance_threshold: float = 1.0
) -> List[CellObject]:
    """
    Merge cells that are very close (oversegmentation correction).
    
    Args:
        cells: List of cells
        distance_threshold: Distance threshold for merging
        
    Returns:
        List of cells after merging
    """
    # Simple implementation: merge cells with centroids very close
    merged = []
    merged_ids = set()
    
    for i, cell_i in enumerate(cells):
        if cell_i.object_id in merged_ids:
            continue
        
        # Find cells to merge
        to_merge = [cell_i]
        
        for j, cell_j in enumerate(cells[i+1:], i+1):
            if cell_j.object_id in merged_ids:
                continue
            
            # Check distance
            dist = np.linalg.norm(
                np.array(cell_i.centroid) - np.array(cell_j.centroid)
            )
            
            if dist < distance_threshold:
                to_merge.append(cell_j)
                merged_ids.add(cell_j.object_id)
        
        # Merge if multiple cells
        if len(to_merge) > 1:
            merged_cell = _merge_cell_list(to_merge)
            merged.append(merged_cell)
        else:
            merged.append(cell_i)
    
    return merged


def _merge_cell_list(cells: List[CellObject]) -> CellObject:
    """Merge multiple cells into one."""
    # Take properties from largest cell
    largest = max(cells, key=lambda c: c.area)
    
    # Update area (sum)
    largest.area = sum(c.area for c in cells)
    
    # Update centroid (weighted average)
    total_area = largest.area
    centroid_x = sum(c.centroid[0] * c.area for c in cells) / total_area
    centroid_y = sum(c.centroid[1] * c.area for c in cells) / total_area
    largest.centroid = (centroid_x, centroid_y)
    
    return largest