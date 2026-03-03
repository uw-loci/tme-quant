"""
Distance calculation utilities for TME analysis.

Provides efficient distance computations between TME components.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, Polygon


def compute_pairwise_distances(
    points1: np.ndarray,
    points2: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two point sets.
    
    Args:
        points1: Array of shape (n, 2) or (n, 3)
        points2: Array of shape (m, 2) or (m, 3)
        
    Returns:
        Distance matrix of shape (n, m)
    """
    from scipy.spatial.distance import cdist
    return cdist(points1, points2)


def find_nearest_neighbors(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    k: int = 1,
    max_distance: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors using KD-tree.
    
    Args:
        query_points: Query points (n, 2) or (n, 3)
        reference_points: Reference points (m, 2) or (m, 3)
        k: Number of nearest neighbors
        max_distance: Maximum search distance (optional)
        
    Returns:
        Tuple of (distances, indices)
    """
    tree = cKDTree(reference_points)
    
    if max_distance is not None:
        distances, indices = tree.query(
            query_points,
            k=k,
            distance_upper_bound=max_distance
        )
    else:
        distances, indices = tree.query(query_points, k=k)
    
    return distances, indices


def compute_distance_to_boundary(
    points: np.ndarray,
    boundary: LineString
) -> np.ndarray:
    """
    Compute distances from points to a boundary line.
    
    Args:
        points: Array of points (n, 2)
        boundary: Shapely LineString boundary
        
    Returns:
        Array of distances (n,)
    """
    distances = np.array([
        Point(p).distance(boundary) for p in points
    ])
    return distances


def points_within_distance(
    query_point: np.ndarray,
    reference_points: np.ndarray,
    max_distance: float
) -> np.ndarray:
    """
    Find all points within a given distance.
    
    Args:
        query_point: Single query point
        reference_points: Reference points (n, 2)
        max_distance: Maximum distance
        
    Returns:
        Indices of points within distance
    """
    tree = cKDTree(reference_points)
    indices = tree.query_ball_point(query_point, max_distance)
    return np.array(indices)


def compute_distance_map(
    region_shape: Tuple[int, int],
    reference_points: np.ndarray,
    pixel_size: float = 1.0
) -> np.ndarray:
    """
    Compute distance map from reference points.
    
    Args:
        region_shape: Shape of output map (height, width)
        reference_points: Reference points (n, 2)
        pixel_size: Size of each pixel in microns
        
    Returns:
        Distance map array
    """
    h, w = region_shape
    
    # Create grid of coordinates
    y, x = np.mgrid[0:h, 0:w]
    grid_points = np.column_stack([x.ravel(), y.ravel()]) * pixel_size
    
    # Build KD-tree
    tree = cKDTree(reference_points)
    
    # Query distances
    distances, _ = tree.query(grid_points)
    
    # Reshape to map
    distance_map = distances.reshape(region_shape)
    
    return distance_map