"""
Feature computation processor.

This module provides high-level feature computation orchestration,
implementing the algorithms from getCT.m and related functions.
"""

from typing import List, Optional, Sequence
import numpy as np
from scipy.spatial import cKDTree

from ...types import Curvelet, FeatureTable, FeatureOptions


def compute_features(
    curvelets: Sequence[Curvelet],
    options: Optional[FeatureOptions] = None,
) -> FeatureTable:
    """
    Compute fiber features from curvelets.
    
    This implements the feature computation from getCT.m, including:
    - Nearest neighbor density features
    - Local alignment features  
    - Box-based density and alignment
    
    Parameters
    ----------
    curvelets : Sequence[Curvelet]
        List of extracted curvelets
    options : FeatureOptions, optional
        Feature computation parameters
        
    Returns
    -------
    FeatureTable
        Dictionary of computed features for each curvelet
    """
    if options is None:
        options = FeatureOptions()
    
    if not curvelets:
        return {}
    
    n_curvelets = len(curvelets)
    
    # Extract positions and angles
    centers = np.array([[c.center_row, c.center_col] for c in curvelets])
    angles = np.array([c.angle_deg for c in curvelets])
    weights = np.array([c.weight or 1.0 for c in curvelets])
    
    # Build spatial index for efficient neighbor queries
    tree = cKDTree(centers)
    
    # Initialize feature arrays
    features = {
        'center_row': centers[:, 0],
        'center_col': centers[:, 1], 
        'angle_deg': angles,
        'weight': weights,
        'density_nn': np.zeros(n_curvelets),
        'alignment_nn': np.zeros(n_curvelets),
        'density_box': np.zeros(n_curvelets),
        'alignment_box': np.zeros(n_curvelets)
    }
    
    # Compute nearest neighbor features
    for i in range(n_curvelets):
        center = centers[i]
        
        # Find k nearest neighbors (excluding self)
        distances, neighbor_indices = tree.query(
            center, 
            k=options.minimum_nearest_fibers + 1
        )
        
        # Remove self from neighbors
        neighbor_indices = neighbor_indices[1:]
        distances = distances[1:]
        
        if len(neighbor_indices) >= options.minimum_nearest_fibers:
            # Density: inverse of mean distance to k-NN
            mean_distance = np.mean(distances)
            features['density_nn'][i] = 1.0 / (mean_distance + 1e-6)
            
            # Alignment: coherence of angles with neighbors
            neighbor_angles = angles[neighbor_indices]
            alignment = _compute_angle_coherence(angles[i], neighbor_angles)
            features['alignment_nn'][i] = alignment
    
    # Compute box-based features
    for i in range(n_curvelets):
        center = centers[i]
        
        # Define box around curvelet
        box_size = options.minimum_box_size
        box_neighbors = tree.query_ball_point(center, box_size / 2.0)
        
        # Remove self
        box_neighbors = [j for j in box_neighbors if j != i]
        
        if len(box_neighbors) > 0:
            # Box density: number of neighbors in box
            features['density_box'][i] = len(box_neighbors) / (box_size * box_size)
            
            # Box alignment
            neighbor_angles = angles[box_neighbors]
            alignment = _compute_angle_coherence(angles[i], neighbor_angles)
            features['alignment_box'][i] = alignment
    
    return features


def _compute_angle_coherence(reference_angle: float, neighbor_angles: np.ndarray) -> float:
    """
    Compute alignment coherence between reference angle and neighbors.
    
    Parameters
    ----------
    reference_angle : float
        Reference angle in degrees
    neighbor_angles : np.ndarray
        Neighbor angles in degrees
        
    Returns
    -------
    float
        Alignment coherence score (0-1, higher = more aligned)
    """
    if len(neighbor_angles) == 0:
        return 0.0
    
    # Convert to radians for circular statistics
    ref_rad = np.radians(reference_angle)
    neighbor_rad = np.radians(neighbor_angles)
    
    # Compute angle differences (handling periodicity)
    angle_diffs = np.abs(neighbor_rad - ref_rad)
    angle_diffs = np.minimum(angle_diffs, np.pi - angle_diffs)
    
    # Convert to coherence score (1 = perfect alignment, 0 = random)
    coherence = np.mean(np.cos(2 * angle_diffs))
    return max(0.0, coherence)
