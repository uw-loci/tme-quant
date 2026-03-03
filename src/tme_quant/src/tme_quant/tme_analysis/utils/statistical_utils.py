"""
Statistical utilities for TME analysis.

Provides statistical tests and summary functions.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats


def compute_summary_statistics(
    data: np.ndarray
) -> Dict[str, float]:
    """
    Compute summary statistics for a dataset.
    
    Args:
        data: 1D array of values
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'count': len(data)
    }


def compute_circular_statistics(
    angles_deg: np.ndarray
) -> Dict[str, float]:
    """
    Compute circular statistics for angles.
    
    Args:
        angles_deg: Array of angles in degrees
        
    Returns:
        Dictionary with circular mean, variance, etc.
    """
    # Convert to radians
    angles_rad = np.radians(angles_deg)
    
    # Circular mean
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    circular_mean = np.degrees(np.arctan2(mean_sin, mean_cos))
    
    # Circular variance (1 - R where R is mean resultant length)
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    circular_variance = 1 - R
    
    # Circular standard deviation
    if R > 0:
        circular_std = np.degrees(np.sqrt(-2 * np.log(R)))
    else:
        circular_std = 180.0
    
    return {
        'circular_mean': float(circular_mean),
        'mean_resultant_length': float(R),
        'circular_variance': float(circular_variance),
        'circular_std': float(circular_std)
    }


def test_spatial_randomness(
    points: np.ndarray,
    region_area: float
) -> Dict[str, Any]:
    """
    Test if points are spatially random using nearest neighbor analysis.
    
    Args:
        points: Point coordinates (n, 2)
        region_area: Area of the region
        
    Returns:
        Dictionary with test results
    """
    from scipy.spatial.distance import cdist
    
    n = len(points)
    
    # Compute nearest neighbor distances
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    nn_distances = distances.min(axis=1)
    
    # Observed mean nearest neighbor distance
    observed_mean = np.mean(nn_distances)
    
    # Expected mean for random distribution
    density = n / region_area
    expected_mean = 0.5 / np.sqrt(density)
    
    # Nearest neighbor index
    nn_index = observed_mean / expected_mean
    
    # Z-score
    se = 0.26136 / np.sqrt(n * density)
    z_score = (observed_mean - expected_mean) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'observed_mean_nn_distance': float(observed_mean),
        'expected_mean_nn_distance': float(expected_mean),
        'nn_index': float(nn_index),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'interpretation': 'clustered' if nn_index < 1 else 'dispersed' if nn_index > 1 else 'random'
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_statistics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        stat = statistic_func(sample)
        bootstrap_statistics.append(stat)
    
    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))
    
    return lower, upper