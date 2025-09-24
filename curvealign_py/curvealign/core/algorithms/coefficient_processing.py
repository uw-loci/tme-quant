"""
Curvelet coefficient processing algorithms.

This module provides functions for processing curvelet coefficients,
including thresholding, filtering, and structure manipulation.
"""

from typing import List
import numpy as np

from ...types import CtCoeffs


def threshold_coefficients_at_scale(scale_coeffs: List[np.ndarray], keep: float) -> List[np.ndarray]:
    """
    Threshold coefficients at a specific scale to keep only the strongest.
    
    This implements the coefficient selection logic from newCurv.m.
    
    Parameters
    ----------
    scale_coeffs : List[np.ndarray]
        Coefficients for all wedges at one scale
    keep : float
        Fraction of coefficients to keep (e.g., 0.001 = top 0.1%)
        
    Returns
    -------
    List[np.ndarray]
        Thresholded coefficients (zeros where below threshold)
    """
    thresholded = []
    
    for coeff in scale_coeffs:
        if coeff.size == 0:
            thresholded.append(coeff.copy())
            continue
        
        # Take absolute values for thresholding
        abs_coeff = np.abs(coeff)
        
        # Find threshold value (keep top 'keep' fraction)
        n_total = abs_coeff.size
        n_keep = max(1, int(n_total * keep))
        
        # Get threshold value
        thresh_val = np.partition(abs_coeff.ravel(), -n_keep)[-n_keep]
        
        # Apply threshold
        mask = abs_coeff >= thresh_val
        thresholded_coeff = np.zeros_like(coeff)
        thresholded_coeff[mask] = coeff[mask]
        
        thresholded.append(thresholded_coeff)
    
    return thresholded


def create_empty_coeffs_like(coeffs: CtCoeffs) -> CtCoeffs:
    """
    Create an empty coefficient structure with the same layout as input.
    
    Parameters
    ----------
    coeffs : CtCoeffs
        Reference coefficient structure
        
    Returns
    -------
    CtCoeffs
        Empty coefficient structure with same dimensions
    """
    empty_coeffs = []
    
    for scale_coeffs in coeffs:
        scale_empty = []
        for coeff in scale_coeffs:
            scale_empty.append(np.zeros_like(coeff))
        empty_coeffs.append(scale_empty)
    
    return empty_coeffs


def get_nonzero_coefficient_positions(coeffs: List[np.ndarray]) -> List[tuple]:
    """
    Get positions of non-zero coefficients in a scale.
    
    Parameters
    ----------
    coeffs : List[np.ndarray]
        Coefficients for all wedges at one scale
        
    Returns
    -------
    List[tuple]
        List of (wedge_idx, row, col) positions with non-zero coefficients
    """
    positions = []
    
    for wedge_idx, coeff in enumerate(coeffs):
        if coeff.size == 0:
            continue
        
        # Find non-zero positions
        nonzero_rows, nonzero_cols = np.nonzero(coeff)
        
        for row, col in zip(nonzero_rows, nonzero_cols):
            positions.append((wedge_idx, row, col))
    
    return positions
