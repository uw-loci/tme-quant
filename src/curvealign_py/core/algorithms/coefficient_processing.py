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
    Threshold coefficients at a specific scale using MATLAB histogram-based approach.
    
    This implements the exact coefficient selection logic from newCurv.m lines 61-76.
    
    Parameters
    ----------
    scale_coeffs : List[np.ndarray]
        Coefficients for all wedges at one scale
    keep : float
        Fraction of coefficients to keep (e.g., 0.01 = top 1%)
        
    Returns
    -------
    List[np.ndarray]
        Thresholded coefficients (zeros where below threshold)
    """
    if not scale_coeffs or all(coeff.size == 0 for coeff in scale_coeffs):
        return [coeff.copy() for coeff in scale_coeffs]
    
    # MATLAB logic: find the maximum coefficient value
    # absMax = max(cellfun(@max,cellfun(@max,C{s},'UniformOutput',0)));
    abs_max = max(max(np.max(np.abs(coeff)) for coeff in scale_coeffs), 1e-10)
    
    # MATLAB logic: bins = 0:.01*absMax:absMax;
    bins = np.arange(0, abs_max + 0.01 * abs_max, 0.01 * abs_max)
    
    # MATLAB logic: histVals = cellfun(@(x) hist(x,bins),C{s},'UniformOutput',0);
    hist_vals = []
    for coeff in scale_coeffs:
        abs_coeff = np.abs(coeff)
        hist_val, _ = np.histogram(abs_coeff.ravel(), bins=bins)
        hist_vals.append(hist_val)
    
    # MATLAB logic: sumHist = cellfun(@(x) sum(x,2),histVals,'UniformOutput',0);
    sum_hist = [np.sum(hist_val) for hist_val in hist_vals]
    
    # MATLAB logic: totHist = horzcat(sumHist{aa});
    # sumVals = sum(totHist,2);
    sum_vals = np.sum(hist_vals, axis=0)
    
    # MATLAB logic: cumVals = cumsum(sumVals);
    cum_vals = np.cumsum(sum_vals)
    
    # MATLAB logic: cumMax = max(cumVals);
    # loc = find(cumVals > (1-keep)*cumMax,1,'first');
    cum_max = np.max(cum_vals)
    threshold_idx = np.where(cum_vals > (1 - keep) * cum_max)[0]
    
    if len(threshold_idx) > 0:
        loc = threshold_idx[0]
        max_val = bins[loc]
    else:
        max_val = abs_max
    
    # MATLAB logic: Ct{s} = cellfun(@(x)(x .* abs(x >= maxVal)),C{s},'UniformOutput',0);
    thresholded = []
    for coeff in scale_coeffs:
        abs_coeff = np.abs(coeff)
        thresholded_coeff = np.where(abs_coeff >= max_val, coeff, 0.0)
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
