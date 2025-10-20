"""
Main curvelet processing orchestrator.

This module provides high-level processing functions that coordinate
the various algorithms to extract and process curvelets from images.
"""

from typing import List, Tuple, Optional, Sequence
import numpy as np

from ...types import Curvelet, CtCoeffs
from ..algorithms import (
    apply_fdct, apply_ifdct, extract_parameters,
    threshold_coefficients_at_scale, create_empty_coeffs_like,
    extract_curvelets_from_coeffs, group_curvelets, 
    normalize_angles, filter_edge_curvelets
)


def extract_curvelets(
    image: np.ndarray,
    keep: float = 0.001,
    scale: Optional[int] = None,
    group_radius: Optional[float] = None,
) -> Tuple[List[Curvelet], CtCoeffs]:
    """
    Extract curvelets from an image using FDCT.
    
    This function implements the core curvelet extraction algorithm from newCurv.m:
    1. Apply forward FDCT to the image
    2. Threshold coefficients to keep only the strongest ones
    3. Extract center positions and angles from remaining coefficients
    4. Group nearby curvelets and compute mean angles
    5. Filter edge curvelets
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image
    keep : float, default 0.001
        Fraction of coefficients to keep (e.g., 0.001 = top 0.1%)
    scale : int, optional
        Specific scale to analyze (default: auto-select appropriate scale)
    group_radius : float, optional
        Radius for grouping nearby curvelets (pixels)
        
    Returns
    -------
    Tuple[List[Curvelet], CtCoeffs]
        Extracted curvelets and thresholded coefficient structure
    """
    # Apply forward FDCT - equivalent to: C = fdct_wrapping(IMG,0,2)
    # Note: MATLAB's third parameter (2) is nbangles_coarsest=2
    C = apply_fdct(image, finest=0, nbangles_coarsest=2)
    
    # Create empty coefficient structure for thresholding
    Ct = create_empty_coeffs_like(C)
    
    # Select scale to analyze
    if scale is None:
        # Default: second finest scale (scale=1 in MATLAB indexing)
        scale = 1
    # MATLAB scale indexing: 1=second finest, 2=third finest, etc.
    # MATLAB formula: s = length(C) - Sscale
    # So MATLAB scale=1 corresponds to Python index (len(C) - 1)
    s = len(C) - scale
    
    # Take absolute values of coefficients at selected scale
    for wedge_idx in range(len(C[s])):
        C[s][wedge_idx] = np.abs(C[s][wedge_idx])
    
    # Threshold coefficients - keep only the strongest 'keep' fraction
    Ct[s] = threshold_coefficients_at_scale(C[s], keep)
    
    # Extract center positions and angles (pass image shape for better scaling)
    X_rows, Y_cols = extract_parameters(Ct, img_shape=image.shape)
    
    # Convert coefficient positions to curvelet objects
    curvelets = extract_curvelets_from_coeffs(Ct[s], X_rows[s], Y_cols[s], s)
    
    # Group nearby curvelets if radius specified
    if group_radius is not None:
        curvelets = group_curvelets(curvelets, group_radius)
    
    # Normalize angles to 0-180 degree range (fiber symmetry)
    curvelets = normalize_angles(curvelets)
    
    # Remove curvelets too close to image edges
    curvelets = filter_edge_curvelets(curvelets, image.shape)
    
    return curvelets, Ct


def reconstruct_image(coeffs: CtCoeffs, scales: Optional[Sequence[int]] = None, img_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Reconstruct an image from curvelet coefficients.
    
    This implements the inverse FDCT reconstruction from CTrec.m and processImage.m.
    Equivalent to: Y = ifdct_wrapping(Ct, 0)
    
    Parameters
    ----------
    coeffs : CtCoeffs
        Curvelet coefficient structure from forward transform
    scales : Sequence[int], optional
        Specific scales to include in reconstruction (default: all scales)
    img_shape : Tuple[int, int], optional
        Target image shape for reconstruction
        
    Returns
    -------
    np.ndarray
        Reconstructed image
    """
    if scales is not None:
        # Create filtered coefficient structure with only specified scales
        filtered_coeffs = create_empty_coeffs_like(coeffs)
        for scale_idx in scales:
            if 0 <= scale_idx < len(coeffs):
                filtered_coeffs[scale_idx] = coeffs[scale_idx]
        coeffs = filtered_coeffs
    
    # Apply inverse FDCT (pass image shape for better reconstruction)
    return apply_ifdct(coeffs, finest=0, img_shape=img_shape)
