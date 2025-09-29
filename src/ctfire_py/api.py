"""
CT-FIRE Python API - High-level interface for individual fiber analysis.

This module provides the main user-facing API for CT-FIRE functionality,
including individual fiber extraction, network analysis, and batch operations.
"""

from typing import Optional, Sequence, Iterable, List, Union
from pathlib import Path
import numpy as np

from .types import Fiber, FiberNetwork, CTFireResult, CTFireOptions, FiberMetrics
from .core.processors import (
    analyze_image_ctfire, analyze_fiber_network,
    compute_fiber_metrics, compute_ctfire_statistics
)


def analyze_image(
    image: np.ndarray,
    options: Optional[CTFireOptions] = None
) -> CTFireResult:
    """
    Analyze a single image for individual fiber extraction using CT-FIRE.
    
    This is the main high-level function for CT-FIRE analysis, combining
    curvelet enhancement with FIRE algorithm for individual fiber extraction.
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to analyze
    options : CTFireOptions, optional
        CT-FIRE analysis parameters and options
        
    Returns
    -------
    CTFireResult
        Complete analysis results including individual fibers and network
        
    Examples
    --------
    >>> import numpy as np
    >>> import ctfire
    >>> 
    >>> # Load your image
    >>> image = np.random.randn(256, 256)  # Replace with real image
    >>> 
    >>> # Basic CT-FIRE analysis
    >>> result = ctfire.analyze_image(image)
    >>> print(f"Found {len(result.fibers)} individual fibers")
    >>> 
    >>> # With custom options
    >>> options = ctfire.CTFireOptions(run_mode="ctfire", thresh_flen=50.0)
    >>> result = ctfire.analyze_image(image, options=options)
    """
    return analyze_image_ctfire(image, options)


def batch_analyze(
    inputs: Iterable[Union[Path, np.ndarray]],
    options: Optional[CTFireOptions] = None
) -> List[CTFireResult]:
    """
    Analyze multiple images in batch using CT-FIRE.
    
    Parameters
    ----------
    inputs : Iterable[Path | np.ndarray]
        Image paths or arrays to analyze
    options : CTFireOptions, optional
        CT-FIRE analysis parameters and options
        
    Returns
    -------
    List[CTFireResult]
        Analysis results for each input image
    """
    if options is None:
        options = CTFireOptions()
    
    results = []
    
    for input_item in inputs:
        # Load image if path provided
        if isinstance(input_item, (str, Path)):
            from skimage import io
            image = io.imread(str(input_item))
        else:
            image = input_item
        
        # Analyze image
        result = analyze_image(image, options=options)
        results.append(result)
    
    return results


def extract_fibers(
    image: np.ndarray,
    enhance_with_curvelets: bool = True,
    options: Optional[CTFireOptions] = None
) -> List[Fiber]:
    """
    Extract individual fibers from an image (mid-level API).
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image
    enhance_with_curvelets : bool, default True
        Whether to enhance image with curvelet transform
    options : CTFireOptions, optional
        FIRE algorithm parameters
        
    Returns
    -------
    List[Fiber]
        List of extracted individual fibers
    """
    if options is None:
        options = CTFireOptions()
    
    # Set run mode based on enhancement choice
    if enhance_with_curvelets:
        options.run_mode = "ctfire"
    else:
        options.run_mode = "fire"
    
    result = analyze_image_ctfire(image, options)
    return result.fibers


def get_fiber_metrics(fibers: List[Fiber]) -> FiberMetrics:
    """
    Compute detailed metrics for fibers (mid-level API).
    
    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers to analyze
        
    Returns
    -------
    FiberMetrics
        Detailed metrics for each fiber
    """
    return compute_fiber_metrics(fibers)


def get_network_analysis(fibers: List[Fiber]) -> FiberNetwork:
    """
    Analyze fiber network connectivity (mid-level API).
    
    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers to analyze
        
    Returns
    -------
    FiberNetwork
        Network analysis results
    """
    return analyze_fiber_network(fibers)
