"""
Validation utilities for TME analysis.

Provides input validation and error checking.
"""

import numpy as np
from typing import List, Optional


def validate_analysis_inputs(
    cells: Optional[List],
    fibers: Optional[List],
    tumor_regions: Optional[List],
    mode: str
) -> bool:
    """
    Validate inputs for TME analysis.
    
    Args:
        cells: List of CellObject instances
        fibers: List of FiberObject instances
        tumor_regions: List of TumorRegion instances
        mode: Analysis mode
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Check mode-specific requirements
    if mode == 'tumor_based':
        if not fibers:
            raise ValueError("Tumor-based analysis requires fibers")
        if not tumor_regions:
            raise ValueError("Tumor-based analysis requires tumor regions")
    
    elif mode == 'cell_based':
        if not cells:
            raise ValueError("Cell-based analysis requires cells")
    
    elif mode == 'fiber_based':
        if not fibers:
            raise ValueError("Fiber-based analysis requires fibers")
    
    return True


def validate_distance_threshold(
    distance: float,
    min_val: float = 0.0,
    max_val: float = 1000.0
) -> bool:
    """
    Validate distance threshold.
    
    Args:
        distance: Distance value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If distance is invalid
    """
    if distance < min_val or distance > max_val:
        raise ValueError(
            f"Distance must be between {min_val} and {max_val}, got {distance}"
        )
    
    return True


def validate_interaction_pairs(
    interaction_pairs: List
) -> bool:
    """
    Validate interaction pairs.
    
    Args:
        interaction_pairs: List of InteractionPair objects
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If pairs are invalid
    """
    if not interaction_pairs:
        return True  # Empty list is valid
    
    # Check that each pair has required attributes
    for pair in interaction_pairs:
        if not hasattr(pair, 'source_id') or not hasattr(pair, 'target_id'):
            raise ValueError("InteractionPair missing required attributes")
        
        if not hasattr(pair, 'distance'):
            raise ValueError("InteractionPair missing distance attribute")
    
    return True