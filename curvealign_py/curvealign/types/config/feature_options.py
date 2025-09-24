"""
Feature computation configuration options.

This module defines configuration options specifically for controlling
feature computation algorithms.
"""

from dataclasses import dataclass


@dataclass 
class FeatureOptions:
    """
    Options for feature computation.
    
    Parameters
    ----------
    minimum_nearest_fibers : int, default 4
        Minimum number of nearest neighbors for density/alignment
    minimum_box_size : int, default 16
        Minimum box size for local computations
    """
    minimum_nearest_fibers: int = 4
    minimum_box_size: int = 16
