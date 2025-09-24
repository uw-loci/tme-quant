"""
Fiber network data structure.

This module defines data structures for representing networks of
extracted fibers and their connectivity.
"""

from typing import NamedTuple, List, Dict, Optional
import numpy as np

from .fiber import Fiber


class FiberNetwork(NamedTuple):
    """
    Represents a network of interconnected fibers.
    
    Attributes
    ----------
    fibers : List[Fiber]
        List of individual fibers in the network
    intersections : List[tuple]
        Intersection points between fibers
    connectivity : Dict[int, List[int]]
        Connectivity matrix showing which fibers connect
    network_stats : Dict[str, float]
        Network-level statistics (density, connectivity, etc.)
    """
    fibers: List[Fiber]
    intersections: List[tuple]
    connectivity: Dict[int, List[int]]
    network_stats: Dict[str, float]


class FiberGraph(NamedTuple):
    """
    Graph representation of fiber network for analysis.
    
    Attributes
    ----------
    vertices : np.ndarray
        Array of vertex coordinates
    edges : np.ndarray
        Array of edge connections
    fiber_mapping : Dict[int, int]
        Mapping from fiber ID to graph vertices
    """
    vertices: np.ndarray
    edges: np.ndarray
    fiber_mapping: Dict[int, int]
