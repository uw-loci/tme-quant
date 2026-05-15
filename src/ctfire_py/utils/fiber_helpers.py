"""
Helper functions for fiber analysis

Contains utility functions for calculating fiber properties and converting
between different fiber representations.
"""

import numpy as np
from typing import List, Dict, Tuple


def calc_fiberlen(
    X: np.ndarray,
    F: List[Dict],
    R: np.ndarray
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Calculate the length of each fiber.
    
    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates, shape (N, 2) or (N, 3)
    F : list of dict
        Fiber structures, each with 'v' field containing vertex indices
    R : np.ndarray
        Vertex radii
        
    Returns
    -------
    F : list of dict
        Fiber structures with added 'len' field
    L : np.ndarray
        Array of fiber lengths
    RF : np.ndarray
        Average radius for each fiber
        
    Notes
    -----
    Length is calculated as the sum of Euclidean distances between
    consecutive vertices in the fiber.
    """
    L = np.zeros(len(F))
    RF = np.zeros(len(F))
    
    for fi, fiber in enumerate(F):
        vertices = fiber['v']
        
        if len(vertices) < 2:
            L[fi] = 0.0
            RF[fi] = 0.0 if len(vertices) == 0 else R[vertices[0]]
            continue

        # Calculate length as sum of segment lengths
        length = 0.0
        for i in range(len(vertices) - 1):
            v1_idx = vertices[i]      # already 0-based
            v2_idx = vertices[i + 1]

            segment_length = np.linalg.norm(X[v2_idx] - X[v1_idx])
            length += segment_length

        L[fi] = length

        # Calculate average radius for this fiber
        fiber_radii = [R[v] for v in vertices]
        RF[fi] = np.mean(fiber_radii)
        
        # Add length to fiber structure
        F[fi]['len'] = length
    
    return F, L, RF


def fiber2edge(
    F: List[Dict],
    V: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert fiber structure to edge matrix representation.
    
    Parameters
    ----------
    F : list of dict
        Fiber structures
    V : list of dict
        Vertex structures
        
    Returns
    -------
    E : np.ndarray
        Edge matrix, shape (M, 2) where each row is [start_vertex, end_vertex]
    Ve : np.ndarray
        Vertex edge list - for each vertex, which edges connect to it
        
    Notes
    -----
    An edge connects each pair of consecutive vertices in a fiber.
    The edge matrix has one row per edge.
    """
    # Build edge list
    edges = []
    vertex_edges = [[] for _ in range(len(V))]
    
    edge_idx = 0
    for fi, fiber in enumerate(F):
        vertices = fiber['v']
        
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            
            edges.append([v1, v2])
            
            # Track which edges connect to each vertex
            vertex_edges[v1 - 1].append(edge_idx)
            vertex_edges[v2 - 1].append(edge_idx)
            
            edge_idx += 1
    
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=int), np.array(vertex_edges, dtype=object)
    
    E = np.array(edges, dtype=int)
    Ve = np.array(vertex_edges, dtype=object)
    
    return E, Ve


def calc_persistent2(
    X: np.ndarray,
    F: List[Dict]
) -> np.ndarray:
    """
    Calculate persistence length for each fiber.
    
    The persistence length is a measure of how straight a fiber is.
    It represents the length scale over which the fiber's direction
    changes significantly.
    
    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates
    F : list of dict
        Fiber structures
        
    Returns
    -------
    Lp : np.ndarray
        Persistence length for each fiber
        
    Notes
    -----
    For now, this returns a placeholder implementation.
    The full persistence length calculation requires correlation
    analysis of tangent vectors along the fiber.
    """
    # Placeholder implementation
    # TODO: Implement full persistence length calculation
    Lp = np.zeros(len(F))
    
    for fi, fiber in enumerate(F):
        vertices = fiber['v']
        
        if len(vertices) < 3:
            Lp[fi] = 0.0
            continue
        
        # Simple approximation: ratio of end-to-end distance to contour length
        v_start = vertices[0] - 1
        v_end = vertices[-1] - 1
        
        end_to_end = np.linalg.norm(X[v_end] - X[v_start])
        
        # Calculate contour length
        contour = 0.0
        for i in range(len(vertices) - 1):
            v1 = vertices[i] - 1
            v2 = vertices[i + 1] - 1
            contour += np.linalg.norm(X[v2] - X[v1])
        
        if contour > 0:
            # Persistence length approximation
            # Higher ratio = straighter fiber = longer persistence length
            Lp[fi] = contour * (end_to_end / contour)
        else:
            Lp[fi] = 0.0
    
    return Lp
