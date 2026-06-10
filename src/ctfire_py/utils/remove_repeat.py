"""
Remove repeated vertices in fiber structures

This function identifies and removes vertices that appear multiple times
at the same location, merging their connections.
"""

import numpy as np
from typing import List, Dict, Tuple


def remove_repeat(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: np.ndarray
) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]:
    """
    Remove repeated vertices that occupy the same position.
    
    When multiple vertices exist at the same location, they are merged
    into a single vertex, and all fiber references are updated accordingly.
    
    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates, shape (N, 2) or (N, 3)
    F : list of dict
        Fiber structures
    V : list of dict
        Vertex structures
    R : np.ndarray
        Vertex radii
        
    Returns
    -------
    X : np.ndarray
        Coordinates with duplicates removed
    F : list of dict
        Updated fiber structures
    V : list of dict
        Updated vertex structures
    R : np.ndarray
        Updated radii
        
    Notes
    -----
    Vertices are considered duplicates if they are within a small tolerance
    (1e-10) of each other.
    """
    if len(X) == 0:
        return X, F, V, R
    
    n_vertices = len(X)
    
    # Find duplicate vertices
    # Map each vertex to its representative (first occurrence)
    vertex_map = np.arange(n_vertices) + 1  # 1-based
    tolerance = 1e-10
    
    # Check each pair of vertices for duplicates
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            # Check if vertices are at the same location
            if np.linalg.norm(X[i] - X[j]) < tolerance:
                # Map j to i's representative
                representative = vertex_map[i]
                vertex_map[j] = representative
    
    # Check if any duplicates were found
    unique_indices = np.unique(vertex_map)
    if len(unique_indices) == n_vertices:
        # No duplicates found
        return X, F, V, R
    
    # Create new vertex coordinates (keep first occurrence of each unique vertex)
    new_to_old = {}
    old_to_new = {}
    X_new = []
    R_new = []
    
    current_new_idx = 1
    for old_idx in range(1, n_vertices + 1):
        representative = vertex_map[old_idx - 1]
        
        if representative not in old_to_new:
            # This is a new unique vertex
            old_to_new[representative] = current_new_idx
            new_to_old[current_new_idx] = representative
            X_new.append(X[representative - 1])
            R_new.append(R[representative - 1])
            current_new_idx += 1
        
        # Map this old index to its new index
        old_to_new[old_idx] = old_to_new[representative]
    
    X_new = np.array(X_new)
    R_new = np.array(R_new)
    
    # Update fiber vertices
    F_new = []
    for fiber in F:
        fiber_new = fiber.copy()
        v_old = fiber['v']
        v_new = [old_to_new[v] for v in v_old]
        
        # Remove consecutive duplicates in fiber
        v_unique = []
        for v in v_new:
            if len(v_unique) == 0 or v != v_unique[-1]:
                v_unique.append(v)
        
        fiber_new['v'] = v_unique
        
        # Only keep fiber if it has at least 2 vertices
        if len(v_unique) >= 2:
            F_new.append(fiber_new)
    
    # Reconstruct V structure
    V_new = []
    for i in range(len(X_new)):
        V_new.append({
            'f': [],
            'fe': [],
            'vall': []
        })
    
    for fi, fiber in enumerate(F_new):
        fiber_idx = fi + 1
        vertices = fiber['v']
        
        for vi in vertices:
            V_new[vi - 1]['f'].append(fiber_idx)
            V_new[vi - 1]['vall'].extend(vertices)
        
        if len(vertices) > 0:
            V_new[vertices[0] - 1]['fe'].append(fiber_idx)
            V_new[vertices[-1] - 1]['fe'].append(fiber_idx)
    
    # Remove duplicates in V
    for v in V_new:
        v['f'] = sorted(list(set(v['f'])))
        v['fe'] = sorted(list(set(v['fe'])))
        v['vall'] = sorted(list(set(v['vall'])))
    
    return X_new, F_new, V_new, R_new
