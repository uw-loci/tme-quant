"""
Trim and reorganize X, F, V data structures

This function removes empty fibers and reorganizes vertex and fiber data structures
to maintain consistency.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union


def trimxfv(
    X: np.ndarray,
    F: List[Dict],
    V: Optional[List[Dict]] = None,
    R: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, List[Dict], List[Dict]], 
           Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]]:
    """
    Trim and reorganize X, F, V, R data structures.
    
    Removes empty fibers and vertices, and renumbers everything to maintain
    consistency. This is called after operations that remove fibers.
    
    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates array, shape (N, 2) or (N, 3)
        Each row is [y, x] or [z, y, x] in MATLAB convention
    F : list of dict
        Fiber structure, each element contains:
        - 'v': list of vertex indices (1-based from MATLAB)
        - Optional: 'r', 'a', 'f' fields
    V : list of dict, optional
        Vertex structure, each element contains:
        - 'f': list of fiber indices that include this vertex
        - 'fe': list of fiber indices where this vertex is an endpoint
        - 'vall': list of all vertices in connected fibers
    R : np.ndarray, optional
        Radius array for each vertex
        
    Returns
    -------
    X : np.ndarray
        Trimmed vertex array
    F : list of dict
        Trimmed and renumbered fiber array
    V : list of dict
        Trimmed and renumbered vertex array (if provided)
    R : np.ndarray
        Trimmed radius array (if provided)
        
    Notes
    -----
    This function handles the conversion between MATLAB's 1-based indexing
    and Python's 0-based indexing internally.
    """
    X = np.asarray(X)
    if R is not None:
        R = np.asarray(R)

    # Remove empty fibers (fibers with no vertices)
    F_trimmed = []
    for fi, fiber in enumerate(F):
        if 'v' in fiber and len(fiber['v']) > 0:
            F_trimmed.append(fiber.copy())
    
    if len(F_trimmed) == 0:
        # No fibers remain
        empty_X = np.zeros((0, X.shape[1]))
        empty_F = []
        empty_V = [] if V is not None else None
        empty_R = np.zeros(0) if R is not None else None
        
        if R is not None and V is not None:
            return empty_X, empty_F, empty_V, empty_R
        elif V is not None:
            return empty_X, empty_F, empty_V
        else:
            return empty_X, empty_F
    
    # Find all vertices that are used by remaining fibers
    vertices_used = set()
    for fiber in F_trimmed:
        vertices_used.update(fiber['v'])
    
    # Convert to sorted list for consistent ordering
    vertices_used = sorted(list(vertices_used))
    
    if len(vertices_used) == 0:
        # No vertices used
        empty_X = np.zeros((0, X.shape[1]))
        empty_F = []
        empty_V = [] if V is not None else None
        empty_R = np.zeros(0) if R is not None else None
        
        if R is not None and V is not None:
            return empty_X, empty_F, empty_V, empty_R
        elif V is not None:
            return empty_X, empty_F, empty_V
        else:
            return empty_X, empty_F
    
    # Vertex v is stored at X[v] throughout the pipeline (the C++ trimxfv_cpp
    # never renumbers, so indices are direct numpy indices into X).
    # Output uses 0-based new indices (new_idx = position in sorted vertices_used).
    old_to_new = {}
    for new_idx, old_idx in enumerate(vertices_used):
        old_to_new[old_idx] = new_idx  # 0-based output

    # Trim X: vertex old_idx lives at X[old_idx] (direct 0-based numpy index).
    X_trimmed = X[list(vertices_used), :]

    # Renumber fiber vertices
    F_renumbered = []
    for fiber in F_trimmed:
        fiber_new = fiber.copy()
        fiber_new['v'] = [old_to_new[v] for v in fiber['v']]
        F_renumbered.append(fiber_new)

    # Trim and renumber R if provided
    R_trimmed = None
    if R is not None:
        R_trimmed = R[list(vertices_used)]

    # Reconstruct V if provided
    V_reconstructed = None
    if V is not None:
        V_reconstructed = []
        for new_idx in range(len(vertices_used)):
            v_new = {
                'f': [],
                'fe': [],
                'vall': []
            }
            V_reconstructed.append(v_new)

        # Populate V based on F; vertex and fiber indices are both 0-based.
        for fi, fiber in enumerate(F_renumbered):
            vertices = fiber['v']

            for vi in vertices:
                V_reconstructed[vi]['f'].append(fi)
                V_reconstructed[vi]['vall'].extend(vertices)

            if len(vertices) > 0:
                V_reconstructed[vertices[0]]['fe'].append(fi)
                V_reconstructed[vertices[-1]]['fe'].append(fi)

        # Remove duplicates and sort
        for v in V_reconstructed:
            v['f'] = sorted(list(set(v['f'])))
            v['fe'] = sorted(list(set(v['fe'])))
            v['vall'] = sorted(list(set(v['vall'])))
    
    # Return based on what was provided
    if R_trimmed is not None and V_reconstructed is not None:
        return X_trimmed, F_renumbered, V_reconstructed, R_trimmed
    elif V_reconstructed is not None:
        return X_trimmed, F_renumbered, V_reconstructed
    else:
        return X_trimmed, F_renumbered
