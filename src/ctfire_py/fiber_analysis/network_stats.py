"""
Network Statistics Calculation

Computes various statistical properties of the extracted fiber network.
"""

import numpy as np
from typing import List, Dict, Any

from ctfire_py.utils.fiber_helpers import calc_fiberlen, fiber2edge, calc_persistent2


def network_statK(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate network statistics from fiber data.
    
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
    M : dict
        Dictionary containing network statistics:
        - fiber_num : int
            Number of fibers
        - L : np.ndarray
            Length of each fiber
        - avgL : float
            Average fiber length
        - totL : float
            Total length of all fibers
        - Ldens : float
            Length density (total length / volume)
        - fibervol : float
            Total volume of all fibers (assuming cylindrical)
        - vol : float
            Total volume of bounding box
        - volfrac : float
            Volume fraction occupied by fibers
        - dens : float
            Density (kg/m³), assuming collagen density 1360 kg/m³
        - angle_xz : np.ndarray
            Angle in xz plane for each fiber
        - angle_xy : np.ndarray
            Angle in xy plane for each fiber
        - coord : np.ndarray
            Coordination number (number of fibers meeting at each vertex)
        - avgcoord : float
            Average coordination number
        - xlinkdens : float
            Cross-link density (cross-links per unit length)
        - xlinkspace : np.ndarray
            Spacing between cross-links
        - Lp3 : np.ndarray
            Persistence length for each fiber
    """
    M = {}
    
    # Construct edge matrix
    E = np.zeros((len(F), 2), dtype=int)
    for fi in range(len(F)):
        if len(F[fi]['v']) > 0:
            E[fi, 0] = F[fi]['v'][0]
            E[fi, 1] = F[fi]['v'][-1]
    
    # Count number of fibers
    N = len(E)
    M['fiber_num'] = N
    
    # Calculate length of all fibers
    F_with_len, L, RF = calc_fiberlen(X, F, R)
    M['L'] = L
    M['avgL'] = np.mean(L) if len(L) > 0 else 0.0
    M['totL'] = np.sum(L)
    
    # Calculate volume fraction and density of network
    if len(X) > 0:
        bbox_min = np.min(X, axis=0)
        bbox_max = np.max(X, axis=0)
        bbox_size = bbox_max - bbox_min
        M['vol'] = np.prod(bbox_size)
    else:
        M['vol'] = 0.0
    
    M['Ldens'] = M['totL'] / M['vol'] if M['vol'] > 0 else 0.0
    
    # Calculate fiber volume (assuming cylindrical fibers)
    M['fibervol'] = np.sum(L * np.pi * RF**2)
    M['volfrac'] = M['fibervol'] / M['vol'] if M['vol'] > 0 else 0.0
    M['dens'] = M['volfrac'] * 1360  # Collagen density
    
    # Calculate angle orientation of fibers
    v1 = E[:, 0]
    v2 = E[:, 1]
    
    # Indices are already 0-based
    v1_idx = v1
    v2_idx = v2
    
    # Handle invalid indices
    valid_mask = (v1_idx >= 0) & (v1_idx < len(X)) & (v2_idx >= 0) & (v2_idx < len(X))
    
    x1 = X[v1_idx[valid_mask], :]
    x2 = X[v2_idx[valid_mask], :]
    
    # Calculate angles.
    # MATLAB: atan((x2(:,3)-x1(:,3)) ./ (x2(:,1)-x1(:,1)+eps))
    # Using single-argument arctan (range [-π/2, π/2]) to match MATLAB's `atan`.
    eps = np.finfo(float).eps
    if X.shape[1] >= 3:
        M['angle_xz'] = np.arctan(
            (x2[:, 2] - x1[:, 2]) / (x2[:, 0] - x1[:, 0] + eps)
        )
    else:
        M['angle_xz'] = np.zeros(len(x1))
    
    M['angle_xy'] = np.arctan(
        (x2[:, 1] - x1[:, 1]) / (x2[:, 0] - x1[:, 0] + eps)
    )
    
    # Calculate coordination number
    if len(F) > 2:
        E_edges, Ve = fiber2edge(F, V)
        coord = np.zeros(max(len(X), 1), dtype=int)
        
        if len(E_edges) > 0:
            # MATLAB: `LenE = length(E)-1` in both if/else branches, so the
            # loop runs 1:LenE = 1:n_edges-1 and skips the last edge row.
            # This is a MATLAB quirk preserved here for faithfulness.
            for k in range(len(E_edges) - 1):
                v = E_edges[k, :]
                for vi in v:
                    if 0 <= vi < len(coord):
                        coord[vi] += 1
        
        M['coord'] = coord
        M['avgcoord'] = np.mean(coord != 0) if len(coord) > 0 else 0.0
    else:
        M['coord'] = np.zeros(len(X), dtype=int)
        M['avgcoord'] = 0.0
    
    # Cross-link density and number
    xlink = 0
    vflag = np.zeros(len(V), dtype=bool)
    
    for i in range(len(V)):
        if len(V[i]['f']) > 1:
            xlink += 1
            vflag[i] = True
    
    M['xlinkdens'] = xlink / M['totL'] if M['totL'] > 0 else 0.0
    
    # Mean cross-link spacing distribution
    xlinkspace = []
    
    for fi in range(len(F)):
        v = F[fi]['v']
        ind = [i for i, vi in enumerate(v) if vflag[vi]]
        
        if len(ind) > 1:
            # There is more than one cross-link in this fiber
            for j in range(len(ind) - 1):
                i1 = ind[j]
                i2 = ind[j + 1]
                
                # Calculate length between cross-links
                length = 0.0
                for ii in range(i1, i2):
                    v1_idx = v[ii]
                    v2_idx = v[ii + 1]
                    if v1_idx < len(X) and v2_idx < len(X):
                        length += np.linalg.norm(X[v2_idx] - X[v1_idx])
                
                xlinkspace.append(length)
    
    M['xlinkspace'] = np.array(xlinkspace) if len(xlinkspace) > 0 else np.array([])
    
    # Persistence length
    Lp3 = calc_persistent2(X, F)
    M['Lp3'] = Lp3
    
    return M
