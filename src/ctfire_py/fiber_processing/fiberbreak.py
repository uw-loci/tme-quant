"""
fiberbreak - Breaks fibers at cross-links

Based on MATLAB fiberproc/fiberbreak.m
Splits fibers at internal vertices that are crosslinks (vertices with >1 fiber)
"""

import numpy as np
from typing import List, Dict


def fiberbreak(X: np.ndarray, F: List[Dict], V: List[Dict]) -> tuple:
    """
    Break fibers at cross-links for better network topology
    
    Args:
        X: Vertex coordinates (N x 3)
        F: Fiber list, each with 'v' (vertex indices)
        V: Vertex list, each with 'f' (fiber indices)
        
    Returns:
        X, F, V: Updated arrays with fibers split at crosslinks
    """
    
    print(f"  fiberbreak: Breaking {len(F)} fibers at crosslinks...")
    
    # Create new fiber list
    F_new = []
    
    # For each original fiber
    for fi, fiber in enumerate(F):
        v = fiber['v']  # Vertex indices for this fiber
        
        # Ensure v is a list of integers
        if not isinstance(v, (list, np.ndarray)):
            v = [v]
        v = [int(vi) for vi in v]
        
        if len(v) < 2:
            # Keep short fibers as-is
            new_fiber = fiber.copy()
            new_fiber['v'] = v
            F_new.append(new_fiber)
            continue
            
        # Split fiber at crosslinks.
        # MATLAB: `for j = 2:length(v-1)` — in MATLAB, `v-1` is element-wise
        # subtraction so length(v-1) == length(v), making the range 2:length(v)
        # (1-based), i.e. 0-based range(1, len(v)). The original Python used
        # range(1, len(v)-1) which silently excluded the last vertex.
        # If the last vertex is a crosslink, MATLAB would emit a trivial
        # single-vertex tail fiber; trimxfv discards it. We replicate that.
        vstart = 0
        for j in range(1, len(v)):
            vj = int(v[j])               # 0-based vertex ID

            # Check if this vertex is a crosslink (has multiple fibers).
            if vj < len(V) and len(V[vj]['f']) > 1:
                # Split here - add fiber segment from vstart to current vertex
                new_fiber = {
                    'v': v[vstart:j+1].copy() if isinstance(v, np.ndarray) else v[vstart:j+1]
                }
                if 'r' in fiber:
                    new_fiber['r'] = fiber['r']
                F_new.append(new_fiber)
                vstart = j
        
        # Add final segment
        final_fiber = {
            'v': v[vstart:].copy() if isinstance(v, np.ndarray) else v[vstart:]
        }
        if 'r' in fiber:
            final_fiber['r'] = fiber['r']
        F_new.append(final_fiber)
    
    print(f"    Before break: {len(F)} fibers")
    print(f"    After break: {len(F_new)} fibers")
    
    # Rebuild V structure
    from ctfire_py.utils.trimxfv import trimxfv
    X_out, F_out, V_out = trimxfv(X, F_new, V)
    
    return X_out, F_out, V_out
