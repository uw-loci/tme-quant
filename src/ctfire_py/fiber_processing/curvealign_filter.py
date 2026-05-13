"""
CurveAlign-style filtering for fiber networks

Filters fibers based on quality criteria used in CurveAlign:
- Minimum length threshold
- Straightness (end-to-end distance / total path length)
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_fiber_length(X: np.ndarray, fiber_vertices: List[int]) -> float:
    """
    Calculate total path length of a fiber
    
    Args:
        X: Vertex coordinates (N x 3) or (N x 2)
        fiber_vertices: List of vertex indices (1-based)
        
    Returns:
        Total length along fiber path
    """
    if len(fiber_vertices) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(fiber_vertices) - 1):
        v1_idx = int(fiber_vertices[i]) - 1      # 1-based → 0-based
        v2_idx = int(fiber_vertices[i+1]) - 1

        if v1_idx < 0 or v1_idx >= len(X) or v2_idx < 0 or v2_idx >= len(X):
            continue

        p1 = X[v1_idx]
        p2 = X[v2_idx]
        segment_length = np.linalg.norm(p2 - p1)
        total_length += segment_length
    
    return total_length


def calculate_fiber_straightness(X: np.ndarray, fiber_vertices: List[int]) -> float:
    """
    Calculate fiber straightness (end-to-end distance / total path length)
    
    Args:
        X: Vertex coordinates (N x 3) or (N x 2)
        fiber_vertices: List of vertex indices (1-based)
        
    Returns:
        Straightness value between 0 and 1
    """
    if len(fiber_vertices) < 2:
        return 0.0
    
    # End-to-end distance
    v_start_idx = int(fiber_vertices[0]) - 1    # 1-based → 0-based
    v_end_idx = int(fiber_vertices[-1]) - 1

    if v_start_idx < 0 or v_start_idx >= len(X) or v_end_idx < 0 or v_end_idx >= len(X):
        return 0.0

    end_to_end_dist = np.linalg.norm(X[v_end_idx] - X[v_start_idx])
    
    # Total path length
    total_length = calculate_fiber_length(X, fiber_vertices)
    
    if total_length == 0:
        return 0.0
    
    straightness = end_to_end_dist / total_length
    return min(straightness, 1.0)  # Cap at 1.0


def curvealign_filter(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    min_length: float = 30.0,
    min_straightness: float = 0.8
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Filter fibers using CurveAlign-style quality criteria
    
    Based on MATLAB CurveAlign/getFIRE.m filtering logic:
    - Filters by minimum length (LL1 parameter, typically 30 pixels)
    - Filters by straightness (typically ≥ 0.8)
    
    Args:
        X: Vertex coordinates (N x 3)
        F: Fiber list with 'v' (vertex indices)
        V: Vertex list
        min_length: Minimum fiber length threshold (default: 30.0)
        min_straightness: Minimum straightness threshold (default: 0.8)
        
    Returns:
        X, F, V: Filtered arrays
    """
    
    print(f"  CurveAlign filter: min_length={min_length:.1f}, min_straightness={min_straightness:.2f}")
    print(f"    Before filter: {len(F)} fibers")
    
    # Calculate fiber statistics
    fiber_stats = []
    for fi, fiber in enumerate(F):
        v = fiber['v']
        if not isinstance(v, (list, np.ndarray)):
            v = [v]
        v = [int(vi) for vi in v]
        
        if len(v) < 2:
            fiber_stats.append({
                'index': fi,
                'length': 0.0,
                'straightness': 0.0,
                'keep': False
            })
            continue
        
        length = calculate_fiber_length(X, v)
        straightness = calculate_fiber_straightness(X, v)
        
        # Apply filters (both conditions must be met)
        keep = (length >= min_length) and (straightness >= min_straightness)
        
        fiber_stats.append({
            'index': fi,
            'length': length,
            'straightness': straightness,
            'keep': keep
        })
    
    # Filter fibers
    F_filtered = []
    for i, stats in enumerate(fiber_stats):
        if stats['keep']:
            F_filtered.append(F[i])
    
    # Count how many failed each criterion
    failed_length = sum(1 for s in fiber_stats if s['length'] < min_length)
    failed_straightness = sum(1 for s in fiber_stats if s['length'] >= min_length and s['straightness'] < min_straightness)
    
    print(f"    Removed {failed_length} fibers (length < {min_length:.1f})")
    print(f"    Removed {failed_straightness} fibers (straightness < {min_straightness:.2f})")
    print(f"    After filter: {len(F_filtered)} fibers ({100*len(F_filtered)/len(F):.1f}%)")
    
    # Rebuild data structures
    from ctfire_py.utils.trimxfv import trimxfv
    X_out, F_out, V_out = trimxfv(X, F_filtered, V)
    
    return X_out, F_out, V_out


def print_fiber_statistics(X: np.ndarray, F: List[Dict]):
    """
    Print statistics about fiber network
    
    Args:
        X: Vertex coordinates
        F: Fiber list
    """
    if len(F) == 0:
        print("  No fibers to analyze")
        return
    
    lengths = []
    straightnesses = []
    
    for fiber in F:
        v = fiber['v']
        if not isinstance(v, (list, np.ndarray)):
            v = [v]
        v = [int(vi) for vi in v]
        
        if len(v) >= 2:
            length = calculate_fiber_length(X, v)
            straightness = calculate_fiber_straightness(X, v)
            lengths.append(length)
            straightnesses.append(straightness)
    
    if lengths:
        print(f"\n  Fiber Statistics:")
        print(f"    Length: min={min(lengths):.1f}, max={max(lengths):.1f}, mean={np.mean(lengths):.1f}")
        print(f"    Straightness: min={min(straightnesses):.3f}, max={max(straightnesses):.3f}, mean={np.mean(straightnesses):.3f}")
