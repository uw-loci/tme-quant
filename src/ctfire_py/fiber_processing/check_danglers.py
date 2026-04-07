"""
check_danglers - Remove dangling fiber segments

Identifies and removes fiber segments that:
1. Connect to only one cross-link (dangler)
2. Run parallel to another fiber (redundant)
3. Are very short and not legitimate fiber extensions

IMPORTANT: The original MATLAB implementation has critical bugs that prevent it from
working. This Python implementation corrects those bugs and properly removes danglers.

Corrected Bugs from MATLAB version:
- MATLAB line 12-15: Logic error where condition is impossible to satisfy
- MATLAB line 16: setdiff(vi,vi) always returns empty
- Result: MATLAB check_danglers NEVER removes any fibers!

This corrected version implements the intended algorithm properly.
"""

import numpy as np
from typing import List, Dict, Tuple


def check_danglers(
    vertices: np.ndarray,
    fibers: List[Dict],
    vertex_info: List[Dict],
    radii: np.ndarray,
    params: Dict
) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]:
    """
    Remove dangling fiber segments based on connectivity and geometry.
    
    A "dangler" is a fiber that connects to only one cross-link (junction point).
    Such fibers are candidates for removal if they are:
    - Running parallel to another fiber (redundant coverage)
    - Very short and not extending a legitimate fiber
    - Short and connected to a well-established cross-link
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (num_vertices, 3)
    fibers : List[Dict]
        List of fiber dictionaries with 'v' (vertex indices)
    vertex_info : List[Dict]
        List of vertex dictionaries with 'f' (fiber indices)
    radii : np.ndarray
        Fiber radii
    params : Dict
        Parameters including:
        - threshold_dangler_angle_extension: Min dot product to consider extension (default: 0.5)
        - threshold_dangler_length: Max length for short dangler removal (default: 10.0)
    
    Returns
    -------
    vertices : np.ndarray
        Updated vertex coordinates
    fibers : List[Dict]
        Updated fiber list (with danglers removed)
    vertex_info : List[Dict]
        Updated vertex info
    radii : np.ndarray
        Updated radii
        
    Notes
    -----
    The original MATLAB implementation has bugs:
    - Line 12: checks `length(V(vi).f)>1` but should be `==1` for danglers
    - Line 16: uses `setdiff(vi,vi)` which always returns empty
    
    This Python implementation corrects these bugs while maintaining the intended logic.
    """
    from ctfire_py.utils import trimxfv
    
    # Get parameters - use MATLAB parameter names for compatibility
    threshold_angle_parallel = params.get('thresh_dang_aextend', 0.9848)  # cos(10°)
    threshold_short_length = params.get('thresh_dang_L', 15.0)
    
    # Determine if fiber indices are 0-based or 1-based
    max_fiber_idx = -1
    for vertex in vertex_info:
        if 'f' in vertex and len(vertex['f']) > 0:
            max_fiber_idx = max(max_fiber_idx, max(vertex['f']))
    indices_are_one_based = (max_fiber_idx >= len(fibers))
    
    # Also check if vertex indices in fibers are 1-based
    max_vertex_idx = -1
    for fiber in fibers:
        if 'v' in fiber and len(fiber['v']) > 0:
            max_vertex_idx = max(max_vertex_idx, max(fiber['v']))
    vertex_indices_are_one_based = (max_vertex_idx >= len(vertex_info))
    
    # Step 1: For each fiber, count the number of crosslinks it has
    num_crosslinks_per_fiber = np.zeros(len(fibers), dtype=int)
    crosslink_vertex_per_fiber = np.full(len(fibers), -1, dtype=int)  # Store the crosslink vertex for danglers
    
    for fiber_idx in range(len(fibers)):
        fiber = fibers[fiber_idx]
        if 'v' not in fiber or len(fiber['v']) == 0:
            continue
        
        fiber_vertices = fiber['v']
        num_crosslinks = 0
        last_crosslink_vertex = -1
        
        for v_orig in fiber_vertices:
            v_idx = v_orig - 1 if vertex_indices_are_one_based else v_orig
            if v_idx < 0 or v_idx >= len(vertex_info):
                continue
            
            # Count fibers at this vertex
            num_fibers_at_vertex = len(vertex_info[v_idx].get('f', []))
            if num_fibers_at_vertex > 1:
                num_crosslinks += 1
                last_crosslink_vertex = v_idx
        
        num_crosslinks_per_fiber[fiber_idx] = num_crosslinks
        if num_crosslinks == 1:
            crosslink_vertex_per_fiber[fiber_idx] = last_crosslink_vertex
    
    # Track which fibers to remove
    fibers_to_remove = np.zeros(len(fibers), dtype=bool)
    
    # Step 2: Loop through fibers and check danglers (fibers with exactly 1 crosslink)
    for fiber_idx in range(len(fibers)):
        # Skip if not a dangler
        if num_crosslinks_per_fiber[fiber_idx] != 1:
            continue
        
        # Skip if already marked for removal
        if fibers_to_remove[fiber_idx]:
            continue
        
        fiber = fibers[fiber_idx]
        if 'v' not in fiber or len(fiber['v']) < 2:
            continue
        
        # Get crosslink vertex (in 0-based indexing)
        crosslink_vertex_idx = crosslink_vertex_per_fiber[fiber_idx]
        if crosslink_vertex_idx < 0 or crosslink_vertex_idx >= len(vertex_info):
            continue
        
        # Find the free end (not the crosslink)
        fiber_vertices = fiber['v']
        v_start = fiber_vertices[0] - 1 if vertex_indices_are_one_based else fiber_vertices[0]
        v_end = fiber_vertices[-1] - 1 if vertex_indices_are_one_based else fiber_vertices[-1]
        
        if v_start == crosslink_vertex_idx:
            free_end_vertex_idx = v_end
        elif v_end == crosslink_vertex_idx:
            free_end_vertex_idx = v_start
        else:
            # Crosslink is in the middle - not a typical dangler
            continue
        
        if free_end_vertex_idx < 0 or free_end_vertex_idx >= len(vertices):
            continue
        
        # Calculate dangler properties
        crosslink_pos = vertices[crosslink_vertex_idx]
        free_end_pos = vertices[free_end_vertex_idx]
        
        # Dangler vector (from crosslink to free end)
        dangler_vector = free_end_pos - crosslink_pos
        dangler_length = np.linalg.norm(dangler_vector)
        
        if dangler_length < 1e-10:
            fibers_to_remove[fiber_idx] = True
            continue
        
        dangler_direction = dangler_vector / dangler_length
        
        # Get other fibers at the crosslink (excluding this dangler)
        crosslink_vertex = vertex_info[crosslink_vertex_idx]
        orig_fiber_idx_in_list = fiber_idx + 1 if indices_are_one_based else fiber_idx
        
        crosslink_fiber_indices = []
        for f_orig in crosslink_vertex.get('f', []):
            f_idx = f_orig - 1 if indices_are_one_based else f_orig
            if f_idx != fiber_idx and f_idx >= 0 and f_idx < len(fibers):
                crosslink_fiber_indices.append(f_idx)
        
        if len(crosslink_fiber_indices) == 0:
            continue
        
        # Check dangler against neighboring fibers
        max_dot_product = -np.inf
        num_legitimate_neighbors = 0
        dot_products = []
        
        for neighbor_fiber_idx in crosslink_fiber_indices:
            if fibers_to_remove[neighbor_fiber_idx]:
                continue
            
            neighbor_fiber = fibers[neighbor_fiber_idx]
            if 'v' not in neighbor_fiber or len(neighbor_fiber['v']) < 2:
                continue
            
            neighbor_vertices = neighbor_fiber['v']
            
            # Find the end of neighbor fiber that's NOT at the crosslink
            nv_start_orig = neighbor_vertices[0]
            nv_end_orig = neighbor_vertices[-1]
            
            # Convert to 0-based
            nv_start = nv_start_orig - 1 if vertex_indices_are_one_based else nv_start_orig
            nv_end = nv_end_orig - 1 if vertex_indices_are_one_based else nv_end_orig
            
            # Determine which end is away from crosslink
            if nv_start == crosslink_vertex_idx:
                neighbor_far_vertex_idx = nv_end
            elif nv_end == crosslink_vertex_idx:
                neighbor_far_vertex_idx = nv_start
            else:
                # Crosslink not at ends - find closer end
                if nv_start >= 0 and nv_start < len(vertices) and nv_end >= 0 and nv_end < len(vertices):
                    dist_start = np.linalg.norm(vertices[nv_start] - crosslink_pos)
                    dist_end = np.linalg.norm(vertices[nv_end] - crosslink_pos)
                    neighbor_far_vertex_idx = nv_end if dist_start < dist_end else nv_start
                else:
                    continue
            
            if neighbor_far_vertex_idx < 0 or neighbor_far_vertex_idx >= len(vertices):
                continue
            
            # Direction of neighbor fiber (from crosslink outward)
            neighbor_vector = vertices[neighbor_far_vertex_idx] - crosslink_pos
            neighbor_length = np.linalg.norm(neighbor_vector)
            
            if neighbor_length < 1e-10:
                continue
            
            neighbor_direction = neighbor_vector / neighbor_length
            
            # Compute dot product
            dot_product = np.dot(dangler_direction, neighbor_direction)
            dot_products.append(dot_product)
            max_dot_product = max(max_dot_product, dot_product)
            
            # Check if neighbor is legitimate (has crosslinks at its far end)
            if neighbor_far_vertex_idx < len(vertex_info):
                neighbor_far_vertex = vertex_info[neighbor_far_vertex_idx]
                if len(neighbor_far_vertex.get('f', [])) >= 2:
                    num_legitimate_neighbors += 1
        
        # Apply removal rules (matching MATLAB logic)
        remove_dangler = False
        
        # Rule 1: Parallel to another fiber
        if max_dot_product > threshold_angle_parallel:
            remove_dangler = True
        
        # Rule 2: Short AND crosslink has 2+ legitimate neighbors
        elif dangler_length < threshold_short_length and num_legitimate_neighbors >= 2:
            remove_dangler = True
        
        # Rule 3: Short AND not an extension of incoming fiber
        elif dangler_length < threshold_short_length:
            if len(dot_products) > 0:
                min_dot_product = min(dot_products)
                if -min_dot_product < threshold_angle_parallel:
                    remove_dangler = True
            else:
                # No neighbors to compare, remove if short
                remove_dangler = True
        
        if remove_dangler:
            fibers_to_remove[fiber_idx] = True
    
    # Remove marked fibers
    num_removed = np.sum(fibers_to_remove)
    if num_removed > 0:
        print(f"  Removing {num_removed} dangling fibers")
        fibers = [f for i, f in enumerate(fibers) if not fibers_to_remove[i]]
    
    # Clean up and renumber
    vertices, fibers, vertex_info, radii = trimxfv(vertices, fibers, vertex_info, radii)
    
    return vertices, fibers, vertex_info, radii
