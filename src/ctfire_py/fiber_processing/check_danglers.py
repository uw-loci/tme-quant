"""
check_danglers - Remove dangling fiber segments

Identifies and removes fiber segments that:
1. Connect to only one cross-link (dangler)
2. Run parallel to another fiber (redundant)
3. Are very short and not legitimate fiber extensions

This function significantly reduces fiber count by removing artifacts.
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
    
    # Get parameters with defaults matching MATLAB
    threshold_angle_extension = params.get('threshold_dangler_angle_extension', 0.5)
    threshold_angle_parallel = params.get('threshold_dangler_angle_parallel', 0.5)
    threshold_short_length = params.get('threshold_dangler_length', 10.0)
    
    # Determine if fiber indices are 0-based or 1-based
    # Check the maximum fiber index in vertex_info
    max_fiber_idx = -1
    for vertex in vertex_info:
        if 'f' in vertex and len(vertex['f']) > 0:
            max_fiber_idx = max(max_fiber_idx, max(vertex['f']))
    
    # If max_fiber_idx >= len(fibers), indices are 1-based
    indices_are_one_based = (max_fiber_idx >= len(fibers))
    
    # Also check if vertex indices in fibers are 1-based
    max_vertex_idx = -1
    for fiber in fibers:
        if 'v' in fiber and len(fiber['v']) > 0:
            max_vertex_idx = max(max_vertex_idx, max(fiber['v']))
    
    vertex_indices_are_one_based = (max_vertex_idx >= len(vertex_info))
    
    # Track which fibers to remove
    fibers_to_remove = np.zeros(len(fibers), dtype=bool)
    
    # Loop through all vertices to find danglers
    for vertex_idx in range(len(vertex_info)):
        vertex = vertex_info[vertex_idx]
        
        # Check if this vertex has only 1 fiber (potential dangler endpoint)
        if len(vertex['f']) == 1:
            # This is a dangler endpoint
            fiber_idx = vertex['f'][0]
            
            # Convert from 1-based to 0-based if necessary
            if indices_are_one_based:
                fiber_idx = fiber_idx - 1
            
            if fiber_idx < 0 or fiber_idx >= len(fibers):
                continue  # Invalid index
            
            # Skip if already marked for removal
            if fibers_to_remove[fiber_idx]:
                continue
            
            # Get the fiber's vertex list
            fiber_vertices = fibers[fiber_idx]['v']
            
            # Find the OTHER end of the fiber (the cross-link end)
            # Need to compare using the original vertex index from fiber (may be 1-based)
            orig_vertex_idx = vertex_idx + 1 if vertex_indices_are_one_based else vertex_idx
            
            if fiber_vertices[0] == orig_vertex_idx:
                crosslink_vertex_idx = fiber_vertices[-1]
            elif fiber_vertices[-1] == orig_vertex_idx:
                crosslink_vertex_idx = fiber_vertices[0]
            else:
                # This vertex is in the middle - not a dangler
                continue
            
            # Convert crosslink vertex index to 0-based for accessing vertex_info
            crosslink_vertex_idx_access = crosslink_vertex_idx - 1 if vertex_indices_are_one_based else crosslink_vertex_idx
            
            if crosslink_vertex_idx_access < 0 or crosslink_vertex_idx_access >= len(vertex_info):
                continue  # Invalid index
            
            # Check if the other end is a cross-link (has multiple fibers)
            crosslink_vertex = vertex_info[crosslink_vertex_idx_access]
            if len(crosslink_vertex['f']) <= 1:
                # Not a dangler - both ends are free
                continue
            
            # This is a true dangler: one end free, other end at cross-link
            # Calculate dangler properties
            dangler_start_pos = vertices[vertex_idx]
            crosslink_pos = vertices[crosslink_vertex_idx_access]
            
            # Dangler vector (from cross-link to free end)
            dangler_vector = dangler_start_pos - crosslink_pos
            dangler_length = np.linalg.norm(dangler_vector)
            
            if dangler_length < 1e-10:
                # Zero-length dangler
                fibers_to_remove[fiber_idx] = True
                continue
            
            dangler_direction = dangler_vector / dangler_length
            
            # Get fibers at the cross-link (excluding this dangler)
            # Need to compare using the same indexing convention
            orig_fiber_idx = fiber_idx + 1 if indices_are_one_based else fiber_idx
            crosslink_fiber_indices = [
                (f - 1 if indices_are_one_based else f)
                for f in crosslink_vertex['f']
                if f != orig_fiber_idx
            ]
            
            if len(crosslink_fiber_indices) == 0:
                # No other fibers to compare against
                continue
            
            # Check dangler against neighboring fibers
            max_dot_product = -np.inf
            num_legitimate_neighbors = 0
            
            for neighbor_fiber_idx in crosslink_fiber_indices:
                # Skip if neighbor is already marked for removal
                if fibers_to_remove[neighbor_fiber_idx]:
                    continue
                
                # Find the vertex at the OTHER end of the neighbor fiber
                neighbor_vertices = fibers[neighbor_fiber_idx]['v']
                
                # Find which vertex is NOT the cross-link
                if neighbor_vertices[0] == crosslink_vertex_idx:
                    neighbor_far_vertex_idx = neighbor_vertices[-1]
                elif neighbor_vertices[-1] == crosslink_vertex_idx:
                    neighbor_far_vertex_idx = neighbor_vertices[0]
                else:
                    # Cross-link is in the middle
                    # Use the closer end
                    v0_access = neighbor_vertices[0] - 1 if vertex_indices_are_one_based else neighbor_vertices[0]
                    vn_access = neighbor_vertices[-1] - 1 if vertex_indices_are_one_based else neighbor_vertices[-1]
                    
                    if v0_access >= 0 and v0_access < len(vertices) and vn_access >= 0 and vn_access < len(vertices):
                        dist_to_start = np.linalg.norm(vertices[v0_access] - crosslink_pos)
                        dist_to_end = np.linalg.norm(vertices[vn_access] - crosslink_pos)
                        if dist_to_start < dist_to_end:
                            neighbor_far_vertex_idx = neighbor_vertices[-1]
                        else:
                            neighbor_far_vertex_idx = neighbor_vertices[0]
                    else:
                        continue  # Invalid indices
                
                # Convert neighbor far vertex index to 0-based for accessing vertices
                neighbor_far_vertex_idx_access = neighbor_far_vertex_idx - 1 if vertex_indices_are_one_based else neighbor_far_vertex_idx
                
                if neighbor_far_vertex_idx_access < 0 or neighbor_far_vertex_idx_access >= len(vertices):
                    continue  # Invalid index
                
                # Direction of neighbor fiber (from cross-link outward)
                neighbor_vector = vertices[neighbor_far_vertex_idx_access] - crosslink_pos
                neighbor_length = np.linalg.norm(neighbor_vector)
                
                if neighbor_length < 1e-10:
                    continue
                
                neighbor_direction = neighbor_vector / neighbor_length
                
                # Compute dot product (>0 means same direction, <0 opposite)
                dot_product = np.dot(dangler_direction, neighbor_direction)
                max_dot_product = max(max_dot_product, dot_product)
                
                # Check if neighbor is legitimate (has cross-links)
                if neighbor_far_vertex_idx_access >= 0 and neighbor_far_vertex_idx_access < len(vertex_info):
                    neighbor_far_vertex = vertex_info[neighbor_far_vertex_idx_access]
                    if len(neighbor_far_vertex['f']) >= 2:
                        num_legitimate_neighbors += 1
            
            # Decision rules for dangler removal
            remove_dangler = False
            
            # Rule 1: If parallel to another fiber (not marked for removal), remove this dangler
            if max_dot_product > threshold_angle_parallel:
                remove_dangler = True
            
            # Rule 2: If very short and cross-link has >= 2 legitimate fibers, remove
            elif dangler_length < threshold_short_length and num_legitimate_neighbors >= 2:
                remove_dangler = True
            
            # Rule 3: If short and NOT an extension of an incoming fiber, remove
            elif dangler_length < threshold_short_length and (-max_dot_product) < threshold_angle_extension:
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
