"""
Main CT-FIRE processing orchestrator.

This module provides high-level processing functions that coordinate
CT-FIRE analysis workflows, implementing ctFIRE_1.m functionality.
"""

from typing import List, Optional, Dict
import numpy as np

from ...types import Fiber, FiberNetwork, CTFireResult, CTFireOptions, FiberMetrics
from ..algorithms import extract_fibers_fire, enhance_image_with_curvelets


def analyze_image_ctfire(
    image: np.ndarray,
    options: Optional[CTFireOptions] = None
) -> CTFireResult:
    """
    Analyze image using CT-FIRE for individual fiber extraction.
    
    This implements the main CT-FIRE workflow from ctFIRE_1.m:
    1. Optionally enhance image with curvelet transform
    2. Apply FIRE algorithm for fiber extraction  
    3. Compute fiber metrics and network analysis
    4. Generate summary statistics
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image
    options : CTFireOptions, optional
        CT-FIRE analysis parameters
        
    Returns
    -------
    CTFireResult
        Complete CT-FIRE analysis results
    """
    if options is None:
        options = CTFireOptions()
    
    enhanced_image = None
    
    # Step 1: Curvelet enhancement (if ctfire mode)
    if options.run_mode in ["ctfire", "both"]:
        enhanced_image = enhance_image_with_curvelets(
            image, 
            keep=options.keep,
            scale=options.scale
        )
        processing_image = enhanced_image
    else:
        processing_image = image
    
    # Step 2: FIRE fiber extraction
    if options.run_mode in ["fire", "ctfire", "both"]:
        fibers = extract_fibers_fire(processing_image, options)
    else:
        fibers = []
    
    # Step 3: Network analysis
    network = analyze_fiber_network(fibers)
    
    # Step 4: Compute summary statistics
    stats = compute_ctfire_statistics(fibers, network)
    
    return CTFireResult(
        fibers=fibers,
        network=network,
        stats=stats,
        enhanced_image=enhanced_image
    )


def analyze_fiber_network(fibers: List[Fiber]) -> FiberNetwork:
    """
    Analyze connectivity and network properties of extracted fibers.
    
    Parameters
    ----------
    fibers : List[Fiber]
        List of extracted fibers
        
    Returns
    -------
    FiberNetwork
        Network analysis results
    """
    if not fibers:
        return FiberNetwork(
            fibers=fibers,
            intersections=[],
            connectivity={},
            network_stats={}
        )
    
    # Find intersections between fibers
    intersections = _find_fiber_intersections(fibers)
    
    # Build connectivity graph
    connectivity = _build_connectivity_graph(fibers, intersections)
    
    # Compute network statistics
    network_stats = _compute_network_statistics(fibers, intersections, connectivity)
    
    return FiberNetwork(
        fibers=fibers,
        intersections=intersections,
        connectivity=connectivity,
        network_stats=network_stats
    )


def compute_fiber_metrics(fibers: List[Fiber]) -> FiberMetrics:
    """
    Compute detailed metrics for all fibers.
    
    Parameters
    ----------
    fibers : List[Fiber]
        List of fibers to analyze
        
    Returns
    -------
    FiberMetrics
        Detailed metrics for each fiber
    """
    if not fibers:
        return FiberMetrics(
            lengths=[], widths=[], angles=[], 
            straightness=[], curvatures=[]
        )
    
    return FiberMetrics(
        lengths=[f.length for f in fibers],
        widths=[f.width for f in fibers],
        angles=[f.angle_deg for f in fibers],
        straightness=[f.straightness for f in fibers],
        curvatures=[f.curvature or 0.0 for f in fibers]
    )


def compute_ctfire_statistics(fibers: List[Fiber], network: FiberNetwork) -> Dict[str, float]:
    """
    Compute summary statistics for CT-FIRE analysis.
    
    Parameters
    ----------
    fibers : List[Fiber]
        Extracted fibers
    network : FiberNetwork
        Network analysis results
        
    Returns
    -------
    Dict[str, float]
        Summary statistics
    """
    if not fibers:
        return {
            'n_fibers': 0,
            'mean_length': 0.0,
            'mean_width': 0.0,
            'mean_angle': 0.0,
            'mean_straightness': 0.0,
            'total_length': 0.0,
            'fiber_density': 0.0
        }
    
    lengths = [f.length for f in fibers]
    widths = [f.width for f in fibers]
    angles = [f.angle_deg for f in fibers]
    straightness = [f.straightness for f in fibers]
    
    stats = {
        'n_fibers': len(fibers),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_width': np.mean(widths),
        'std_width': np.std(widths),
        'mean_angle': np.mean(angles),
        'std_angle': np.std(angles),
        'mean_straightness': np.mean(straightness),
        'std_straightness': np.std(straightness),
        'total_length': np.sum(lengths),
        'max_length': np.max(lengths),
        'min_length': np.min(lengths),
    }
    
    # Add network statistics
    stats.update(network.network_stats)
    
    return stats


def _find_fiber_intersections(fibers: List[Fiber], threshold: float = 5.0) -> List[tuple]:
    """Find intersection points between fibers."""
    intersections = []
    
    for i, fiber1 in enumerate(fibers):
        for j, fiber2 in enumerate(fibers[i+1:], i+1):
            intersection = _find_fiber_pair_intersection(fiber1, fiber2, threshold)
            if intersection:
                intersections.append(intersection)
    
    return intersections


def _find_fiber_pair_intersection(fiber1: Fiber, fiber2: Fiber, threshold: float) -> Optional[tuple]:
    """Find intersection point between two fibers."""
    # Simplified intersection detection
    # Check if any points are within threshold distance
    for p1 in fiber1.points:
        for p2 in fiber2.points:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist <= threshold:
                # Return midpoint as intersection
                intersection = (
                    (p1[0] + p2[0]) / 2,
                    (p1[1] + p2[1]) / 2
                )
                return intersection
    
    return None


def _build_connectivity_graph(fibers: List[Fiber], intersections: List[tuple]) -> Dict[int, List[int]]:
    """Build connectivity graph showing which fibers connect."""
    connectivity = {i: [] for i in range(len(fibers))}
    
    # For each intersection, find which fibers are connected
    for intersection in intersections:
        connected_fibers = []
        for i, fiber in enumerate(fibers):
            # Check if fiber passes through this intersection
            for point in fiber.points:
                dist = np.linalg.norm(np.array(point) - np.array(intersection))
                if dist <= 5.0:  # Within threshold
                    connected_fibers.append(i)
                    break
        
        # Add connections
        for i in connected_fibers:
            for j in connected_fibers:
                if i != j and j not in connectivity[i]:
                    connectivity[i].append(j)
    
    return connectivity


def _compute_network_statistics(
    fibers: List[Fiber], 
    intersections: List[tuple], 
    connectivity: Dict[int, List[int]]
) -> Dict[str, float]:
    """Compute network-level statistics."""
    if not fibers:
        return {}
    
    # Basic network stats
    n_intersections = len(intersections)
    n_connections = sum(len(connections) for connections in connectivity.values()) // 2
    
    # Network density
    total_connections_possible = len(fibers) * (len(fibers) - 1) // 2
    connection_density = n_connections / total_connections_possible if total_connections_possible > 0 else 0.0
    
    return {
        'n_intersections': n_intersections,
        'n_connections': n_connections,
        'connection_density': connection_density,
        'mean_connections_per_fiber': n_connections / len(fibers) if fibers else 0.0
    }
