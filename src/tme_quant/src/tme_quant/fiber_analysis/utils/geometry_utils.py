# File: tme_quant/utils/geometry_utils.py

import numpy as np
from typing import Tuple, Optional, List
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


def find_nearest_boundary_point(
    fiber_point: np.ndarray,
    tumor_boundary: 'ROI',
    pixel_size: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Find the nearest point on the tumor boundary to a fiber point.
    
    Args:
        fiber_point: [x, y] coordinates of fiber point
        tumor_boundary: Tumor boundary ROI
        pixel_size: Pixel size in microns
        
    Returns:
        Tuple of (nearest_point, distance)
            - nearest_point: [x, y] coordinates on boundary
            - distance: Distance in microns
    """
    # Convert fiber point to Shapely Point
    point = Point(fiber_point)
    
    # Get boundary geometry
    boundary_geom = tumor_boundary.to_shapely_geometry()
    
    # Handle different boundary types
    if isinstance(boundary_geom, Polygon):
        # Use exterior ring for polygon
        boundary_line = boundary_geom.exterior
    elif isinstance(boundary_geom, LineString):
        boundary_line = boundary_geom
    else:
        raise ValueError(f"Unsupported boundary geometry type: {type(boundary_geom)}")
    
    # Find nearest point on boundary
    nearest_geom = nearest_points(point, boundary_line)[1]
    nearest_point = np.array([nearest_geom.x, nearest_geom.y])
    
    # Calculate distance
    distance = np.linalg.norm(fiber_point - nearest_point) * pixel_size
    
    return nearest_point, distance


def compute_boundary_normal(
    point_on_boundary: np.ndarray,
    tumor_boundary: 'ROI',
    epsilon: float = 1.0
) -> float:
    """
    Compute the normal vector angle at a point on the tumor boundary.
    
    The normal points inward (toward tumor interior).
    
    Args:
        point_on_boundary: [x, y] coordinates on boundary
        tumor_boundary: Tumor boundary ROI
        epsilon: Distance for numerical differentiation
        
    Returns:
        Normal angle in degrees (0-360)
    """
    # Get boundary geometry
    boundary_geom = tumor_boundary.to_shapely_geometry()
    
    if isinstance(boundary_geom, Polygon):
        boundary_line = boundary_geom.exterior
    else:
        boundary_line = boundary_geom
    
    # Get coordinates along boundary
    coords = np.array(boundary_line.coords)
    
    # Find closest segment
    point = Point(point_on_boundary)
    min_dist = float('inf')
    closest_idx = 0
    
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        dist = point.distance(segment)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # Get tangent vector from closest segment
    p1 = coords[closest_idx]
    p2 = coords[closest_idx + 1]
    tangent = p2 - p1
    tangent = tangent / np.linalg.norm(tangent)
    
    # Normal is perpendicular to tangent (rotated 90°)
    # For inward normal, choose direction toward centroid
    normal_option1 = np.array([-tangent[1], tangent[0]])
    normal_option2 = np.array([tangent[1], -tangent[0]])
    
    # Determine which normal points inward
    if isinstance(boundary_geom, Polygon):
        centroid = np.array([boundary_geom.centroid.x, boundary_geom.centroid.y])
        to_centroid = centroid - point_on_boundary
        
        # Choose normal that aligns better with direction to centroid
        if np.dot(normal_option1, to_centroid) > np.dot(normal_option2, to_centroid):
            normal = normal_option1
        else:
            normal = normal_option2
    else:
        # For non-polygon boundaries, default to option 1
        normal = normal_option1
    
    # Convert to angle
    normal_angle = np.degrees(np.arctan2(normal[1], normal[0]))
    
    # Normalize to 0-360
    if normal_angle < 0:
        normal_angle += 360
    
    return normal_angle


def compute_relative_angles(
    fiber_angle: float,
    boundary_normal_angle: float
) -> Dict[str, float]:
    """
    Compute relative angles between fiber and boundary.
    
    Args:
        fiber_angle: Fiber orientation in degrees
        boundary_normal_angle: Boundary normal angle in degrees
        
    Returns:
        Dictionary with:
            - angle_to_normal: 0° = perpendicular to boundary (TACS-3)
            - angle_to_tangent: 0° = parallel to boundary (TACS-2)
    """
    # Normalize angles to 0-180 range for orientation
    def normalize_orientation(angle):
        angle = angle % 180
        return angle
    
    fiber_orientation = normalize_orientation(fiber_angle)
    normal_orientation = normalize_orientation(boundary_normal_angle)
    
    # Angle to normal (0° = fiber perpendicular to boundary)
    angle_to_normal = abs(fiber_orientation - normal_orientation)
    if angle_to_normal > 90:
        angle_to_normal = 180 - angle_to_normal
    
    # Angle to tangent (0° = fiber parallel to boundary)
    angle_to_tangent = 90 - angle_to_normal
    
    return {
        'angle_to_normal': angle_to_normal,
        'angle_to_tangent': angle_to_tangent
    }


def compute_fiber_to_boundary_alignment(
    fiber_centerline: np.ndarray,
    boundary: 'ROI',
    pixel_size: float = 1.0
) -> Dict[str, float]:
    """
    Compute comprehensive fiber-to-boundary alignment metrics.
    
    This is the main function that combines all metrics.
    
    Args:
        fiber_centerline: Nx2 array of fiber coordinates
        boundary: Tumor boundary ROI
        pixel_size: Pixel size in microns
        
    Returns:
        Dictionary with all alignment metrics
    """
    # Get fiber midpoint
    mid_idx = len(fiber_centerline) // 2
    fiber_point = fiber_centerline[mid_idx]
    
    # Find nearest boundary point
    nearest_point, distance = find_nearest_boundary_point(
        fiber_point, boundary, pixel_size
    )
    
    # Compute boundary normal
    normal_angle = compute_boundary_normal(nearest_point, boundary)
    
    # Compute fiber orientation
    fiber_vector = fiber_centerline[-1] - fiber_centerline[0]
    fiber_angle = np.degrees(np.arctan2(fiber_vector[1], fiber_vector[0]))
    
    # Compute relative angles
    relative_angles = compute_relative_angles(fiber_angle, normal_angle)
    
    # Compute alignment score (0 = perpendicular, 1 = parallel)
    alignment_score = abs(relative_angles['angle_to_tangent']) / 90.0
    
    return {
        'distance': distance,
        'nearest_point': nearest_point,
        'boundary_normal_angle': normal_angle,
        'fiber_angle': fiber_angle,
        'angle_to_normal': relative_angles['angle_to_normal'],
        'angle_to_tangent': relative_angles['angle_to_tangent'],
        'alignment_score': alignment_score
    }