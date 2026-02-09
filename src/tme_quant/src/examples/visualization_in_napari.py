"""
Napari layers for visualizing fiber-tumor boundary relationships.
"""

import napari
import numpy as np

def visualize_fiber_tumor_alignment(
    viewer: napari.Viewer,
    project: TMEProject,
    tumor_region_id: str
):
    """
    Visualize fiber alignment to tumor boundary in Napari.
    """
    # Get tumor and fibers
    tumor = project.hierarchy.get_object(tumor_region_id)
    fibers = project.get_fibers_in_region(tumor_region_id)
    
    # Layer 1: Base image (collagen channel)
    image = project.get_image_data(tumor.parent_id, channel='collagen')
    viewer.add_image(image, name='Collagen', colormap='gray')
    
    # Layer 2: Tumor boundary ROI
    boundary_mask = tumor.roi.get_mask(image.shape)
    viewer.add_labels(boundary_mask, name='Tumor Boundary')
    
    # Layer 3: Orientation map (heatmap)
    orientation_map_obj = project.orientation_maps.get(f"{tumor_region_id}_boundary_orientation")
    if orientation_map_obj:
        orientation_map = orientation_map_obj.orientation_result.orientation_map
        viewer.add_image(
            orientation_map,
            name='Fiber Orientation',
            colormap='hsv',
            blending='additive',
            opacity=0.5
        )
    
    # Layer 4: Individual fibers colored by alignment
    fiber_shapes = []
    fiber_properties = {
        'alignment_angle': [],
        'fiber_type': [],
        'color': []
    }
    
    for fiber in fibers:
        if fiber.in_tumor_boundary:
            # Color by alignment to boundary
            if fiber.angle_to_tumor_boundary is not None:
                angle = abs(fiber.angle_to_tumor_boundary)
                
                if angle < 30:  # Parallel (TACS-2)
                    color = 'green'
                    fiber_type = 'parallel'
                elif angle > 60:  # Perpendicular (TACS-3)
                    color = 'red'
                    fiber_type = 'perpendicular'
                else:  # Intermediate
                    color = 'yellow'
                    fiber_type = 'intermediate'
            else:
                color = 'gray'
                fiber_type = 'unknown'
            
            fiber_shapes.append(fiber.centerline)
            fiber_properties['alignment_angle'].append(fiber.angle_to_tumor_boundary)
            fiber_properties['fiber_type'].append(fiber_type)
            fiber_properties['color'].append(color)
    
    # Add shapes layer
    if fiber_shapes:
        viewer.add_shapes(
            fiber_shapes,
            shape_type='path',
            edge_color=fiber_properties['color'],
            edge_width=2,
            name='Fibers by Alignment',
            properties=fiber_properties
        )
    
    # Layer 5: Fiber endpoints (for visualization)
    endpoints = []
    for fiber in fibers:
        if fiber.in_tumor_boundary:
            endpoints.append(fiber.start_point.coords[0])
            endpoints.append(fiber.end_point.coords[0])
    
    if endpoints:
        viewer.add_points(
            np.array(endpoints),
            name='Fiber Endpoints',
            size=3,
            face_color='cyan'
        )