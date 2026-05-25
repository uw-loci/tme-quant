"""
Parameter name mapping for readable variable names.

This module provides utilities to convert between legacy cryptic parameter names
and new readable parameter names, enabling gradual refactoring while maintaining
backward compatibility.
"""

# Mapping from old (cryptic) names to new (readable) names
PARAMETER_NAME_MAPPING = {
    # Image processing
    'sigma_im': 'image_smoothing_sigma',
    'dtype': 'distance_transform_method',
    'thresh_im': 'image_intensity_threshold',
    'thresh_im2': 'background_threshold',
    
    # Cross-link detection
    'thresh_Dxlink': 'crosslink_distance_threshold',
    's_xlinkbox': 'crosslink_search_box_size',
    
    # Local maxima detection
    'thresh_LMP': 'local_max_threshold',
    'thresh_LMPdist': 'local_max_distance_threshold',
    
    # Fiber extension
    'thresh_ext': 'fiber_extension_threshold',
    'lam_dirdecay': 'direction_decay_lambda',
    's_minstep': 'min_step_size',
    's_maxstep': 'max_step_size',
    
    # Dangler removal
    'thresh_dang_aextend': 'dangler_angle_extension_threshold',
    'thresh_dang_L': 'dangler_length_threshold',
    'thresh_dang_aclose': 'dangler_angle_parallel_threshold',
    
    # Fiber processing
    'thresh_short_L': 'short_fiber_length_threshold',
    's_fiberdir': 'fiber_direction_window_size',
    'thresh_linkd': 'fiber_link_distance_threshold',
    'thresh_linka': 'fiber_link_angle_threshold',
    'thresh_flen': 'min_fiber_length',
    'thresh_numv': 'min_vertices_per_fiber',
    
    # Boundary and spacing
    's_boundthick': 'boundary_thickness',
    'blist': 'boundary_list',
    's_maxspace': 'max_gap_size',
    
    # Advanced parameters
    'lambda': 'curvature_penalty',
    'ang_interval': 'angle_bin_size',
    'scale': 'scale',  # No change
}

# Reverse mapping (new names to old names)
REVERSE_PARAMETER_MAPPING = {v: k for k, v in PARAMETER_NAME_MAPPING.items()}


def normalize_parameters(params):
    """
    Convert parameter dictionary to use new readable names internally.
    
    Accepts both old and new parameter names for backward compatibility.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary with either old or new names
    
    Returns
    -------
    dict
        Parameter dictionary with new readable names
    """
    if params is None:
        return {}
    
    normalized = {}
    
    for key, value in params.items():
        # If it's an old name, convert to new name
        if key in PARAMETER_NAME_MAPPING:
            new_key = PARAMETER_NAME_MAPPING[key]
            normalized[new_key] = value
        # If it's already a new name, keep it
        elif key in REVERSE_PARAMETER_MAPPING:
            normalized[key] = value
        # Unknown parameter, keep as is
        else:
            normalized[key] = value
    
    return normalized


def get_parameter(params, new_name, default=None):
    """
    Get parameter value using new readable name, with fallback to old name.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary
    new_name : str
        New readable parameter name
    default : any, optional
        Default value if parameter not found
    
    Returns
    -------
    any
        Parameter value
    """
    # Try new name first
    if new_name in params:
        return params[new_name]
    
    # Fall back to old name for backward compatibility
    old_name = REVERSE_PARAMETER_MAPPING.get(new_name)
    if old_name and old_name in params:
        return params[old_name]
    
    return default


def get_default_parameters():
    """
    Get default ctFIRE parameters with readable names.
    
    Returns
    -------
    dict
        Default parameters with new readable names
    """
    return {
        # Image processing
        'image_smoothing_sigma': 1,
        'distance_transform_method': 'cityblock',
        'image_intensity_threshold': [],
        'background_threshold': 0,
        
        # Cross-link detection
        'crosslink_distance_threshold': 1.5,
        'crosslink_search_box_size': 8,
        
        # Local maxima detection (nucleation points)
        'local_max_threshold': 0.2,
        'local_max_distance_threshold': 2,
        
        # Fiber extension
        'fiber_extension_threshold': 0.342,  # cos(70°)
        'direction_decay_lambda': 0.5,
        'min_step_size': 2,
        'max_step_size': 6,
        
        # Dangler removal
        'dangler_angle_extension_threshold': 0.9848,  # cos(10°)
        'dangler_angle_parallel_threshold': 0.5,  # cos(60°)
        'dangler_length_threshold': 15,
        
        # Fiber processing
        'short_fiber_length_threshold': 15,
        'fiber_direction_window_size': 4,
        'fiber_link_distance_threshold': 15,
        'fiber_link_angle_threshold': -0.866,  # cos(150°)
        'min_fiber_length': 15,
        'min_vertices_per_fiber': 3,
        
        # Boundary and spacing
        'boundary_thickness': 10,
        'boundary_list': 1,
        'max_gap_size': 5,
        
        # Advanced parameters
        'curvature_penalty': 0.01,
        'angle_bin_size': 3,
        'scale': [1.0, 1.0, 1.0],
    }
