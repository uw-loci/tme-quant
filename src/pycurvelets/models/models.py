from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd


# Custom Exception Classes for Fiber Analysis
class FiberAnalysisError(Exception):
    """Base exception for fiber analysis operations."""

    pass


class ROIProcessingError(FiberAnalysisError):
    """Exception raised during ROI processing."""

    pass


class BoundaryAnalysisError(FiberAnalysisError):
    """Exception raised during boundary analysis."""

    pass


class FeatureExtractionError(FiberAnalysisError):
    """Exception raised during feature extraction."""

    pass


class ImageProcessingError(FiberAnalysisError):
    """Exception raised during image processing operations."""

    pass


@dataclass
class CurveletControlParameters:
    keep: float
    scale: float
    radius: float


@dataclass
class FeatureControlParameters:
    minimum_nearest_fibers: int
    minimum_box_size: int
    fiber_midpoint_estimate: int


@dataclass
class Fiber:
    center_row: float
    center_col: float
    angle: float


@dataclass
class FiberFeatures:
    object: tuple
    fiber_key: int
    total_length: float
    end_length: float
    curvature: float
    width: float
    density: float
    alignment: float
    curvelet_coefficients: list


@dataclass
class ROI:
    coordinates: tuple
    image_width: int
    image_height: int
    distance: Optional[float]
    index_to_object: Optional[int]


@dataclass
class ROIList:
    coordinates: list[tuple]
    image_width: int
    image_height: int


@dataclass
class ImageInputParameters:
    """Parameters related to input image and identification."""
    img: np.ndarray
    img_name: str
    slice_num: int = 1
    num_sections: int = 1


@dataclass
class BoundaryParameters:
    """Parameters for boundary analysis."""
    coordinates: Optional[Dict] = None
    distance_threshold: float = 100.0
    tif_boundary: int = 0  # 0=none, 1/2=CSV, 3=TIFF
    boundary_img: Optional[np.ndarray] = None


@dataclass
class FiberAnalysisParameters:
    """Parameters for fiber/curvelet analysis method."""
    fiber_mode: int = 0  # 0=curvelet, 1/2/3=FIRE
    keep: float = 0.05  # Percentage of curvelets to keep
    fire_directory: Optional[str] = None


@dataclass
class OutputControlParameters:
    """Control flags for output generation."""
    output_directory: str = "."
    make_associations: bool = False
    make_map: bool = False
    make_overlay: bool = False
    make_feature_file: bool = False


@dataclass
class AdvancedAnalysisOptions:
    """Advanced options for analysis customization."""
    # Fiber exclusion and grouping
    exclude_fibers_in_mask_flag: int = 0
    curvelets_group_radius: float = 10.0
    selected_scale: int = 1
    min_dist: Union[List, float] = None
    
    # Feature extraction
    minimum_nearest_fibers: int = 2
    minimum_box_size: int = 32
    fiber_midpoint_estimate: int = 1
    
    # Heatmap visualization
    heatmap_STD_filter_size: int = 24
    heatmap_SQUARE_max_filter_size: int = 12
    heatmap_GAUSSIAN_disc_filter_sigma: float = 4.0
    
    def __post_init__(self):
        """Initialize min_dist to empty list if None."""
        if self.min_dist is None:
            self.min_dist = []
