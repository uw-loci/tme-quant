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
