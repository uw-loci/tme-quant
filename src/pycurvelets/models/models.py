from dataclasses import dataclass


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
