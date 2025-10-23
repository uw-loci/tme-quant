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
