from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple


@dataclass
class FiberSegment:
    center: Tuple[float, float]
    angle: float
    weight: Optional[float] = None


def process_fibers(fibers: List, feature_CP: dict) -> Tuple[
    List[FiberSegment], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared processing logic for fiber/curvelet features:
    - Calculates segment/fiber centers, angles
    - Computes length, curvature, width
    - Computes density and alignment features

    Args:
        fibers: list of fiber data (can come from FIRE or CT)
        feature_CP: feature control parameters

    Returns:
        object: list of FiberSegment
        totLengthList, endLengthList, curvatureList, widthList, denList, alignList: numpy arrays of features
    """
    num_fib = len(fibers)
    object_ = []
    totLengthList = np.zeros(num_fib)
    endLengthList = np.zeros(num_fib)
    curvatureList = np.zeros(num_fib)
    widthList = np.zeros(num_fib)
    # Placeholders for density/alignment
    denList = np.zeros((num_fib, 5))  # adjust size as needed
    alignList = np.zeros((num_fib, 5))

    for i, f in enumerate(fibers):
        # Example: compute center, angle, etc.
        cen = f.get('center', (0, 0))
        angle = f.get('angle', 0)
        object_.append(FiberSegment(center=cen, angle=angle, weight=None))

        # placeholder computations
        totLengthList[i] = f.get('totLength', 0)
        endLengthList[i] = f.get('endLength', 0)
        curvatureList[i] = f.get('curvature', 0)
        widthList[i] = f.get('width', 0)

    # TODO: implement density/alignment calculation
    return object_, totLengthList, endLengthList, curvatureList, widthList, denList, alignList


def get_FIRE(img_name: str, fire_dir: str, fiber_mode: int, feature_cp: dict) -> Tuple[
    List[FiberSegment], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CT-FIRE fibers and process them.
    """
    # TODO: Load FIRE .mat file and extract fiber list
    fibers = []  # replace with actual loading logic

    return process_fibers(fibers, featCP)


def get_CT(img_name: str, img_array: np.ndarray, curve_cp: dict, feature_cp: dict) -> Tuple[
    List[FiberSegment], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply curvelet processing and process fibers.
    """
    # TODO: Run newCurv or equivalent Python implementation
    fibers, Ct = [], None  # replace with actual curvelet extraction

    processed = process_fibers(fibers, featCP)
    return (*processed, Ct)
