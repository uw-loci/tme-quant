"""
Abstract base class for cell segmentation methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from skimage import measure
from scipy.ndimage import binary_fill_holes

from ..config import SegmentationParams, SegmentationResult
from tme_quant.core.tme_models.cell_model import CellProperties


class BaseSegmentationMethod(ABC):
    """
    Abstract base class for segmentation methods.

    Subclasses must implement:
        - segment_2d(): 2D segmentation
        - segment_3d(): 3D segmentation
        - supports_3d(): Whether method supports 3D
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        pass

    @abstractmethod
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        pass

    @abstractmethod
    def supports_3d(self) -> bool:
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _labels_to_cells(
        self,
        labels: np.ndarray,
        pixel_size: float = 1.0,
    ) -> List[CellProperties]:
        """Convert a 2D label image to a list of CellProperties."""
        cells = []
        for region in measure.regionprops(labels):
            area = region.area * (pixel_size ** 2)
            perimeter = region.perimeter * pixel_size
            centroid = (
                region.centroid[1] * pixel_size,
                region.centroid[0] * pixel_size,
            )
            circularity = (
                (4 * np.pi * region.area) / (region.perimeter ** 2)
                if region.perimeter > 0
                else 0.0
            )
            boundary = region.coords
            boundary_scaled = boundary[:, ::-1] * pixel_size

            cells.append(CellProperties(
                cell_id=region.label,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                circularity=min(circularity, 1.0),
                eccentricity=region.eccentricity,
                solidity=region.solidity,
                extent=region.extent,
                major_axis_length=region.major_axis_length * pixel_size,
                minor_axis_length=region.minor_axis_length * pixel_size,
                orientation=np.degrees(region.orientation),
                boundary=boundary_scaled,
                mask=labels == region.label,
            ))
        return cells

    def _labels_to_cells_3d(
        self,
        labels: np.ndarray,
        pixel_size: float = 1.0,
    ) -> List[CellProperties]:
        """Convert a 3D label image to a list of CellProperties (simplified)."""
        cells = []
        for region in measure.regionprops(labels):
            volume = region.area * (pixel_size ** 3)
            area = volume ** (2 / 3)
            centroid_3d = region.centroid
            centroid = (centroid_3d[2] * pixel_size, centroid_3d[1] * pixel_size)

            cells.append(CellProperties(
                cell_id=region.label,
                area=area,
                perimeter=0.0,
                centroid=centroid,
                circularity=0.0,
                eccentricity=0.0,
                solidity=region.solidity,
                extent=region.extent,
                major_axis_length=0.0,
                minor_axis_length=0.0,
                orientation=0.0,
                boundary=np.array([]),
                mask=None,
            ))
        return cells

    def _post_process_cells(
        self,
        cells: List[CellProperties],
        params: SegmentationParams,
    ) -> List[CellProperties]:
        """Filter cells by size."""
        return [
            c for c in cells
            if params.min_cell_size <= c.area <= params.max_cell_size
        ]

    def _prepare_image(
        self,
        image: np.ndarray,
        target_channel: Optional[int] = None,
    ) -> np.ndarray:
        """Select channel and normalise to [0, 1]."""
        if image.ndim == 3 and image.shape[-1] <= 4:
            image = image[:, :, target_channel if target_channel is not None else 0]
        if image.max() > 1.0:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
        return image
