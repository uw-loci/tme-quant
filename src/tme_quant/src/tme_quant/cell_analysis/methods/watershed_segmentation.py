"""
Watershed segmentation method (2D and 3D).
"""

import numpy as np
from skimage import measure
from scipy.ndimage import binary_fill_holes

from ..config import SegmentationParams, SegmentationResult
from .base_segmentation import BaseSegmentationMethod


class WatershedSegmentation(BaseSegmentationMethod):
    """
    Watershed segmentation for separating touching cells.

    Uses distance transform or image intensity peaks as markers.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Watershed segmentation in 2D."""
        from skimage import filters, morphology, segmentation
        from skimage.feature import peak_local_max
        from scipy import ndimage

        image_prep = self._prepare_image(image)

        binary = image_prep > filters.threshold_otsu(image_prep)
        binary = binary_fill_holes(binary)
        distance = ndimage.distance_transform_edt(binary)

        if params.watershed_markers == "distance":
            local_max = peak_local_max(
                distance,
                min_distance=params.watershed_min_distance,
                labels=binary,
            )
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            markers = morphology.dilation(markers, morphology.disk(2))

        elif params.watershed_markers == "peaks":
            local_max = peak_local_max(
                image_prep,
                min_distance=params.watershed_min_distance,
                labels=binary,
            )
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

        else:
            raise ValueError(f"Unknown marker method: {params.watershed_markers}")

        if self.verbose:
            print(f"Running watershed with {markers.max()} markers...")

        labels = segmentation.watershed(-distance, markers, mask=binary)
        cells = self._labels_to_cells(labels, params.pixel_size)
        cells = self._post_process_cells(cells, params)

        if self.verbose:
            print(f"Detected {len(cells)} cells")

        return SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            pixel_size=params.pixel_size,
        )

    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """3D watershed segmentation."""
        from skimage import filters, segmentation
        from skimage.feature import peak_local_max
        from scipy import ndimage

        binary = image > filters.threshold_otsu(image)
        distance = ndimage.distance_transform_edt(binary)

        local_max = peak_local_max(
            distance,
            min_distance=params.watershed_min_distance,
            labels=binary,
        )
        markers = np.zeros_like(binary, dtype=int)
        for i, coord in enumerate(local_max):
            markers[tuple(coord)] = i + 1

        labels = segmentation.watershed(-distance, markers, mask=binary)
        cells = self._labels_to_cells_3d(labels, params.pixel_size)
        cells = self._post_process_cells(cells, params)

        return SegmentationResult(
            mode=params.mode,
            dimension="3D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            pixel_size=params.pixel_size,
        )

    def supports_3d(self) -> bool:
        return True
