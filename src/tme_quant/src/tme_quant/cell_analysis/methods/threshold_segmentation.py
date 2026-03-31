"""
Classical thresholding segmentation method (2D and 3D).
"""

import numpy as np
from skimage import measure
from scipy.ndimage import binary_fill_holes

from ..config import SegmentationParams, SegmentationResult
from .base_segmentation import BaseSegmentationMethod


class ThresholdingSegmentation(BaseSegmentationMethod):
    """
    Thresholding-based segmentation.

    Supports Otsu, adaptive, and manual threshold methods.
    Good for high-contrast images.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Segment using thresholding + optional watershed."""
        from skimage import filters, morphology, segmentation
        from scipy import ndimage

        image_prep = self._prepare_image(image)

        if params.threshold_method == "otsu":
            if self.verbose:
                print("Applying Otsu threshold...")
            binary = image_prep > filters.threshold_otsu(image_prep)

        elif params.threshold_method == "adaptive":
            if self.verbose:
                print("Applying adaptive threshold...")
            image_uint8 = (image_prep * 255).astype(np.uint8)
            from skimage.filters import threshold_local
            threshold = threshold_local(
                image_uint8,
                block_size=params.adaptive_block_size,
            )
            binary = image_uint8 > threshold

        elif params.threshold_method == "manual":
            if params.threshold_value is None:
                raise ValueError("threshold_value required for manual threshold")
            binary = image_prep > params.threshold_value

        else:
            raise ValueError(f"Unknown threshold method: {params.threshold_method}")

        if params.fill_holes:
            binary = binary_fill_holes(binary)

        min_size = int(params.min_cell_size / (params.pixel_size ** 2))
        binary = morphology.remove_small_objects(binary, min_size=min_size)

        if params.watershed_markers == "distance":
            from skimage.feature import peak_local_max
            distance = ndimage.distance_transform_edt(binary)
            local_max = peak_local_max(
                distance,
                min_distance=params.watershed_min_distance,
                labels=binary,
            )
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            markers = morphology.dilation(markers, morphology.disk(2))
            labels = segmentation.watershed(-distance, markers, mask=binary)
        else:
            labels = measure.label(binary)

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
        """3D thresholding segmentation (Otsu only)."""
        from skimage import filters, morphology
        from scipy import ndimage

        if params.threshold_method != "otsu":
            raise NotImplementedError("Only Otsu threshold is supported for 3D")

        binary = image > filters.threshold_otsu(image)

        if params.fill_holes:
            binary = ndimage.binary_fill_holes(binary)

        labels = measure.label(binary)
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
