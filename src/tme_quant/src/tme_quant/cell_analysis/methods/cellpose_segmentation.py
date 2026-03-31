"""
Cellpose segmentation method (2D and 3D).
"""

import numpy as np

from ..config import SegmentationParams, SegmentationResult
from .base_segmentation import BaseSegmentationMethod


class CellposeSegmentation(BaseSegmentationMethod):
    """
    Cellpose segmentation.

    Versatile across imaging modalities and cell types.
    Supports both 2D and 3D segmentation.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model = None
        self._check_installation()

    def _check_installation(self) -> None:
        try:
            import cellpose  # noqa: F401
            self.cellpose_available = True
        except ImportError:
            self.cellpose_available = False

    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Segment cells using Cellpose."""
        if not self.cellpose_available:
            raise ImportError(
                "Cellpose not installed. Install with: pip install cellpose"
            )

        from cellpose import models

        if self.model is None:
            if self.verbose:
                print(f"Loading Cellpose model: {params.cellpose_model}")
            self.model = models.Cellpose(
                gpu=params.use_gpu,
                model_type=params.cellpose_model,
            )

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            channels = [0, 0]
        else:
            channels = [1, 2] if params.target == "whole_cell" else [0, 0]

        if self.verbose:
            print(f"Running Cellpose with channels {channels}...")

        masks, _, _, _ = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            flow_threshold=params.cellpose_flow_threshold,
            cellprob_threshold=params.cellpose_cellprob_threshold,
            channels=channels,
        )

        cells = self._labels_to_cells(masks, params.pixel_size)
        cells = self._post_process_cells(cells, params)

        if self.verbose:
            print(f"Detected {len(cells)} cells")

        return SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=masks,
            pixel_size=params.pixel_size,
        )

    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Segment 3D image with Cellpose (do_3D=True)."""
        if not self.cellpose_available:
            raise ImportError("Cellpose not installed")

        from cellpose import models

        if self.model is None:
            import torch
            self.model = models.Cellpose(
                gpu=torch.cuda.is_available(),
                model_type=params.cellpose_model,
            )

        masks, _, _, _ = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            do_3D=True,
            channels=[0, 0],
        )

        cells = self._labels_to_cells_3d(masks, params.pixel_size)
        cells = self._post_process_cells(cells, params)

        return SegmentationResult(
            mode=params.mode,
            dimension="3D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=masks,
            pixel_size=params.pixel_size,
        )

    def supports_3d(self) -> bool:
        return True
