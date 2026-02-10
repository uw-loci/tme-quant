# tme_quant/cell_analysis/methods/segmentation/cellpose.py
"""
Cellpose deep learning-based cell segmentation.

Cellpose is versatile and works across many imaging modalities.
"""

import numpy as np
from .base_segmentation import BaseSegmentationMethod
from ...config.segmentation_params import SegmentationParams, SegmentationResult

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


class CellposeSegmentation(BaseSegmentationMethod):
    """
    Cellpose segmentation method.
    
    Supports nucleus and cytoplasm segmentation across multiple modalities.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model = None
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells using Cellpose.
        
        Args:
            image: 2D image
            params: Segmentation parameters
            
        Returns:
            SegmentationResult with detected cells
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is not installed. Install with: pip install cellpose"
            )
        
        # Load model
        if self.model is None:
            if self.verbose:
                print(f"Loading Cellpose model: {params.cellpose_model}")
            
            gpu = False  # Set to True if GPU available
            self.model = models.Cellpose(gpu=gpu, model_type=params.cellpose_model)
        
        # Run segmentation
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            flow_threshold=params.cellpose_flow_threshold,
            cellprob_threshold=params.cellpose_cellprob_threshold,
            channels=[0, 0] if image.ndim == 2 else [1, 2]  # grayscale or multi-channel
        )
        
        # Convert to cells
        cells = self._labels_to_cells(masks, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        result = SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=masks,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Segment 3D image with Cellpose (processes as stack)."""
        # Cellpose 3D support
        if self.model is None:
            self.model = models.Cellpose(gpu=False, model_type=params.cellpose_model)
        
        masks, _, _, _ = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            do_3D=True
        )
        
        cells = self._labels_to_cells_3d(masks, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        result = SegmentationResult(
            mode=params.mode,
            dimension="3D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=masks,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def supports_3d(self) -> bool:
        return True