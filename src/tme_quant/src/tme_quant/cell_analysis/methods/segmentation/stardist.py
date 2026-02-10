#
"""
StarDist deep learning-based cell segmentation.

StarDist predicts star-convex polygons, making it efficient for dense,
roundish objects like nuclei.
"""

import numpy as np
from typing import Optional
from .base_segmentation import BaseSegmentationMethod
from ...config.segmentation_params import SegmentationParams, SegmentationResult
from ....core.tme_models.cell_model import CellProperties, ImageModality

try:
    from stardist.models import StarDist2D, StarDist3D
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False


class StarDistSegmentation(BaseSegmentationMethod):
    """
    StarDist segmentation method.
    
    Uses pre-trained or custom models for nucleus/cell segmentation.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model_2d = None
        self.model_3d = None
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells using StarDist 2D.
        
        Args:
            image: 2D image (grayscale or multi-channel)
            params: Segmentation parameters
            
        Returns:
            SegmentationResult with detected cells
        """
        if not STARDIST_AVAILABLE:
            raise ImportError(
                "StarDist is not installed. Install with: pip install stardist"
            )
        
        # Load model
        if self.model_2d is None:
            if self.verbose:
                print(f"Loading StarDist model: {params.stardist_model}")
            self.model_2d = StarDist2D.from_pretrained(params.stardist_model)
        
        # Prepare image (StarDist expects single channel for nucleus)
        if image.ndim == 3:
            # Take first channel or convert to grayscale
            image_gray = image[:, :, 0] if image.shape[2] > 1 else image.squeeze()
        else:
            image_gray = image
        
        # Normalize image
        from stardist import normalize
        image_norm = normalize(image_gray, 1, 99.8)
        
        # Run prediction
        labels, details = self.model_2d.predict_instances(
            image_norm,
            prob_thresh=params.stardist_prob_thresh,
            nms_thresh=params.stardist_nms_thresh
        )
        
        # Convert to CellProperties
        cells = self._labels_to_cells(labels, params.pixel_size)
        
        # Post-processing
        cells = self._post_process_cells(cells, params)
        
        # Create result
        result = SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            probability_map=details['prob'] if params.return_probabilities else None,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Segment cells using StarDist 3D."""
        if not STARDIST_AVAILABLE:
            raise ImportError("StarDist is not installed")
        
        # Load model
        if self.model_3d is None:
            self.model_3d = StarDist3D.from_pretrained(params.stardist_model)
        
        # Run prediction
        labels, details = self.model_3d.predict_instances(image)
        
        # Convert to cells
        cells = self._labels_to_cells_3d(labels, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        result = SegmentationResult(
            mode=params.mode,
            dimension="3D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def supports_3d(self) -> bool:
        return True