"""
StarDist segmentation method (2D and 3D).
"""

import numpy as np

from ..config import SegmentationParams, SegmentationResult
from .base_segmentation import BaseSegmentationMethod


class StarDistSegmentation(BaseSegmentationMethod):
    """
    StarDist segmentation using star-convex polygons.

    Excellent for dense, roundish nuclei/cells.
    Supports both 2D (StarDist2D) and 3D (StarDist3D) models.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model_2d = None
        self.model_3d = None
        self._check_installation()

    def _check_installation(self) -> None:
        try:
            import stardist  # noqa: F401
            self.stardist_available = True
        except ImportError:
            self.stardist_available = False

    @staticmethod
    def _get_normalize():
        """Return a percentile-normalisation function, preferring csbdeep."""
        try:
            from csbdeep.utils import normalize
            return normalize
        except ImportError:
            pass
        try:
            from stardist.utils import normalize
            return normalize
        except ImportError:
            pass

        def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False):
            lo = np.percentile(x, pmin, axis=axis, keepdims=True)
            hi = np.percentile(x, pmax, axis=axis, keepdims=True)
            out = (x - lo) / (np.maximum(hi - lo, 1e-20))
            return np.clip(out, 0, 1) if clip else out

        return normalize

    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Segment cells using StarDist2D."""
        if not self.stardist_available:
            raise ImportError(
                "StarDist not installed. Install with: pip install stardist"
            )

        from stardist.models import StarDist2D

        if not params.use_gpu:
            try:
                import tensorflow as tf
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass

        if self.model_2d is None:
            if self.verbose:
                print(f"Loading StarDist model: {params.stardist_model}")
            try:
                self.model_2d = StarDist2D.from_pretrained(params.stardist_model)
            except Exception:
                if self.verbose:
                    print(f"Failed to load {params.stardist_model}, using default")
                self.model_2d = StarDist2D.from_pretrained('2D_versatile_fluo')

        normalize = self._get_normalize()
        he_model = 'he' in params.stardist_model.lower()
        if he_model and image.ndim == 3 and image.shape[-1] == 3:
            image_norm = normalize(
                image.astype(np.float32), 1, 99.8, axis=(0, 1)
            )
        else:
            image_prep = self._prepare_image(image)
            image_norm = normalize(image_prep, 1, 99.8)

        if self.verbose:
            print("Running StarDist prediction...")

        labels, details = self.model_2d.predict_instances(
            image_norm,
            prob_thresh=params.stardist_prob_thresh,
            nms_thresh=params.stardist_nms_thresh,
        )

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
            probability_map=details['prob'] if params.return_probabilities else None,
            pixel_size=params.pixel_size,
        )

    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
    ) -> SegmentationResult:
        """Segment cells using StarDist3D."""
        if not self.stardist_available:
            raise ImportError("StarDist not installed")

        from stardist.models import StarDist3D

        if self.model_3d is None:
            if self.verbose:
                print("Loading StarDist3D model")
            try:
                self.model_3d = StarDist3D.from_pretrained('3D_demo')
            except Exception:
                raise ValueError("StarDist3D model not available")

        normalize = self._get_normalize()
        image_norm = normalize(image, 1, 99.8)
        labels, _ = self.model_3d.predict_instances(image_norm)

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
