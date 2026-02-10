# Cell Analysis Module - Part 4: Base Classes and Complete Method Implementations

## Base Segmentation Method

### Location: `tme_quant/cell_analysis/methods/segmentation/base_segmentation.py`

```python
"""
Base class for cell segmentation methods.

All segmentation methods inherit from this class and implement
the abstract methods for 2D and 3D segmentation.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from skimage import measure
from scipy.ndimage import binary_fill_holes

from ...config.segmentation_params import SegmentationParams, SegmentationResult
from ....core.tme_models.cell_model import CellProperties


class BaseSegmentationMethod(ABC):
    """
    Abstract base class for segmentation methods.
    
    Subclasses must implement:
        - segment_2d(): 2D segmentation
        - segment_3d(): 3D segmentation (optional)
        - supports_3d(): Whether method supports 3D
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize base segmentation method.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
    
    @abstractmethod
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells in 2D image.
        
        Args:
            image: 2D image
            params: Segmentation parameters
            
        Returns:
            SegmentationResult with detected cells
        """
        pass
    
    @abstractmethod
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment cells in 3D image.
        
        Args:
            image: 3D image
            params: Segmentation parameters
            
        Returns:
            SegmentationResult with detected cells
        """
        pass
    
    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this method supports 3D segmentation."""
        pass
    
    # ============================================================
    # HELPER METHODS (shared across all segmentation methods)
    # ============================================================
    
    def _labels_to_cells(
        self,
        labels: np.ndarray,
        pixel_size: float = 1.0
    ) -> List[CellProperties]:
        """
        Convert label image to list of CellProperties.
        
        Args:
            labels: Label image (2D)
            pixel_size: Pixel size in microns
            
        Returns:
            List of CellProperties
        """
        cells = []
        
        # Get region properties
        regions = measure.regionprops(labels)
        
        for region in regions:
            # Geometric properties
            area = region.area * (pixel_size ** 2)
            perimeter = region.perimeter * pixel_size
            centroid = (region.centroid[1] * pixel_size, region.centroid[0] * pixel_size)  # (x, y)
            
            # Shape properties
            if region.perimeter > 0:
                circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            else:
                circularity = 0.0
            
            eccentricity = region.eccentricity
            solidity = region.solidity
            extent = region.extent
            
            major_axis = region.major_axis_length * pixel_size
            minor_axis = region.minor_axis_length * pixel_size
            orientation = np.degrees(region.orientation)
            
            # Boundary coordinates
            boundary = region.coords  # Nx2 array (row, col)
            boundary_scaled = boundary[:, ::-1] * pixel_size  # Convert to (x, y) and scale
            
            # Create CellProperties
            cell = CellProperties(
                cell_id=region.label,
                area=area,
                perimeter=perimeter,
                centroid=centroid,
                circularity=min(circularity, 1.0),  # Clamp to [0, 1]
                eccentricity=eccentricity,
                solidity=solidity,
                extent=extent,
                major_axis_length=major_axis,
                minor_axis_length=minor_axis,
                orientation=orientation,
                boundary=boundary_scaled,
                mask=labels == region.label
            )
            
            cells.append(cell)
        
        return cells
    
    def _labels_to_cells_3d(
        self,
        labels: np.ndarray,
        pixel_size: float = 1.0
    ) -> List[CellProperties]:
        """
        Convert 3D label image to list of CellProperties.
        
        Args:
            labels: Label image (3D)
            pixel_size: Voxel size in microns
            
        Returns:
            List of CellProperties (with simplified 2D projection)
        """
        cells = []
        
        # Get region properties in 3D
        regions = measure.regionprops(labels)
        
        for region in regions:
            # Volume (convert to approximate 2D area for consistency)
            volume = region.area * (pixel_size ** 3)
            area = volume ** (2/3)  # Approximate 2D area from volume
            
            # Centroid (use x, y, ignore z for 2D compatibility)
            centroid_3d = region.centroid
            centroid = (centroid_3d[2] * pixel_size, centroid_3d[1] * pixel_size)
            
            # Simplified shape properties
            solidity = region.solidity
            extent = region.extent
            
            # Create CellProperties
            cell = CellProperties(
                cell_id=region.label,
                area=area,
                perimeter=0.0,  # Not meaningful in 3D
                centroid=centroid,
                circularity=0.0,
                eccentricity=0.0,
                solidity=solidity,
                extent=extent,
                major_axis_length=0.0,
                minor_axis_length=0.0,
                orientation=0.0,
                boundary=np.array([]),  # Not computed for 3D
                mask=None
            )
            
            cells.append(cell)
        
        return cells
    
    def _post_process_cells(
        self,
        cells: List[CellProperties],
        params: SegmentationParams
    ) -> List[CellProperties]:
        """
        Post-process segmented cells (filtering, cleanup).
        
        Args:
            cells: List of segmented cells
            params: Segmentation parameters
            
        Returns:
            Filtered list of cells
        """
        filtered_cells = []
        
        for cell in cells:
            # Filter by size
            if cell.area < params.min_cell_size:
                continue
            if cell.area > params.max_cell_size:
                continue
            
            # Keep cell
            filtered_cells.append(cell)
        
        return filtered_cells
    
    def _prepare_image(
        self,
        image: np.ndarray,
        target_channel: Optional[int] = None
    ) -> np.ndarray:
        """
        Prepare image for segmentation (channel selection, normalization).
        
        Args:
            image: Input image
            target_channel: Channel to use (None for grayscale or auto-select)
            
        Returns:
            Prepared image
        """
        # Handle multi-channel images
        if image.ndim == 3 and image.shape[-1] <= 4:  # Assume (H, W, C)
            if target_channel is not None:
                image = image[:, :, target_channel]
            else:
                # Use first channel by default
                image = image[:, :, 0]
        
        # Normalize to 0-1
        if image.max() > 1.0:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
        
        return image


# ============================================================
# COMPLETE STARDIST IMPLEMENTATION
# ============================================================

class StarDistSegmentation(BaseSegmentationMethod):
    """
    Complete StarDist segmentation implementation.
    
    StarDist uses star-convex polygons for cell/nucleus segmentation.
    Excellent for dense, roundish objects.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model_2d = None
        self.model_3d = None
        self._check_installation()
    
    def _check_installation(self):
        """Check if StarDist is installed."""
        try:
            import stardist
            self.stardist_available = True
        except ImportError:
            self.stardist_available = False
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Segment cells using StarDist 2D."""
        if not self.stardist_available:
            raise ImportError(
                "StarDist not installed. Install with: pip install stardist"
            )
        
        from stardist.models import StarDist2D
        from stardist import normalize
        
        # Load model
        if self.model_2d is None:
            if self.verbose:
                print(f"Loading StarDist model: {params.stardist_model}")
            
            try:
                self.model_2d = StarDist2D.from_pretrained(params.stardist_model)
            except Exception as e:
                # Fallback to default model
                if self.verbose:
                    print(f"Failed to load {params.stardist_model}, using default")
                self.model_2d = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        # Prepare image
        image_prep = self._prepare_image(image)
        
        # Normalize
        image_norm = normalize(image_prep, 1, 99.8)
        
        # Predict
        if self.verbose:
            print("Running StarDist prediction...")
        
        labels, details = self.model_2d.predict_instances(
            image_norm,
            prob_thresh=params.stardist_prob_thresh,
            nms_thresh=params.stardist_nms_thresh
        )
        
        # Convert to cells
        cells = self._labels_to_cells(labels, params.pixel_size)
        
        # Post-process
        cells = self._post_process_cells(cells, params)
        
        if self.verbose:
            print(f"Detected {len(cells)} cells")
        
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
        if not self.stardist_available:
            raise ImportError("StarDist not installed")
        
        from stardist.models import StarDist3D
        from stardist import normalize
        
        # Load model
        if self.model_3d is None:
            if self.verbose:
                print("Loading StarDist3D model")
            
            try:
                self.model_3d = StarDist3D.from_pretrained('3D_demo')
            except:
                raise ValueError("StarDist3D model not available")
        
        # Normalize
        image_norm = normalize(image, 1, 99.8)
        
        # Predict
        labels, details = self.model_3d.predict_instances(image_norm)
        
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


# ============================================================
# COMPLETE CELLPOSE IMPLEMENTATION
# ============================================================

class CellposeSegmentation(BaseSegmentationMethod):
    """
    Complete Cellpose segmentation implementation.
    
    Cellpose is versatile and works across many imaging modalities
    and cell types.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.model = None
        self._check_installation()
    
    def _check_installation(self):
        """Check if Cellpose is installed."""
        try:
            import cellpose
            self.cellpose_available = True
        except ImportError:
            self.cellpose_available = False
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Segment cells using Cellpose."""
        if not self.cellpose_available:
            raise ImportError(
                "Cellpose not installed. Install with: pip install cellpose"
            )
        
        from cellpose import models
        
        # Load model
        if self.model is None:
            if self.verbose:
                print(f"Loading Cellpose model: {params.cellpose_model}")
            
            # Check for GPU
            import torch
            gpu = torch.cuda.is_available()
            
            self.model = models.Cellpose(
                gpu=gpu,
                model_type=params.cellpose_model
            )
        
        # Determine channels
        # Cellpose uses [0, 0] for grayscale, [1, 2] for [cytoplasm, nucleus]
        if image.ndim == 2:
            channels = [0, 0]  # Grayscale
        elif image.ndim == 3 and image.shape[2] == 1:
            channels = [0, 0]  # Single channel
        else:
            # Multi-channel: assume first is cytoplasm, second is nucleus
            channels = [1, 2] if params.target == "whole_cell" else [0, 0]
        
        if self.verbose:
            print(f"Running Cellpose with channels {channels}...")
        
        # Run segmentation
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            flow_threshold=params.cellpose_flow_threshold,
            cellprob_threshold=params.cellpose_cellprob_threshold,
            channels=channels
        )
        
        # Convert to cells
        cells = self._labels_to_cells(masks, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        if self.verbose:
            print(f"Detected {len(cells)} cells")
        
        # Create result
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
        """Segment 3D image with Cellpose."""
        if not self.cellpose_available:
            raise ImportError("Cellpose not installed")
        
        from cellpose import models
        
        # Load model
        if self.model is None:
            import torch
            gpu = torch.cuda.is_available()
            self.model = models.Cellpose(gpu=gpu, model_type=params.cellpose_model)
        
        # Run 3D segmentation
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=params.cellpose_diameter,
            do_3D=True,
            channels=[0, 0]
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


# ============================================================
# COMPLETE THRESHOLDING IMPLEMENTATION
# ============================================================

class ThresholdingSegmentation(BaseSegmentationMethod):
    """
    Classical thresholding-based segmentation.
    
    Supports Otsu, adaptive thresholding, and manual threshold.
    Good for high-contrast images.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Segment using thresholding."""
        from skimage import filters, morphology, segmentation
        from scipy import ndimage
        
        # Prepare image
        image_prep = self._prepare_image(image)
        
        # Apply threshold
        if params.threshold_method == "otsu":
            if self.verbose:
                print("Applying Otsu threshold...")
            threshold = filters.threshold_otsu(image_prep)
            binary = image_prep > threshold
        
        elif params.threshold_method == "adaptive":
            if self.verbose:
                print("Applying adaptive threshold...")
            # Convert to uint8 for adaptive threshold
            image_uint8 = (image_prep * 255).astype(np.uint8)
            from skimage.filters import threshold_local
            threshold = threshold_local(
                image_uint8,
                block_size=params.adaptive_block_size
            )
            binary = image_uint8 > threshold
        
        elif params.threshold_method == "manual":
            if params.threshold_value is None:
                raise ValueError("threshold_value required for manual threshold")
            binary = image_prep > params.threshold_value
        
        else:
            raise ValueError(f"Unknown threshold method: {params.threshold_method}")
        
        # Post-processing
        if params.fill_holes:
            binary = binary_fill_holes(binary)
        
        # Remove small objects
        min_size = int(params.min_cell_size / (params.pixel_size ** 2))
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Separate touching objects (optional watershed)
        if params.watershed_markers == "distance":
            # Distance transform watershed
            distance = ndimage.distance_transform_edt(binary)
            
            # Find local maxima
            from skimage.feature import peak_local_max
            local_max = peak_local_max(
                distance,
                min_distance=params.watershed_min_distance,
                labels=binary
            )
            
            # Create markers
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            markers = morphology.dilation(markers, morphology.disk(2))
            
            # Watershed
            labels = segmentation.watershed(-distance, markers, mask=binary)
        else:
            # Simple connected components
            labels = measure.label(binary)
        
        # Convert to cells
        cells = self._labels_to_cells(labels, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        if self.verbose:
            print(f"Detected {len(cells)} cells")
        
        result = SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """3D thresholding segmentation."""
        from skimage import filters, morphology
        from scipy import ndimage
        
        # Threshold
        if params.threshold_method == "otsu":
            threshold = filters.threshold_otsu(image)
            binary = image > threshold
        else:
            raise NotImplementedError("Only Otsu supported for 3D")
        
        # Fill holes
        if params.fill_holes:
            binary = ndimage.binary_fill_holes(binary)
        
        # Label
        labels = measure.label(binary)
        
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


# ============================================================
# COMPLETE WATERSHED IMPLEMENTATION
# ============================================================

class WatershedSegmentation(BaseSegmentationMethod):
    """
    Watershed segmentation for separating touching cells.
    
    Uses distance transform or provided markers.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
    
    def segment_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """Watershed segmentation."""
        from skimage import filters, morphology, segmentation
        from scipy import ndimage
        from skimage.feature import peak_local_max
        
        # Prepare image
        image_prep = self._prepare_image(image)
        
        # Initial thresholding
        threshold = filters.threshold_otsu(image_prep)
        binary = image_prep > threshold
        
        # Fill holes
        binary = binary_fill_holes(binary)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find markers
        if params.watershed_markers == "distance":
            # Use distance transform peaks
            local_max = peak_local_max(
                distance,
                min_distance=params.watershed_min_distance,
                labels=binary
            )
            
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            markers = morphology.dilation(markers, morphology.disk(2))
        
        elif params.watershed_markers == "peaks":
            # Use image intensity peaks
            local_max = peak_local_max(
                image_prep,
                min_distance=params.watershed_min_distance,
                labels=binary
            )
            
            markers = np.zeros_like(binary, dtype=int)
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        
        else:
            raise ValueError(f"Unknown marker method: {params.watershed_markers}")
        
        # Watershed
        if self.verbose:
            print(f"Running watershed with {markers.max()} markers...")
        
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        # Convert to cells
        cells = self._labels_to_cells(labels, params.pixel_size)
        cells = self._post_process_cells(cells, params)
        
        if self.verbose:
            print(f"Detected {len(cells)} cells")
        
        result = SegmentationResult(
            mode=params.mode,
            dimension="2D",
            image_modality=params.image_modality,
            cells=cells,
            label_mask=labels,
            pixel_size=params.pixel_size
        )
        
        return result
    
    def segment_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """3D watershed segmentation."""
        from skimage import filters, segmentation
        from scipy import ndimage
        from skimage.feature import peak_local_max
        
        # Threshold
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find markers
        local_max = peak_local_max(
            distance,
            min_distance=params.watershed_min_distance,
            labels=binary
        )
        
        markers = np.zeros_like(binary, dtype=int)
        for i, coord in enumerate(local_max):
            markers[tuple(coord)] = i + 1
        
        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
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