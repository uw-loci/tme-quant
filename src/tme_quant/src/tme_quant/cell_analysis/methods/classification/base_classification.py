# Cell Analysis Module - Part 5: Classification and Quantification Methods

## Base Classification Method

### Location: `tme_quant/cell_analysis/methods/classification/base_classification.py`


"""
Base class for cell classification methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ...config.classification_params import ClassificationParams, ClassificationResult
from ....core.tme_models.cell_model import SegmentationResult


class BaseClassificationMethod(ABC):
    """
    Abstract base class for classification methods.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    @abstractmethod
    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None
    ) -> ClassificationResult:
        """
        Classify segmented cells.
        
        Args:
            segmentation_result: Segmentation result
            params: Classification parameters
            image: Original image (for intensity-based methods)
            
        Returns:
            ClassificationResult with cell type assignments
        """
        pass


# ============================================================
# MORPHOLOGY-BASED CLASSIFIER
# ============================================================

class MorphologyClassifier(BaseClassificationMethod):
    """
    Classify cells based on morphological features.
    
    Uses area, shape, and texture to distinguish cell types.
    """
    
    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None
    ) -> ClassificationResult:
        """Classify cells using morphology."""
        from ....core.tme_models.cell_model import CellType
        
        result = ClassificationResult(mode=params.mode)
        
        for cell in segmentation_result.cells:
            # Extract features
            features = self._extract_morphology_features(cell)
            
            # Classify based on rules
            cell_type, confidence = self._classify_morphology(
                features, params
            )
            
            result.cell_types[cell.cell_id] = cell_type
            result.confidences[cell.cell_id] = confidence
        
        return result
    
    def _extract_morphology_features(self, cell) -> dict:
        """Extract morphological features."""
        features = {}
        
        if hasattr(cell, 'area'):
            features['area'] = cell.area
        if hasattr(cell, 'perimeter'):
            features['perimeter'] = cell.perimeter
        if hasattr(cell, 'circularity'):
            features['circularity'] = cell.circularity
        if hasattr(cell, 'eccentricity'):
            features['eccentricity'] = cell.eccentricity
        if hasattr(cell, 'solidity'):
            features['solidity'] = cell.solidity
        if hasattr(cell, 'extent'):
            features['extent'] = cell.extent
        
        # Derived features
        if 'area' in features and features['area'] > 0:
            features['equivalent_diameter'] = 2 * np.sqrt(features['area'] / np.pi)
        
        return features
    
    def _classify_morphology(self, features: dict, params) -> tuple:
        """
        Classify cell based on morphology.
        
        Simple rule-based classification:
        - Tumor cells: Large, irregular (low circularity)
        - Immune cells: Small, round (high circularity)
        - Fibroblasts: Elongated (high eccentricity)
        """
        from ....core.tme_models.cell_model import CellType
        
        area = features.get('area', 0)
        circularity = features.get('circularity', 0)
        eccentricity = features.get('eccentricity', 0)
        
        # Default classification rules
        if area > 200 and circularity < 0.7:
            # Large, irregular → Tumor cell
            return CellType.TUMOR, 0.7
        
        elif area < 100 and circularity > 0.8:
            # Small, round → Immune cell
            return CellType.IMMUNE, 0.6
        
        elif eccentricity > 0.7:
            # Elongated → Fibroblast
            return CellType.FIBROBLAST, 0.6
        
        else:
            # Unknown
            return CellType.STROMAL, 0.4


# ============================================================
# MARKER-BASED CLASSIFIER
# ============================================================

class MarkerClassifier(BaseClassificationMethod):
    """
    Classify cells based on immunofluorescence markers.
    
    Uses marker expression (e.g., CD3, CD8, CD68) to identify cell types.
    """
    
    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None
    ) -> ClassificationResult:
        """Classify cells using marker expression."""
        from ....core.tme_models.cell_model import CellType
        
        if image is None:
            raise ValueError("Image required for marker-based classification")
        
        if not params.marker_channels:
            raise ValueError("marker_channels required for marker classification")
        
        result = ClassificationResult(mode=params.mode)
        
        # Extract marker intensities for all cells
        marker_intensities = self._extract_marker_intensities(
            segmentation_result, image, params
        )
        
        # Classify each cell
        for cell in segmentation_result.cells:
            cell_markers = marker_intensities.get(cell.cell_id, {})
            
            cell_type, confidence = self._classify_by_markers(
                cell_markers, params
            )
            
            result.cell_types[cell.cell_id] = cell_type
            result.confidences[cell.cell_id] = confidence
        
        return result
    
    def _extract_marker_intensities(
        self,
        segmentation_result: SegmentationResult,
        image: np.ndarray,
        params: ClassificationParams
    ) -> dict:
        """Extract mean marker intensities for each cell."""
        marker_intensities = {}
        
        labels = segmentation_result.label_mask
        
        for cell in segmentation_result.cells:
            cell_mask = labels == cell.cell_id
            cell_intensities = {}
            
            for marker_name, channel_idx in params.marker_channels.items():
                if image.ndim == 2:
                    # Single channel
                    if channel_idx == 0:
                        marker_image = image
                    else:
                        continue
                else:
                    # Multi-channel
                    if channel_idx >= image.shape[-1]:
                        continue
                    marker_image = image[:, :, channel_idx]
                
                # Calculate mean intensity
                mean_intensity = np.mean(marker_image[cell_mask])
                
                # Normalize to 0-1
                if marker_image.max() > 1.0:
                    mean_intensity = mean_intensity / marker_image.max()
                
                cell_intensities[marker_name] = mean_intensity
            
            marker_intensities[cell.cell_id] = cell_intensities
        
        return marker_intensities
    
    def _classify_by_markers(
        self,
        marker_values: dict,
        params: ClassificationParams
    ) -> tuple:
        """
        Classify cell based on marker expression.
        
        Classification logic:
        - CD3+ → T cell
        - CD3+ CD8+ → Cytotoxic T cell
        - CD3+ CD8- → Helper T cell
        - CD68+ → Macrophage
        - No markers → Stromal or Tumor (default)
        """
        from ....core.tme_models.cell_model import CellType
        
        # Get thresholds
        thresholds = params.marker_thresholds or {}
        
        # Check each marker
        cd3_pos = marker_values.get('CD3', 0) > thresholds.get('CD3', 0.3)
        cd8_pos = marker_values.get('CD8', 0) > thresholds.get('CD8', 0.25)
        cd68_pos = marker_values.get('CD68', 0) > thresholds.get('CD68', 0.4)
        cd20_pos = marker_values.get('CD20', 0) > thresholds.get('CD20', 0.3)
        
        # Classification logic
        if cd3_pos and cd8_pos:
            # Cytotoxic T cell
            return CellType.T_CELL, 0.9
        
        elif cd3_pos:
            # T cell (general)
            return CellType.T_CELL, 0.8
        
        elif cd20_pos:
            # B cell
            return CellType.B_CELL, 0.85
        
        elif cd68_pos:
            # Macrophage
            return CellType.MACROPHAGE, 0.85
        
        else:
            # No immune markers → Likely tumor or stromal
            # Could use additional criteria (size, morphology)
            return CellType.TUMOR, 0.5


# ============================================================
# COMPLETE QUANTIFICATION METHODS
# ============================================================

### Location: `tme_quant/cell_analysis/methods/quantification/morphological_features.py`

class MorphologicalFeatureCalculator:
    """
    Calculate morphological features (CellProfiler-like).
    
    Includes:
    - Area, perimeter, circularity
    - Shape descriptors (eccentricity, solidity, extent)
    - Moments and orientations
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate(
        self,
        cell,
        pixel_size: float = 1.0
    ) -> dict:
        """Calculate all morphological features for a cell."""
        features = {}
        
        # Basic measurements (already computed)
        features['area'] = cell.area
        features['perimeter'] = cell.perimeter
        features['circularity'] = cell.circularity
        features['eccentricity'] = cell.eccentricity
        features['solidity'] = cell.solidity
        features['extent'] = cell.extent
        
        # Axis lengths
        features['major_axis_length'] = cell.major_axis_length
        features['minor_axis_length'] = cell.minor_axis_length
        
        # Orientation
        features['orientation'] = cell.orientation
        
        # Derived features
        if cell.major_axis_length > 0:
            features['aspect_ratio'] = cell.minor_axis_length / cell.major_axis_length
        else:
            features['aspect_ratio'] = 0.0
        
        # Equivalent diameter (diameter of circle with same area)
        features['equivalent_diameter'] = 2 * np.sqrt(cell.area / np.pi)
        
        # Compactness (perimeter^2 / area)
        if cell.area > 0:
            features['compactness'] = (cell.perimeter ** 2) / (4 * np.pi * cell.area)
        else:
            features['compactness'] = 0.0
        
        # Form factor
        if cell.perimeter > 0:
            features['form_factor'] = (4 * np.pi * cell.area) / (cell.perimeter ** 2)
        else:
            features['form_factor'] = 0.0
        
        return features


### Location: `tme_quant/cell_analysis/methods/quantification/intensity_features.py`

class IntensityFeatureCalculator:
    """
    Calculate intensity features per channel.
    
    Includes:
    - Mean, median, std, min, max
    - Integrated intensity
    - Mass displacement
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate(
        self,
        cell,
        image: np.ndarray,
        label_mask: np.ndarray
    ) -> dict:
        """Calculate intensity features for a cell."""
        features = {}
        
        # Get cell mask
        cell_mask = label_mask == cell.cell_id
        
        # Handle multi-channel
        if image.ndim == 2:
            channels = {'intensity': image}
        else:
            channels = {f'ch_{i}': image[:, :, i] for i in range(image.shape[-1])}
        
        for ch_name, ch_image in channels.items():
            # Get pixels in cell
            pixels = ch_image[cell_mask]
            
            if len(pixels) == 0:
                continue
            
            # Basic statistics
            features[f'{ch_name}_mean'] = float(np.mean(pixels))
            features[f'{ch_name}_median'] = float(np.median(pixels))
            features[f'{ch_name}_std'] = float(np.std(pixels))
            features[f'{ch_name}_min'] = float(np.min(pixels))
            features[f'{ch_name}_max'] = float(np.max(pixels))
            
            # Integrated intensity (sum)
            features[f'{ch_name}_integrated'] = float(np.sum(pixels))
            
            # Mean absolute deviation
            features[f'{ch_name}_mad'] = float(np.mean(np.abs(pixels - np.mean(pixels))))
            
            # Quartiles
            features[f'{ch_name}_q25'] = float(np.percentile(pixels, 25))
            features[f'{ch_name}_q75'] = float(np.percentile(pixels, 75))
            
            # Intensity range
            features[f'{ch_name}_range'] = float(np.ptp(pixels))
        
        return features


### Location: `tme_quant/cell_analysis/methods/quantification/texture_features.py`

class TextureFeatureCalculator:
    """
    Calculate texture features using Haralick/GLCM.
    
    Includes:
    - Contrast, correlation, energy, homogeneity
    - At multiple scales
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate(
        self,
        cell,
        image: np.ndarray,
        label_mask: np.ndarray,
        scales: list = [1, 3, 5]
    ) -> dict:
        """Calculate texture features for a cell."""
        from skimage.feature import graycomatrix, graycoprops
        
        features = {}
        
        # Get cell region
        cell_mask = label_mask == cell.cell_id
        
        # Get bounding box
        bbox = cell.get_bounding_box()
        y_min, y_max = int(bbox.min_y), int(bbox.max_y)
        x_min, x_max = int(bbox.min_x), int(bbox.max_x)
        
        # Extract region
        if image.ndim == 2:
            cell_region = image[y_min:y_max, x_min:x_max]
            cell_mask_region = cell_mask[y_min:y_max, x_min:x_max]
        else:
            # Use first channel
            cell_region = image[y_min:y_max, x_min:x_max, 0]
            cell_mask_region = cell_mask[y_min:y_max, x_min:x_max]
        
        # Mask out non-cell pixels
        cell_region_masked = cell_region.copy()
        cell_region_masked[~cell_mask_region] = 0
        
        # Convert to uint8
        cell_region_uint8 = (
            (cell_region_masked - cell_region_masked.min()) /
            (cell_region_masked.max() - cell_region_masked.min() + 1e-10) * 255
        ).astype(np.uint8)
        
        # Calculate GLCM at different scales
        for scale in scales:
            try:
                # Compute GLCM
                glcm = graycomatrix(
                    cell_region_uint8,
                    distances=[scale],
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                
                # Compute properties
                features[f'contrast_scale_{scale}'] = float(
                    graycoprops(glcm, 'contrast').mean()
                )
                features[f'correlation_scale_{scale}'] = float(
                    graycoprops(glcm, 'correlation').mean()
                )
                features[f'energy_scale_{scale}'] = float(
                    graycoprops(glcm, 'energy').mean()
                )
                features[f'homogeneity_scale_{scale}'] = float(
                    graycoprops(glcm, 'homogeneity').mean()
                )
                features[f'asm_scale_{scale}'] = float(
                    graycoprops(glcm, 'ASM').mean()
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to compute texture at scale {scale}: {e}")
        
        return features


### Location: `tme_quant/cell_analysis/methods/quantification/spatial_features.py`

class SpatialFeatureCalculator:
    """
    Calculate spatial distribution features.
    
    Includes:
    - Nearest neighbor distances
    - Cell density
    - Clustering metrics
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate_neighbors(
        self,
        cells: list,
        max_distance: float = 50.0
    ) -> dict:
        """Calculate neighbor features for all cells."""
        from scipy.spatial import cKDTree
        
        if len(cells) < 2:
            return {}
        
        # Build KD-tree
        centroids = np.array([c.centroid for c in cells])
        tree = cKDTree(centroids)
        
        # Query neighbors
        distances, indices = tree.query(centroids, k=min(6, len(cells)))
        
        features = {}
        for i, cell in enumerate(cells):
            cell_features = {}
            
            # Nearest neighbor distance (exclude self)
            nn_dist = distances[i, 1] if len(distances[i]) > 1 else 0.0
            cell_features['nearest_neighbor_distance'] = nn_dist
            
            # Number of neighbors within max_distance
            neighbors_within = np.sum(distances[i, 1:] < max_distance)
            cell_features['neighbors_count'] = neighbors_within
            
            # Mean distance to k nearest neighbors
            k_neighbors = min(5, len(distances[i]) - 1)
            if k_neighbors > 0:
                cell_features['mean_k_neighbor_distance'] = float(
                    np.mean(distances[i, 1:k_neighbors+1])
                )
            
            features[cell.cell_id] = cell_features
        
        return features
    
    def calculate_density(
        self,
        cells: list,
        region_area: float
    ) -> dict:
        """Calculate cell density metrics."""
        features = {}
        
        if region_area > 0:
            # Cells per square mm
            features['cell_density'] = len(cells) / (region_area / 1e6)
        
        # Local density (mean reciprocal of NN distance)
        nn_distances = []
        for cell in cells:
            if hasattr(cell, 'nearest_neighbor_distance') and cell.nearest_neighbor_distance:
                nn_distances.append(cell.nearest_neighbor_distance)
        
        if nn_distances:
            features['local_density'] = float(1.0 / np.mean(nn_distances))
        
        return features


### Location: `tme_quant/cell_analysis/methods/quantification/relationship_features.py`

class RelationshipFeatureCalculator:
    """
    Calculate cell-cell and cell-fiber relationship features.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate_cell_cell_relationships(
        self,
        cells: list,
        max_distance: float = 50.0
    ) -> dict:
        """Calculate cell-cell relationship features."""
        from scipy.spatial import cKDTree
        from ....core.tme_models.cell_model import CellType
        
        features = {}
        
        if len(cells) < 2:
            return features
        
        # Get cell types
        cell_types = [c.cell_type for c in cells]
        
        # Count by type
        type_counts = {}
        for ct in cell_types:
            if ct:
                type_counts[ct.value] = type_counts.get(ct.value, 0) + 1
        
        features['type_distribution'] = type_counts
        
        # Build KD-tree
        centroids = np.array([c.centroid for c in cells])
        tree = cKDTree(centroids)
        
        # For each cell type, find neighbors of other types
        for i, cell in enumerate(cells):
            if not cell.cell_type:
                continue
            
            # Find neighbors within max_distance
            neighbor_indices = tree.query_ball_point(centroids[i], max_distance)
            neighbor_indices = [j for j in neighbor_indices if j != i]
            
            # Count neighbors by type
            neighbor_types = {}
            for j in neighbor_indices:
                neighbor_type = cells[j].cell_type
                if neighbor_type:
                    type_val = neighbor_type.value
                    neighbor_types[type_val] = neighbor_types.get(type_val, 0) + 1
            
            cell_features = {
                f'neighbors_{ntype}': count
                for ntype, count in neighbor_types.items()
            }
            
            features[cell.cell_id] = cell_features
        
        return features
    
    def calculate_cell_fiber_relationships(
        self,
        cells: list,
        fibers: list,
        max_distance: float = 30.0
    ) -> dict:
        """Calculate cell-fiber relationship features."""
        from scipy.spatial import cKDTree
        
        features = {}
        
        if not cells or not fibers:
            return features
        
        # Build KD-tree for fibers
        fiber_points = []
        for fiber in fibers:
            # Use fiber midpoint
            mid_idx = len(fiber.centerline) // 2
            fiber_points.append(fiber.centerline[mid_idx])
        
        fiber_tree = cKDTree(np.array(fiber_points))
        
        # For each cell, find nearby fibers
        for cell in cells:
            # Query fibers within max_distance
            fiber_indices = fiber_tree.query_ball_point(cell.centroid, max_distance)
            
            cell_features = {
                'nearby_fibers_count': len(fiber_indices),
            }
            
            if fiber_indices:
                # Compute distances
                distances = []
                for j in fiber_indices:
                    dist = np.linalg.norm(
                        np.array(cell.centroid) - np.array(fiber_points[j])
                    )
                    distances.append(dist)
                
                cell_features['min_fiber_distance'] = float(np.min(distances))
                cell_features['mean_fiber_distance'] = float(np.mean(distances))
            
            features[cell.cell_id] = cell_features
        
        return features