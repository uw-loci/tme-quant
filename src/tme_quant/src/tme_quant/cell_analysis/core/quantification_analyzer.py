"""
Cell quantification analyzer.

Implements CellProfiler-like quantification of morphological, intensity,
texture, and spatial features.
"""

import numpy as np
from typing import Optional, Dict, List
import time
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from ..config.quantification_params import QuantificationParams, QuantificationResult
from ...core.tme_models.cell_model import SegmentationResult


class CellQuantificationAnalyzer:
    """
    Quantifies cell features: morphology, intensity, texture, spatial relationships.
    
    Implements measurements similar to CellProfiler.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize quantification analyzer."""
        self.verbose = verbose
    
    def quantify(
        self,
        segmentation_result: SegmentationResult,
        params: QuantificationParams,
        image: Optional[np.ndarray] = None
    ) -> QuantificationResult:
        """
        Quantify cell features.
        
        Args:
            segmentation_result: Segmentation result
            params: Quantification parameters
            image: Original image for intensity measurements
            
        Returns:
            QuantificationResult with measurements
        """
        start_time = time.time()
        
        result = QuantificationResult()
        cells = segmentation_result.cells
        
        if not cells:
            return result
        
        # Per-cell measurements
        for cell in cells:
            measurements = {}
            
            # Morphological measurements (already computed in segmentation)
            if params.measure_area:
                measurements['area'] = cell.area
            if params.measure_perimeter:
                measurements['perimeter'] = cell.perimeter
            if params.measure_circularity:
                measurements['circularity'] = cell.circularity
            if params.measure_eccentricity:
                measurements['eccentricity'] = cell.eccentricity
            if params.measure_solidity:
                measurements['solidity'] = cell.solidity
            if params.measure_extent:
                measurements['extent'] = cell.extent
            if params.measure_orientation:
                measurements['orientation'] = cell.orientation
            
            # Intensity measurements (if image provided)
            if image is not None:
                intensity_features = self._compute_intensity_features(
                    cell, image, params
                )
                measurements.update(intensity_features)
            
            # Texture measurements
            if params.measure_texture and image is not None:
                texture_features = self._compute_texture_features(
                    cell, image, params.texture_scales
                )
                measurements.update(texture_features)
            
            result.measurements[cell.cell_id] = measurements
        
        # Population statistics
        result.population_stats = self._compute_population_statistics(
            cells, params
        )
        
        # Spatial statistics
        if params.measure_distances or params.measure_density:
            result.spatial_stats = self._compute_spatial_statistics(
                cells, params
            )
        
        result.processing_time = time.time() - start_time
        result.parameters = params.to_dict()
        
        return result
    
    def _compute_intensity_features(
        self,
        cell,
        image: np.ndarray,
        params: QuantificationParams
    ) -> Dict[str, float]:
        """Compute intensity features for a cell."""
        features = {}
        
        # Get cell region
        if cell.mask is not None:
            cell_pixels = image[cell.mask > 0]
        else:
            # Approximate with bounding box
            bbox = cell.get_bounding_box()
            cell_region = image[
                int(bbox.min_y):int(bbox.max_y),
                int(bbox.min_x):int(bbox.max_x)
            ]
            cell_pixels = cell_region.flatten()
        
        if len(cell_pixels) == 0:
            return features
        
        # Multi-channel support
        if image.ndim == 2:
            channels = {'intensity': cell_pixels}
        else:
            channels = {
                f'ch_{i}': cell_pixels[:, i]
                for i in range(image.shape[-1])
            }
        
        for ch_name, pixels in channels.items():
            if params.measure_mean_intensity:
                features[f'mean_{ch_name}'] = float(np.mean(pixels))
            
            if params.measure_integrated_intensity:
                features[f'integrated_{ch_name}'] = float(np.sum(pixels))
            
            if params.measure_std_intensity:
                features[f'std_{ch_name}'] = float(np.std(pixels))
            
            if params.measure_min_max_intensity:
                features[f'min_{ch_name}'] = float(np.min(pixels))
                features[f'max_{ch_name}'] = float(np.max(pixels))
        
        return features
    
    def _compute_texture_features(
        self,
        cell,
        image: np.ndarray,
        scales: List[int]
    ) -> Dict[str, float]:
        """Compute texture features (Haralick-like)."""
        from skimage.feature import graycomatrix, graycoprops
        
        features = {}
        
        # Get cell region
        bbox = cell.get_bounding_box()
        cell_region = image[
            int(bbox.min_y):int(bbox.max_y),
            int(bbox.min_x):int(bbox.max_x)
        ]
        
        # Convert to uint8
        cell_region_scaled = (
            (cell_region - cell_region.min()) /
            (cell_region.max() - cell_region.min()) * 255
        ).astype(np.uint8)
        
        for scale in scales:
            # Compute GLCM
            glcm = graycomatrix(
                cell_region_scaled,
                distances=[scale],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Compute texture properties
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
        
        return features
    
    def _compute_population_statistics(
        self,
        cells: List,
        params: QuantificationParams
    ) -> Dict[str, float]:
        """Compute population-level statistics."""
        stats = {}
        
        if params.measure_area:
            areas = [c.area for c in cells]
            stats['mean_area'] = float(np.mean(areas))
            stats['std_area'] = float(np.std(areas))
            stats['median_area'] = float(np.median(areas))
        
        if params.measure_circularity:
            circularities = [c.circularity for c in cells]
            stats['mean_circularity'] = float(np.mean(circularities))
        
        if params.measure_eccentricity:
            eccentricities = [c.eccentricity for c in cells]
            stats['mean_eccentricity'] = float(np.mean(eccentricities))
        
        return stats
    
    def _compute_spatial_statistics(
        self,
        cells: List,
        params: QuantificationParams
    ) -> Dict[str, float]:
        """Compute spatial statistics."""
        stats = {}
        
        if len(cells) < 2:
            return stats
        
        # Get centroids
        centroids = np.array([c.centroid for c in cells])
        
        # Build KD-tree for nearest neighbor search
        tree = cKDTree(centroids)
        
        # Compute nearest neighbor distances
        distances, indices = tree.query(centroids, k=2)  # k=2 for self + nearest
        nn_distances = distances[:, 1]  # Exclude self (distance 0)
        
        stats['mean_nearest_neighbor_distance'] = float(np.mean(nn_distances))
        stats['std_nearest_neighbor_distance'] = float(np.std(nn_distances))
        stats['median_nearest_neighbor_distance'] = float(np.median(nn_distances))
        
        # Cell density (requires region area - placeholder)
        # stats['cell_density'] = len(cells) / region_area
        
        # Clustering index (simplified)
        expected_distance = 1 / (2 * np.sqrt(len(cells) / 10000))  # Approximate
        if expected_distance > 0:
            stats['clustering_index'] = float(
                np.mean(nn_distances) / expected_distance
            )
        
        return stats