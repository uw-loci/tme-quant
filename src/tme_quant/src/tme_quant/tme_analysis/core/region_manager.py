"""
Manage regions of interest and automated tumor detection.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

from ..config.analysis_params import TumorDetectionParams, TumorDetectionMethod
from ...core.tme_models.cell_model import CellObject, CellType
from ...core.tme_models.tumor_model import TumorRegion
from ...core.geometry import ROI


class RegionManager:
    """
    Manage ROIs and automated tumor region detection.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize region manager."""
        self.verbose = verbose
    
    # ============================================================
    # TUMOR REGION DETECTION
    # ============================================================
    
    def detect_tumor_regions(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams
    ) -> List[TumorRegion]:
        """
        Automatically detect tumor regions from cell data.
        
        Args:
            cells: List of cells
            params: Detection parameters
            
        Returns:
            List of TumorRegion objects
        """
        if params.method == TumorDetectionMethod.CLUSTERING:
            return self._detect_by_clustering(cells, params)
        
        elif params.method == TumorDetectionMethod.DENSITY:
            return self._detect_by_density(cells, params)
        
        elif params.method == TumorDetectionMethod.CELL_TYPE:
            return self._detect_by_cell_type(cells, params)
        
        elif params.method == TumorDetectionMethod.DEEP_LEARNING:
            return self._detect_by_deep_learning(cells, params)
        
        elif params.method == TumorDetectionMethod.MANUAL:
            # User must provide annotations separately
            return []
        
        else:
            raise ValueError(f"Unknown detection method: {params.method}")
    
    def _detect_by_clustering(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams
    ) -> List[TumorRegion]:
        """Detect tumor regions using DBSCAN clustering."""
        if len(cells) < params.dbscan_min_samples:
            return []
        
        # Get cell centroids
        centroids = np.array([c.centroid for c in cells])
        
        # Run DBSCAN
        clustering = DBSCAN(
            eps=params.dbscan_eps,
            min_samples=params.dbscan_min_samples
        ).fit(centroids)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if self.verbose:
            print(f"DBSCAN found {n_clusters} clusters")
        
        # Create tumor regions from clusters
        tumor_regions = []
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get cells in this cluster
            cluster_mask = labels == cluster_id
            cluster_centroids = centroids[cluster_mask]
            
            # Create convex hull as boundary
            points = MultiPoint(cluster_centroids)
            boundary_polygon = points.convex_hull.buffer(50)  # 50 micron buffer
            
            # Check minimum area
            area = boundary_polygon.area
            if area < params.min_tumor_area:
                continue
            
            # Smooth boundary if requested
            if params.smooth_boundary:
                # Simplify polygon
                boundary_polygon = boundary_polygon.simplify(
                    tolerance=params.smoothing_sigma
                )
            
            # Create TumorRegion
            tumor_region = TumorRegion(
                geometry=boundary_polygon,
                metadata={
                    'detection_method': 'clustering',
                    'n_cells': int(np.sum(cluster_mask)),
                    'area': float(area),
                    'cluster_id': cluster_id
                }
            )
            
            tumor_regions.append(tumor_region)
        
        return tumor_regions
    
    def _detect_by_cell_type(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams
    ) -> List[TumorRegion]:
        """Detect tumor regions based on classified tumor cells."""
        # Filter for tumor cells
        tumor_cells = [
            c for c in cells
            if c.cell_type and c.cell_type.value in params.tumor_cell_types
        ]
        
        if not tumor_cells:
            if self.verbose:
                print("No tumor cells found for region detection")
            return []
        
        # Use clustering on tumor cells
        return self._detect_by_clustering(tumor_cells, params)
    
    def _detect_by_density(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams
    ) -> List[TumorRegion]:
        """Detect tumor regions using kernel density estimation."""
        from scipy.stats import gaussian_kde
        
        # Get centroids
        centroids = np.array([c.centroid for c in cells])
        
        # Compute KDE
        kde = gaussian_kde(centroids.T, bw_method=params.density_bandwidth)
        
        # Create grid for evaluation
        x_min, y_min = centroids.min(axis=0) - 100
        x_max, y_max = centroids.max(axis=0) + 100
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kde(positions), xx.shape)
        
        # Threshold to get tumor regions
        tumor_mask = density > params.density_threshold
        
        # Convert to contours/polygons
        # ... (implementation using skimage.measure.find_contours)
        
        # Create TumorRegion objects
        # ... (implementation)
        
        return []  # Placeholder
    
    def _detect_by_deep_learning(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams
    ) -> List[TumorRegion]:
        """Detect tumor regions using deep learning model."""
        # Load model
        if params.dl_model_path is None:
            raise ValueError("Deep learning model path required")
        
        # Run inference
        # ... (implementation depends on specific DL framework)
        
        return []  # Placeholder
    
    # ============================================================
    # ZONE GENERATION
    # ============================================================
    
    def generate_tumor_zones(
        self,
        tumor_regions: List[TumorRegion],
        invasive_margin_width: float = 50.0,
        stroma_width: float = 200.0
    ) -> Dict[str, List[ROI]]:
        """
        Generate zones around tumor regions.
        
        Zones:
            - Invasive margin: 0-50 microns from boundary
            - Peri-tumor stroma: 50-250 microns from boundary
            - Tumor core: Inside tumor
        
        Args:
            tumor_regions: List of tumor regions
            invasive_margin_width: Width of invasive margin zone
            stroma_width: Width of stroma zone
            
        Returns:
            Dictionary of zone_type -> List[ROI]
        """
        zones = {
            'tumor_core': [],
            'invasive_margin': [],
            'stroma': []
        }
        
        for tumor in tumor_regions:
            # Tumor core (original region)
            zones['tumor_core'].append(tumor.roi)
            
            # Invasive margin (buffer zone)
            boundary = tumor.roi.polygon.boundary
            invasive_zone = boundary.buffer(invasive_margin_width)
            zones['invasive_margin'].append(ROI.from_shapely(invasive_zone))
            
            # Stroma zone (further buffer)
            stroma_zone = boundary.buffer(invasive_margin_width + stroma_width)
            # Subtract invasive margin to get ring
            stroma_ring = stroma_zone.difference(invasive_zone)
            zones['stroma'].append(ROI.from_shapely(stroma_ring))
        
        return zones
    
    # ============================================================
    # ROI FILTERING
    # ============================================================
    
    def filter_cells_by_roi(
        self,
        cells: Optional[List[CellObject]],
        roi: ROI
    ) -> List[CellObject]:
        """Filter cells that fall within ROI."""
        if not cells:
            return []
        
        filtered = []
        roi_polygon = roi.polygon
        
        for cell in cells:
            cell_point = Point(cell.centroid)
            if roi_polygon.contains(cell_point):
                filtered.append(cell)
        
        return filtered
    
    def filter_fibers_by_roi(
        self,
        fibers: Optional[List],
        roi: ROI
    ) -> List:
        """Filter fibers that intersect ROI."""
        if not fibers:
            return []
        
        filtered = []
        roi_polygon = roi.polygon
        
        for fiber in fibers:
            fiber_geom = fiber.geometry
            if roi_polygon.intersects(fiber_geom):
                filtered.append(fiber)
        
        return filtered