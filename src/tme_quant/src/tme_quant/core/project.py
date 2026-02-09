from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

from .tme_models.fiber_model import (
    FiberObject, RegionOrientationMap, FiberPopulation,
    OrientationResult, ExtractionResult
)
from .hierarchy import ObjectHierarchy
from ..fiber_analysis import FiberAnalyzer
from ..fiber_analysis.config.orientation_params import OrientationParams
from ..fiber_analysis.config.extraction_params import ExtractionParams


class TMEProject:
    """
    TME analysis project managing hierarchical objects and analysis results.
    """
    
    def __init__(self, name: str, base_path: Optional[Path] = None):
        self.name = name
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Hierarchical object storage
        self.hierarchy = ObjectHierarchy()
        
        # Image entries
        self.images: Dict[str, ImageEntry] = {}
        
        # Region-level orientation maps
        self.orientation_maps: Dict[str, RegionOrientationMap] = {}
        
        # Fiber populations by region
        self.fiber_populations: Dict[str, FiberPopulation] = {}
        
        # Analyzers
        self.fiber_analyzer = FiberAnalyzer()
    
    # ============================================================
    # FIBER ANALYSIS METHODS
    # ============================================================
    
    def analyze_region_orientation(
        self,
        image_id: str,
        region_id: str,
        params: OrientationParams,
        region_type: str = "tumor_boundary"
    ) -> RegionOrientationMap:
        """
        Analyze fiber orientation within a specific region.
        
        Args:
            image_id: ID of the image
            region_id: ID of the region (e.g., tumor boundary ROI)
            params: Orientation analysis parameters
            region_type: Type of region ("tumor_boundary", "tumor_core", "stroma")
            
        Returns:
            RegionOrientationMap with orientation results
            
        Example:
            >>> # Analyze fiber orientation at tumor boundary
            >>> params = OrientationParams(
            ...     mode=OrientationMode.CURVEALIGN,
            ...     window_size=128
            ... )
            >>> orientation_map = project.analyze_region_orientation(
            ...     image_id="image_001",
            ...     region_id="tumor_1_boundary",
            ...     params=params,
            ...     region_type="tumor_boundary"
            ... )
        """
        # Get the region ROI
        region = self.hierarchy.get_object(region_id)
        if region is None:
            raise ValueError(f"Region {region_id} not found")
        
        # Get the image
        image_entry = self.images[image_id]
        
        # Extract region from image using ROI mask
        region_image = self._extract_region_image(
            image_entry, region.roi
        )
        
        # Run orientation analysis
        orientation_result = self.fiber_analyzer.analyze_orientation_2d(
            region_image,
            params,
            image_id=f"{image_id}_{region_id}"
        )
        
        # Create RegionOrientationMap object
        orientation_map = RegionOrientationMap(
            object_id=f"{region_id}_orientation",
            parent_id=region_id,
            orientation_result=orientation_result,
            region_type=region_type,
            roi=region.roi
        )
        
        # Add to hierarchy and storage
        self.hierarchy.add_object(orientation_map, parent=region)
        self.orientation_maps[orientation_map.object_id] = orientation_map
        
        return orientation_map
    
    def extract_region_fibers(
        self,
        image_id: str,
        region_id: str,
        params: ExtractionParams,
        region_type: str = "tumor_boundary"
    ) -> List[FiberObject]:
        """
        Extract individual fibers within a specific region.
        
        Args:
            image_id: ID of the image
            region_id: ID of the region
            params: Extraction parameters
            region_type: Type of region
            
        Returns:
            List of FiberObject instances
            
        Example:
            >>> # Extract fibers at tumor boundary
            >>> params = ExtractionParams(
            ...     mode=ExtractionMode.CTFIRE,
            ...     min_fiber_length=10.0
            ... )
            >>> fibers = project.extract_region_fibers(
            ...     image_id="image_001",
            ...     region_id="tumor_1_boundary",
            ...     params=params,
            ...     region_type="tumor_boundary"
            ... )
            >>> print(f"Extracted {len(fibers)} fibers")
        """
        # Get the region
        region = self.hierarchy.get_object(region_id)
        if region is None:
            raise ValueError(f"Region {region_id} not found")
        
        # Get the image
        image_entry = self.images[image_id]
        
        # Extract region from image
        region_image = self._extract_region_image(
            image_entry, region.roi
        )
        
        # Run fiber extraction
        extraction_result = self.fiber_analyzer.extract_fibers_2d(
            region_image,
            params,
            image_id=f"{image_id}_{region_id}"
        )
        
        # Convert to FiberObject instances and add to hierarchy
        fiber_objects = []
        for fiber_props in extraction_result.fibers:
            # Transform coordinates from region to image space
            fiber_coords_global = self._transform_coords_to_image_space(
                fiber_props.centerline,
                region.roi
            )
            
            # Create FiberObject
            fiber_obj = FiberObject(
                object_id=f"{region_id}_fiber_{fiber_props.fiber_id}",
                parent_id=region_id,
                centerline=fiber_coords_global,
                length=fiber_props.length,
                width=fiber_props.width,
                angle=fiber_props.angle,
                mean_orientation=fiber_props.angle,
                straightness=fiber_props.straightness,
                curvature=fiber_props.curvature,
                extraction_mode=params.mode.value,
                roi=region.roi,
                metadata={
                    'region_type': region_type,
                    'extraction_params': params.__dict__
                }
            )
            
            # Add to hierarchy
            self.hierarchy.add_object(fiber_obj, parent=region)
            fiber_objects.append(fiber_obj)
        
        # Create fiber population summary
        self._create_fiber_population(
            region_id, region_type, fiber_objects
        )
        
        return fiber_objects
    
    def analyze_fiber_alignment_to_tumor_boundary(
        self,
        tumor_region_id: str,
        boundary_width: float = 50.0
    ) -> Dict[str, Any]:
        """
        Analyze fiber alignment relative to tumor boundary.
        
        This is a key analysis for TACS classification.
        
        Args:
            tumor_region_id: ID of the tumor region
            boundary_width: Width of boundary zone in microns
            
        Returns:
            Dictionary with alignment analysis results
            
        Example:
            >>> # Analyze fiber-tumor boundary alignment
            >>> alignment = project.analyze_fiber_alignment_to_tumor_boundary(
            ...     tumor_region_id="tumor_1",
            ...     boundary_width=50.0
            ... )
            >>> print(f"TACS type: {alignment['tacs_type']}")
            >>> print(f"Alignment score: {alignment['alignment_score']:.3f}")
        """
        # Get tumor region
        tumor = self.hierarchy.get_object(tumor_region_id)
        if tumor is None:
            raise ValueError(f"Tumor region {tumor_region_id} not found")
        
        # Get or create tumor boundary ROI
        boundary_roi = self._get_or_create_boundary_roi(
            tumor, boundary_width
        )
        
        # Get all fibers in or near the tumor
        fibers = self.get_fibers_in_region(
            tumor_region_id,
            include_children=True
        )
        
        # Classify fibers by location
        boundary_fibers = []
        core_fibers = []
        
        for fiber in fibers:
            # Compute alignment to boundary
            alignment = fiber.compute_alignment_to_boundary(boundary_roi)
            
            # Classify fiber location
            if alignment['distance'] <= boundary_width:
                fiber.in_tumor_boundary = True
                fiber.in_tumor_core = False
                boundary_fibers.append(fiber)
            else:
                fiber.in_tumor_boundary = False
                fiber.in_tumor_core = True
                core_fibers.append(fiber)
        
        # Compute boundary fiber statistics
        if boundary_fibers:
            boundary_angles = [f.angle_to_tumor_boundary for f in boundary_fibers]
            mean_alignment_angle = np.mean(boundary_angles)
            
            # Parallel fibers (angle close to 0°)
            parallel_count = sum(1 for a in boundary_angles if abs(a) < 30)
            
            # Perpendicular fibers (angle close to 90°)
            perp_count = sum(1 for a in boundary_angles if abs(a) > 60)
            
            parallel_ratio = parallel_count / len(boundary_fibers)
            perp_ratio = perp_count / len(boundary_fibers)
            
            # TACS classification
            tacs_type, tacs_score = self._classify_tacs(
                boundary_fibers, mean_alignment_angle, perp_ratio
            )
        else:
            mean_alignment_angle = None
            parallel_ratio = 0
            perp_ratio = 0
            tacs_type = None
            tacs_score = 0
        
        results = {
            'tumor_region_id': tumor_region_id,
            'boundary_width': boundary_width,
            'total_fibers': len(fibers),
            'boundary_fibers': len(boundary_fibers),
            'core_fibers': len(core_fibers),
            'mean_alignment_angle': mean_alignment_angle,
            'parallel_ratio': parallel_ratio,
            'perpendicular_ratio': perp_ratio,
            'tacs_type': tacs_type,
            'tacs_score': tacs_score,
            'fibers': {
                'boundary': boundary_fibers,
                'core': core_fibers
            }
        }
        
        # Store results
        tumor.metadata['fiber_alignment'] = results
        
        return results
    
    def compare_orientation_across_regions(
        self,
        region_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare fiber orientation across multiple regions.
        
        Example:
            >>> # Compare orientation in boundary vs core vs stroma
            >>> comparison = project.compare_orientation_across_regions([
            ...     "tumor_1_boundary",
            ...     "tumor_1_core",
            ...     "stroma_region_1"
            ... ])
        """
        results = {}
        
        for region_id in region_ids:
            # Get orientation map for region
            orientation_map = self.orientation_maps.get(
                f"{region_id}_orientation"
            )
            
            if orientation_map:
                results[region_id] = {
                    'mean_orientation': orientation_map.get_dominant_orientation(),
                    'alignment_score': orientation_map.get_alignment_score(),
                    'region_type': orientation_map.region_type
                }
        
        # Compute differences
        if len(results) > 1:
            orientations = [r['mean_orientation'] for r in results.values()]
            results['orientation_variance'] = np.var(orientations)
            results['max_orientation_difference'] = np.ptp(orientations)
        
        return results
    
    # ============================================================
    # QUERY METHODS
    # ============================================================
    
    def get_fibers_in_region(
        self,
        region_id: str,
        include_children: bool = True
    ) -> List[FiberObject]:
        """Get all fibers in a specific region."""
        fibers = []
        
        if include_children:
            # Get all descendant objects of type FIBER
            descendants = self.hierarchy.get_descendants(region_id)
            fibers = [
                obj for obj in descendants
                if isinstance(obj, FiberObject)
            ]
        else:
            # Get only direct children
            children = self.hierarchy.get_children(region_id)
            fibers = [
                obj for obj in children
                if isinstance(obj, FiberObject)
            ]
        
        return fibers
    
    def get_fibers_by_criteria(
        self,
        min_length: Optional[float] = None,
        max_length: Optional[float] = None,
        min_straightness: Optional[float] = None,
        region_type: Optional[str] = None,
        in_tumor_boundary: Optional[bool] = None
    ) -> List[FiberObject]:
        """
        Query fibers by specific criteria.
        
        Example:
            >>> # Get long, straight fibers in tumor boundary
            >>> fibers = project.get_fibers_by_criteria(
            ...     min_length=20.0,
            ...     min_straightness=0.8,
            ...     in_tumor_boundary=True
            ... )
        """
        all_fibers = self.hierarchy.get_objects_by_type(ObjectType.FIBER)
        filtered = []
        
        for fiber in all_fibers:
            # Apply filters
            if min_length and fiber.length < min_length:
                continue
            if max_length and fiber.length > max_length:
                continue
            if min_straightness and fiber.straightness < min_straightness:
                continue
            if region_type and fiber.metadata.get('region_type') != region_type:
                continue
            if in_tumor_boundary is not None and fiber.in_tumor_boundary != in_tumor_boundary:
                continue
            
            filtered.append(fiber)
        
        return filtered
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _extract_region_image(
        self,
        image_entry: 'ImageEntry',
        roi: 'ROI'
    ) -> np.ndarray:
        """Extract image region using ROI mask."""
        # Get image data
        image = image_entry.get_channel_data('collagen')  # or appropriate channel
        
        # Get ROI mask and bounding box
        mask = roi.get_mask(image.shape)
        bbox = roi.get_bounding_box()
        
        # Crop to bounding box
        region_image = image[
            bbox.min_y:bbox.max_y,
            bbox.min_x:bbox.max_x
        ]
        
        # Apply mask
        region_mask = mask[
            bbox.min_y:bbox.max_y,
            bbox.min_x:bbox.max_x
        ]
        region_image = region_image * region_mask
        
        return region_image
    
    def _transform_coords_to_image_space(
        self,
        coords: np.ndarray,
        roi: 'ROI'
    ) -> np.ndarray:
        """Transform coordinates from ROI space to image space."""
        bbox = roi.get_bounding_box()
        offset = np.array([bbox.min_x, bbox.min_y])
        return coords + offset
    
    def _get_or_create_boundary_roi(
        self,
        tumor: TMEObject,
        boundary_width: float
    ) -> 'ROI':
        """Get or create tumor boundary ROI."""
        # Check if boundary ROI already exists
        boundary_id = f"{tumor.object_id}_boundary"
        boundary = self.hierarchy.get_object(boundary_id)
        
        if boundary is None:
            # Create boundary ROI by dilating tumor ROI
            from ..utils.geometry_utils import create_boundary_roi
            boundary_roi = create_boundary_roi(
                tumor.roi,
                width=boundary_width
            )
            
            # Create boundary object
            boundary = TumorRegion(
                object_id=boundary_id,
                parent_id=tumor.object_id,
                roi=boundary_roi,
                region_type="tumor_boundary"
            )
            self.hierarchy.add_object(boundary, parent=tumor)
        
        return boundary.roi
    
    def _create_fiber_population(
        self,
        region_id: str,
        region_type: str,
        fibers: List[FiberObject]
    ):
        """Create FiberPopulation summary for a region."""
        if not fibers:
            return
        
        # Compute statistics
        fiber_ids = [f.object_id for f in fibers]
        lengths = [f.length for f in fibers]
        widths = [f.width for f in fibers]
        straightnesses = [f.straightness for f in fibers]
        orientations = [f.angle for f in fibers]
        
        # Compute alignment score (order parameter)
        angles_rad = np.deg2rad(orientations)
        mean_cos = np.mean(np.cos(2 * angles_rad))
        mean_sin = np.mean(np.sin(2 * angles_rad))
        alignment_score = np.sqrt(mean_cos**2 + mean_sin**2)
        
        # Create population object
        population = FiberPopulation(
            fiber_ids=fiber_ids,
            region_id=region_id,
            region_type=region_type,
            count=len(fibers),
            mean_length=np.mean(lengths),
            mean_width=np.mean(widths),
            mean_straightness=np.mean(straightnesses),
            mean_orientation=np.rad2deg(np.arctan2(mean_sin, mean_cos) / 2),
            alignment_score=alignment_score,
            length_distribution=np.histogram(lengths, bins=20)[0],
            orientation_distribution=np.histogram(orientations, bins=36, range=(-90, 90))[0]
        )
        
        self.fiber_populations[f"{region_id}_population"] = population
    
    def _classify_tacs(
        self,
        fibers: List[FiberObject],
        mean_alignment_angle: float,
        perp_ratio: float
    ) -> tuple:
        """
        Classify Tumor-Associated Collagen Signatures (TACS).
        
        TACS-1: Loose, curly collagen (high curvature, low alignment)
        TACS-2: Straightened, aligned parallel to boundary (high alignment, low angle)
        TACS-3: Perpendicular alignment (high perpendicular ratio)
        """
        # Compute metrics
        mean_straightness = np.mean([f.straightness for f in fibers])
        alignment_scores = [
            1 - abs(f.angle_to_tumor_boundary) / 90
            for f in fibers
            if f.angle_to_tumor_boundary is not None
        ]
        mean_parallel_alignment = np.mean(alignment_scores) if alignment_scores else 0
        
        # TACS classification logic
        if perp_ratio > 0.4:  # More than 40% perpendicular
            tacs_type = "TACS-3"
            tacs_score = perp_ratio
        elif mean_straightness > 0.7 and mean_parallel_alignment > 0.6:
            tacs_type = "TACS-2"
            tacs_score = (mean_straightness + mean_parallel_alignment) / 2
        else:
            tacs_type = "TACS-1"
            tacs_score = 1 - mean_straightness
        
        return tacs_type, tacs_score