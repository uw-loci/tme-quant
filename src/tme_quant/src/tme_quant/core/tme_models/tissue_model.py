# tissue_model.py
"""
Comprehensive tissue sample and region models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from ..base_models import TMEObject, Geometry, TMEMetadata, Measurement

class TissueZone(Enum):
    """Tissue microenvironment zones"""
    TUMOR_CORE = "tumor_core"
    TUMOR_INVASIVE_FRONT = "tumor_invasive_front"
    PERITUMORAL_STROMA = "peritumoral_stroma"
    DISTANT_STROMA = "distant_stroma"
    NORMAL_TISSUE = "normal_tissue"
    NECROTIC_REGION = "necrotic_region"
    HYPOXIC_REGION = "hypoxic_region"

@dataclass
class TissueRegion(TMEObject):
    """Annotated tissue region with zone classification"""
    geometry: Geometry
    zone_type: TissueZone = TissueZone.NORMAL_TISSUE
    tissue_type: str = ""  # epithelium, connective, muscle, nervous
    annotations: List[str] = field(default_factory=list)  # Text annotations
    
    # Spatial relationships
    adjacent_regions: List['TissueRegion'] = field(default_factory=list)
    distance_to_tumor: float = float('inf')
    
    def __post_init__(self):
        """Initialize tissue region"""
        super().__post_init__()
        self.type = TMEType.REGION

@dataclass
class TissueSample(TMEObject):
    """Comprehensive tissue sample representation with QuPath-like hierarchy"""
    metadata: TMEMetadata = field(default_factory=TMEMetadata)
    image_data: Optional[np.ndarray] = None
    mask_data: Optional[np.ndarray] = None
    
    # Hierarchical components
    annotations: List[TissueRegion] = field(default_factory=list)
    tumor: Optional['Tumor'] = None
    stroma: Optional['Stroma'] = None
    cells: List['Cell'] = field(default_factory=list)
    fibers: List['Fiber'] = field(default_factory=list)
    vessels: List['Vessel'] = field(default_factory=list)
    
    # Derived properties
    tissue_area: float = 0.0
    tumor_stroma_ratio: float = 0.0
    
    def __post_init__(self):
        """Initialize tissue sample with hierarchy"""
        super().__post_init__()
        self.type = TMEType.SAMPLE
        
        # Build hierarchy
        if self.tumor:
            self.add_child(self.tumor)
        
        if self.stroma:
            self.add_child(self.stroma)
        
        # Add annotations
        for annotation in self.annotations:
            self.add_child(annotation)
        
        # Add cells, fibers, vessels
        for cell in self.cells:
            self.add_child(cell)
        
        for fiber in self.fibers:
            self.add_child(fiber)
        
        for vessel in self.vessels:
            self.add_child(vessel)
    
    def calculate_tme_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive TME metrics"""
        metrics = {
            'sample_id': self.metadata.sample_id,
            'tissue_area': self.tissue_area,
            'tumor_stroma_ratio': self.tumor_stroma_ratio,
            'cell_density': self._calculate_cell_density(),
            'fiber_density': self._calculate_fiber_density(),
            'vessel_density': self._calculate_vessel_density(),
            'spatial_heterogeneity': self._calculate_spatial_heterogeneity()
        }
        
        # Add tumor-specific metrics if available
        if self.tumor:
            tumor_metrics = {
                'tumor_area': self.tumor.get_total_area(),
                'tumor_grade': self.tumor.dominant_grade.value,
                'necrosis_percentage': self._calculate_necrosis_percentage(),
                'invasion_front_length': self._calculate_invasion_front_length()
            }
            metrics.update(tumor_metrics)
        
        return metrics
    
    def _calculate_cell_density(self) -> float:
        """Calculate cell density per tissue area"""
        if self.tissue_area == 0:
            return 0.0
        return len(self.cells) / self.tissue_area
    
    def _calculate_fiber_density(self) -> float:
        """Calculate fiber density"""
        if self.tissue_area == 0 or not self.fibers:
            return 0.0
        
        total_fiber_length = sum(fiber.length for fiber in self.fibers)
        return total_fiber_length / self.tissue_area
    
    def _calculate_vessel_density(self) -> float:
        """Calculate vessel density"""
        if self.tissue_area == 0 or not self.vessels:
            return 0.0
        
        total_vessel_area = sum(vessel.lumen_area for vessel in self.vessels)
        return total_vessel_area / self.tissue_area
    
    def _calculate_spatial_heterogeneity(self) -> float:
        """Calculate spatial heterogeneity index"""
        # Calculate coefficient of variation for cell densities in quadrants
        if not self.cells:
            return 0.0
        
        # Divide tissue into quadrants
        quadrants = self._divide_into_quadrants()
        quadrant_densities = []
        
        for quadrant in quadrants:
            cells_in_quadrant = self._count_cells_in_region(quadrant)
            density = cells_in_quadrant / quadrant.geometry.area() if quadrant.geometry.area() > 0 else 0
            quadrant_densities.append(density)
        
        if np.mean(quadrant_densities) > 0:
            return np.std(quadrant_densities) / np.mean(quadrant_densities)
        
        return 0.0
    
    def _divide_into_quadrants(self) -> List[TissueRegion]:
        """Divide tissue sample into quadrants"""
        # Implementation for spatial partitioning
        quadrants = []
        bounds = self._get_tissue_bounds()
        
        if bounds is None:
            return quadrants
        
        x_min, y_min, x_max, y_max = bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        # Create 4 quadrants
        quadrants_coords = [
            ([x_min, y_min], [x_mid, y_mid]),  # Bottom-left
            ([x_mid, y_min], [x_max, y_mid]),  # Bottom-right
            ([x_min, y_mid], [x_mid, y_max]),  # Top-left
            ([x_mid, y_mid], [x_max, y_max])   # Top-right
        ]
        
        for i, (min_coord, max_coord) in enumerate(quadrants_coords):
            quadrants.append(TissueRegion(
                name=f"Quadrant_{i+1}",
                geometry=Geometry(
                    type=GeometryType.RECTANGLE,
                    coordinates=np.array([min_coord, max_coord])
                )
            ))
        
        return quadrants
    
    def _count_cells_in_region(self, region: TissueRegion) -> int:
        """Count cells within a region"""
        count = 0
        region_bounds = region.geometry.bounds
        
        for cell in self.cells:
            cell_centroid = cell.centroid
            if (region_bounds[0] <= cell_centroid[0] <= region_bounds[3] and
                region_bounds[1] <= cell_centroid[1] <= region_bounds[4]):
                count += 1
        
        return count
    
    def _get_tissue_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get tissue bounding box"""
        if self.mask_data is not None:
            # Use mask to determine bounds
            y_coords, x_coords = np.where(self.mask_data > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                return (x_coords.min(), y_coords.min(), 
                       x_coords.max(), y_coords.max())
        
        # Fallback to annotation bounds
        if self.annotations:
            all_bounds = [ann.geometry.bounds for ann in self.annotations]
            x_min = min(b[0] for b in all_bounds)
            y_min = min(b[1] for b in all_bounds)
            x_max = max(b[3] for b in all_bounds)
            y_max = max(b[4] for b in all_bounds)
            return (x_min, y_min, x_max, y_max)
        
        return None
    
    def _calculate_necrosis_percentage(self) -> float:
        """Calculate necrosis percentage in tumor"""
        if not self.tumor:
            return 0.0
        
        total_tumor_area = self.tumor.get_total_area()
        if total_tumor_area == 0:
            return 0.0
        
        necrotic_area = 0
        for region in self.tumor.regions:
            necrotic_area += region.geometry.area() * (region.necrosis_percentage / 100)
        
        return (necrotic_area / total_tumor_area) * 100
    
    def _calculate_invasion_front_length(self) -> float:
        """Calculate total length of tumor invasion front"""
        if not self.tumor:
            return 0.0
        
        invasion_front_regions = self.tumor.get_invasion_front_regions()
        total_length = 0
        
        for region in invasion_front_regions:
            total_length += region._calculate_perimeter()
        
        return total_length