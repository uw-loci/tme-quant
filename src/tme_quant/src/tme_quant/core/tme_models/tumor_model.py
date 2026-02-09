# tumor_model.py
"""
Tumor-related data models
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum
from ..base_models import TMEObject, Geometry, Classification, Measurement

class TumorGrade(Enum):
    """Tumor grading classifications"""
    G1 = "Well differentiated"
    G2 = "Moderately differentiated"
    G3 = "Poorly differentiated"
    G4 = "Undifferentiated"
    GX = "Cannot be assessed"

@dataclass
class TumorRegion(TMEObject):
    """Represents a tumor region with spatial characteristics"""
    geometry: Geometry
    grade: TumorGrade = TumorGrade.GX
    necrosis_percentage: float = 0.0
    proliferation_index: float = 0.0
    invasion_front: bool = False
    subregions: List['TumorRegion'] = field(default_factory=list)
    
    # Spatial metrics
    border_irregularity: float = 0.0
    compactness: float = 0.0
    distance_to_margin: float = 0.0
    
    def __post_init__(self):
        """Initialize as tumor region"""
        super().__post_init__()
        self.type = TMEType.REGION
        self.properties['is_tumor'] = True
    
    def calculate_morphometrics(self) -> Dict[str, float]:
        """Calculate tumor morphology metrics"""
        metrics = {
            'area': self.geometry.area(),
            'perimeter': self._calculate_perimeter(),
            'compactness': self.compactness,
            'border_irregularity': self.border_irregularity,
            'circularity': self._calculate_circularity(),
            'solidity': self._calculate_solidity()
        }
        return metrics
    
    def _calculate_perimeter(self) -> float:
        """Calculate tumor perimeter"""
        # Implementation depends on geometry type
        if self.geometry.type == GeometryType.POLYGON:
            # Calculate polygon perimeter
            coords = self.geometry.coordinates
            if len(coords) > 1:
                perimeter = 0
                for i in range(len(coords)):
                    j = (i + 1) % len(coords)
                    perimeter += np.linalg.norm(coords[j] - coords[i])
                return perimeter
        return 0.0
    
    def _calculate_circularity(self) -> float:
        """Calculate circularity index (4π*area/perimeter²)"""
        area = self.geometry.area()
        perimeter = self._calculate_perimeter()
        if perimeter > 0:
            return (4 * np.pi * area) / (perimeter ** 2)
        return 0.0
    
    def _calculate_solidity(self) -> float:
        """Calculate solidity (area/convex_hull_area)"""
        # Implementation would require convex hull calculation
        return 1.0  # Placeholder

@dataclass
class Tumor(TMEObject):
    """Comprehensive tumor representation"""
    regions: List[TumorRegion] = field(default_factory=list)
    dominant_grade: TumorGrade = TumorGrade.GX
    spatial_distribution: str = ""
    molecular_subtype: str = ""
    
    # Tumor microenvironment context
    tumor_stroma_ratio: float = 0.0
    immune_infiltrate_density: float = 0.0
    angiogenesis_index: float = 0.0
    
    def __post_init__(self):
        """Initialize tumor object"""
        super().__post_init__()
        self.type = TMEType.TISSUE
        self.properties['object_type'] = 'tumor'
        
        # Add regions as children
        for region in self.regions:
            self.add_child(region)
    
    def get_total_area(self) -> float:
        """Calculate total tumor area"""
        return sum(region.geometry.area() for region in self.regions)
    
    def get_average_grade(self) -> float:
        """Calculate weighted average grade"""
        if not self.regions:
            return 0.0
        
        total_area = self.get_total_area()
        if total_area == 0:
            return 0.0
        
        weighted_sum = 0
        for region in self.regions:
            grade_value = float(region.grade.value.split()[0][1])  # Extract G1 -> 1, etc.
            weighted_sum += grade_value * region.geometry.area()
        
        return weighted_sum / total_area
    
    def get_invasion_front_regions(self) -> List[TumorRegion]:
        """Get regions identified as invasion front"""
        return [region for region in self.regions if region.invasion_front]