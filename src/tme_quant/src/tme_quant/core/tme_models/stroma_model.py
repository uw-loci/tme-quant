# stroma_model.py
"""
Stromal compartment data models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from enum import Enum
from ..base_models import TMEObject, Geometry, Measurement

class ECMComponent(Enum):
    """Extracellular matrix components"""
    COLLAGEN = "collagen"
    ELASTIN = "elastin"
    FIBRONECTIN = "fibronectin"
    LAMININ = "laminin"
    PROTEOGLYCAN = "proteoglycan"
    GLYCOSAMINOGLYCAN = "glycosaminoglycan"
    HYALURONAN = "hyaluronan"

@dataclass
class StromaRegion(TMEObject):
    """Stromal region with ECM composition"""
    geometry: Geometry
    ecm_composition: Dict[ECMComponent, float] = field(default_factory=dict)
    cellularity: float = 0.0
    fibrosis_score: float = 0.0
    inflammation_score: float = 0.0
    
    # Architectural patterns
    pattern_type: str = ""  # desmoplastic, myxoid, hyalinized, etc.
    organization_score: float = 0.0
    
    def __post_init__(self):
        """Initialize stroma region"""
        super().__post_init__()
        self.type = TMEType.REGION
        self.properties['is_stroma'] = True
    
    @property
    def total_ecm_density(self) -> float:
        """Calculate total ECM density"""
        return sum(self.ecm_composition.values())
    
    def get_collagen_content(self) -> float:
        """Get collagen content in stroma"""
        return self.ecm_composition.get(ECMComponent.COLLAGEN, 0.0)

@dataclass
class Stroma(TMEObject):
    """Comprehensive stromal compartment representation"""
    regions: List[StromaRegion] = field(default_factory=list)
    fibroblast_density: float = 0.0
    immune_cell_density: float = 0.0
    vascular_density: float = 0.0
    
    # Mechanical properties
    stiffness: float = 0.0
    tensile_strength: float = 0.0
    
    def __post_init__(self):
        """Initialize stroma object"""
        super().__post_init__()
        self.type = TMEType.STROMA
        
        # Add regions as children
        for region in self.regions:
            self.add_child(region)
    
    def get_total_area(self) -> float:
        """Calculate total stroma area"""
        return sum(region.geometry.area() for region in self.regions)
    
    def calculate_ecm_profile(self) -> Dict[str, float]:
        """Calculate average ECM composition"""
        if not self.regions:
            return {}
        
        total_area = self.get_total_area()
        if total_area == 0:
            return {}
        
        ecm_profile = {}
        for region in self.regions:
            weight = region.geometry.area() / total_area
            for component, value in region.ecm_composition.items():
                ecm_profile[component.value] = ecm_profile.get(component.value, 0) + value * weight
        
        return ecm_profile