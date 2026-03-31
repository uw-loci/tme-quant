# stroma_model.py
"""
Stromal compartment data models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from enum import Enum
from ..base_models import TMEObject, TMEType, ObjectType, Geometry, Measurement

class ECMComponent(Enum):
    """Extracellular matrix components"""
    COLLAGEN = "collagen"
    ELASTIN = "elastin"
    FIBRONECTIN = "fibronectin"
    LAMININ = "laminin"
    PROTEOGLYCAN = "proteoglycan"
    GLYCOSAMINOGLYCAN = "glycosaminoglycan"
    HYALURONAN = "hyaluronan"

class StromaRegion(TMEObject):
    """Stromal region with ECM composition."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi=None,
        parent=None,
        metadata=None,
        geometry: Geometry = None,
        ecm_composition: Dict[ECMComponent, float] = None,
        cellularity: float = 0.0,
        fibrosis_score: float = 0.0,
        inflammation_score: float = 0.0,
        pattern_type: str = "",
        organization_score: float = 0.0,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.REGION,
            object_type=ObjectType.REGION,
            roi=roi,
            parent=parent,
            metadata=metadata,
        )
        self.geometry = geometry
        self.ecm_composition: Dict[ECMComponent, float] = (
            ecm_composition if ecm_composition is not None else {}
        )
        self.cellularity = cellularity
        self.fibrosis_score = fibrosis_score
        self.inflammation_score = inflammation_score
        self.pattern_type = pattern_type
        self.organization_score = organization_score
        self.properties['is_stroma'] = True
    
    @property
    def total_ecm_density(self) -> float:
        """Calculate total ECM density"""
        return sum(self.ecm_composition.values())
    
    def get_collagen_content(self) -> float:
        """Get collagen content in stroma"""
        return self.ecm_composition.get(ECMComponent.COLLAGEN, 0.0)

class Stroma(TMEObject):
    """Comprehensive stromal compartment representation."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi=None,
        parent=None,
        metadata=None,
        regions: List[StromaRegion] = None,
        fibroblast_density: float = 0.0,
        immune_cell_density: float = 0.0,
        vascular_density: float = 0.0,
        stiffness: float = 0.0,
        tensile_strength: float = 0.0,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.STROMA,
            object_type=ObjectType.REGION,
            roi=roi,
            parent=parent,
            metadata=metadata,
        )
        self.regions: List[StromaRegion] = regions if regions is not None else []
        self.fibroblast_density = fibroblast_density
        self.immune_cell_density = immune_cell_density
        self.vascular_density = vascular_density
        self.stiffness = stiffness
        self.tensile_strength = tensile_strength
        for region in self.regions:
            self.add_child(region)
    
    def get_total_area(self) -> float:
        """Calculate total stroma area."""
        return sum(
            r.geometry.area() for r in self.regions if r.geometry is not None
        )
    
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