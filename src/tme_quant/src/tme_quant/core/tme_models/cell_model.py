# cell_model.py
"""
Cell-related data models with comprehensive cell types and states
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum, auto
from ..base_models import TMEObject, Geometry, Classification, Measurement

class CellType(Enum):
    """Comprehensive cell type classification"""
    TUMOR = "tumor_cell"
    IMMUNE = "immune_cell"
    STROMAL = "stromal_cell"
    ENDOTHELIAL = "endothelial_cell"
    FIBROBLAST = "fibroblast"
    MYOFIBROBLAST = "myofibroblast"
    ADIPOCYTE = "adipocyte"
    NEURONAL = "neuronal"
    EPITHELIAL = "epithelial"
    UNKNOWN = "unknown"

class ImmuneCellSubtype(Enum):
    """Immune cell subtypes"""
    T_CELL = "t_cell"
    T_HELPER = "t_helper"
    T_CYTOTOXIC = "t_cytotoxic"
    T_REGULATORY = "t_regulatory"
    B_CELL = "b_cell"
    NK_CELL = "nk_cell"
    MACROPHAGE = "macrophage"
    DENDRITIC = "dendritic"
    NEUTROPHIL = "neutrophil"
    MAST = "mast"

class CellState(Enum):
    """Cell functional states"""
    PROLIFERATING = "proliferating"
    APOPTOTIC = "apoptotic"
    NECROTIC = "necrotic"
    SENESCENT = "senescent"
    QUIESCENT = "quiescent"
    ACTIVATED = "activated"
    EXHAUSTED = "exhausted"
    STEM_LIKE = "stem_like"
    EMT = "emt"  # Epithelial-mesenchymal transition

@dataclass
class Cell(TMEObject):
    """Base cell model with comprehensive properties"""
    geometry: Geometry
    cell_type: CellType = CellType.UNKNOWN
    subtype: Optional[str] = None
    cell_state: CellState = CellState.QUIESCENT
    
    # Morphological features
    nucleus_area: float = 0.0
    cytoplasm_area: float = 0.0
    nuclear_cytoplasmic_ratio: float = 0.0
    eccentricity: float = 0.0
    
    # Spatial features
    neighbors: List[str] = field(default_factory=list)  # IDs of neighboring cells
    voronoi_region: Optional[Geometry] = None
    delaunay_triangulation: Optional[List[Tuple[int, int, int]]] = None
    
    # Molecular markers
    markers: Dict[str, float] = field(default_factory=dict)  # Marker: intensity
    expression_profile: Dict[str, float] = field(default_factory=dict)
    
    # Measurements
    measurements: List[Measurement] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize cell object"""
        super().__post_init__()
        self.type = TMEType.CELL
        self.properties['object_type'] = 'cell'
    
    @property
    def centroid(self) -> np.ndarray:
        """Get cell centroid"""
        return self.geometry.centroid()
    
    @property
    def area(self) -> float:
        """Get cell area"""
        return self.geometry.area()
    
    def get_marker_expression(self, marker_name: str) -> float:
        """Get marker expression level"""
        return self.markers.get(marker_name, 0.0)
    
    def is_positive_for(self, marker_name: str, threshold: float = 0.5) -> bool:
        """Check if cell is positive for marker"""
        return self.get_marker_expression(marker_name) > threshold
    
    def calculate_morphometrics(self) -> Dict[str, float]:
        """Calculate comprehensive morphological metrics"""
        return {
            'total_area': self.area,
            'nucleus_area': self.nucleus_area,
            'cytoplasm_area': self.cytoplasm_area,
            'nuclear_cytoplasmic_ratio': self.nuclear_cytoplasmic_ratio,
            'eccentricity': self.eccentricity,
            'perimeter': self._calculate_perimeter(),
            'circularity': self._calculate_circularity()
        }
    
    def _calculate_perimeter(self) -> float:
        """Calculate cell perimeter"""
        # Implementation depends on geometry
        if self.geometry.type == GeometryType.POLYGON:
            coords = self.geometry.coordinates
            if len(coords) > 1:
                perimeter = 0
                for i in range(len(coords)):
                    j = (i + 1) % len(coords)
                    perimeter += np.linalg.norm(coords[j] - coords[i])
                return perimeter
        return 0.0
    
    def _calculate_circularity(self) -> float:
        """Calculate circularity"""
        area = self.area
        perimeter = self._calculate_perimeter()
        if perimeter > 0:
            return (4 * np.pi * area) / (perimeter ** 2)
        return 0.0

@dataclass
class ImmuneCell(Cell):
    """Immune cell specialization"""
    immune_subtype: ImmuneCellSubtype = ImmuneCellSubtype.T_CELL
    activation_state: str = "naive"
    exhaustion_markers: Dict[str, float] = field(default_factory=dict)
    checkpoint_expression: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize as immune cell"""
        super().__post_init__()
        self.cell_type = CellType.IMMUNE
        self.properties['immune_subtype'] = self.immune_subtype.value
    
    def get_exhaustion_score(self) -> float:
        """Calculate immune exhaustion score"""
        if not self.exhaustion_markers:
            return 0.0
        return sum(self.exhaustion_markers.values()) / len(self.exhaustion_markers)

@dataclass
class TumorCell(Cell):
    """Tumor cell specialization"""
    grade: TumorGrade = TumorGrade.GX
    proliferation_marker: float = 0.0
    stemness_score: float = 0.0
    emt_score: float = 0.0
    drug_resistance_markers: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize as tumor cell"""
        super().__post_init__()
        self.cell_type = CellType.TUMOR
        self.properties['tumor_grade'] = self.grade.value

@dataclass
class StromalCell(Cell):
    """Stromal cell specialization"""
    stromal_subtype: str = ""
    activation_state: str = ""
    ecm_production_markers: Dict[str, float] = field(default_factory=dict)
    contractility_markers: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize as stromal cell"""
        super().__post_init__()
        self.cell_type = CellType.STROMAL
        self.properties['stromal_subtype'] = self.stromal_subtype