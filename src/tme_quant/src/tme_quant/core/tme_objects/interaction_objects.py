# interaction_models.py
# Contains:
# - CellFiberInteraction: Individual interaction between cell and fiber
# - InteractionNetwork: Network of interactions with graph analysis
# - InteractionCategory enum (physical_contact, spatial_proximity, etc.)
# - InteractionStrength enum (weak, moderate, strong, very_strong)

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from enum import Enum

class InteractionType(Enum):
    """Types of cell-fiber interactions"""
    FIBER_ALIGNMENT = "fiber_alignment"
    FIBER_PROXIMITY = "fiber_proximity"
    FIBER_ADHESION = "fiber_adhesion"
    FIBER_GUIDANCE = "fiber_guidance"
    COLLAGEN_REMODELING = "collagen_remodeling"

@dataclass
class Interaction:
    """Represents a cell-fiber interaction"""
    cell_id: str
    fiber_id: str
    interaction_type: InteractionType
    distance: float  # Distance between cell and fiber
    alignment_angle: Optional[float] = None  # Angle between cell orientation and fiber
    spatial_context: Optional[str] = None  # e.g., "tumor_boundary", "stroma"
    
    # Quantifiable metrics
    interaction_strength: float = 0.0
    contact_area: Optional[float] = None
    temporal_duration: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'cell_id': self.cell_id,
            'fiber_id': self.fiber_id,
            'interaction_type': self.interaction_type.value,
            'distance': self.distance,
            'alignment_angle': self.alignment_angle,
            'spatial_context': self.spatial_context,
            'interaction_strength': self.interaction_strength
        }

@dataclass 
class InteractionNetwork:
    """Network of cell-fiber interactions in a tissue sample"""
    interactions: List[Interaction]
    cells: Dict[str, 'Cell']  # Reference to cell objects
    fibers: Dict[str, 'Fiber']  # Reference to fiber objects
    
    def get_interactions_by_cell(self, cell_id: str) -> List[Interaction]:
        """Get all interactions for a specific cell"""
        return [i for i in self.interactions if i.cell_id == cell_id]
    
    def get_interactions_by_fiber(self, fiber_id: str) -> List[Interaction]:
        """Get all interactions for a specific fiber"""
        return [i for i in self.interactions if i.fiber_id == fiber_id]
    
    def calculate_network_metrics(self) -> Dict:
        """Calculate network-level metrics"""
        metrics = {
            'total_interactions': len(self.interactions),
            'unique_cells': len(set(i.cell_id for i in self.interactions)),
            'unique_fibers': len(set(i.fiber_id for i in self.interactions)),
            'average_distance': np.mean([i.distance for i in self.interactions]),
            'interaction_density': len(self.interactions) / (len(self.cells) * len(self.fibers))
        }
        return metrics