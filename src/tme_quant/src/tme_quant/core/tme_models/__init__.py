# __init__.py
"""
TME Models package with comprehensive TME object representations
"""
from .tumor_model import Tumor, TumorRegion, TumorGrade
from .cell_model import Cell, CellType, CellState, ImmuneCell, TumorCell, StromalCell
from .fiber_model import Fiber, CollagenFiber, ReticularFiber, ElasticFiber, FiberNetwork
from .vessel_model import Vessel, BloodVessel, LymphaticVessel, VesselNetwork
from .stroma_model import Stroma, StromaRegion, ECMComponent
from .tissue_model import TissueSample, TissueRegion, TissueZone
from .interaction_models import Interaction, InteractionNetwork, InteractionType

__all__ = [
    'Tumor', 'TumorRegion', 'TumorGrade',
    'Cell', 'CellType', 'CellState', 'ImmuneCell', 'TumorCell', 'StromalCell',
    'Fiber', 'CollagenFiber', 'ReticularFiber', 'ElasticFiber', 'FiberNetwork',
    'Vessel', 'BloodVessel', 'LymphaticVessel', 'VesselNetwork',
    'Stroma', 'StromaRegion', 'ECMComponent',
    'TissueSample', 'TissueRegion', 'TissueZone',
    'Interaction', 'InteractionNetwork', 'InteractionType'
]