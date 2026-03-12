"""
Deep learning registration methods.

Provides unsupervised deep learning methods for multimodal registration:
- CoMIR: Contrastive Multimodal Image Registration
- VoxelMorph: Learning-based deformable registration

Installation:
    # CoMIR
    git clone https://github.com/MIDA-group/CoMIR_INSPIRE.git
    cd CoMIR_INSPIRE
    pip install -r requirements.txt
    
    # VoxelMorph
    pip install voxelmorph

Example:
    >>> from tme_quant.image_registration.methods.deep_learning import CoMIRRegistration
    >>> 
    >>> comir = CoMIRRegistration(
    ...     comir_path="/path/to/CoMIR_INSPIRE",
    ...     model_path="/path/to/model.pth"
    ... )
    >>> result = comir.register(fixed, moving, params)
"""

from .comir_registration import CoMIRRegistration
from .voxelmorph_registration import VoxelMorphRegistration

__all__ = [
    'CoMIRRegistration',
    'VoxelMorphRegistration',
]