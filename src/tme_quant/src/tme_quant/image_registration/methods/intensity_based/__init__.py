"""
Intensity-based registration methods.

Methods that use pixel intensity values directly for registration:
- Mutual Information (general multimodal)
- Cross-Correlation (similar modalities)
- H&E-SHG (specialized MI-based method for H&E and SHG alignment)
"""

from .mutual_information import MutualInformationRegistration
from .cross_correlation import CrossCorrelationRegistration
from .he_shg_registration_python import HESHGRegistration

__all__ = [
    'MutualInformationRegistration',
    'CrossCorrelationRegistration',
    'HESHGRegistration',
]