"""
Feature-based registration methods.
"""

from .sift_registration import SIFTRegistration
from .orb_registration import ORBRegistration

__all__ = [
    'SIFTRegistration',
    'ORBRegistration',
]