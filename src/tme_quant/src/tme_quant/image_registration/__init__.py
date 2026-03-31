"""
Image registration module for multimodal microscopy alignment.

Available methods: intensity-based, feature-based, landmark-based, deep learning.
"""

from .registration_manager import RegistrationManager

__all__ = ['RegistrationManager']
__version__ = '1.0.0'
