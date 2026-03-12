"""
Image registration module for multimodal microscopy alignment.

Provides registration methods for aligning cell images (H&E, fluorescence, etc.)
with fiber images (SHG, polarized light, etc.).

Available Methods:
    Intensity-based:
        - MutualInformationRegistration (general multimodal)
        - CrossCorrelationRegistration (fast, similar modalities)
        - HESHGRegistration (H&E ↔ SHG, Keikhosravi 2020)
    
    Feature-based:
        - SIFTRegistration (scale-invariant features)
        - ORBRegistration (fast binary features)
    
    Landmark-based:
        - ManualLandmarkRegistration (user-selected points)
        - ThinPlateSpline (non-rigid warping)
    
    Deep Learning:
        - CoMIRRegistration (unsupervised multimodal DL)
        - VoxelMorphRegistration (deformable DL)

Example:
    >>> from tme_quant.image_registration import RegistrationManager
    >>> from tme_quant.image_registration.config import RegistrationParams, RegistrationMethod
    >>> 
    >>> # Use H&E-SHG method
    >>> params = RegistrationParams(method=RegistrationMethod.HE_SHG)
    >>> manager = RegistrationManager(verbose=True)
    >>> result = manager.register(shg_image, he_image, params)
    >>> 
    >>> registered_he = result.registered_image
"""

from .core.registration_manager import RegistrationManager

__all__ = ['RegistrationManager']

__version__ = '1.0.0'