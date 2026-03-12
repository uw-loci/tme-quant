# ============================================================
# MUTUAL INFORMATION REGISTRATION
# Location: methods/intensity_based/mutual_information.py
# ============================================================

import importlib
import time
from typing import Any, cast

import numpy as np

from ..base_registration import BaseRegistration

sitk: Any

try:
    sitk = importlib.import_module("SimpleITK")
    SITK_AVAILABLE = True
except ImportError:
    sitk = cast(Any, None)
    SITK_AVAILABLE = False


class MutualInformationRegistration(BaseRegistration):
    """
    Mutual information registration using SimpleITK.
    
    Works for any modality pair (multimodal registration).
    Uses Mattes mutual information metric with gradient descent.
    
    Example:
        >>> from tme_quant.image_registration.methods.intensity_based import MutualInformationRegistration
        >>> from tme_quant.image_registration.config import RegistrationParams
        >>> 
        >>> registration = MutualInformationRegistration(verbose=True)
        >>> result = registration.register(fixed, moving, params)
        >>> 
        >>> registered_image = result.registered_image
        >>> mi_score = result.mutual_information
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.method_name = "mutual_information"
        
        if not SITK_AVAILABLE:
            raise ImportError(
                "SimpleITK required for MI registration. "
                "Install with: pip install SimpleITK"
            )
    
    def register(self, fixed_image, moving_image, params):
        """
        Register using mutual information.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            params: Registration parameters
            
        Returns:
            RegistrationResult
        """
        self.validate_images(fixed_image, moving_image)
        
        if self.verbose:
            print("Starting Mutual Information registration")
        
        start_time = time.time()
        
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Mutual information metric
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=params.mi_bins
        )
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        
        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=params.learning_rate,
            numberOfIterations=params.num_iterations,
            convergenceMinimumValue=params.convergence_threshold,
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Multi-resolution
        if params.use_multiresolution:
            shrink_factors = [4, 2, 1][:params.pyramid_levels]
            smoothing_sigmas = [2, 1, 0][:params.pyramid_levels]
            registration_method.SetShrinkFactorsPerLevel(shrink_factors)
            registration_method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Initial transform
        initial_transform = self._get_initial_transform(
            fixed_sitk, moving_sitk, params.transform_type
        )
        registration_method.SetInitialTransform(initial_transform)
        
        # Execute registration
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # Apply transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        registered_sitk = resampler.Execute(moving_sitk)
        
        registered_image = sitk.GetArrayFromImage(registered_sitk)
        
        # Get transform matrix
        transform_matrix = self._sitk_to_matrix(final_transform)
        
        # Get final MI
        final_mi = registration_method.GetMetricValue()
        
        if self.verbose:
            print(f"  ✓ MI registration complete")
            print(f"  Final MI: {-final_mi:.4f}")
            print(f"  Time: {time.time() - start_time:.2f}s")
        
        # Create result
        from ...config.registration_params import RegistrationResult, Transform, RegistrationMethod
        
        result = RegistrationResult(
            transform=Transform(
                transform_type=params.transform_type,
                matrix=transform_matrix
            ),
            registered_image=registered_image,
            final_metric_value=-final_mi,
            mutual_information=-final_mi,
            num_iterations=params.num_iterations,
            converged=True,
            optimization_time=time.time() - start_time,
            method=RegistrationMethod.MUTUAL_INFORMATION,
            transform_type=params.transform_type
        )
        
        return result
    
    def _get_initial_transform(self, fixed, moving, transform_type):
        """Get initial transform based on type."""
        from ...config.registration_params import TransformType
        
        if transform_type == TransformType.TRANSLATION:
            transform = sitk.TranslationTransform(2)
        elif transform_type == TransformType.RIGID:
            transform = sitk.Euler2DTransform()
        elif transform_type == TransformType.SIMILARITY:
            transform = sitk.Similarity2DTransform()
        elif transform_type == TransformType.AFFINE:
            transform = sitk.AffineTransform(2)
        else:
            transform = sitk.AffineTransform(2)
        
        # Initialize with centering
        initial = sitk.CenteredTransformInitializer(
            fixed, moving, transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        return initial
    
    def _sitk_to_matrix(self, transform):
        """Convert SimpleITK transform to matrix."""
        params = transform.GetParameters()
        
        # For 2D affine: [m00, m01, m10, m11, tx, ty]
        if len(params) == 6:
            matrix = np.array([
                [params[0], params[1], params[4]],
                [params[2], params[3], params[5]],
                [0, 0, 1]
            ])
        else:
            matrix = np.eye(3)
        
        return matrix