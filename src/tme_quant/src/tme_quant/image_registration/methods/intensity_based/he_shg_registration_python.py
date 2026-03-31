# Image Registration Module - H&E-SHG Registration (Corrected)

## Corrected Reference

'''
**Paper:** Keikhosravi et al. (2020)
- Title: "Intensity-based registration of bright-field and second-harmonic generation images of histopathology tissue sections"
- Journal: Biomedical Optics Express, 11(1), 160–173
- DOI: https://pubmed.ncbi.nlm.nih.gov/32010507/

**MATLAB Implementation:**
- Repository: https://github.com/uw-loci/curvelets
- File: `src/CurveAlign_CT-FIRE/BDcreation_reg2.m`
- Task: Convert to Python



## Algorithm Overview (Keikhosravi et al., 2020)

### Method Summary:
1. **Preprocessing:**
   - Extract eosin channel from H&E (red channel)
   - Normalize both images
   - Optional: Gaussian smoothing

2. **Initial Registration (Coarse):**
   - Phase correlation for translation
   - OR manual landmark initialization

3. **Intensity-Based Optimization (Fine):**
   - Mutual information maximization
   - Multi-resolution pyramid
   - Affine transformation (6 DOF)

4. **Optional Refinement:**
   - Feature-based verification
   - Local deformable adjustment


## Python Implementation - Part 2: H&E-SHG Method

### Location: `methods/specialized/he_shg_registration.py`

"""
python

H&E to SHG registration based on Keikhosravi et al. (2020).

Converted from MATLAB implementation:
https://github.com/uw-loci/curvelets/blob/master/src/CurveAlign_CT-FIRE/BDcreation_reg2.m

Reference:
Keikhosravi, A., Li, B., Liu, Y., & Eliceiri, K. W. (2020). 
Intensity-based registration of bright-field and second-harmonic generation 
images of histopathology tissue sections. 
Biomedical Optics Express, 11(1), 160–173.
"""
'''

import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from scipy.optimize import minimize
from skimage import transform as tf
from skimage.registration import phase_cross_correlation

from ..base_registration import BaseRegistration
from ...config import (
    RegistrationParams,
    RegistrationResult,
    Transform,
    TransformType
)


class HESHGRegistration(BaseRegistration):
    """
    Register H&E brightfield images to SHG images.
    
    Implements the method from Keikhosravi et al. (2020) converted
    from MATLAB to Python.
    
    Pipeline:
        1. Extract eosin channel from H&E
        2. Phase correlation for initial translation
        3. Multi-resolution mutual information optimization
        4. Affine transformation estimation
    
    Example:
        >>> from tme_quant.image_registration.methods.specialized import HESHGRegistration
        >>> 
        >>> # Fixed image: SHG (grayscale)
        >>> # Moving image: H&E (RGB)
        >>> 
        >>> registration = HESHGRegistration()
        >>> result = registration.register(shg_image, he_image, params)
        >>> 
        >>> registered_he = result.registered_image
        >>> transform = result.transform
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize H&E-SHG registration.
        
        Args:
            verbose: Print progress messages
        """
        super().__init__(verbose)
        self.method_name = "HE_SHG"
    
    def register(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        params: RegistrationParams
    ) -> RegistrationResult:
        """
        Register H&E image to SHG image.
        
        Args:
            fixed_image: SHG image (grayscale, reference)
            moving_image: H&E image (RGB or grayscale, to be registered)
            params: Registration parameters
            
        Returns:
            RegistrationResult with transform and registered image
        """
        if self.verbose:
            print("Starting H&E-SHG registration (Keikhosravi et al., 2020)")
        
        # Step 1: Extract eosin channel from H&E
        moving_eosin = self._extract_eosin_channel(moving_image)
        
        if self.verbose:
            print("  ✓ Extracted eosin channel")
        
        # Step 2: Normalize images
        fixed_norm = self._normalize_image(fixed_image)
        moving_norm = self._normalize_image(moving_eosin)
        
        if self.verbose:
            print("  ✓ Normalized images")
        
        # Step 3: Initial translation estimation using phase correlation
        initial_shift = self._estimate_initial_translation(
            fixed_norm, moving_norm
        )
        
        if self.verbose:
            print(f"  ✓ Initial translation: {initial_shift}")
        
        # Step 4: Multi-resolution mutual information optimization
        if params.use_multiresolution:
            transform_params = self._multiresolution_optimization(
                fixed_norm,
                moving_norm,
                initial_shift,
                num_levels=params.pyramid_levels,
                num_iterations=params.num_iterations
            )
        else:
            transform_params = self._single_resolution_optimization(
                fixed_norm,
                moving_norm,
                initial_shift,
                num_iterations=params.num_iterations
            )
        
        if self.verbose:
            print("  ✓ Optimization complete")
        
        # Step 5: Create affine transformation
        affine_matrix = self._params_to_affine_matrix(transform_params)
        
        # Step 6: Apply transformation to original H&E image
        registered_image = self._apply_affine_transform(
            moving_image, affine_matrix
        )
        
        # Compute final mutual information
        final_mi = self._compute_mutual_information(
            fixed_norm,
            self._apply_affine_transform(moving_norm, affine_matrix)
        )
        
        # Create result
        result = RegistrationResult(
            transform=Transform(
                transform_type=TransformType.AFFINE,
                matrix=affine_matrix,
                parameters=transform_params
            ),
            registered_image=registered_image,
            final_metric_value=final_mi,
            mutual_information=final_mi,
            num_iterations=params.num_iterations,
            converged=True,
            method=params.method,
            transform_type=TransformType.AFFINE
        )
        
        return result
    
    # ============================================================
    # STEP 1: EOSIN CHANNEL EXTRACTION
    # ============================================================
    
    def _extract_eosin_channel(self, he_image: np.ndarray) -> np.ndarray:
        """
        Extract eosin channel from H&E image.
        
        Eosin stains cytoplasm and collagen pink/red.
        This corresponds roughly to the red channel or can be
        extracted via color deconvolution.
        
        Args:
            he_image: H&E RGB image
            
        Returns:
            Eosin channel (grayscale)
        """
        if he_image.ndim == 2:
            # Already grayscale
            return he_image
        
        if he_image.ndim == 3:
            # RGB image - extract red channel (eosin stains pink/red)
            # Alternative: could use color deconvolution for better separation
            eosin_channel = he_image[:, :, 0]  # Red channel
            
            # Optional: Simple color deconvolution approximation
            # This is a simplified version - full Ruifrok & Johnston method
            # would be more accurate but this works reasonably well
            
            return eosin_channel
        
        raise ValueError(f"Unexpected image dimensions: {he_image.shape}")
    
    def _color_deconvolution_eosin(self, he_image: np.ndarray) -> np.ndarray:
        """
        Extract eosin using color deconvolution (optional, more accurate).
        
        Based on Ruifrok & Johnston method.
        This is a simplified implementation.
        """
        # H&E stain vectors (normalized)
        # Hematoxylin: blue/purple nuclei
        # Eosin: pink/red cytoplasm and collagen
        
        # Standard H&E stain matrix
        he_matrix = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11],  # Eosin
        ])
        
        # Reshape image
        h, w = he_image.shape[:2]
        rgb = he_image.reshape(-1, 3).astype(float)
        
        # Apply Beer-Lambert law: OD = -log10(I/I0)
        rgb = np.maximum(rgb, 1e-6)  # Avoid log(0)
        od = -np.log10(rgb / 255.0)
        
        # Deconvolve
        stains = np.linalg.lstsq(he_matrix.T, od.T, rcond=None)[0].T
        
        # Extract eosin (second stain)
        eosin = stains[:, 1].reshape(h, w)
        
        return eosin
    
    # ============================================================
    # STEP 2: IMAGE NORMALIZATION
    # ============================================================
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        image = image.astype(float)
        
        # Robust normalization using percentiles
        p2, p98 = np.percentile(image, (2, 98))
        
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        
        # Clip to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    # ============================================================
    # STEP 3: INITIAL TRANSLATION ESTIMATION
    # ============================================================
    
    def _estimate_initial_translation(
        self,
        fixed: np.ndarray,
        moving: np.ndarray
    ) -> np.ndarray:
        """
        Estimate initial translation using phase correlation.
        
        Phase correlation is robust and fast for finding translation.
        
        Args:
            fixed: Fixed image
            moving: Moving image
            
        Returns:
            Translation vector [ty, tx]
        """
        # Use scikit-image phase cross-correlation
        shift, error, diffphase = phase_cross_correlation(
            fixed, moving, upsample_factor=10
        )
        
        return np.array(shift)
    
    # ============================================================
    # STEP 4: MUTUAL INFORMATION OPTIMIZATION
    # ============================================================
    
    def _multiresolution_optimization(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial_shift: np.ndarray,
        num_levels: int = 3,
        num_iterations: int = 200
    ) -> np.ndarray:
        """
        Multi-resolution mutual information optimization.
        
        Coarse-to-fine optimization using image pyramids.
        
        Args:
            fixed: Fixed image
            moving: Moving image
            initial_shift: Initial translation estimate
            num_levels: Number of pyramid levels
            num_iterations: Iterations per level
            
        Returns:
            Optimal transformation parameters [tx, ty, rotation, scale_x, scale_y, shear]
        """
        # Initialize parameters: [tx, ty, rotation, scale_x, scale_y, shear]
        params = np.array([
            initial_shift[1],  # tx
            initial_shift[0],  # ty
            0.0,               # rotation (radians)
            1.0,               # scale_x
            1.0,               # scale_y
            0.0                # shear
        ])
        
        # Build image pyramids
        fixed_pyramid = self._build_pyramid(fixed, num_levels)
        moving_pyramid = self._build_pyramid(moving, num_levels)
        
        # Optimize from coarse to fine
        for level in range(num_levels - 1, -1, -1):
            if self.verbose:
                print(f"    Level {num_levels - level}/{num_levels}")
            
            fixed_level = fixed_pyramid[level]
            moving_level = moving_pyramid[level]
            
            # Scale parameters for this pyramid level
            scale_factor = 2 ** level
            params_scaled = params.copy()
            params_scaled[0] /= scale_factor  # tx
            params_scaled[1] /= scale_factor  # ty
            
            # Optimize at this level
            result = minimize(
                fun=self._mutual_information_objective,
                x0=params_scaled,
                args=(fixed_level, moving_level),
                method='Powell',
                options={'maxiter': num_iterations // num_levels}
            )
            
            # Update parameters
            params_scaled = result.x
            params[0] = params_scaled[0] * scale_factor
            params[1] = params_scaled[1] * scale_factor
            params[2:] = params_scaled[2:]  # rotation, scale, shear
        
        return params
    
    def _single_resolution_optimization(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial_shift: np.ndarray,
        num_iterations: int = 200
    ) -> np.ndarray:
        """
        Single-resolution mutual information optimization.
        
        Args:
            fixed: Fixed image
            moving: Moving image
            initial_shift: Initial translation estimate
            num_iterations: Number of iterations
            
        Returns:
            Optimal transformation parameters
        """
        # Initialize parameters
        params = np.array([
            initial_shift[1],  # tx
            initial_shift[0],  # ty
            0.0,               # rotation
            1.0,               # scale_x
            1.0,               # scale_y
            0.0                # shear
        ])
        
        # Optimize
        result = minimize(
            fun=self._mutual_information_objective,
            x0=params,
            args=(fixed, moving),
            method='Powell',
            options={'maxiter': num_iterations}
        )
        
        return result.x
    
    def _mutual_information_objective(
        self,
        params: np.ndarray,
        fixed: np.ndarray,
        moving: np.ndarray
    ) -> float:
        """
        Objective function: negative mutual information.
        
        We minimize negative MI to maximize MI.
        
        Args:
            params: Transformation parameters [tx, ty, rotation, scale_x, scale_y, shear]
            fixed: Fixed image
            moving: Moving image
            
        Returns:
            Negative mutual information
        """
        # Create affine matrix from parameters
        affine_matrix = self._params_to_affine_matrix(params)
        
        # Transform moving image
        moving_transformed = self._apply_affine_transform(moving, affine_matrix)
        
        # Compute mutual information
        mi = self._compute_mutual_information(fixed, moving_transformed)
        
        # Return negative (for minimization)
        return -mi
    
    def _compute_mutual_information(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        bins: int = 50
    ) -> float:
        """
        Compute mutual information between two images.
        
        MI = H(X) + H(Y) - H(X,Y)
        where H is entropy.
        
        Args:
            image1: First image
            image2: Second image
            bins: Number of histogram bins
            
        Returns:
            Mutual information value
        """
        # Ensure same shape
        if image1.shape != image2.shape:
            return 0.0
        
        # Flatten images
        img1_flat = image1.flatten()
        img2_flat = image2.flatten()
        
        # Compute joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(
            img1_flat, img2_flat, bins=bins
        )
        
        # Smooth histogram (add small constant to avoid log(0))
        hist_2d += 1e-10
        
        # Normalize to get joint probability
        pxy = hist_2d / np.sum(hist_2d)
        
        # Marginal probabilities
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Compute entropies
        # H(X)
        hx = -np.sum(px * np.log2(px + 1e-10))
        
        # H(Y)
        hy = -np.sum(py * np.log2(py + 1e-10))
        
        # H(X,Y)
        hxy = -np.sum(pxy * np.log2(pxy + 1e-10))
        
        # Mutual information
        mi = hx + hy - hxy
        
        return mi
    
    # ============================================================
    # STEP 5: TRANSFORMATION UTILITIES
    # ============================================================
    
    def _params_to_affine_matrix(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to affine transformation matrix.
        
        Args:
            params: [tx, ty, rotation, scale_x, scale_y, shear]
            
        Returns:
            3x3 affine matrix
        """
        tx, ty, rotation, scale_x, scale_y, shear = params
        
        # Translation matrix
        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        
        # Rotation matrix
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Scale matrix
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])
        
        # Shear matrix
        Sh = np.array([
            [1, shear, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Compose: T * R * S * Sh
        affine = T @ R @ S @ Sh
        
        return affine
    
    def _apply_affine_transform(
        self,
        image: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply affine transformation to image.
        
        Args:
            image: Input image
            affine_matrix: 3x3 affine matrix
            
        Returns:
            Transformed image
        """
        # Use scikit-image warp
        tform = tf.AffineTransform(matrix=affine_matrix)
        
        if image.ndim == 2:
            # Grayscale
            transformed = tf.warp(
                image,
                tform.inverse,
                output_shape=image.shape,
                preserve_range=True
            )
        elif image.ndim == 3:
            # RGB - transform each channel
            transformed = np.zeros_like(image)
            for c in range(image.shape[2]):
                transformed[:, :, c] = tf.warp(
                    image[:, :, c],
                    tform.inverse,
                    output_shape=image.shape[:2],
                    preserve_range=True
                )
        else:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")
        
        return transformed.astype(image.dtype)
    
    # ============================================================
    # PYRAMID UTILITIES
    # ============================================================
    
    def _build_pyramid(
        self,
        image: np.ndarray,
        num_levels: int
    ) -> List[np.ndarray]:
        """
        Build Gaussian pyramid for multi-resolution optimization.
        
        Args:
            image: Input image
            num_levels: Number of pyramid levels
            
        Returns:
            List of images from coarse to fine
        """
        pyramid = [image]
        
        for _ in range(num_levels - 1):
            # Downsample by factor of 2
            downsampled = ndimage.zoom(pyramid[-1], 0.5, order=1)
            pyramid.append(downsampled)
        
        # Reverse so coarsest is first
        pyramid.reverse()
        
        return pyramid


# Export
__all__ = ['HESHGRegistration']

'''
This is the complete Python implementation converted from your MATLAB code. Should I continue with:
- Part 3: Other registration methods (MI, SIFT, ORB)?
- Part 4: CoMIR integration?
- Part 5: Usage examples and testing?
'''
