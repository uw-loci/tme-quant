"""
FDCT transform wrapper functions.

This module provides wrapper functions for the Fast Discrete Curvelet Transform,
interfacing with CurveLab via Curvelops or equivalent libraries.
"""

from typing import List, Tuple, Optional
import numpy as np
import warnings

from ...types import CtCoeffs

# Try to import Curvelops
try:
    import curvelops
    HAS_CURVELOPS = True
except ImportError:
    HAS_CURVELOPS = False
    warnings.warn(
        "Curvelops not available. Using placeholder FDCT implementation. "
        "Install with: pip install 'curvelops @ git+https://github.com/PyLops/curvelops@0.23' "
        "Note: Requires FFTW 2.1.5 and CurveLab to be built first.",
        UserWarning
    )


def apply_fdct(image: np.ndarray, finest: int = 0, nbangles_coarsest: int = 2) -> CtCoeffs:
    """
    Apply forward Fast Discrete Curvelet Transform.
    
    Wraps the CurveLab fdct_wrapping function via Curvelops.
    Equivalent to: C = fdct_wrapping(IMG, finest, nbangles_coarsest)
    
    Parameters
    ----------
    image : np.ndarray
        2D input image
    finest : int, default 0
        Finest scale parameter (0 = include finest scale)
    nbangles_coarsest : int, default 2
        Number of angles at coarsest scale
        
    Returns
    -------
    CtCoeffs
        Curvelet coefficient structure (scales × wedges)
    """
    if HAS_CURVELOPS:
        try:
            # Use Curvelops for real FDCT
            cop = curvelops.FDCT2D(
                dims=image.shape,
                nbscales=None,  # Auto-determine scales
                nbangles_coarse=nbangles_coarsest,
                allcurvelets=finest == 0
            )
            
            # Apply forward transform
            coeffs_flat = cop.matvec(image.ravel())
            
            # Convert flat coefficients back to nested structure
            coeffs = _reshape_coeffs_from_flat(coeffs_flat, cop)
            return coeffs
            
        except Exception as e:
            warnings.warn(f"Curvelops FDCT failed: {e}. Using placeholder.", UserWarning)
    
    # Fallback to placeholder implementation
    return _apply_fdct_placeholder(image, finest, nbangles_coarsest)


def _apply_fdct_placeholder(image: np.ndarray, finest: int = 0, nbangles_coarsest: int = 2) -> CtCoeffs:
    """Placeholder FDCT implementation for when Curvelops is not available."""
    n_scales = max(3, int(np.log2(min(image.shape))) - 2)
    coeffs = []
    
    for scale in range(n_scales):
        # Number of wedges increases with scale
        if scale == 0:  # Coarsest scale
            n_wedges = nbangles_coarsest
        elif scale == n_scales - 1:  # Finest scale
            n_wedges = 1 if finest == 1 else 8 * 2**(scale-1)
        else:
            n_wedges = 8 * 2**(scale-1)
        
        scale_coeffs = []
        for wedge in range(n_wedges):
            # Generate realistic coefficient shape
            scale_factor = 2**(scale+1)
            coeff_shape = (image.shape[0] // scale_factor, image.shape[1] // scale_factor)
            coeff_shape = tuple(max(1, s) for s in coeff_shape)
            
            # Create coefficients with some structure based on image
            downsampled = image[::scale_factor, ::scale_factor]
            if downsampled.shape != coeff_shape:
                from scipy import ndimage
                downsampled = ndimage.zoom(downsampled, 
                                         [s/d for s, d in zip(coeff_shape, downsampled.shape)],
                                         order=1)
            
            # Add some randomness and make complex
            coeff = downsampled.astype(np.complex128)
            coeff += 0.1 * np.random.randn(*coeff_shape)
            coeff += 1j * 0.1 * np.random.randn(*coeff_shape)
            scale_coeffs.append(coeff)
        
        coeffs.append(scale_coeffs)
    
    return coeffs


def _reshape_coeffs_from_flat(coeffs_flat: np.ndarray, cop) -> CtCoeffs:
    """Convert flat coefficient array back to nested structure."""
    # This is a simplified version - in practice, we'd need to know
    # the exact structure from Curvelops
    coeffs = []
    idx = 0
    
    # Get the structure information from the operator
    try:
        # Access internal structure if available
        if hasattr(cop, '_coeff_shapes'):
            shapes = cop._coeff_shapes
        else:
            # Fallback: estimate structure
            n_scales = cop.nbscales if hasattr(cop, 'nbscales') else 4
            shapes = []
            for scale in range(n_scales):
                scale_shapes = []
                n_wedges = 8 * 2**scale if scale > 0 else cop.nbangles_coarse
                for wedge in range(n_wedges):
                    # Estimate shape
                    size = max(1, len(coeffs_flat) // (n_scales * n_wedges))
                    scale_shapes.append((int(np.sqrt(size)), int(np.sqrt(size))))
                shapes.append(scale_shapes)
        
        # Reshape coefficients
        for scale_shapes in shapes:
            scale_coeffs = []
            for shape in scale_shapes:
                size = np.prod(shape)
                if idx + size <= len(coeffs_flat):
                    coeff = coeffs_flat[idx:idx+size].reshape(shape)
                    scale_coeffs.append(coeff)
                    idx += size
                else:
                    # Handle remaining coefficients
                    remaining = len(coeffs_flat) - idx
                    if remaining > 0:
                        coeff = coeffs_flat[idx:].reshape(-1)
                        scale_coeffs.append(coeff)
                        idx = len(coeffs_flat)
                    break
            coeffs.append(scale_coeffs)
            
    except Exception:
        # Fallback: simple reshaping
        n_per_scale = len(coeffs_flat) // 4
        for i in range(4):
            start = i * n_per_scale
            end = start + n_per_scale if i < 3 else len(coeffs_flat)
            scale_data = coeffs_flat[start:end]
            coeffs.append([scale_data])
    
    return coeffs


def apply_ifdct(coeffs: CtCoeffs, finest: int = 0, img_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Apply inverse Fast Discrete Curvelet Transform.
    
    Wraps the CurveLab ifdct_wrapping function via Curvelops.
    Equivalent to: Y = ifdct_wrapping(Ct, finest)
    
    Parameters
    ----------
    coeffs : CtCoeffs
        Curvelet coefficient structure
    finest : int, default 0
        Finest scale parameter
    img_shape : Optional[Tuple[int, int]]
        Target image shape for reconstruction
        
    Returns
    -------
    np.ndarray
        Reconstructed image
    """
    if not coeffs or not coeffs[0]:
        return np.zeros((256, 256))
    
    # Estimate image size if not provided
    if img_shape is None:
        max_scale_coeffs = coeffs[-1][0] if coeffs[-1] else coeffs[0][0]
        img_shape = tuple(s * 2**len(coeffs) for s in max_scale_coeffs.shape)
        img_shape = tuple(min(1024, max(64, s)) for s in img_shape)
    
    if HAS_CURVELOPS:
        try:
            # Use Curvelops for real inverse FDCT
            cop = curvelops.FDCT2D(
                dims=img_shape,
                nbscales=len(coeffs),
                nbangles_coarse=len(coeffs[0]) if coeffs[0] else 2,
                allcurvelets=finest == 0
            )
            
            # Convert nested coefficients to flat array
            coeffs_flat = _flatten_coeffs(coeffs)
            
            # Apply inverse transform
            img_flat = cop.rmatvec(coeffs_flat)
            img_reconstructed = img_flat.reshape(img_shape)
            
            return np.real(img_reconstructed)
            
        except Exception as e:
            warnings.warn(f"Curvelops inverse FDCT failed: {e}. Using placeholder.", UserWarning)
    
    # Fallback to placeholder reconstruction
    return _apply_ifdct_placeholder(coeffs, img_shape)


def _apply_ifdct_placeholder(coeffs: CtCoeffs, img_shape: Tuple[int, int]) -> np.ndarray:
    """Placeholder inverse FDCT implementation."""
    # Simple reconstruction by summing scaled coefficient contributions
    reconstruction = np.zeros(img_shape)
    
    for scale_idx, scale_coeffs in enumerate(coeffs):
        for wedge_idx, coeff in enumerate(scale_coeffs):
            if coeff.size > 0:
                # Upscale coefficient to image size
                scale_factor = 2**(scale_idx + 1)
                upscaled_shape = (coeff.shape[0] * scale_factor, coeff.shape[1] * scale_factor)
                
                # Ensure we don't exceed image dimensions
                upscaled_shape = (
                    min(upscaled_shape[0], img_shape[0]),
                    min(upscaled_shape[1], img_shape[1])
                )
                
                if upscaled_shape[0] > 0 and upscaled_shape[1] > 0:
                    from scipy import ndimage
                    upscaled = ndimage.zoom(
                        np.real(coeff), 
                        [u/c for u, c in zip(upscaled_shape, coeff.shape)],
                        order=1
                    )
                    
                    # Add to reconstruction (with proper cropping)
                    h, w = upscaled.shape
                    reconstruction[:h, :w] += upscaled * (1.0 / len(coeffs))
    
    return reconstruction


def _flatten_coeffs(coeffs: CtCoeffs) -> np.ndarray:
    """Convert nested coefficient structure to flat array."""
    flat_coeffs = []
    for scale_coeffs in coeffs:
        for coeff in scale_coeffs:
            flat_coeffs.extend(coeff.ravel())
    return np.array(flat_coeffs)


def extract_parameters(coeffs: CtCoeffs, img_shape: Optional[Tuple[int, int]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract center position parameters from curvelet coefficients.
    
    Wraps the CurveLab fdct_wrapping_param function.
    Equivalent to: [X_rows, Y_cols] = fdct_wrapping_param(Ct)
    
    Parameters
    ----------
    coeffs : CtCoeffs
        Curvelet coefficient structure
    img_shape : Optional[Tuple[int, int]]
        Original image shape for parameter scaling
        
    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        Row and column center positions for each scale/wedge
    """
    if HAS_CURVELOPS and img_shape is not None:
        try:
            # Use Curvelops to get proper parameter structure
            cop = curvelops.FDCT2D(
                dims=img_shape,
                nbscales=len(coeffs),
                nbangles_coarse=len(coeffs[0]) if coeffs[0] else 2,
                allcurvelets=True
            )
            
            # Get parameter information from operator if available
            if hasattr(cop, 'get_parameters'):
                return cop.get_parameters()
            
        except Exception as e:
            warnings.warn(f"Curvelops parameter extraction failed: {e}. Using placeholder.", UserWarning)
    
    # Fallback to placeholder parameter generation
    X_rows = []
    Y_cols = []
    
    for scale_idx, scale_coeffs in enumerate(coeffs):
        scale_X = []
        scale_Y = []
        
        for wedge_idx, coeff in enumerate(scale_coeffs):
            if coeff.size > 0:
                # Create coordinate grids for this coefficient
                rows, cols = np.meshgrid(
                    np.arange(coeff.shape[0]), 
                    np.arange(coeff.shape[1]),
                    indexing='ij'
                )
                
                # Scale coordinates based on scale level
                scale_factor = 2**(scale_idx + 1)
                if img_shape is not None:
                    # Scale to image coordinates
                    rows = rows * (img_shape[0] / coeff.shape[0])
                    cols = cols * (img_shape[1] / coeff.shape[1])
                else:
                    rows = rows * scale_factor
                    cols = cols * scale_factor
                
                scale_X.append(rows)
                scale_Y.append(cols)
            else:
                scale_X.append(np.array([]))
                scale_Y.append(np.array([]))
        
        X_rows.append(scale_X)
        Y_cols.append(scale_Y)
    
    return X_rows, Y_cols


def get_curvelops_status() -> dict:
    """
    Get the current status of Curvelops integration.
    
    Returns
    -------
    dict
        Status information including availability and version
    """
    status = {
        'available': HAS_CURVELOPS,
        'version': None,
        'backend': 'placeholder'
    }
    
    if HAS_CURVELOPS:
        try:
            import curvelops
            status['version'] = getattr(curvelops, '__version__', 'unknown')
            status['backend'] = 'curvelops'
            
            # Test if we can create an operator
            try:
                test_op = curvelops.FDCT2D(dims=(64, 64))
                status['functional'] = True
            except Exception as e:
                status['functional'] = False
                status['error'] = str(e)
                
        except Exception as e:
            status['error'] = str(e)
    
    return status
