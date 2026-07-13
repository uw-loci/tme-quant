"""
Preprocessing module for CurveAlign Napari plugin.

Provides preprocessing options including:
- Bio-Formats import (via aicsimageio or napari-imagej)
- Tubeness filter (via skimage.filters)
- Frangi filter (via skimage.filters)
- Autothreshold options (Otsu, Triangle, Isodata, etc.)
"""

import numpy as np
from typing import Optional, Tuple, Literal
from enum import Enum

from skimage import filters, io
from skimage.filters import (
    frangi,
    meijering,
    threshold_isodata,
    threshold_li,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

try:
    import aicsimageio
    HAS_AICSIO = True
except ImportError:
    HAS_AICSIO = False

try:
    import napari_imagej
    HAS_IMAGEJ = True
except ImportError:
    HAS_IMAGEJ = False


class ThresholdMethod(Enum):
    """Threshold methods available for preprocessing."""
    OTSU = "Otsu"
    TRIANGLE = "Triangle"
    ISODATA = "Isodata"
    MEAN = "Mean"
    MINIMUM = "Minimum"
    LI = "Li"
    YEN = "Yen"
    MANUAL = "Manual"


class PreprocessingOptions:
    """Options for image preprocessing."""
    
    def __init__(
        self,
        apply_tubeness: bool = False,
        tubeness_sigma: float = 1.0,
        apply_frangi: bool = False,
        frangi_sigma_range: Tuple[float, float] = (1.0, 10.0),
        frangi_beta: float = 0.5,
        frangi_gamma: float = 15.0,
        apply_threshold: bool = False,
        threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
        threshold_value: Optional[float] = None,
        apply_gaussian: bool = False,
        gaussian_sigma: float = 1.0,
    ):
        """
        Initialize preprocessing options.

        Parameters
        ----------
        apply_tubeness : bool, default False
            Whether to apply tubeness-style fiber enhancement.
        tubeness_sigma : float, default 1.0
            Sigma value used by the tubeness filter.
        apply_frangi : bool, default False
            Whether to apply the Frangi vesselness filter.
        frangi_sigma_range : tuple of float, default (1.0, 10.0)
            Minimum and maximum sigma values used by the Frangi filter.
        frangi_beta : float, default 0.5
            Frangi beta correction parameter.
        frangi_gamma : float, default 15.0
            Frangi gamma correction parameter.
        apply_threshold : bool, default False
            Whether to threshold the image after enhancement.
        threshold_method : ThresholdMethod, default ThresholdMethod.OTSU
            Thresholding algorithm to use when thresholding is enabled.
        threshold_value : float, optional
            Manual threshold value used when ``threshold_method`` is
            ``ThresholdMethod.MANUAL``.
        apply_gaussian : bool, default False
            Whether to smooth the image before enhancement.
        gaussian_sigma : float, default 1.0
            Sigma value used for Gaussian smoothing.
        """
        self.apply_tubeness = apply_tubeness
        self.tubeness_sigma = tubeness_sigma
        self.apply_frangi = apply_frangi
        self.frangi_sigma_range = frangi_sigma_range
        self.frangi_beta = frangi_beta
        self.frangi_gamma = frangi_gamma
        self.apply_threshold = apply_threshold
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.apply_gaussian = apply_gaussian
        self.gaussian_sigma = gaussian_sigma


def load_image_with_bioformats(
    file_path: str,
    use_napari_imagej: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Load image using Bio-Formats (via aicsimageio or napari-imagej).
    
    Parameters
    ----------
    file_path : str
        Path to image file
    use_napari_imagej : bool, default False
        If True, use napari-imagej bridge. Otherwise use aicsimageio.
        
    Returns
    -------
    Tuple[np.ndarray, dict]
        Image data and metadata dictionary
    """
    if use_napari_imagej and HAS_IMAGEJ:
        # Use napari-imagej bridge to Fiji
        try:
            ij = napari_imagej.init()
            reader = ij.scifio().datasetIO().open(file_path)
            image_data = np.array(reader.data())
            metadata = {"source": "napari-imagej", "shape": image_data.shape}
            return image_data, metadata
        except Exception as e:
            print(f"napari-imagej loading failed: {e}, falling back to aicsimageio")
    
    if HAS_AICSIO:
        # Use aicsimageio (pure Python Bio-Formats)
        try:
            reader = aicsimageio.AICSImage(file_path)
            image_data = reader.get_image_data("YX")  # Get first 2D slice
            if image_data.ndim > 2:
                image_data = image_data[0]  # Take first slice
            metadata = {
                "source": "aicsimageio",
                "shape": image_data.shape,
                "dims": reader.dims,
            }
            return image_data, metadata
        except Exception as e:
            print(f"aicsimageio loading failed: {e}, falling back to skimage")
    
    # Fallback to skimage
    image_data = io.imread(file_path)
    if image_data.ndim > 2:
        image_data = image_data[0]
    metadata = {"source": "skimage", "shape": image_data.shape}
    return image_data, metadata


def apply_tubeness(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Tubeness filter for fiber enhancement.
    
    Uses Meijering filter from skimage which is similar to Tubeness.
    
    Parameters
    ----------
    image : np.ndarray
        Input grayscale image
    sigma : float, default 1.0
        Standard deviation for Gaussian kernel
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    if image.ndim != 2:
        raise ValueError("Tubeness filter requires 2D grayscale image")
    
    # Meijering filter is similar to Tubeness
    return meijering(image, sigmas=sigma, black_ridges=False)


def apply_frangi(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (1.0, 10.0),
    beta: float = 0.5,
    gamma: float = 15.0
) -> np.ndarray:
    """
    Apply Frangi vesselness filter for fiber enhancement.
    
    Parameters
    ----------
    image : np.ndarray
        Input grayscale image
    sigma_range : Tuple[float, float], default (1.0, 10.0)
        Range of sigmas to use
    beta : float, default 0.5
        Frangi correction constant
    gamma : float, default 15.0
        Frangi correction constant
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    if image.ndim != 2:
        raise ValueError("Frangi filter requires 2D grayscale image")
    
    return frangi(
        image,
        sigmas=np.arange(sigma_range[0], sigma_range[1], 1.0),
        beta=beta,
        gamma=gamma,
        black_ridges=False
    )


def apply_threshold(
    image: np.ndarray,
    method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold_value: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Apply automatic thresholding to image.
    
    Parameters
    ----------
    image : np.ndarray
        Input grayscale image
    method : ThresholdMethod, default OTSU
        Threshold method to use
    threshold_value : float, optional
        Manual threshold value (only used if method is MANUAL)
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Binary thresholded image and threshold value used
    """
    if image.ndim != 2:
        raise ValueError("Thresholding requires 2D grayscale image")
    
    if method == ThresholdMethod.MANUAL:
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for MANUAL method")
        thresh = threshold_value
    elif method == ThresholdMethod.OTSU:
        thresh = threshold_otsu(image)
    elif method == ThresholdMethod.TRIANGLE:
        thresh = threshold_triangle(image)
    elif method == ThresholdMethod.ISODATA:
        thresh = threshold_isodata(image)
    elif method == ThresholdMethod.MEAN:
        thresh = np.mean(image)
    elif method == ThresholdMethod.MINIMUM:
        thresh = threshold_minimum(image)
    elif method == ThresholdMethod.LI:
        thresh = threshold_li(image)
    elif method == ThresholdMethod.YEN:
        thresh = threshold_yen(image)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    binary = image > thresh
    return binary.astype(np.uint8) * 255, thresh


def preprocess_image(
    image: np.ndarray,
    options: PreprocessingOptions
) -> np.ndarray:
    """
    Apply preprocessing pipeline to image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    options : PreprocessingOptions
        Preprocessing options
        
    Returns
    -------
    np.ndarray
        Preprocessed image
    """
    result = image.copy()
    
    # Ensure 2D grayscale
    if result.ndim > 2:
        if result.ndim == 3 and result.shape[2] == 3:
            # Convert RGB to grayscale
            result = 0.2125 * result[:, :, 0] + \
                    0.7154 * result[:, :, 1] + \
                    0.0721 * result[:, :, 2]
        else:
            result = result[0]  # Take first slice
    
    # Normalize to 0-1 range
    if result.max() > 1.0:
        result = result.astype(np.float32) / 255.0
    
    # Apply Gaussian smoothing if requested
    if options.apply_gaussian:
        result = filters.gaussian(result, sigma=options.gaussian_sigma)
    
    # Apply Tubeness filter
    if options.apply_tubeness:
        result = apply_tubeness(result, sigma=options.tubeness_sigma)
        # Renormalize
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    # Apply Frangi filter
    if options.apply_frangi:
        result = apply_frangi(
            result,
            sigma_range=options.frangi_sigma_range,
            beta=options.frangi_beta,
            gamma=options.frangi_gamma
        )
        # Renormalize
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    # Apply thresholding
    if options.apply_threshold:
        binary, thresh = apply_threshold(
            result,
            method=options.threshold_method,
            threshold_value=options.threshold_value
        )
        result = binary.astype(np.float32) / 255.0
    
    # Ensure proper type and range
    result = np.clip(result, 0, 1)
    if result.max() <= 1.0:
        result = (result * 255).astype(np.uint8)
    
    return result
