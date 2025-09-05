import numpy as np
try:
    from curvelops import FDCT2D, curveshow, fdct2d_wrapper  # type: ignore
    HAS_CURVELETS = True
except Exception:
    HAS_CURVELETS = False
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import gaussian
import random
from enum import Enum
from typing import Tuple

def run_analysis(
    image_path: str,
    image_name: str,
    boundary_type: Enum,
    curve_threshold: float,
    distance_boundary: int,
    output_options: dict,
    advanced_params: dict  # Added advanced parameters
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Main analysis function that returns two images and measurements"""
    print("\nRunning analysis with parameters:")
    print(f"Image: {image_name} ({image_path})")
    print(f"Boundary type: {boundary_type.value}")
    print(f"Curvelets threshold: {curve_threshold}")
    print(f"Distance to boundary: {distance_boundary}")
    print("Output options:")
    for option, enabled in output_options.items():
        print(f"  - {option}: {'Enabled' if enabled else 'Disabled'}")
    
    # Print advanced parameters
    print("\nAdvanced parameters:")
    for param, value in advanced_params.items():
        print(f"  - {param}: {value}")
    

    # Add curvelets analysis logic here
    # to be added in the future


    # For now, we will mock the analysis results
    # Load the image
    image_data = imread(image_path)
    
    # For multi-page TIFFs, take the first page
    if image_data.ndim > 2 and image_data.shape[0] > 1:
        image_data = image_data[0]
    
    # Generate mock overlay image (convert to RGB and add green overlay)
    if image_data.ndim == 2:
        rgb_image = gray2rgb(image_data)
    else:
        rgb_image = image_data.copy()
    
    # Create a mock overlay (green highlights)
    overlay = np.zeros_like(rgb_image)
    overlay[:, :, 1] = 200  # Green channel
    
    # Apply overlay only to high-intensity areas
    if image_data.ndim == 2:
        # Use advanced_param1 to adjust the percentile threshold
        percentile = 80 + (advanced_params["advanced_param1"] * 20)
        mask = image_data > np.percentile(image_data, percentile)
        for c in range(3):
            rgb_image[:, :, c] = np.where(mask, 
                                         rgb_image[:, :, c] * 0.5 + overlay[:, :, c] * 0.5,
                                         rgb_image[:, :, c])
    else:
        # For RGB images, just add the green overlay
        rgb_image = np.clip(rgb_image * 0.7 + overlay * 0.3, 0, 255).astype(np.uint8)
    
    # Generate mock heatmap (Gaussian smoothed version)
    if image_data.ndim == 2:
        # Use advanced_param2 to adjust sigma
        sigma = 5 * advanced_params["advanced_param2"]
        heatmap = gaussian(image_data, sigma=sigma)
    else:
        # For RGB, convert to grayscale first
        gray_image = 0.2125 * image_data[:, :, 0] + \
                    0.7154 * image_data[:, :, 1] + \
                    0.0721 * image_data[:, :, 2]
        # Use advanced_param2 to adjust sigma
        sigma = 5 * advanced_params["advanced_param2"]
        heatmap = gaussian(gray_image, sigma=sigma)
    
    # Create mock measurements using iterations parameter
    iterations = advanced_params["iterations"]
    measurements = pd.DataFrame({
        'Feature': ['Curve Density', 'Alignment Score', 'Boundary Proximity', 
                   'Average Intensity', 'Max Intensity', 'Iterations Used'],
        'Value': [
            random.uniform(0.1, 0.9),  # Curve Density
            random.uniform(0.5, 1.0),   # Alignment Score
            random.uniform(0.0, 1.0),   # Boundary Proximity
            np.mean(image_data),        # Average Intensity
            np.max(image_data),         # Max Intensity
            iterations                  # Iterations parameter
        ]
    })
    
    return rgb_image, heatmap, measurements
