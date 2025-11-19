import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import gaussian
import random
from enum import Enum
from typing import Tuple

# Import the new CurveAlign API
try:
    import curvealign_py as curvealign
    HAS_CURVEALIGN = True
except ImportError:
    HAS_CURVEALIGN = False
    print("Warning: curvealign_py not available. Using mock analysis.")

# Legacy curvelops import for backward compatibility
try:
    from curvelops import FDCT2D, curveshow, fdct2d_wrapper  # type: ignore
    HAS_CURVELETS = True
except Exception:
    HAS_CURVELETS = False

def _convert_features_to_dataframe(features: dict, stats: dict) -> pd.DataFrame:
    """Convert CurveAlign features and stats to a DataFrame for display."""
    measurements = []
    
    # Add summary statistics
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            measurements.append({
                'Feature': key.replace('_', ' ').title(),
                'Value': float(value)
            })
    
    # Add feature array summaries (mean values)
    for key, array in features.items():
        if isinstance(array, np.ndarray) and array.size > 0:
            measurements.append({
                'Feature': f'{key.replace("_", " ").title()} (Mean)',
                'Value': float(np.mean(array))
            })
    
    return pd.DataFrame(measurements)


def _convert_features_to_dataframe_full(
    features: dict, 
    stats: dict, 
    curvelets: list
) -> pd.DataFrame:
    """
    Convert CurveAlign features and stats to a comprehensive DataFrame matching MATLAB output.
    
    This includes all ~30 features from MATLAB CurveAlign:
    - Individual fiber features (angle, weight, position)
    - Density features (nearest neighbors: 2, 4, 8, 16; box sizes: 32, 64, 128)
    - Alignment features (nearest neighbors: 2, 4, 8, 16; box sizes: 32, 64, 128)
    - Boundary features (if available)
    - Circular statistics
    """
    measurements = []
    
    # Basic statistics
    n_curvelets = len(curvelets)
    measurements.append({'Feature': 'Number of Curvelets', 'Value': n_curvelets})
    
    if n_curvelets > 0:
        # Extract angles for circular statistics
        angles = np.array([c.angle_deg for c in curvelets])
        angles_rad = np.radians(angles)
        
        # Circular statistics (matching MATLAB CircStat)
        # Circular mean
        complex_angles = np.exp(1j * 2 * angles_rad)  # Factor of 2 for fiber symmetry
        mean_resultant = np.mean(complex_angles)
        circ_mean = np.angle(mean_resultant) / 2.0 * 180.0 / np.pi
        measurements.append({'Feature': 'Circular Mean Angle (deg)', 'Value': circ_mean % 180})
        
        # Circular variance (1 - R, where R is mean resultant length)
        R = np.abs(mean_resultant)
        circ_var = 1.0 - R
        measurements.append({'Feature': 'Circular Variance', 'Value': circ_var})
        
        # Circular standard deviation
        circ_std = np.sqrt(-2 * np.log(R)) * 180.0 / np.pi
        measurements.append({'Feature': 'Circular Std Dev (deg)', 'Value': circ_std})
        
        # Mean resultant length (alignment metric)
        measurements.append({'Feature': 'Mean Resultant Length (R)', 'Value': float(R)})
        
        # Standard statistics
        measurements.append({'Feature': 'Mean Angle (deg)', 'Value': float(np.mean(angles))})
        measurements.append({'Feature': 'Std Angle (deg)', 'Value': float(np.std(angles))})
        
        # Weight statistics
        weights = np.array([c.weight or 1.0 for c in curvelets])
        measurements.append({'Feature': 'Mean Weight', 'Value': float(np.mean(weights))})
        measurements.append({'Feature': 'Total Weight', 'Value': float(np.sum(weights))})
    
    # Add summary statistics from stats dict
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            feature_name = key.replace('_', ' ').title()
            # Avoid duplicates
            if not any(m['Feature'] == feature_name for m in measurements):
                measurements.append({
                    'Feature': feature_name,
                    'Value': float(value)
                })
    
    # Add feature array summaries with full statistics
    for key, array in features.items():
        if isinstance(array, np.ndarray) and array.size > 0:
            feature_name = key.replace('_', ' ').title()
            measurements.append({
                'Feature': f'{feature_name} (Mean)',
                'Value': float(np.mean(array))
            })
            measurements.append({
                'Feature': f'{feature_name} (Std)',
                'Value': float(np.std(array))
            })
            measurements.append({
                'Feature': f'{feature_name} (Min)',
                'Value': float(np.min(array))
            })
            measurements.append({
                'Feature': f'{feature_name} (Max)',
                'Value': float(np.max(array))
            })
    
    # Add boundary metrics if available
    if 'boundary_metrics' in stats or any('boundary' in k.lower() for k in stats.keys()):
        for key, value in stats.items():
            if 'boundary' in key.lower() and isinstance(value, (int, float, np.integer, np.floating)):
                measurements.append({
                    'Feature': key.replace('_', ' ').title(),
                    'Value': float(value)
                })
    
    return pd.DataFrame(measurements)


def _generate_histograms(result, image_name: str):
    """
    Generate histogram visualizations matching MATLAB CurveAlign output.
    
    Creates histograms for:
    - Angle distribution
    - Density distribution
    - Alignment distribution
    """
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        if not result.curvelets:
            return
        
        # Extract data
        angles = np.array([c.angle_deg for c in result.curvelets])
        weights = np.array([c.weight or 1.0 for c in result.curvelets])
        
        # Create output directory
        output_dir = Path("curvealign_output")
        output_dir.mkdir(exist_ok=True)
        
        # Angle histogram (0-180 degrees, matching MATLAB)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Angle histogram
        axes[0].hist(angles, bins=36, range=(0, 180), edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Angle (degrees)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Angle Distribution - {image_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Weight histogram
        if len(weights) > 0:
            axes[1].hist(weights, bins=50, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Weight')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Weight Distribution - {image_name}')
            axes[1].grid(True, alpha=0.3)
        
        # Density histogram (if available)
        if 'density_nn' in result.features:
            density = result.features['density_nn']
            density = density[density > 0]  # Remove zeros
            if len(density) > 0:
                axes[2].hist(density, bins=50, edgecolor='black', alpha=0.7)
                axes[2].set_xlabel('Density')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title(f'Density Distribution - {image_name}')
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / f"{image_name}_histograms.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Histograms saved to: {hist_path}")
        
    except ImportError:
        print("matplotlib not available, skipping histogram generation")
    except Exception as e:
        print(f"Histogram generation failed: {e}")

def run_analysis(
    image_path: str,
    image_name: str,
    boundary_type: Enum,
    curve_threshold: float,
    distance_boundary: int,
    output_options: dict,
    advanced_params: dict,  # Added advanced parameters
    analysis_mode: str = "curvelets"  # "curvelets", "ctfire", or "both"
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
    

    # Load the image
    image_data = imread(image_path)
    
    # For multi-page TIFFs, take the first page
    if image_data.ndim > 2 and image_data.shape[0] > 1:
        image_data = image_data[0]
    
    # Use real CurveAlign analysis if available
    if HAS_CURVEALIGN:
        try:
            # Create CurveAlign options from parameters
            options = curvealign.CurveAlignOptions(
                keep=curve_threshold,
                dist_thresh=distance_boundary,
            )
            
            # Handle boundary if specified
            boundary = None
            if boundary_type.value != "No boundary":
                # TODO: Load boundary from file if TIFF boundary
                # For now, boundary analysis will be skipped
                pass
            
            # Determine analysis mode
            mode = analysis_mode
            if mode == "both":
                # For "both", run curvelets mode first
                # Note: Full "both" mode would require running both analyses and combining results
                # For now, we default to curvelets as it's the primary mode
                print("Note: 'Both' mode not fully implemented. Using curvelets mode.")
                mode = "curvelets"
            
            # Run CurveAlign analysis with specified mode
            result = curvealign.analyze_image(
                image_data, 
                boundary=boundary,
                mode=mode,
                options=options
            )
            
            # Create overlay visualization
            overlay_img = curvealign.overlay(image_data, result.curvelets)
            
            # Create angle map visualization
            angle_map_raw, angle_map_processed = curvealign.angle_map(image_data, result.curvelets)
            
            # Generate histograms if requested
            if output_options.get("histograms", False):
                _generate_histograms(result, image_name)
            
            # Convert features to measurements DataFrame with full feature set
            measurements = _convert_features_to_dataframe_full(result.features, result.stats, result.curvelets)
            
            return overlay_img, angle_map_processed, measurements
            
        except Exception as e:
            print(f"CurveAlign analysis failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to mock analysis...")
    
    # Fallback to mock analysis if CurveAlign not available or failed
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
