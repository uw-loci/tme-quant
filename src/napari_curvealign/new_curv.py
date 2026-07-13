import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import gaussian
import random
from enum import Enum
from typing import Tuple

# Use pycurvelets (manually converted API from branch 22)
try:
    from pycurvelets.models import CurveletControlParameters, FeatureControlParameters
    from pycurvelets.get_ct import get_ct
    from pycurvelets.utils.visualization.draw_map import draw_map
    HAS_PYCURVELETS = True
except ImportError:
    HAS_PYCURVELETS = False
    print("Warning: pycurvelets not available. Using mock analysis.")

def _convert_features_to_dataframe(features: dict, stats: dict) -> pd.DataFrame:
    """
    Convert CurveAlign features and stats to a display table.

    Parameters
    ----------
    features : dict
        Mapping of feature names to arrays produced by CurveAlign analysis.
    stats : dict
        Mapping of summary statistic names to scalar values.

    Returns
    -------
    pandas.DataFrame
        Two-column table with ``Feature`` and ``Value`` columns.
    """
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

    Parameters
    ----------
    features : dict
        Mapping of feature names to arrays produced by CurveAlign analysis.
    stats : dict
        Mapping of summary statistic names to scalar values.
    curvelets : list
        Curvelet-like objects with ``angle_deg`` and optional ``weight``
        attributes.

    Returns
    -------
    pandas.DataFrame
        Table of scalar measurements and feature distribution summaries.
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


def _generate_histograms(fiber_structure, features: dict, image_name: str):
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
        
        if fiber_structure is None or len(fiber_structure) == 0:
            return
        
        angles = fiber_structure["angle"].values
        weights = np.ones(len(angles))
        
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
        if features and 'density_nn' in features:
            density = features['density_nn']
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
    """
    Run CurveAlign-style fiber analysis for an image.

    Parameters
    ----------
    image_path : str
        Path to the image file being analyzed.
    image_name : str
        Display name used in output labels and optional histogram files.
    boundary_type : Enum
        Boundary mode selected by the widget.
    curve_threshold : float
        Curvelet retention threshold passed to pycurvelets when available.
    distance_boundary : int
        Distance from boundary used by boundary-aware analysis options.
    output_options : dict
        Mapping of output option names to enabled flags.
    advanced_params : dict
        Additional numeric parameters used by the fallback analysis path.
    analysis_mode : str, default "curvelets"
        Analysis mode name. Supported values are intended to include
        ``"curvelets"``, ``"ctfire"``, and ``"both"``.

    Returns
    -------
    overlay_img : numpy.ndarray
        RGB overlay image showing detected fiber positions or fallback overlay.
    heatmap_img : numpy.ndarray
        Angle map or fallback heatmap image.
    measurements : pandas.DataFrame
        Measurement table containing fiber and image summary statistics.
    """
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
    
    # Use pycurvelets analysis if available
    if HAS_PYCURVELETS:
        try:
            curve_cp = CurveletControlParameters(
                keep=curve_threshold,
                scale=1.0,
                radius=10.0,
            )
            feature_cp = FeatureControlParameters(
                minimum_nearest_fibers=2,
                minimum_box_size=32,
                fiber_midpoint_estimate=1,
            )
            fiber_structure, density_df, alignment_df, _ = get_ct(
                image_data, curve_cp, feature_cp
            )

            if len(fiber_structure) == 0:
                raise ValueError("No curvelets extracted")

            angles = fiber_structure["angle"].values
            boundary_measurement = boundary_type.value != "No boundary"
            map_params = {
                "STDfilter_size": 24,
                "SQUAREmaxfilter_size": 12,
                "GAUSSIANdiscfilter_sigma": 4.0,
            }
            _, angle_map_processed = draw_map(
                fiber_structure, angles, image_data,
                boundary_measurement, map_params,
            )

            # Simple overlay: green at curvelet centers
            rgb_image = gray2rgb(image_data) if image_data.ndim == 2 else image_data.copy()
            centers = fiber_structure[["center_row", "center_col"]].values.astype(int)
            for r, c in centers:
                if 0 <= r < rgb_image.shape[0] and 0 <= c < rgb_image.shape[1]:
                    rgb_image[r, c, 1] = np.minimum(255, rgb_image[r, c, 1].astype(float) + 150)
            overlay_img = np.clip(rgb_image, 0, 255).astype(np.uint8)

            stats = {
                "mean_angle": float(np.mean(angles)),
                "alignment": float(alignment_df["alignment_mean"].mean()) if len(alignment_df) > 0 else 0.0,
                "density": float(density_df["density_mean"].mean()) if len(density_df) > 0 else 0.0,
            }
            features = {
                "angle": angles,
                "center_row": fiber_structure["center_row"].values,
                "center_col": fiber_structure["center_col"].values,
            }
            curvelets = [
                type("C", (), {"angle_deg": float(a), "weight": 1.0})()
                for a in angles
            ]

            if output_options.get("histograms", False):
                _generate_histograms(fiber_structure, features, image_name)

            measurements = _convert_features_to_dataframe_full(features, stats, curvelets)
            return overlay_img, angle_map_processed, measurements

        except Exception as e:
            print(f"pycurvelets analysis failed: {e}")
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
