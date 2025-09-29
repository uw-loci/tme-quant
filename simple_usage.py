"""
CurveAlign API Usage Examples
Demonstrates basic and advanced usage of the modern CurveAlign Python API.
"""

import numpy as np
import curvealign_py as curvealign
import ctfire_py as ctfire

def basic_analysis_example():
    """Basic fiber analysis with CurveAlign API."""
    print("=== Basic CurveAlign Analysis ===")
    
    # Create test image (replace with: io.imread('your_image.tif'))
    image = np.random.rand(256, 256)
    print(f"Image shape: {image.shape}")
    
    # Run core analysis
    print("Running curvelet-based analysis...")
    result = curvealign.analyze_image(image)
    
    # Display results
    print(f"\nResults:")
    print(f"  Fiber segments detected: {len(result.curvelets)}")
    print(f"  Mean angle: {result.stats['mean_angle']:.1f}°")
    print(f"  Alignment score: {result.stats['alignment']:.3f}")
    print(f"  Fiber density: {result.stats['density']:.6f}")
    
    return result

def ctfire_analysis_example():
    """Individual fiber extraction with CT-FIRE integration."""
    print("\n=== CT-FIRE Analysis ===")
    
    image = np.random.rand(256, 256)
    
    # Option 1: Use unified CurveAlign interface
    print("Using unified interface (CT-FIRE mode)...")
    result_unified = curvealign.analyze_image(image, mode="ctfire")
    print(f"  Unified interface: {len(result_unified.curvelets)} features")
    
    # Option 2: Use CT-FIRE directly
    print("Using CT-FIRE API directly...")
    result_ctfire = ctfire.analyze_image(image)
    print(f"  Direct CT-FIRE: {len(result_ctfire.fibers)} fibers")
    print(f"  Network analysis: {len(result_ctfire.network.intersections)} intersections")
    
    return result_unified, result_ctfire

def visualization_example():
    """Create visualizations with different backends."""
    print("\n=== Visualization Examples ===")
    
    image = np.random.rand(128, 128)
    result = curvealign.analyze_image(image)
    
    # Create overlay visualization
    try:
        overlay = curvealign.overlay(image, result.curvelets)
        print(f"Overlay created: {overlay.shape}")
        
        # Create angle map
        angle_map_raw, angle_map_processed = curvealign.angle_map(image, result.curvelets)
        print(f"Angle maps created: {angle_map_processed.shape}")
        
    except ImportError as e:
        print(f"Visualization backend not available: {e}")

def batch_processing_example():
    """Process multiple images efficiently."""
    print("\n=== Batch Processing ===")
    
    # Create sample images
    images = [np.random.rand(128, 128) for _ in range(3)]
    print(f"Processing {len(images)} images...")
    
    # Batch analysis with curvelet method
    results_curvelets = curvealign.batch_analyze(images, mode="curvelets")
    print(f"Curvelet analysis: {len(results_curvelets)} results")
    
    # Batch analysis with CT-FIRE method
    results_ctfire = curvealign.batch_analyze(images, mode="ctfire")
    print(f"CT-FIRE analysis: {len(results_ctfire)} results")
    
    # Summary statistics
    total_features = sum(len(r.curvelets) for r in results_curvelets)
    print(f"Total features detected: {total_features}")

def advanced_options_example():
    """Demonstrate custom analysis parameters."""
    print("\n=== Advanced Configuration ===")
    
    image = np.random.rand(256, 256)
    
    # Configure custom analysis options
    options = curvealign.CurveAlignOptions(
        keep=0.002,                    # Stricter coefficient threshold
        dist_thresh=150.0,             # Boundary analysis distance
        minimum_nearest_fibers=6,      # Feature computation requirements
        minimum_box_size=24            # Local analysis window size
    )
    
    result = curvealign.analyze_image(image, options=options)
    print(f"Custom analysis: {len(result.curvelets)} features with strict parameters")
    
    # CT-FIRE with custom options
    ctfire_options = ctfire.CTFireOptions(
        run_mode="ctfire",
        thresh_flen=20.0,             # Minimum fiber length
        sigma_im=1.5                  # Image smoothing
    )
    
    ctfire_result = ctfire.analyze_image(image, ctfire_options)
    print(f"Custom CT-FIRE: {len(ctfire_result.fibers)} fibers")

def main():
    """Run all examples to demonstrate API capabilities."""
    print("CurveAlign Python API - Usage Examples")
    print("=" * 50)
    
    # Run examples
    basic_result = basic_analysis_example()
    ctfire_results = ctfire_analysis_example()
    visualization_example()
    batch_processing_example()
    advanced_options_example()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nNext steps:")
    print("1. Load your own images with: from skimage import io; image = io.imread('path.tif')")
    print("2. Explore visualization backends: matplotlib, napari, imagej")
    print("3. Customize analysis with CurveAlignOptions and CTFireOptions")
    print("4. Process datasets with batch_analyze() for efficiency")

if __name__ == "__main__":
    main()
