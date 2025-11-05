"""
Quick test script for napari-curvealign plugin.
Run with: python examples/test_plugin.py
"""

import napari
import numpy as np
from skimage import data, filters
import sys

def create_synthetic_fibers(size=512):
    """Create a synthetic fiber-like image for testing."""
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Create oriented structures
    fibers = np.sin(X) * np.cos(Y/2) + 0.5 * np.cos(X/1.5) * np.sin(Y)
    
    # Add some noise
    fibers += np.random.rand(size, size) * 0.2
    
    # Normalize to 0-1
    fibers = (fibers - fibers.min()) / (fibers.max() - fibers.min())
    
    # Apply some smoothing
    fibers = filters.gaussian(fibers, sigma=1.5)
    
    return fibers.astype(np.float32)


def main():
    """Launch napari with test image and CurveAlign plugin."""
    
    print("=" * 60)
    print("CurveAlign Napari Plugin Test")
    print("=" * 60)
    
    # Create viewer
    viewer = napari.Viewer()
    
    # Create test images
    print("\n📊 Creating synthetic fiber image...")
    fiber_image = create_synthetic_fibers(size=512)
    
    # Add images to viewer
    viewer.add_image(fiber_image, name='Synthetic Fibers', colormap='gray')
    
    # Add the plugin widget directly
    print("🔌 Loading CurveAlign plugin widget...")
    try:
        # Import and instantiate the widget directly
        from napari_curvealign.widget import CurveAlignWidget
        
        # Create widget instance with viewer
        widget_instance = CurveAlignWidget(viewer)
        
        # Add to viewer as dock widget
        viewer.window.add_dock_widget(
            widget_instance,
            name='CurveAlign',
            area='right'
        )
        print("✅ Plugin loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load plugin: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure plugin is installed: pip install -e '.[napari]'")
        print("2. Check napari version: python -c 'import napari; print(napari.__version__)'")
        print("3. Check widget import:")
        print("   python -c 'from napari_curvealign.widget import CurveAlignWidget'")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🧪 Testing Instructions:")
    print("=" * 60)
    print("\n1. MAIN TAB:")
    print("   - Select 'Curvelets' mode")
    print("   - Click 'Run Analysis'")
    print("   - Check for results (angle histogram, heatmap)")
    
    print("\n2. PREPROCESSING TAB:")
    print("   - Try 'Apply Gaussian Smoothing'")
    print("   - Try 'Apply Frangi Filter'")
    print("   - Try 'Apply Thresholding' with Otsu method")
    print("   - Run analysis again to see effect")
    
    print("\n3. ROI MANAGER TAB:")
    print("   - Click 'Create Rectangle'")
    print("   - Draw a rectangle on the image")
    print("   - Click 'Create Polygon'")
    print("   - Draw a polygon (Enter to finish)")
    print("   - Select ROI and click 'Save ROI'")
    print("   - Try saving in different formats:")
    print("     * JSON (.json) - recommended")
    print("     * Fiji ROI (.zip)")
    print("     * CSV (.csv)")
    print("   - Delete ROIs and reload them")
    print("   - Select ROI and click 'Analyze Selected ROI'")
    print("   - Click 'Show ROI Table' to see results")
    
    print("\n4. VISUALIZATION:")
    print("   - After analysis, check viewer layers")
    print("   - Toggle visibility of angle heatmap")
    print("   - Examine HSV colormap (colors = angles)")
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("=" * 60)
    print("- Use mouse wheel to zoom")
    print("- Hold Shift and drag to pan")
    print("- Check terminal for debug messages")
    print("- Layer controls on the left side")
    
    print("\n▶️  Starting napari...")
    print("=" * 60 + "\n")
    
    # Run napari
    napari.run()


if __name__ == "__main__":
    main()

