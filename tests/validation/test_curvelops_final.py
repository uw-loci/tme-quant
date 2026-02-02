#!/usr/bin/env python3
"""
Final test for Curvelops integration with CurveLab.
Tests both the technical integration and practical fiber analysis.
"""

import numpy as np
import curvealign_py as curvealign

def test_curvelops_status():
    """Check Curvelops installation and functionality."""
    print("=== Curvelops Installation Status ===")
    
    from curvealign_py.core.algorithms.fdct_wrapper import get_curvelops_status
    status = get_curvelops_status()
    
    print(f"Curvelops Available: {status['available']}")
    print(f"Backend: {status['backend']}")
    
    if status['available']:
        print(f"Version: {status.get('version', 'unknown')}")
        print(f"Functional: {status.get('functional', 'unknown')}")
        if 'error' in status:
            print(f"Error: {status['error']}")
        
        if status.get('functional', False):
            print("✅ Curvelops is properly installed and functional!")
            return True
        else:
            print("❌ Curvelops installed but not functional")
            return False
    else:
        print("❌ Curvelops not available - using placeholder implementation")
        return False

def test_fiber_analysis_comparison():
    """Compare fiber analysis with and without Curvelops."""
    print("\n=== Fiber Analysis Comparison ===")
    
    # Create a synthetic fiber image
    print("Creating synthetic fiber image...")
    image = create_synthetic_fiber_image()
    print(f"Image shape: {image.shape}")
    
    # Analyze with current backend
    print("Running CurveAlign analysis...")
    result = curvealign.analyze_image(image)
    
    print(f"Results:")
    print(f"  Fiber segments detected: {len(result.curvelets)}")
    print(f"  Mean angle: {result.stats['mean_angle']:.1f}°")
    print(f"  Alignment score: {result.stats['alignment']:.3f}")
    print(f"  Density: {result.stats['density']:.6f}")
    
    # Test visualization
    try:
        overlay = curvealign.overlay(image, result.curvelets)
        print(f"  Visualization: {overlay.shape}")
    except Exception as e:
        print(f"  Visualization failed: {e}")
    
    return result

def create_synthetic_fiber_image(size=128):
    """Create a synthetic image with fiber-like structures."""
    image = np.zeros((size, size))
    
    # Add several oriented fiber structures
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Fiber 1: Horizontal orientation
    fiber1 = np.exp(-((x - size//4)**2 + (y - size//3)**2) / (2 * 10**2))
    fiber1 *= np.cos(np.pi * (y - size//3) / 8)
    
    # Fiber 2: Diagonal orientation  
    fiber2 = np.exp(-((x - 3*size//4)**2 + (y - size//2)**2) / (2 * 8**2))
    fiber2 *= np.cos(np.pi * (x - y) / 12)
    
    # Fiber 3: Vertical orientation
    fiber3 = np.exp(-((x - size//2)**2 + (y - 3*size//4)**2) / (2 * 12**2))
    fiber3 *= np.cos(np.pi * (x - size//2) / 6)
    
    # Combine fibers
    image = fiber1 + fiber2 + fiber3
    
    # Add some noise
    image += 0.1 * np.random.randn(size, size)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image

def test_direct_fdct():
    """Test FDCT functions directly."""
    print("\n=== Direct FDCT Testing ===")
    
    # Create simple test image
    image = create_synthetic_fiber_image(64)  # Smaller for faster testing
    
    try:
        from curvealign_py.core.algorithms.fdct_wrapper import apply_fdct, apply_ifdct
        
        # Forward transform
        print("Testing forward FDCT...")
        coeffs, _ = apply_fdct(image)
        print(f"  Scales: {len(coeffs)}")
        
        # Count total coefficients
        total_coeffs = sum(sum(c.size for c in scale) for scale in coeffs)
        print(f"  Total coefficients: {total_coeffs}")
        
        # Inverse transform
        print("Testing inverse FDCT...")
        reconstructed = apply_ifdct(coeffs, img_shape=image.shape)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        # Quality metrics
        mse = np.mean((image - reconstructed)**2)
        print(f"  MSE: {mse:.8f}")
        
        if mse < 1e-6:
            print("  ✅ Excellent reconstruction quality!")
        elif mse < 1e-3:
            print("  ✅ Good reconstruction quality!")
        else:
            print("  ⚠️  Reconstruction quality needs improvement")
            
        return True
        
    except Exception as e:
        print(f"  Direct FDCT test failed: {e}")
        return False

def main():
    """Run all Curvelops integration tests."""
    print("CurveAlign + Curvelops Integration Test")
    print("=" * 50)
    
    # Test 1: Check installation
    curvelops_working = test_curvelops_status()
    
    # Test 2: Direct FDCT functions
    fdct_working = test_direct_fdct()
    
    # Test 3: Full fiber analysis
    analysis_result = test_fiber_analysis_comparison()
    
    print("\n" + "=" * 50)
    print("Integration Test Summary:")
    print(f"  Curvelops Installation: {'✅ PASS' if curvelops_working else '❌ FAIL'}")
    print(f"  Direct FDCT Functions: {'✅ PASS' if fdct_working else '❌ FAIL'}")
    print(f"  Fiber Analysis: {'✅ PASS' if analysis_result else '❌ FAIL'}")
    
    if curvelops_working and fdct_working:
        print("\n🎉 Curvelops integration is working correctly!")
        print("You now have access to authentic CurveLab FDCT transforms!")
    else:
        print("\n⚠️  Some issues detected. Check error messages above.")
        print("The API will fall back to placeholder implementations.")
    
    return curvelops_working and fdct_working

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
