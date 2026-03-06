"""
Test script for FIRE 2D fiber extraction
"""

import numpy as np
import sys

sys.path.insert(0, "CPP")

from fire_2d_ang1 import fire_2d_ang1, create_default_params


def create_synthetic_fiber_image(size=(256, 256), num_fibers=5):
    """
    Create a synthetic image with fiber-like structures for testing

    Args:
        size: Image dimensions (height, width)
        num_fibers: Number of fibers to generate

    Returns:
        Synthetic fiber image
    """
    image = np.zeros(size, dtype=np.float32)

    for _ in range(num_fibers):
        # Random start and end points
        y1, x1 = np.random.randint(20, size[0] - 20), np.random.randint(
            20, size[1] - 20
        )
        y2, x2 = np.random.randint(20, size[0] - 20), np.random.randint(
            20, size[1] - 20
        )

        # Create line
        length = int(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        y_coords = np.linspace(y1, y2, length).astype(int)
        x_coords = np.linspace(x1, x2, length).astype(int)

        # Add fiber with some width
        for y, x in zip(y_coords, x_coords):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size[0] and 0 <= nx < size[1]:
                        # Gaussian-like intensity
                        dist = np.sqrt(dy**2 + dx**2)
                        intensity = 200 * np.exp(-(dist**2) / 2)
                        image[ny, nx] = max(image[ny, nx], intensity)

    # Add some noise
    image += np.random.normal(0, 10, size).astype(np.float32)
    image = np.clip(image, 0, 255)

    return image


def test_fire_2d_basic():
    """Test basic FIRE 2D functionality"""
    print("=" * 60)
    print("Testing FIRE 2D Fiber Extraction")
    print("=" * 60)

    # Create synthetic test image
    print("\n1. Creating synthetic fiber image...")
    image = create_synthetic_fiber_image(size=(128, 128), num_fibers=3)
    print(f"   Image shape: {image.shape}")
    print(f"   Image range: [{image.min():.1f}, {image.max():.1f}]")

    # Create parameters
    print("\n2. Setting up parameters...")
    params = create_default_params()

    # Adjust parameters for synthetic image
    params["sigma_im"] = 1.0
    params["thresh_im"] = 0.2
    params["sigma_d"] = 1.5
    params["s_xlinkbox"] = 2
    params["thresh_Dxlink"] = 0.5

    print("   Key parameters:")
    print(f"   - sigma_im: {params['sigma_im']}")
    print(f"   - thresh_im: {params['thresh_im']}")
    print(f"   - s_xlinkbox: {params['s_xlinkbox']}")

    # Run FIRE
    print("\n3. Running FIRE extraction...")
    try:
        data = fire_2d_ang1(params, image, plotflag=0)

        print("\n4. Results:")
        print(f"   - Nucleation points: {data['xlink'].shape[0]}")
        print(f"   - Vertices: {data['Xa'].shape[0]}")
        print(f"   - Fibers: {len(data['Fa'])}")
        print(f"   - Edges: {data['Ea'].shape[0]}")

        # Print some fiber details
        if len(data["Fa"]) > 0:
            print("\n   First few fibers:")
            for i in range(min(3, len(data["Fa"]))):
                if isinstance(data["Fa"][i], dict) and "v" in data["Fa"][i]:
                    v_list = data["Fa"][i]["v"]
                    print(f"   - Fiber {i}: {len(v_list)} vertices")

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fire_2d_with_real_image():
    """Test with a real image if available"""
    print("\n" + "=" * 60)
    print("Testing with Real Image (if available)")
    print("=" * 60)

    # Try to load a real image
    try:
        # Try common test image locations
        test_paths = [
            "tests/test_images/real1.tif",
            "test_images/real1.tif",
            "../tests/test_images/real1.tif",
        ]

        image = None
        for path in test_paths:
            try:
                from PIL import Image

                img = Image.open(path)
                image = np.array(img, dtype=np.float32)
                print(f"\n✓ Loaded image from: {path}")
                break
            except:
                continue

        if image is None:
            print("\n⊘ No test image found, skipping real image test")
            return None

        print(f"   Image shape: {image.shape}")
        print(f"   Image range: [{image.min():.1f}, {image.max():.1f}]")

        # Create parameters
        params = create_default_params()

        # Run FIRE
        print("\nRunning FIRE extraction on real image...")
        data = fire_2d_ang1(params, image, plotflag=0)

        print("\nResults:")
        print(f"   - Nucleation points: {data['xlink'].shape[0]}")
        print(f"   - Vertices: {data['Xa'].shape[0]}")
        print(f"   - Fibers: {len(data['Fa'])}")

        print("\n✓ Real image test completed!")
        return True

    except Exception as e:
        print(f"\n✗ Real image test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FIRE 2D Test Suite")
    print("=" * 60)

    # Run tests
    test1 = test_fire_2d_basic()
    test2 = test_fire_2d_with_real_image()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Basic test: {'PASSED' if test1 else 'FAILED'}")
    if test2 is not None:
        print(f"Real image test: {'PASSED' if test2 else 'FAILED'}")
    else:
        print("Real image test: SKIPPED")
    print("=" * 60)
