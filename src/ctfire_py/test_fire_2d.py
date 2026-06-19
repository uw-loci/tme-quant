"""
Test script for FIRE 2D fiber extraction
"""

import matplotlib
matplotlib.use("Agg")  # headless: no display server required (WSL2)
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from ctfire_py.fire_2d_angle import fire_2d_angle


def plot_fiber_overlay(im, X, F, title="Fiber Overlay", save_path=None):
    """
    Overlay fiber centerlines on the original image.

    Each fiber is a 1-pixel-thick polyline in a unique HSV color drawn with
    Bresenham's line algorithm.

    Args:
        im: 2D grayscale image array (H×W)
        X: (N, 2+) vertex array; X[:, 0] = row, X[:, 1] = col (0-based)
        F: list of fiber dicts with key 'v' = list of 0-based vertex indices
        title: axes title
        save_path: optional path to save the figure

    Returns:
        (fig, ax)
    """
    img2d = im[0] if im.ndim == 3 else im
    H, W = img2d.shape

    img_norm = img2d.astype(np.float32)
    peak = img_norm.max()
    if peak > 0:
        img_norm /= peak
    canvas = np.stack([img_norm, img_norm, img_norm], axis=-1)

    n_fibers = len(F)
    if n_fibers > 0:
        cmap = plt.get_cmap("hsv", n_fibers)
        colors = [cmap(i)[:3] for i in range(n_fibers)]
        X_arr = np.asarray(X)

        for fi, fiber in enumerate(F):
            v_list = fiber.get("v", []) if isinstance(fiber, dict) else list(fiber)
            if len(v_list) < 2:
                continue
            rc, gc, bc = colors[fi]
            for seg in range(len(v_list) - 1):
                v0, v1 = v_list[seg], v_list[seg + 1]
                if v0 < 0 or v0 >= len(X_arr) or v1 < 0 or v1 >= len(X_arr):
                    continue
                r0 = int(round(float(X_arr[v0, 0])))
                c0 = int(round(float(X_arr[v0, 1])))
                r1 = int(round(float(X_arr[v1, 0])))
                c1 = int(round(float(X_arr[v1, 1])))
                r0, c0 = np.clip(r0, 0, H - 1), np.clip(c0, 0, W - 1)
                r1, c1 = np.clip(r1, 0, H - 1), np.clip(c1, 0, W - 1)
                rr, cc = draw_line(r0, c0, r1, c1)
                mask = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                canvas[rr[mask], cc[mask]] = (rc, gc, bc)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, origin="upper")
    ax.set_title(f"{title} ({n_fibers} fibers)")
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Saved overlay to: {save_path}")

    return fig, ax


def create_default_params():
    """Fallback dictionary providing the legacy parameter keys expected by fire_2d_angle"""
    return {
        "sigma_im": 0,
        "sigma_d": 0.3,
        "dtype": "cityblock",
        "thresh_im": [],
        "thresh_im2": 5,
        "thresh_Dxlink": 1.5,
        "s_xlinkbox": 8,
        "thresh_LMP": 0.2,
        "thresh_LMPdist": 2,
        "thresh_ext": 0.342,
        "lam_dirdecay": 0.5,
        "s_minstep": 2,
        "s_maxstep": 6,
        "thresh_dang_aextend": 0.9848,
        "thresh_dang_L": 15,
        "thresh_short_L": 15,
        "s_fiberdir": 4,
        "thresh_linkd": 15,
        "thresh_linka": -0.866,
        "thresh_flen": 15,
        "thresh_numv": 3,
        "scale": [1.0, 1.0, 1.0],
        "s_boundthick": 10,
        "blist": 1,
        "s_maxspace": 5,
        "lambda": 0.01,
        "ang_interval": 3,
    }


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
        data = fire_2d_angle(params, image, plotflag=0)

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

        print("\n4b. Generating fiber overlay...")
        plot_fiber_overlay(
            image,
            data["Xas"],#["Xf"],
            data["Fas"],#["Ff"],
            title= "processed extraction",#"Synthetic — filtered fibers",
            save_path="fiber_overlay_synthetic.png",
        )

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
        data = fire_2d_angle(params, image, plotflag=0)

        print("\nResults:")
        print(f"   - Nucleation points: {data['xlink'].shape[0]}")
        print(f"   - Vertices: {data['Xa'].shape[0]}")
        print(f"   - Fibers: {len(data['Fa'])}")

        print("\nGenerating fiber overlay...")
        plot_fiber_overlay(
            image,
            data["Xf"],
            data["Ff"],
            title="Real image — filtered fibers",
            save_path="fiber_overlay_real.png",
        )

        print("\n✓ Real image test completed!")
        return True

    except Exception as e:
        print(f"\n✗ Real image test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fire_2d_with_nonsquare_image():
    """Test with a non-square real image if available"""
    print("\n" + "=" * 60)
    print("Testing with Non-Square Image (if available)")
    print("=" * 60)

    # Try to load the non-square image
    try:
        # Try common test image locations
        test_paths = [
            "tests/test_images/real1_rect.tif",
            "test_images/real1_rect.tif",
            "../tests/test_images/real1_rect.tif",
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
            print("\n⊘ No non-square test image found, skipping test")
            return None

        print(f"   Image shape: {image.shape}")
        print(f"   Image range: [{image.min():.1f}, {image.max():.1f}]")
        h, w = image.shape[:2]
        print(f"   Non-square check: height={h}, width={w}, equal={h == w}")

        # Create parameters
        params = create_default_params()

        # Run FIRE
        print("\nRunning FIRE extraction on non-square image...")
        data = fire_2d_angle(params, image, plotflag=0)

        print("\nResults:")
        print(f"   - Nucleation points: {data['xlink'].shape[0]}")
        print(f"   - Vertices: {data['Xa'].shape[0]}")
        print(f"   - Fibers: {len(data['Fa'])}")

        print("\nGenerating fiber overlay...")
        plot_fiber_overlay(
            image,
            data["Xf"],
            data["Ff"],
            title="Non-square image — filtered fibers",
            save_path="fiber_overlay_nonsquare.png",
        )

        print("\n✓ Non-square image test completed!")
        return True

    except Exception as e:
        print(f"\n✗ Non-square image test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


TESTS = {
    "basic": ("Basic test", test_fire_2d_basic),
    "real": ("Real image test", test_fire_2d_with_real_image),
    "nonsquare": ("Non-square image test", test_fire_2d_with_nonsquare_image),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="FIRE 2D Test Suite")
    parser.add_argument(
        "tests",
        nargs="*",
        choices=list(TESTS.keys()) + ["all"],
        default=["all"],
        help="Which test(s) to run (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = list(TESTS.keys()) if "all" in args.tests else args.tests

    print("\n" + "=" * 60)
    print("FIRE 2D Test Suite")
    print("=" * 60)

    # Run selected tests
    results = {}
    for key in selected:
        label, test_fn = TESTS[key]
        results[key] = test_fn()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for key in selected:
        label, _ = TESTS[key]
        result = results[key]
        if result is None:
            print(f"{label}: SKIPPED")
        else:
            print(f"{label}: {'PASSED' if result else 'FAILED'}")
    print("=" * 60)
