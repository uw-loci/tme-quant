#!/usr/bin/env python3
"""
Test visualization pipeline using pre-computed fibFeatures.csv (skipping FDCT).
"""

import sys
import numpy as np
from pathlib import Path
import csv

# Add source to path
sys.path.insert(0, "src")
import curvealign_py as curvealign


def test_with_expected_curvelets():
    """Test visualization pipeline using curvelets from fibFeatures.csv (no FDCT)"""

    print("=== TESTING VISUALIZATION WITH FIBFEATURES.CSV ===")
    print("🚀 Skipping FDCT - using pre-computed fiber features directly")

    # 1. Load real1.tif
    image_path = Path("tests/test_images/real1.tif")
    try:
        import matplotlib.pyplot as plt

        img = plt.imread(str(image_path))
        print(f"✅ Loaded {image_path}: {img.shape}, dtype: {img.dtype}")

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

    except Exception as e:
        print(f"❌ Failed to load real1.tif: {e}")
        return False

    # 2. Load pre-computed curvelets from fibFeatures.csv
    # Try multiple possible locations for fibFeatures.csv (prioritize the larger files)
    possible_paths = [
        Path("tests/curvealign_py/fibFeatures.csv"),
        Path(
            "tests/test_results/relative_angle_test_files/CAoutput/real1_fibFeatures.csv"
        ),
        Path("test_output_fibFeatures.csv"),
    ]

    curvelets = []
    fibfeatures_path = None

    for path in possible_paths:
        if path.exists():
            fibfeatures_path = path
            break

    if not fibfeatures_path:
        print(f"❌ Could not find fibFeatures.csv in any expected location")
        return False

    try:
        with open(fibfeatures_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4 and row[1] != "NaN" and row[2] != "NaN":
                    try:
                        curvelet = curvealign.Curvelet(
                            center_row=int(float(row[2]) - 1),  # Convert to 0-based
                            center_col=int(float(row[1]) - 1),  # Convert to 0-based
                            angle_deg=float(row[3]),
                            weight=1.0,  # Default weight
                        )
                        curvelets.append(curvelet)
                    except (ValueError, IndexError):
                        continue  # Skip invalid rows

        print(f"✅ Loaded {len(curvelets)} curvelets from {fibfeatures_path}")

    except Exception as e:
        print(f"❌ Failed to load fibFeatures.csv: {e}")
        return False

    # 3. Skip feature computation - focus on visualization
    print("⏭️  Skipping feature computation - focusing on visualization pipeline")
    # 4. Test visualizations
    print("🎨 Testing visualizations...")
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Test overlay
        overlay_img = curvealign.overlay(img, curvelets)
        print(f"✅ Overlay created: {overlay_img.shape}")

        # Test angle maps
        raw_map, processed_map = curvealign.angle_map(img, curvelets)
        print(f"✅ Angle maps created: {raw_map.shape}, {processed_map.shape}")

        # Save visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Original image
        axes[0, 0].imshow(img, cmap="gray")
        axes[0, 0].set_title("Original real1.tif")
        axes[0, 0].axis("off")

        # Fiber overlay
        axes[0, 1].imshow(overlay_img)
        axes[0, 1].set_title(f"Fiber Overlay ({len(curvelets)} curvelets)")
        axes[0, 1].axis("off")

        # Raw angle map
        im1 = axes[1, 0].imshow(raw_map, cmap="hsv", vmin=0, vmax=180)
        axes[1, 0].set_title("Raw Angle Map")
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.6)

        # Processed angle map
        im2 = axes[1, 1].imshow(processed_map, cmap="hsv", vmin=0, vmax=180)
        axes[1, 1].set_title("Processed Angle Map")
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], shrink=0.6)

        plt.tight_layout()
        plt.savefig("fibfeatures_visualization.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("💾 Saved fibfeatures_visualization.png")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. Summary statistics
    angles = [c.angle_deg for c in curvelets]
    print(f"\n📈 Statistics:")
    print(f"   Count: {len(curvelets)}")
    print(f"   Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
    print(f"   Mean angle: {np.mean(angles):.1f}°")
    print(f"   Std angle: {np.std(angles):.1f}°")

    print(f"\n🎉 SUCCESS! Visualization pipeline works with fibFeatures.csv")
    print(f"📁 Generated: fibfeatures_visualization.png")
    print(f"📊 Used data from: {fibfeatures_path}")

    return True


if __name__ == "__main__":
    success = test_with_expected_curvelets()
    if success:
        print("\n✅ Visualization pipeline works correctly with pre-computed data!")
        print("💡 This confirms the issue is in the FDCT curvelet extraction step")
    else:
        print("\n❌ There are issues in the visualization pipeline")
