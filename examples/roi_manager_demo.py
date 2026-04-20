"""
ROI Manager demonstration script.
Run with: python examples/roi_manager_demo.py
"""

import numpy as np
import os
from pathlib import Path
import sys

# Add src directory for direct example execution from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from napari_curvealign.roi_manager import ROIManager, ROIShape


def demo_basic_operations() -> ROIManager:
    """Show basic ROI create/rename/delete operations."""
    rm = ROIManager()
    rm.set_image_shape((512, 512))

    rect_roi = rm.add_roi(np.array([[50, 50], [150, 150]], dtype=float), ROIShape.RECTANGLE, "demo_rect")
    poly_roi = rm.add_roi(
        np.array([[200, 200], [250, 200], [250, 250], [200, 250]], dtype=float),
        ROIShape.POLYGON,
        "demo_poly",
    )
    rm.add_roi(np.array([[300, 300], [380, 380]], dtype=float), ROIShape.ELLIPSE, "demo_ellipse")

    rm.rename_roi(rect_roi.id, "demo_rect_renamed")
    rm.delete_roi(poly_roi.id)
    return rm


def demo_save_load_json(rm: ROIManager) -> None:
    """Show JSON save/load roundtrip."""
    output = "demo_rois.json"
    rm.save_rois(output, format="json")
    rm.clear_rois()
    rm.load_rois(output, format="json")
    if os.path.exists(output):
        os.remove(output)


def main():
    """Run the ROI manager demo."""
    print("Running ROI manager demo...")
    manager = demo_basic_operations()
    demo_save_load_json(manager)
    print(f"Done. ROI count after demo: {len(manager.rois)}")


if __name__ == "__main__":
    main()
