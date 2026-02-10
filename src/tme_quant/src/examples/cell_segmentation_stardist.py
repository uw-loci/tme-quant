"""
Example 1: Basic cell segmentation with StarDist
"""

from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config.segmentation_params import (
    SegmentationParams, SegmentationMode, ImageModality
)
import numpy as np
from skimage import io

# Load fluorescence image (nuclei channel)
image = io.imread("data/nuclei_image.tif")

# Create analyzer
analyzer = CellAnalyzer(verbose=True)

# Configure segmentation
seg_params = SegmentationParams(
    mode=SegmentationMode.STARDIST,
    image_modality=ImageModality.FLUORESCENCE,
    stardist_model="2D_versatile_fluo",
    pixel_size=0.5,  # microns per pixel
    target="nucleus",
    min_cell_size=20.0,  # square microns
    max_cell_size=500.0,
    remove_border_cells=True
)

# Run segmentation
seg_result = analyzer.segment_cells_2d(image, seg_params, image_id="sample_001")

print(f"Segmented {seg_result.total_cell_count} cells")
print(f"Mean cell area: {seg_result.mean_cell_area:.2f} µm²")
print(f"Mean circularity: {seg_result.mean_circularity:.3f}")

# Export results
analyzer.export_results(
    output_dir="output/segmentation/",
    formats=["csv", "json"],
    prefix="nuclei_segmentation"
)