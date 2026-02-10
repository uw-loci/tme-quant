"""
Example 2: Segment, classify, and quantify cells
"""

from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config.segmentation_params import (
    SegmentationParams, SegmentationMode, ImageModality
)
from tme_quant.cell_analysis.config.classification_params import (
    ClassificationParams, ClassificationMode, CellType
)
from tme_quant.cell_analysis.config.quantification_params import (
    QuantificationParams
)

# Load multi-channel IF image
image = io.imread("data/if_image.tif")  # Shape: (H, W, C)
# Channels: 0=DAPI, 1=CD3, 2=CD8, 3=CD68

# Create analyzer
analyzer = CellAnalyzer(verbose=True)

# 1. Segmentation parameters
seg_params = SegmentationParams(
    mode=SegmentationMode.CELLPOSE,
    image_modality=ImageModality.FLUORESCENCE,
    cellpose_model="nuclei",
    pixel_size=0.65,
    target="nucleus"
)

# 2. Classification parameters (marker-based)
class_params = ClassificationParams(
    mode=ClassificationMode.MARKER,
    cell_types=[CellType.TUMOR, CellType.T_CELL, CellType.MACROPHAGE, CellType.STROMAL],
    marker_channels={
        'CD3': 1,   # T cell marker
        'CD8': 2,   # Cytotoxic T cell marker
        'CD68': 3   # Macrophage marker
    },
    marker_thresholds={
        'CD3': 0.3,
        'CD8': 0.25,
        'CD68': 0.4
    }
)

# 3. Quantification parameters
quant_params = QuantificationParams(
    measure_area=True,
    measure_perimeter=True,
    measure_circularity=True,
    measure_mean_intensity=True,
    measure_integrated_intensity=True,
    measure_distances=True,
    measure_density=True,
    neighbor_distance_threshold=30.0
)

# Run full pipeline
result = analyzer.analyze_full_pipeline_2d(
    image,
    segmentation_params=seg_params,
    classification_params=class_params,
    quantification_params=quant_params,
    image_id="patient_001"
)

# Access results
print("\n=== Segmentation Results ===")
print(f"Total cells: {result.segmentation_result.total_cell_count}")

print("\n=== Classification Results ===")
for cell_type, count in result.classification_result.type_counts.items():
    ratio = result.classification_result.type_ratios[cell_type]
    print(f"{cell_type.value}: {count} ({ratio:.1%})")

print("\n=== Quantification Results ===")
pop_stats = result.quantification_result.population_stats
print(f"Mean cell area: {pop_stats['mean_area']:.2f} µm²")
print(f"Mean circularity: {pop_stats['mean_circularity']:.3f}")

spatial_stats = result.quantification_result.spatial_stats
print(f"Mean NN distance: {spatial_stats['mean_nearest_neighbor_distance']:.2f} µm")

# Export all results
analyzer.export_results(
    output_dir="output/full_analysis/",
    formats=["csv", "excel", "json"],
    prefix="tme_cell_analysis"
)