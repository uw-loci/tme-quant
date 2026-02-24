"""
Example 1: Cell-based fiber interaction analysis
"""

from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams,
    AnalysisMode,
    InteractionStrategy
)
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.fiber_analysis import FiberAnalyzer

# Segment cells
cell_analyzer = CellAnalyzer()
cell_result = cell_analyzer.segment_cells_2d(cell_image, cell_seg_params)
cells = [
    create_cell_object_from_segmentation(c, parent_id="region_1")
    for c in cell_result.cells
]

# Extract fibers
fiber_analyzer = FiberAnalyzer()
fiber_result = fiber_analyzer.extract_fibers_2d(fiber_image, fiber_extract_params)
fibers = [
    create_fiber_object_from_extraction(f, parent_id="region_1")
    for f in fiber_result.fibers
]

# Configure cell-based analysis
params = TMEAnalysisParams(
    mode=AnalysisMode.CELL_BASED,
    interaction_strategy=InteractionStrategy.RADIUS,
    cell_fiber_distance=50.0,  # 50 micron radius around each cell
    compute_morphology=True,
    compute_spatial=True,
    compute_orientation=True
)

# Run analysis
tme_analyzer = TMEAnalyzer(verbose=True)
result = tme_analyzer.analyze(
    cells=cells,
    fibers=fibers,
    params=params,
    analysis_id="cell_based_001"
)

# Access results
print(f"Found {len(result.interaction_pairs)} cell-fiber interactions")
print(f"Mean distance: {result.spatial_features['mean_interaction_distance']:.2f} µm")

# Export
tme_analyzer.export_results("output/cell_based/", formats=["csv", "json"])