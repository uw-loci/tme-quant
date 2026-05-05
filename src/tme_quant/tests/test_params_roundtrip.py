"""
Round-trip serialization tests for all *Params dataclasses.

Verifies that to_dict() → from_dict() produces an object whose to_dict()
output is identical to the original, covering all fields (not just the subset
that older partial implementations serialized).
"""

import sys
import unittest.mock as mock

# Only mock what cell_objects.py needs at import time (shapely).
# Do NOT mock curvelops — that would disable skip guards in other test files.
for lib in ['shapely', 'shapely.geometry', 'shapely.ops']:
    sys.modules.setdefault(lib, mock.MagicMock())

import pytest

from tme_quant.fiber_analysis.config import (
    ExtractionParams, ExtractionMode,
    CTFireParams, RidgeDetectionParams, SkeletonParams,
    OrientationParams, OrientationMode,
    CurveAlignParams, CurveAlignAnalysisMode,
    OrientationJParams, GradientParams, StructureTensorParams,
    FiberFeatureParams,
)
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams, AnalysisMode,
    TumorDetectionParams, TumorDetectionMethod, InteractionStrategy,
)
from tme_quant.core.tme_objects.cell_objects import (
    SegmentationParams, ClassificationParams, QuantificationParams,
    SegmentationMode, ClassificationMode, CellType, ImageModality,
)
from tme_quant.image_registration.config import (
    RegistrationParams, RegistrationMethod, TransformType, MicroscopyModality,
)


def _roundtrip(params):
    """Serialize → deserialize → re-serialize and compare dicts."""
    d = params.to_dict()
    restored = type(params).from_dict(d)
    assert params.to_dict() == restored.to_dict(), (
        f"Round-trip failed for {type(params).__name__}.\n"
        f"Original : {params.to_dict()}\n"
        f"Restored : {restored.to_dict()}"
    )
    return restored


# ─────────────────────────────────────────────────────────────────────────────
# Extraction params
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionParamsRoundtrip:
    def test_base_defaults(self):
        _roundtrip(ExtractionParams())

    def test_base_custom(self):
        p = ExtractionParams(
            pixel_size=0.5,
            min_fiber_length=10.0,
            max_fiber_width=8.0,
            extract_centerlines=False,
        )
        r = _roundtrip(p)
        assert r.pixel_size == 0.5
        assert r.min_fiber_length == 10.0
        assert r.extract_centerlines is False

    def test_ctfire(self):
        p = CTFireParams(ctfire_threshold=0.2, spur_length_px=12, z_spacing=0.5)
        r = _roundtrip(p)
        assert r.mode == ExtractionMode.CTFIRE
        assert r.ctfire_threshold == 0.2
        assert r.spur_length_px == 12
        assert r.z_spacing == 0.5

    def test_ridge_detection(self):
        p = RidgeDetectionParams(ridge_sigma=3.0, extend_line=False)
        r = _roundtrip(p)
        assert r.mode == ExtractionMode.RIDGE_DETECTION
        assert r.ridge_sigma == 3.0
        assert r.extend_line is False

    def test_skeleton(self):
        p = SkeletonParams(min_branch_length=5.0, skeleton_method="zhang")
        r = _roundtrip(p)
        assert r.mode == ExtractionMode.SKELETON
        assert r.min_branch_length == 5.0
        assert r.skeleton_method == "zhang"


# ─────────────────────────────────────────────────────────────────────────────
# Orientation params
# ─────────────────────────────────────────────────────────────────────────────

class TestOrientationParamsRoundtrip:
    def test_base_defaults(self):
        _roundtrip(OrientationParams())

    def test_curvealign_windowed(self):
        p = CurveAlignParams(
            analysis_mode=CurveAlignAnalysisMode.WINDOWED,
            window_size=64,
            curvelet_levels=6,
            candidate_keep=0.1,
        )
        r = _roundtrip(p)
        assert r.analysis_mode == CurveAlignAnalysisMode.WINDOWED
        assert r.window_size == 64
        assert r.curvelet_levels == 6
        assert r.candidate_keep == 0.1

    def test_curvealign_with_feature_params(self):
        fp = FiberFeatureParams(minimum_nearest_fibers=4, minimum_box_size=64)
        p = CurveAlignParams(candidate_feature_params=fp)
        r = _roundtrip(p)
        assert r.candidate_feature_params is not None
        assert r.candidate_feature_params.minimum_nearest_fibers == 4
        assert r.candidate_feature_params.minimum_box_size == 64

    def test_orientationj(self):
        p = OrientationJParams(sigma_tensor=3.0, compute_color_survey=True)
        r = _roundtrip(p)
        assert r.mode == OrientationMode.ORIENTATIONJ
        assert r.sigma_tensor == 3.0
        assert r.compute_color_survey is True

    def test_gradient(self):
        p = GradientParams(gradient_operator="scharr", smoothing_sigma=2.0)
        r = _roundtrip(p)
        assert r.mode == OrientationMode.GRADIENT
        assert r.gradient_operator == "scharr"

    def test_structure_tensor(self):
        p = StructureTensorParams(sigma_spatial=4.0, compute_eigenvalues=True)
        r = _roundtrip(p)
        assert r.mode == OrientationMode.STRUCTURE_TENSOR
        assert r.sigma_spatial == 4.0
        assert r.compute_eigenvalues is True

    def test_fiber_feature_params(self):
        p = FiberFeatureParams(minimum_nearest_fibers=4, fiber_midpoint_estimate=0)
        r = _roundtrip(p)
        assert r.minimum_nearest_fibers == 4
        assert r.fiber_midpoint_estimate == 0


# ─────────────────────────────────────────────────────────────────────────────
# TME analysis params
# ─────────────────────────────────────────────────────────────────────────────

class TestTMEAnalysisParamsRoundtrip:
    def test_defaults(self):
        _roundtrip(TMEAnalysisParams())

    def test_all_fields_preserved(self):
        p = TMEAnalysisParams(
            mode=AnalysisMode.FIBER_BASED,
            interaction_strategy=InteractionStrategy.K_NEAREST,
            k_neighbors=10,
            cell_fiber_distance=75.0,
            compute_morphology=False,
            tacs_angle_threshold_perpendicular=25.0,
            tacs_straightness_threshold=0.8,
            invasive_margin_width=60.0,
            return_distance_maps=False,
        )
        r = _roundtrip(p)
        assert r.mode == AnalysisMode.FIBER_BASED
        assert r.interaction_strategy == InteractionStrategy.K_NEAREST
        assert r.k_neighbors == 10
        assert r.cell_fiber_distance == 75.0
        assert r.compute_morphology is False
        assert r.tacs_angle_threshold_perpendicular == 25.0
        assert r.tacs_straightness_threshold == 0.8
        assert r.invasive_margin_width == 60.0
        assert r.return_distance_maps is False

    def test_tumor_detection_params(self):
        p = TumorDetectionParams(
            method=TumorDetectionMethod.DENSITY,
            dbscan_eps=80.0,
            smooth_boundary=False,
        )
        r = _roundtrip(p)
        assert r.method == TumorDetectionMethod.DENSITY
        assert r.dbscan_eps == 80.0
        assert r.smooth_boundary is False


# ─────────────────────────────────────────────────────────────────────────────
# Cell analysis params
# ─────────────────────────────────────────────────────────────────────────────

class TestCellParamsRoundtrip:
    def test_segmentation_defaults(self):
        # mode is required — provide one
        _roundtrip(SegmentationParams(mode=SegmentationMode.STARDIST))

    def test_segmentation_full(self):
        p = SegmentationParams(
            mode=SegmentationMode.CELLPOSE,
            image_modality=ImageModality.BRIGHTFIELD,
            cellpose_diameter=30.0,
            cellpose_flow_threshold=0.6,
            use_gpu=True,
            remove_border_cells=False,
        )
        r = _roundtrip(p)
        assert r.mode == SegmentationMode.CELLPOSE
        assert r.image_modality == ImageModality.BRIGHTFIELD
        assert r.cellpose_diameter == 30.0
        assert r.use_gpu is True
        assert r.remove_border_cells is False

    def test_classification_with_cell_type_list(self):
        p = ClassificationParams(
            mode=ClassificationMode.MARKER,
            cell_types=[CellType.TUMOR, CellType.T_CELL, CellType.MACROPHAGE],
            min_confidence=0.7,
        )
        r = _roundtrip(p)
        assert r.mode == ClassificationMode.MARKER
        assert r.cell_types == [CellType.TUMOR, CellType.T_CELL, CellType.MACROPHAGE]
        assert r.min_confidence == 0.7

    def test_quantification_defaults(self):
        _roundtrip(QuantificationParams())

    def test_quantification_custom(self):
        p = QuantificationParams(
            measure_texture=True,
            texture_scales=[2, 4, 8],
            neighbor_distance_threshold=100.0,
            measure_cell_fiber_distance=True,
        )
        r = _roundtrip(p)
        assert r.measure_texture is True
        assert r.texture_scales == [2, 4, 8]
        assert r.neighbor_distance_threshold == 100.0
        assert r.measure_cell_fiber_distance is True


# ─────────────────────────────────────────────────────────────────────────────
# Registration params
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistrationParamsRoundtrip:
    def test_defaults(self):
        _roundtrip(RegistrationParams())

    def test_custom(self):
        p = RegistrationParams(
            method=RegistrationMethod.SIFT,
            transform_type=TransformType.RIGID,
            fixed_modality=MicroscopyModality.SHG,
            moving_modality=MicroscopyModality.HE_BRIGHTFIELD,
            num_iterations=500,
            histogram_matching=True,
            return_metrics=False,
        )
        r = _roundtrip(p)
        assert r.method == RegistrationMethod.SIFT
        assert r.transform_type == TransformType.RIGID
        assert r.fixed_modality == MicroscopyModality.SHG
        assert r.moving_modality == MicroscopyModality.HE_BRIGHTFIELD
        assert r.num_iterations == 500
        assert r.histogram_matching is True
        assert r.return_metrics is False
