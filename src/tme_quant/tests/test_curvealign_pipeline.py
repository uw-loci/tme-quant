# -*- coding: utf-8 -*-
"""Tests for tme_analysis/pipelines/curvealign_pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from tme_quant.tme_analysis.pipelines import curvealign_pipeline


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _synthetic_image(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8).astype(np.float32)


def _minimal_fiber_df(n=3, img_h=64, img_w=64, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "center_row": rng.uniform(10, img_h - 10, n),
        "center_col": rng.uniform(10, img_w - 10, n),
        "angle":      rng.uniform(0, 180, n),
        "weight":     np.ones(n),
    })


def _ring_mask(h=64, w=64):
    """Binary mask: filled square with a central hole — tests boundary extraction."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:56, 8:56] = 1
    mask[24:40, 24:40] = 0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCurvealignPipeline:

    def test_no_boundary_returns_result_dict(self):
        """Synthetic image with no boundary → valid result dict, boundary_measurement=False.

        Requires curvelops — skipped automatically when it is absent.
        """
        pytest.importorskip("curvelops", reason="curvelops not installed")
        img = _synthetic_image()
        result = curvealign_pipeline(img)
        if result is None:
            pytest.skip("No fibers detected in synthetic image")
        assert isinstance(result, dict)
        assert result["boundary_measurement"] is False
        assert isinstance(result["fiber_structure"], pd.DataFrame)
        assert isinstance(result["fiber_features_df"], pd.DataFrame)
        assert "fiber_absolute_angle" in result["fiber_features_df"].columns

    def test_empty_fiber_structure_returns_none(self):
        """Pre-built empty fiber_structure → pipeline returns None immediately."""
        img = _synthetic_image()
        result = curvealign_pipeline(img, fiber_structure=pd.DataFrame())
        assert result is None

    def test_precomputed_fiber_structure(self):
        """Pre-built fiber_structure bypasses extraction; same rows in output."""
        img = _synthetic_image()
        fibers = _minimal_fiber_df(n=3)
        result = curvealign_pipeline(img, fiber_structure=fibers)
        assert result is not None
        assert len(result["fiber_structure"]) == 3
        assert result["boundary_measurement"] is False
        assert result["in_curvs_flag"] is not None
        assert len(result["in_curvs_flag"]) == 3

    def test_tif3_boundary_mode(self):
        """tif_boundary=3 with a mask → boundary_measurement=True, nearest_angles set."""
        img = _synthetic_image()
        mask = _ring_mask()
        fibers = _minimal_fiber_df(n=5)
        result = curvealign_pipeline(
            img,
            fiber_structure=fibers,
            boundary_img=mask,
            tif_boundary=3,
            distance_threshold=50.0,
        )
        assert result is not None
        assert result["boundary_measurement"] is True
        assert result["nearest_angles"] is not None

    def test_notimplemented_for_csv_boundary(self):
        """tif_boundary=1 raises NotImplementedError (CSV mode not yet ported)."""
        img = _synthetic_image()
        fibers = _minimal_fiber_df(n=2)
        with pytest.raises(NotImplementedError):
            curvealign_pipeline(img, fiber_structure=fibers, tif_boundary=1)


# ─────────────────────────────────────────────────────────────────────────────
# TestCurveAlignFiberCandidates  (requires curvelops — skipped on Windows)
# ─────────────────────────────────────────────────────────────────────────────

class TestCurveAlignFiberCandidates:
    """Tests for analyze_2d fiber candidate integration (requires curvelops)."""

    @pytest.fixture(autouse=True)
    def _skip_without_curvelops(self):
        pytest.importorskip("curvelops", reason="curvelops not installed")

    def _make_image(self, h=64, w=64, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((h, w)).astype(np.float32)

    def test_returns_fiber_structure_dataframe(self):
        from tme_quant.fiber_analysis.methods.curvealign import CurveAlignOrientation
        from tme_quant.fiber_analysis.config import CurveAlignParams
        img = self._make_image()
        params = CurveAlignParams(
            window_size=32, overlap=0.5,
            return_fiber_segments=True,
            candidate_keep=0.05, candidate_scale=1, candidate_radius=4.0,
        )
        result = CurveAlignOrientation().analyze_2d(img, params)
        assert result.fiber_structure is not None
        assert {'angle', 'center_row', 'center_col'}.issubset(result.fiber_structure.columns)

    def test_density_and_alignment_populated(self):
        from tme_quant.fiber_analysis.methods.curvealign import CurveAlignOrientation
        from tme_quant.fiber_analysis.config import CurveAlignParams
        img = self._make_image()
        params = CurveAlignParams(
            window_size=32, overlap=0.5,
            return_fiber_segments=True, candidate_keep=0.05,
        )
        result = CurveAlignOrientation().analyze_2d(img, params)
        if result.fiber_structure is not None and not result.fiber_structure.empty:
            assert result.fiber_density is not None
            assert result.fiber_alignment is not None

    def test_orientation_map_still_populated(self):
        """Sliding-window orientation map must be present regardless of candidate extraction."""
        from tme_quant.fiber_analysis.methods.curvealign import CurveAlignOrientation
        from tme_quant.fiber_analysis.config import CurveAlignParams
        img = self._make_image()
        params = CurveAlignParams(window_size=32, return_fiber_segments=True)
        result = CurveAlignOrientation().analyze_2d(img, params)
        assert result.orientation_map is not None
        assert result.orientation_map.shape == img.shape


# ─────────────────────────────────────────────────────────────────────────────
# TestCurveAlignFiberCandidatesNoCurvelops  (runs on all platforms)
# ─────────────────────────────────────────────────────────────────────────────

class TestCurveAlignFiberCandidatesNoCurvelops:
    """Graceful fallback when curvelops is absent — runs on all platforms."""

    def _make_image(self, h=48, w=48):
        return np.random.default_rng(1).random((h, w)).astype(np.float32)

    def test_no_curvelops_returns_empty_gracefully(self, monkeypatch):
        """When grouping fails, CURVELETS mode returns empty fiber fields and None maps.

        In CURVELETS mode the orientation_map is sparse (populated only at grouped
        curvelet positions). When build_fiber_structure_from_curvelets raises ImportError,
        no positions are available so both orientation_map and alignment_map remain None.
        Use WINDOWED mode when a dense orientation_map is required regardless of grouping.
        """
        import tme_quant.fiber_analysis.utils.fiber_dataframe_utils as fdu

        def _raise(**kw):
            raise ImportError("curvelops not installed (mocked)")

        monkeypatch.setattr(fdu, "build_fiber_structure_from_curvelets", _raise)
        from tme_quant.fiber_analysis.methods.curvealign import CurveAlignOrientation
        from tme_quant.fiber_analysis.config import CurveAlignParams, CurveAlignAnalysisMode
        img = self._make_image()
        params = CurveAlignParams(
            window_size=32,
            analysis_mode=CurveAlignAnalysisMode.CURVELETS,
        )
        result = CurveAlignOrientation().analyze_2d(img, params)
        assert result.fiber_segments == []
        assert result.fiber_structure is None
        # Sparse maps have no grouped positions to paint — both are None
        assert result.orientation_map is None
        assert result.alignment_map is None
