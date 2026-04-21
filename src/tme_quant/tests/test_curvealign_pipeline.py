# -*- coding: utf-8 -*-
"""Tests for tme_analysis/pipelines/curvealign_pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from tme_quant.tme_analysis.pipelines.curvealign_pipeline import curvealign_pipeline


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
