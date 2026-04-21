# -*- coding: utf-8 -*-
"""Tests for fiber_analysis/visualization/draw_utils.py.

Covers draw_curvs (matplotlib drawing) and draw_map (ndarray heatmap).
draw_curvs tests are gated on matplotlib; draw_map tests have no extra deps.
"""

import numpy as np
import pandas as pd
import pytest

# Gate the entire module's matplotlib-dependent tests
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")   # non-interactive backend — required in headless CI
import matplotlib.pyplot as plt

from tme_quant.fiber_analysis.visualization.draw_utils import (
    draw_curvs,
    draw_map,
    compute_angle_histogram,
    generate_fiber_overlay,
    generate_fiber_heatmap,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_fiber_df(n=3, img_size=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "center_row": rng.integers(10, img_size - 10, n).astype(float),
        "center_col": rng.integers(10, img_size - 10, n).astype(float),
        "angle":      rng.uniform(0, 180, n),
    })


def _make_image(h=50, w=60):
    return np.zeros((h, w), dtype=np.uint8)


def _count_lines(ax):
    """Return number of Line2D artists with more than one point on ax."""
    from matplotlib.lines import Line2D
    return sum(
        1 for a in ax.get_children()
        if isinstance(a, Line2D) and len(a.get_xdata()) > 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestDrawCurvs
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawCurvs:

    @pytest.fixture(autouse=True)
    def ax(self):
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    @pytest.fixture()
    def fiber_df(self):
        return _make_fiber_df(n=3)

    def test_no_fibers_returns_early(self, ax):
        empty = pd.DataFrame(columns=["center_row", "center_col", "angle"])
        draw_curvs(empty, ax, length=10, color_flag=0, angles=[], mark_size=5, line_width=1, boundary_measurement=True)
        assert _count_lines(ax) == 0

    def test_boundary_true_color0_draws_lines(self, ax, fiber_df):
        draw_curvs(fiber_df, ax, length=10, color_flag=0, angles=None, mark_size=5, line_width=1, boundary_measurement=True)
        assert _count_lines(ax) == len(fiber_df)

    def test_boundary_true_color1_draws_lines(self, ax, fiber_df):
        draw_curvs(fiber_df, ax, length=10, color_flag=1, angles=None, mark_size=5, line_width=1, boundary_measurement=True)
        assert _count_lines(ax) == len(fiber_df)

    def test_no_boundary_color0_draws_lines(self, ax, fiber_df):
        angles = fiber_df["angle"].values
        draw_curvs(fiber_df, ax, length=10, color_flag=0, angles=angles, mark_size=5, line_width=1, boundary_measurement=False)
        assert _count_lines(ax) == len(fiber_df)

    def test_no_boundary_color1_skips(self, ax, fiber_df):
        angles = fiber_df["angle"].values
        draw_curvs(fiber_df, ax, length=10, color_flag=1, angles=angles, mark_size=5, line_width=1, boundary_measurement=False)
        assert _count_lines(ax) == 0

    def test_accepts_center_1_center_2_aliases_boundary_true(self, ax):
        """center_1/center_2 aliases must not crash in boundary_measurement=True path."""
        df = pd.DataFrame({
            "center_1": [20.0, 30.0],
            "center_2": [15.0, 25.0],
            "angle":    [45.0, 90.0],
        })
        draw_curvs(df, ax, length=8, color_flag=0, angles=None, mark_size=4, line_width=1, boundary_measurement=True)
        assert _count_lines(ax) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestDrawMap
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawMap:

    MAP_PARAMS = {
        "STDfilter_size": 8,
        "SQUAREmaxfilter_size": 6,
        "GAUSSIANdiscfilter_sigma": 2,
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.img = _make_image(h=50, w=60)
        self.fiber_df = _make_fiber_df(n=5, img_size=40)
        self.angles = self.fiber_df["angle"].values

    def _run(self, boundary_measurement=True, **kw):
        return draw_map(
            self.fiber_df, self.angles, self.img,
            boundary_measurement, self.MAP_PARAMS, **kw
        )

    def test_returns_tuple_of_two_arrays(self):
        result = self._run()
        assert isinstance(result, tuple) and len(result) == 2
        rawmap, procmap = result
        assert isinstance(rawmap, np.ndarray)
        assert isinstance(procmap, np.ndarray)

    def test_shapes_match_image(self):
        rawmap, procmap = self._run()
        assert rawmap.shape == self.img.shape
        assert procmap.shape == self.img.shape

    def test_procmap_is_uint8(self):
        _, procmap = self._run()
        assert procmap.dtype == np.uint8

    def test_rawmap_nan_outside_fiber_centers(self):
        """Only fiber-centre pixels should be non-NaN in rawmap."""
        rawmap, _ = self._run(boundary_measurement=True)
        non_nan_count = np.sum(~np.isnan(rawmap))
        # At most n_fibers pixels should be non-NaN
        assert non_nan_count <= len(self.fiber_df)

    def test_boundary_true_scales_0_to_90_degrees(self):
        """angle=90 → rawmap value = 255 at that fiber's centre."""
        df = pd.DataFrame({
            "center_row": [20.0],
            "center_col": [30.0],
            "angle":      [90.0],
        })
        rawmap, _ = draw_map(df, [90.0], self.img, True, self.MAP_PARAMS)
        assert pytest.approx(rawmap[20, 30], abs=1e-6) == 255.0

    def test_boundary_false_scales_0_to_180_degrees(self):
        """angle=180 → rawmap value = 255 at that fiber's centre."""
        df = pd.DataFrame({
            "center_row": [15.0],
            "center_col": [20.0],
            "angle":      [180.0],
        })
        rawmap, _ = draw_map(df, [180.0], self.img, False, self.MAP_PARAMS)
        assert pytest.approx(rawmap[15, 20], abs=1e-6) == 255.0

    def test_out_of_bounds_centers_skipped(self):
        """Fiber centres outside image bounds should not raise IndexError."""
        df = pd.DataFrame({
            "center_row": [-5.0, 200.0, 25.0],
            "center_col": [-1.0, 300.0, 10.0],
            "angle":      [45.0,  90.0, 30.0],
        })
        rawmap, _ = draw_map(df, df["angle"].values, self.img, True, self.MAP_PARAMS)
        # Only the in-bounds fiber should produce a non-NaN value
        assert np.sum(~np.isnan(rawmap)) == 1

    def test_accepts_center_1_center_2_aliases(self):
        """center_1/center_2 column aliases are accepted."""
        df = self.fiber_df.rename(columns={"center_row": "center_1", "center_col": "center_2"})
        rawmap, procmap = draw_map(df, self.angles, self.img, True, self.MAP_PARAMS)
        assert rawmap.shape == self.img.shape
        assert procmap.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# TestVisualizationWrappers
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizationWrappers:

    BINS = np.arange(2.5, 90, 5)
    MAP_PARAMS = {"STDfilter_size": 8, "SQUAREmaxfilter_size": 6, "GAUSSIANdiscfilter_sigma": 2}

    @pytest.fixture(autouse=True)
    def setup(self):
        self.img = _make_image(h=64, w=64)
        self.fiber_df = _make_fiber_df(n=5, img_size=50)
        self.angles = np.array([10.0, 20.0, 30.0, 60.0, 80.0])  # within bins (2.5–90°)
        self.in_flag = np.array([True, True, False, True, False])
        self.out_flag = ~self.in_flag

    @pytest.fixture(autouse=False)
    def close_figs(self):
        yield
        plt.close("all")

    # ── compute_angle_histogram ───────────────────────────────────────────────

    def test_compute_angle_histogram_no_boundary(self):
        """No boundary → all fiber angles land in histogram."""
        result = compute_angle_histogram(
            self.fiber_df,
            nearest_angles=None,
            in_curvs_flag=None,
            boundary_measurement=False,
            tif_boundary=0,
            bins=np.arange(-0.5, 181, 1),  # 0–180° bins; all angles should land
        )
        assert set(result.keys()) == {"counts", "bin_centers", "hist_data"}
        assert result["counts"].sum() == len(self.fiber_df)
        assert result["hist_data"].shape[0] == 2

    def test_compute_angle_histogram_boundary_mode(self):
        """Boundary mode tif3: only in_curvs_flag fibers counted."""
        result = compute_angle_histogram(
            self.fiber_df,
            nearest_angles=self.angles,
            in_curvs_flag=self.in_flag,
            boundary_measurement=True,
            tif_boundary=3,
            bins=self.BINS,
        )
        assert result["counts"].sum() <= len(self.fiber_df)
        assert result["counts"].sum() == self.in_flag.sum()

    # ── generate_fiber_overlay ────────────────────────────────────────────────

    def test_generate_fiber_overlay_no_boundary(self, close_figs):
        """tif_boundary=0 → (fig, ax) returned; image rendered on axes."""
        fig, ax = generate_fiber_overlay(
            self.img,
            self.fiber_df,
            coordinates=None,
            in_curvs_flag=self.in_flag,
            out_curvs_flag=self.out_flag,
            nearest_angles=self.angles,
            measured_boundary=None,
            fiber_mode=0,
            tif_boundary=0,
            boundary_measurement=False,
        )
        assert fig is not None
        assert ax is not None
        assert len(ax.get_images()) > 0

    def test_generate_fiber_overlay_tif3(self, close_figs):
        """tif_boundary=3 with boundary coords → (fig, ax) returned without error."""
        coords = {"ROI_1": np.column_stack([
            np.linspace(5, 55, 50), np.linspace(5, 55, 50)
        ])}
        fig, ax = generate_fiber_overlay(
            self.img,
            self.fiber_df,
            coordinates=coords,
            in_curvs_flag=self.in_flag,
            out_curvs_flag=self.out_flag,
            nearest_angles=self.angles,
            measured_boundary=None,
            fiber_mode=0,
            tif_boundary=3,
            boundary_measurement=True,
            make_associations=False,
        )
        assert fig is not None
        assert len(ax.get_images()) > 0

    # ── generate_fiber_heatmap ────────────────────────────────────────────────

    def test_generate_fiber_heatmap_returns_arrays(self, close_figs):
        """Returns (fig, rawmap, procmap) with correct shapes and dtype."""
        fig, rawmap, procmap = generate_fiber_heatmap(
            self.img,
            self.fiber_df,
            in_curvs_flag=self.in_flag,
            angles=self.angles,
            distances=None,
            tif_boundary=0,
            boundary_measurement=True,
            map_params=self.MAP_PARAMS,
        )
        assert fig is not None
        assert rawmap.shape == self.img.shape
        assert procmap.shape == self.img.shape
        assert procmap.dtype == np.uint8
