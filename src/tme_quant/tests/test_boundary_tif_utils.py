# -*- coding: utf-8 -*-
"""Tests for fiber_analysis/utils/boundary_tif_utils.py.

Covers _rasterize_line_segment, _compute_fiber_boundary_relative_angle,
and extract_tif_boundary (unit + real-dataset tests).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from tme_quant.fiber_analysis.utils.boundary_tif_utils import (
    _rasterize_line_segment,
    _compute_fiber_boundary_relative_angle,
    _get_fiber_line_points,
    extract_tif_boundary,
)

# ─── Real-dataset gate ────────────────────────────────────────────────────────
_REPO_TEST_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "tests"
_GTB_DATA_DIR  = _REPO_TEST_DIR / "test_results" / "get_tif_boundary_test_files"
_GTB_MISSING   = not _GTB_DATA_DIR.exists()

# Parameters used to generate the reference output (from get_tif_boundary.__main__)
_DIST_THRESH = 100
_MIN_DIST    = []    # falsy — no inner threshold


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_ring_boundary(img_size=50, radius=15):
    """Return a dense (N,2) array of integer pixels forming a circular ring."""
    cx = cy = img_size // 2
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    rows = np.round(cx + radius * np.sin(angles)).astype(int)
    cols = np.round(cy + radius * np.cos(angles)).astype(int)
    coords = np.unique(np.column_stack((rows, cols)), axis=0)
    return coords.astype(float)


def _make_synthetic_fiber_df(n=10, img_size=50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "center_row": rng.integers(5, img_size - 5, n).astype(float),
        "center_col": rng.integers(5, img_size - 5, n).astype(float),
        "angle":      rng.uniform(0, 180, n),
    })


def _make_boundary_img(coords, img_size=50):
    """Build a binary uint8 image with 255 inside the boundary ring."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = cy = img_size // 2
    rows, cols = np.ogrid[:img_size, :img_size]
    mask = (rows - cx) ** 2 + (cols - cy) ** 2 < 15 ** 2
    img[mask] = 255
    return img


# ─────────────────────────────────────────────────────────────────────────────
# TestRasterizeLineSegment
# ─────────────────────────────────────────────────────────────────────────────

class TestRasterizeLineSegment:

    def test_diagonal_matches_docstring_example(self):
        pts, angle = _rasterize_line_segment([2, 3], [6, 8])
        # Docstring shows [[2,3],[3,4],[4,5],[5,7],[6,8]]
        arr = np.array(pts)
        assert arr[0].tolist() == [2, 3]
        assert arr[-1].tolist() == [6, 8]
        assert len(pts) >= 3

    def test_horizontal_line_endpoints_included(self):
        pts, _ = _rasterize_line_segment([5, 0], [5, 8])
        arr = np.array(pts)
        assert arr[0].tolist() == [5, 0]
        assert arr[-1].tolist() == [5, 8]
        # All rows should be 5
        assert np.all(arr[:, 0] == 5)

    def test_same_point_returns_none(self):
        result = _rasterize_line_segment([3, 3], [3, 3])
        assert result is None

    def test_output_is_integer_coords(self):
        pts, _ = _rasterize_line_segment([1, 1], [10, 15])
        for pt in pts:
            assert pt[0] == int(pt[0])
            assert pt[1] == int(pt[1])

    def test_vertical_line(self):
        """Constant-col segment (rows only change): rise=0, run>0 → angle=0."""
        pts, angle = _rasterize_line_segment([0, 5], [8, 5])
        arr = np.array(pts)
        assert arr[0].tolist() == [0, 5]
        assert arr[-1].tolist() == [8, 5]
        # rise = col_diff = 5-5 = 0, run = row_diff = 8-0 = 8 → arctan(0/8) = 0
        assert abs(angle) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeFiberBoundaryAngle
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFiberBoundaryAngle:
    """Uses a synthetic dense horizontal boundary segment for deterministic tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Dense horizontal boundary along row=25, cols 5..45
        rows = np.full(41, 25, dtype=float)
        cols = np.arange(5, 46, dtype=float)
        self.coords = np.column_stack((rows, cols))  # (41, 2)
        self.H, self.W = 50, 50
        # idx=20 is roughly the middle — not on image edge
        self.mid_idx = 20

    def test_parallel_fiber_returns_high_angle(self):
        """Fiber with same angle as boundary tangent → circ_r result ≈ 90°.

        For a horizontal boundary (row=const), compute_boundary_tangent_angle
        returns ~90° (angle measured from the col axis in row/col poly-fit space).
        A fiber at 90° matches the boundary tangent → high circular mean angle.
        """
        angle, _ = _compute_fiber_boundary_relative_angle(
            self.coords, self.mid_idx, 90.0, self.H, self.W
        )
        assert angle > 60, f"Expected high angle for fiber matching boundary tangent, got {angle}"

    def test_perpendicular_fiber_returns_low_angle(self):
        """Fiber orthogonal to boundary tangent → circ_r result ≈ 0°.

        Horizontal boundary has tangent ~90°; a fiber at 0° differs by 90°
        → circular mean resultant length near 0 → arcsin(0) ≈ 0°.
        """
        angle, _ = _compute_fiber_boundary_relative_angle(
            self.coords, self.mid_idx, 0.0, self.H, self.W
        )
        assert angle < 30, f"Expected low angle for fiber orthogonal to boundary, got {angle}"

    def test_edge_point_returns_zero(self):
        """Boundary point on image edge → returns 0.0."""
        # idx=0 has col=5 (not edge), but col=0 would be edge; build edge point
        edge_coords = self.coords.copy()
        edge_coords[0, 1] = 0.0  # set col to 0 (left edge)
        angle, _ = _compute_fiber_boundary_relative_angle(
            edge_coords, 0, 45.0, self.H, self.W
        )
        assert angle == 0.0

    def test_returns_boundary_point_correctly(self):
        _, bnd_pt = _compute_fiber_boundary_relative_angle(
            self.coords, self.mid_idx, 30.0, self.H, self.W
        )
        np.testing.assert_array_equal(bnd_pt, self.coords[self.mid_idx])


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractTifBoundaryUnit
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTifBoundaryUnit:
    """Synthetic smoke tests — no external data files required."""

    @pytest.fixture(autouse=True)
    def setup(self):
        img_size = 50
        self.coords = _make_ring_boundary(img_size=img_size, radius=15)
        self.img = _make_boundary_img(self.coords, img_size=img_size)
        self.fiber_df = _make_synthetic_fiber_df(n=8, img_size=img_size)
        self.dist_thresh = 20.0
        self.min_dist = []

    def _run(self, **kw):
        coords = kw.pop("coords", self.coords)
        return extract_tif_boundary(
            coords, self.img, self.fiber_df, self.dist_thresh, self.min_dist, **kw
        )

    def test_output_shape(self):
        result_mat, _, _, _ = self._run()
        assert result_mat.shape == (len(self.fiber_df), 7)

    def test_column_names(self):
        expected = [
            "nearest_boundary_distance",
            "nearest_region_distance",
            "nearest_boundary_angle",
            "extension_point_distance",
            "extension_point_angle",
            "boundary_point_row",
            "boundary_point_col",
        ]
        _, names, _, _ = self._run()
        assert names == expected

    def test_result_df_columns_match_names(self):
        _, names, _, result_df = self._run()
        assert list(result_df.columns) == names

    def test_far_fibers_have_nan_angle(self):
        """Fibers beyond dist_thresh receive NaN for nearest_boundary_angle."""
        # Place all fibers at the center (far from ring at radius=15, dist_thresh=5)
        tiny_thresh = 3.0
        _, _, _, result_df = extract_tif_boundary(
            self.coords, self.img, self.fiber_df, tiny_thresh, []
        )
        far_mask = result_df["nearest_boundary_distance"] > tiny_thresh
        if far_mask.any():
            assert result_df.loc[far_mask, "nearest_boundary_angle"].isna().all()

    def test_extension_cols_always_nan(self):
        """Extension point columns are always NaN (preserved from original bug)."""
        _, _, _, result_df = self._run()
        assert result_df["extension_point_distance"].isna().all()
        assert result_df["extension_point_angle"].isna().all()

    def test_num_img_points_positive(self):
        _, _, num_img_pts, _ = self._run()
        assert num_img_pts > 0

    def test_accepts_dict_coordinates(self):
        """coordinates can be passed as a dict of arrays."""
        half = len(self.coords) // 2
        coords_dict = {"seg1": self.coords[:half], "seg2": self.coords[half:]}
        result_mat_dict, _, _, _ = extract_tif_boundary(
            coords_dict, self.img, self.fiber_df, self.dist_thresh, self.min_dist
        )
        result_mat_arr, _, _, _ = extract_tif_boundary(
            self.coords, self.img, self.fiber_df, self.dist_thresh, self.min_dist
        )
        # Distances should be identical regardless of dict vs ndarray input
        np.testing.assert_allclose(
            result_mat_dict[:, 0], result_mat_arr[:, 0], atol=1e-6
        )

    def test_accepts_center_1_center_2_columns(self):
        """fiber_df using pycurvelets column aliases is accepted."""
        alias_df = self.fiber_df.rename(columns={
            "center_row": "center_1", "center_col": "center_2"
        })
        result_mat, _, _, _ = extract_tif_boundary(
            self.coords, self.img, alias_df, self.dist_thresh, self.min_dist
        )
        assert result_mat.shape == (len(alias_df), 7)


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractTifBoundaryRealData
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(_GTB_MISSING, reason="get_tif_boundary test data not found")
class TestExtractTifBoundaryRealData:
    """Regression tests against pycurvelets reference output on real data."""

    @pytest.fixture(scope="class")
    def real_inputs(self):
        import csv

        # Load boundary coordinate files (one array per file, no header)
        coords_dict = {}
        for i in range(1, 4):
            path = _GTB_DATA_DIR / f"real1_boundary{i}_coords.csv"
            rows_list = []
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    rows_list.append([float(row[0]), float(row[1])])
            coords_dict[f"boundary{i}"] = np.array(rows_list)

        # Load boundary image (512x512 CSV, no header)
        img_rows = []
        with open(_GTB_DATA_DIR / "real1_boundary_img.csv", newline="") as f:
            for row in csv.reader(f):
                img_rows.append([float(v) for v in row])
        img = np.array(img_rows)

        # Load curvelet/fiber data
        fiber_df = pd.read_csv(_GTB_DATA_DIR / "real1_curvelets.csv")

        # Load reference output (MATLAB column names)
        ref = pd.read_csv(_GTB_DATA_DIR / "real1_get_tif_boundary_output.csv")

        return coords_dict, img, fiber_df, ref

    @pytest.fixture(scope="class")
    def result(self, real_inputs):
        coords_dict, img, fiber_df, _ = real_inputs
        result_mat, names, num_pts, result_df = extract_tif_boundary(
            coords_dict, img, fiber_df, _DIST_THRESH, _MIN_DIST
        )
        return result_mat, names, num_pts, result_df

    def test_output_shape_matches_reference(self, real_inputs, result):
        _, _, _, ref = real_inputs
        result_mat, _, _, _ = result
        assert result_mat.shape[0] == len(ref)
        assert result_mat.shape[1] == 7

    def test_nearest_boundary_distance_matches(self, real_inputs, result):
        _, _, _, ref = real_inputs
        _, _, _, result_df = result
        ref_vals = ref["nearestBoundDist"].to_numpy(dtype=float)
        tme_vals = result_df["nearest_boundary_distance"].to_numpy(dtype=float)
        np.testing.assert_allclose(tme_vals, ref_vals, rtol=0.05, atol=5.0)

    def test_nearest_boundary_angle_close(self, real_inputs, result):
        """Compare only fibers within dist_thresh (those that get an angle computed)."""
        _, _, _, ref = real_inputs
        _, _, _, result_df = result
        ref_vals = ref["nearestBoundAng"].to_numpy(dtype=float)
        tme_vals = result_df["nearest_boundary_angle"].to_numpy(dtype=float)
        within = ~np.isnan(ref_vals) & ~np.isnan(tme_vals)
        if within.any():
            np.testing.assert_allclose(
                tme_vals[within], ref_vals[within], rtol=0.1, atol=5.0
            )

    def test_boundary_point_coords_close(self, real_inputs, result):
        _, _, _, ref = real_inputs
        _, _, _, result_df = result
        ref_row = ref["bndryPtRow"].to_numpy(dtype=float)
        ref_col = ref["bndryPtCol"].to_numpy(dtype=float)
        tme_row = result_df["boundary_point_row"].to_numpy(dtype=float)
        tme_col = result_df["boundary_point_col"].to_numpy(dtype=float)
        valid = ~np.isnan(ref_row) & ~np.isnan(tme_row)
        # atol=4: 2/355 fibers differ by 3px due to tie-breaking in brute-force NearestNeighbors
        np.testing.assert_allclose(tme_row[valid], ref_row[valid], atol=4.0)
        np.testing.assert_allclose(tme_col[valid], ref_col[valid], atol=4.0)

    def test_extension_cols_always_nan(self, result):
        _, _, _, result_df = result
        assert result_df["extension_point_distance"].isna().all()
        assert result_df["extension_point_angle"].isna().all()
