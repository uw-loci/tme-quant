"""
Tests for tme_quant.fiber_analysis.utils.geometry_utils
"""
import math

import numpy as np
import pytest

from tme_quant.fiber_analysis.utils.geometry_utils import (
    # public
    compute_angle_to_boundary_normal,
    compute_angle_to_boundary_normal_simplified,
    compute_boundary_tangent_angle,
    compute_fiber_properties,
    compute_relative_fiber_angles,
    find_nearest_boundary_index,
    # private – exercised through the public surface but also tested directly
    _angle_between_orientations,
    _circ_r,
    _find_connected_pts,
    _get_first_neighbor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _close(a, b, atol=1e-6):
    """Convenience: assert two floats are approximately equal."""
    assert abs(a - b) <= atol, f"{a!r} != {b!r} (atol={atol})"


# ─────────────────────────────────────────────────────────────────────────────
# _angle_between_orientations
# ─────────────────────────────────────────────────────────────────────────────

class TestAngleBetweenOrientations:
    """_angle_between_orientations returns angle_to_tangent ∈ [0°, 90°].

    angle_to_tangent = 0° → fiber parallel to boundary (TACS-2)
    angle_to_tangent = 90° → fiber perpendicular to boundary (TACS-3)
    """

    def test_fiber_parallel_to_boundary(self):
        # Boundary normal = 90° (vertical); fiber = 0° (horizontal)
        # → fiber is parallel to boundary surface → angle_to_tangent = 0°
        result = _angle_between_orientations(0.0, 90.0)
        _close(result, 0.0)

    def test_fiber_perpendicular_to_boundary(self):
        # Boundary normal = 90° (vertical); fiber = 90° (vertical)
        # → fiber is perpendicular to boundary surface → angle_to_tangent = 90°
        result = _angle_between_orientations(90.0, 90.0)
        _close(result, 90.0)

    def test_fiber_45_degrees_to_boundary(self):
        # Normal = 90°, fiber = 45° → angle_to_normal = 45° → angle_to_tangent = 45°
        result = _angle_between_orientations(45.0, 90.0)
        _close(result, 45.0)

    def test_same_angle_returns_90(self):
        # fiber == normal → perfectly perpendicular to boundary
        for angle in (0.0, 30.0, 60.0, 90.0, 135.0):
            result = _angle_between_orientations(angle, angle)
            _close(result, 90.0, atol=1e-5)

    def test_angles_beyond_180_are_normalised(self):
        # 270° is equivalent to 270 % 180 = 90°
        result_raw = _angle_between_orientations(0.0, 90.0)
        result_wrap = _angle_between_orientations(0.0, 270.0)
        _close(result_raw, result_wrap)

    def test_result_within_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = float(rng.uniform(0, 360))
            b = float(rng.uniform(0, 360))
            result = _angle_between_orientations(a, b)
            assert 0.0 <= result <= 90.0, f"Out of bounds: {result} for ({a}, {b})"


# ─────────────────────────────────────────────────────────────────────────────
# compute_angle_to_boundary_normal
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAngleToBoundaryNormal:
    """Returns the acute angle [0°, 90°] between fiber and boundary normal."""

    def test_vertical_fiber_horizontal_boundary(self):
        # Horizontal boundary tangent = 0° → normal = 90°
        # Vertical fiber = 90° → parallel to normal → angle_to_normal = 0°
        result = compute_angle_to_boundary_normal(90.0, (0, 0), (10, 0))
        _close(result, 0.0, atol=1e-5)

    def test_horizontal_fiber_horizontal_boundary(self):
        # Horizontal boundary tangent = 0° → normal = 90°
        # Horizontal fiber = 0° → perpendicular to normal → angle_to_normal = 90°
        result = compute_angle_to_boundary_normal(0.0, (0, 0), (10, 0))
        _close(result, 90.0, atol=1e-5)

    def test_45_degree_fiber(self):
        # Horizontal boundary: normal = 90°; fiber = 45° → diff = 45°
        result = compute_angle_to_boundary_normal(45.0, (0, 0), (10, 0))
        _close(result, 45.0, atol=1e-5)

    def test_vertical_boundary(self):
        # Vertical boundary tangent = 90° → normal = 0°
        # Horizontal fiber = 0° → parallel to normal → angle = 0°
        result = compute_angle_to_boundary_normal(0.0, (0, 0), (0, 10))
        _close(result, 0.0, atol=1e-5)

    def test_degenerate_coincident_points_returns_nan(self):
        result = compute_angle_to_boundary_normal(45.0, (5, 5), (5, 5))
        assert math.isnan(result)

    def test_result_within_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            angle = float(rng.uniform(0, 180))
            p1 = tuple(rng.uniform(-10, 10, 2).tolist())
            # Ensure distinct points
            p2 = tuple((rng.uniform(-10, 10, 2) + np.array([1.0, 0.0])).tolist())
            result = compute_angle_to_boundary_normal(angle, p1, p2)
            assert 0.0 <= result <= 90.0, f"Out of bounds: {result}"

    def test_symmetry_of_boundary_direction(self):
        # Reversing boundary direction should give the same angle
        a = compute_angle_to_boundary_normal(30.0, (0, 0), (10, 5))
        b = compute_angle_to_boundary_normal(30.0, (10, 5), (0, 0))
        _close(a, b, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# compute_angle_to_boundary_normal_simplified
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAngleToBoundaryNormalSimplified:
    """Matches the two-argument variant when pre-computed tangent angle is known."""

    def test_vertical_fiber_horizontal_tangent(self):
        result = compute_angle_to_boundary_normal_simplified(90.0, 0.0)
        _close(result, 0.0, atol=1e-5)

    def test_horizontal_fiber_horizontal_tangent(self):
        result = compute_angle_to_boundary_normal_simplified(0.0, 0.0)
        _close(result, 90.0, atol=1e-5)

    def test_consistent_with_two_point_variant(self):
        # Horizontal boundary: tangent computed from (0,0)→(10,0) is 0°
        two_pt = compute_angle_to_boundary_normal(45.0, (0, 0), (10, 0))
        simplified = compute_angle_to_boundary_normal_simplified(45.0, 0.0)
        _close(two_pt, simplified, atol=1e-5)

    def test_result_within_bounds(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            fiber = float(rng.uniform(0, 180))
            tangent = float(rng.uniform(0, 180))
            result = compute_angle_to_boundary_normal_simplified(fiber, tangent)
            assert 0.0 <= result <= 90.0, f"Out of bounds: {result}"

    def test_wrap_beyond_180(self):
        # 270° tangent ≡ 90° tangent
        r1 = compute_angle_to_boundary_normal_simplified(45.0, 90.0)
        r2 = compute_angle_to_boundary_normal_simplified(45.0, 270.0)
        _close(r1, r2)


# ─────────────────────────────────────────────────────────────────────────────
# _circ_r
# ─────────────────────────────────────────────────────────────────────────────

class TestCircR:
    def test_all_aligned_returns_one(self):
        alpha = np.zeros(10)        # all pointing in the same direction
        r = _circ_r(alpha)
        _close(r, 1.0)

    def test_opposite_directions_returns_zero(self):
        alpha = np.array([0.0, np.pi])
        r = _circ_r(alpha)
        _close(r, 0.0)

    def test_uniform_distribution_near_zero(self):
        alpha = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        r = _circ_r(alpha)
        assert r < 0.01

    def test_single_value_returns_one(self):
        r = _circ_r(np.array([1.23]))
        _close(r, 1.0)

    def test_weighted_result(self):
        # Equal weight on two opposed directions → 0
        alpha = np.array([0.0, np.pi])
        w = np.array([1.0, 1.0])
        r = _circ_r(alpha, w=w)
        _close(r, 0.0)

    def test_weighted_skew(self):
        # Heavy weight on 0°, small weight on π
        alpha = np.array([0.0, np.pi])
        w = np.array([10.0, 1.0])
        r = _circ_r(alpha, w=w)
        assert r > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# find_nearest_boundary_index
# ─────────────────────────────────────────────────────────────────────────────

class TestFindNearestBoundaryIndex:
    def test_exact_match(self):
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        assert find_nearest_boundary_index(coords, 1.0, 0.0) == 1

    def test_closest_of_three(self):
        coords = np.array([[0, 0], [5, 0], [10, 0]])
        # Query (3, 0): closest to (0,0)? No → (5,0)? dist=4; (0,0)? dist=9; (10,0)? dist=49
        # Actually dist from (p, q) = (3,0): to [0,0] = 9, to [5,0] = 4, to [10,0] = 49
        assert find_nearest_boundary_index(coords, 3.0, 0.0) == 1

    def test_equidistant_returns_first(self):
        # (0,0) and (2,0) are equidistant from (1,0)
        coords = np.array([[0, 0], [2, 0]])
        idx = find_nearest_boundary_index(coords, 1.0, 0.0)
        assert idx in (0, 1)

    def test_single_point(self):
        coords = np.array([[7, 3]])
        assert find_nearest_boundary_index(coords, 99.0, 99.0) == 0

    def test_2d_coords(self):
        coords = np.array([[0, 0], [3, 4], [6, 8]])
        # Query (3, 4) → index 1 exact
        assert find_nearest_boundary_index(coords, 3.0, 4.0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# _get_first_neighbor
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFirstNeighbor:
    def _make_line(self):
        """Horizontal line from col 0 to 4 at row 5."""
        return np.array([[5, i] for i in range(5)], dtype=int)

    def test_forward_returns_right_neighbor(self):
        coords = self._make_line()
        visited = np.zeros(len(coords), dtype=bool)
        visited[0] = True          # mark start as visited
        nbr = _get_first_neighbor(coords, 0, visited, direction=2)
        assert nbr == 1

    def test_backward_returns_left_neighbor(self):
        coords = self._make_line()
        visited = np.zeros(len(coords), dtype=bool)
        visited[4] = True
        nbr = _get_first_neighbor(coords, 4, visited, direction=1)
        assert nbr == 3

    def test_all_neighbors_visited_returns_self(self):
        coords = self._make_line()
        visited = np.ones(len(coords), dtype=bool)
        visited[2] = False          # only the query point is unvisited
        nbr = _get_first_neighbor(coords, 2, visited, direction=2)
        assert nbr == 2


# ─────────────────────────────────────────────────────────────────────────────
# _find_connected_pts
# ─────────────────────────────────────────────────────────────────────────────

class TestFindConnectedPts:
    def _make_long_line(self):
        """30-point horizontal line at row 0."""
        return np.array([[0, c] for c in range(30)], dtype=int)

    def test_centre_returns_window(self):
        coords = self._make_long_line()
        pts = _find_connected_pts(coords, idx=15, num=7)
        assert pts.shape == (7, 2)
        assert not np.any(np.isnan(pts))

    def test_near_edge_returns_nan(self):
        # Only 5 points; asking for 7 around index 0 should fail (not enough backward)
        coords = np.array([[0, c] for c in range(5)], dtype=int)
        pts = _find_connected_pts(coords, idx=0, num=7)
        assert np.all(np.isnan(pts))

    def test_window_order(self):
        coords = self._make_long_line()
        pts = _find_connected_pts(coords, idx=15, num=5)
        # Middle element (index 2) should be coords[15]
        assert pts[2, 1] == 15, f"Expected col 15, got {pts[2, 1]}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_boundary_tangent_angle
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBoundaryTangentAngle:
    def _horizontal_line(self, length=60):
        return np.array([[10, c] for c in range(length)], dtype=int)

    def _vertical_line(self, length=60):
        return np.array([[r, 10] for r in range(length)], dtype=int)

    def _diagonal_line(self, length=60):
        """45° diagonal line."""
        return np.array([[i, i] for i in range(length)], dtype=int)

    def test_horizontal_line_tangent_near_90(self):
        # coords are (row, col); a horizontal line has constant row and varying col.
        # arctan2(Δcol, Δrow) = arctan2(large, 0) = 90°.
        coords = self._horizontal_line()
        angle = compute_boundary_tangent_angle(coords, idx=30, num=21)
        assert not math.isnan(angle)
        assert 80.0 < angle < 100.0, f"Expected ~90°, got {angle}"

    def test_vertical_line_tangent_near_zero(self):
        # A vertical line has varying row and constant col.
        # arctan2(Δcol, Δrow) = arctan2(0, large) = 0°.
        coords = self._vertical_line()
        angle = compute_boundary_tangent_angle(coords, idx=30, num=21)
        assert not math.isnan(angle)
        assert angle < 10.0 or angle > 170.0, f"Expected ~0°, got {angle}"

    def test_diagonal_tangent_near_45(self):
        coords = self._diagonal_line()
        angle = compute_boundary_tangent_angle(coords, idx=30, num=21)
        assert not math.isnan(angle)
        assert 35.0 < angle < 55.0, f"Expected ~45°, got {angle}"

    def test_insufficient_neighbourhood_returns_nan(self):
        # Only 10 points; asking for num=21 around the start should return NaN
        coords = np.array([[0, c] for c in range(10)], dtype=int)
        angle = compute_boundary_tangent_angle(coords, idx=0, num=21)
        assert math.isnan(angle)

    def test_result_in_range(self):
        coords = self._horizontal_line()
        angle = compute_boundary_tangent_angle(coords, idx=30, num=21)
        assert math.isnan(angle) or 0.0 <= angle < 180.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_fiber_properties
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFiberProperties:
    def _blank_image(self, h=100, w=100):
        img = np.zeros((h, w), dtype=np.uint8)
        img[40:60, :] = 200        # horizontal bright band
        return img

    def test_straight_horizontal_fiber(self):
        centerline = np.array([[50, c] for c in range(10, 90)], dtype=float)
        image = self._blank_image()
        props = compute_fiber_properties(centerline, image, pixel_size=1.0)
        # Length ≈ 80 px
        assert 70.0 < props['length'] < 90.0, f"length={props['length']}"
        # Straightness ≈ 1 for a perfectly straight fiber
        assert props['straightness'] > 0.99
        # Curvature should be near 0
        assert props['curvature'] < 0.1

    def test_length_scales_with_pixel_size(self):
        centerline = np.array([[50, c] for c in range(0, 10)], dtype=float)
        image = self._blank_image()
        p1 = compute_fiber_properties(centerline, image, pixel_size=1.0)
        p2 = compute_fiber_properties(centerline, image, pixel_size=2.0)
        _close(p2['length'], p1['length'] * 2.0, atol=1e-4)

    def test_degenerate_single_point_returns_zeros(self):
        props = compute_fiber_properties(np.array([[5, 5]]), self._blank_image())
        assert props['length'] == 0.0
        assert props['straightness'] == 0.0

    def test_angle_horizontal_fiber(self):
        centerline = np.array([[50, c] for c in range(0, 20)], dtype=float)
        image = self._blank_image()
        props = compute_fiber_properties(centerline, image, pixel_size=1.0)
        # End-to-end vector is horizontal; angle should be ~0° or ~180°
        assert props['angle'] < 10.0 or props['angle'] > 170.0

    def test_returned_keys(self):
        centerline = np.array([[50, c] for c in range(0, 20)], dtype=float)
        props = compute_fiber_properties(centerline, self._blank_image())
        assert set(props.keys()) == {'length', 'width', 'straightness', 'angle', 'curvature'}

    def test_none_centerline_returns_defaults(self):
        props = compute_fiber_properties(None, self._blank_image())
        assert props['length'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_relative_fiber_angles
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRelativeFiberAngles:
    """Test compute_relative_fiber_angles with a simple rectangular ROI."""

    def _square_roi(self):
        """30×30 square centred at (50, 50); coords in (row, col) order."""
        r, c = 50, 50
        size = 15
        return np.array([
            [r - size, c - size],
            [r - size, c + size],
            [r + size, c + size],
            [r + size, c - size],
            [r - size, c - size],   # closed
        ], dtype=float)

    def test_all_three_angles_returned(self):
        coords = self._square_roi()
        angles, meas = compute_relative_fiber_angles(
            obj_center=(50, 50), obj_angle=45.0, roi_coords=coords,
        )
        assert set(angles.keys()) == {
            'angle_to_boundary_tangent',
            'angle_to_roi_orientation',
            'angle_to_centers_line',
        }

    def test_angles_within_bounds(self):
        coords = self._square_roi()
        angles, _ = compute_relative_fiber_angles(
            obj_center=(50, 50), obj_angle=30.0, roi_coords=coords,
        )
        for name, val in angles.items():
            if val is not None:
                assert 0.0 <= val <= 90.0, f"{name}={val} out of [0,90]"

    def test_angle_option_1_returns_only_boundary_tangent(self):
        coords = self._square_roi()
        angles, _ = compute_relative_fiber_angles(
            obj_center=(60, 50), obj_angle=0.0,
            roi_coords=coords, angle_option=1,
        )
        assert angles['angle_to_boundary_tangent'] is not None
        assert angles['angle_to_roi_orientation'] is None
        assert angles['angle_to_centers_line'] is None

    def test_angle_option_2_returns_only_roi_orientation(self):
        coords = self._square_roi()
        angles, _ = compute_relative_fiber_angles(
            obj_center=(60, 50), obj_angle=0.0,
            roi_coords=coords, angle_option=2,
        )
        assert angles['angle_to_roi_orientation'] is not None
        assert angles['angle_to_boundary_tangent'] is None
        assert angles['angle_to_centers_line'] is None

    def test_angle_option_3_returns_only_centers_line(self):
        coords = self._square_roi()
        angles, _ = compute_relative_fiber_angles(
            obj_center=(60, 50), obj_angle=0.0,
            roi_coords=coords, angle_option=3,
        )
        assert angles['angle_to_centers_line'] is not None
        assert angles['angle_to_boundary_tangent'] is None
        assert angles['angle_to_roi_orientation'] is None

    def test_roi_measurements_keys(self):
        coords = self._square_roi()
        _, meas = compute_relative_fiber_angles(
            obj_center=(50, 50), obj_angle=0.0, roi_coords=coords,
        )
        assert 'center' in meas
        assert 'orientation' in meas
        assert 'area' in meas
        assert 'boundary' in meas

    def test_with_image_size_uses_regionprops(self):
        coords = self._square_roi()
        angles, meas = compute_relative_fiber_angles(
            obj_center=(50, 50), obj_angle=0.0,
            roi_coords=coords, image_size=(100, 100),
        )
        # regionprops gives a real pixel area
        assert meas['area'] > 0
        for val in angles.values():
            if val is not None:
                assert 0.0 <= val <= 90.0

    def test_dense_boundary_path(self):
        """Exercise the dense_boundary=True branch with a horizontal line trace."""
        # 60-point horizontal dense trace at row 50
        coords = np.array([[50, c] for c in range(20, 80)], dtype=float)
        angles, _ = compute_relative_fiber_angles(
            obj_center=(50, 50), obj_angle=45.0,
            roi_coords=coords, angle_option=1,
            dense_boundary=True,
        )
        val = angles['angle_to_boundary_tangent']
        if val is not None:       # NaN path is acceptable for edge points
            assert 0.0 <= val <= 90.0

    def test_fiber_angle_normalised_to_180(self):
        """Angles ≥ 180 should be treated the same as their modulo equivalent."""
        coords = self._square_roi()
        a1, _ = compute_relative_fiber_angles(
            obj_center=(60, 50), obj_angle=30.0, roi_coords=coords,
        )
        a2, _ = compute_relative_fiber_angles(
            obj_center=(60, 50), obj_angle=210.0, roi_coords=coords,
        )
        for key in a1:
            if a1[key] is not None and a2[key] is not None:
                _close(a1[key], a2[key], atol=1e-5)
