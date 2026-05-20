"""Tests for VisualizationController."""

import pytest
import unittest.mock as mock
import numpy as np
import pandas as pd


@pytest.fixture
def state():
    from napari_tme_quant.controllers.state import PluginState
    return PluginState()


@pytest.fixture
def mock_viewer():
    viewer = mock.MagicMock()
    viewer.layers = []
    return viewer


@pytest.fixture
def controller(state, mock_viewer):
    from napari_tme_quant.controllers.visualization_controller import VisualizationController
    return VisualizationController(state, mock_viewer)


class TestVisualizationControllerLayers:

    def test_on_committed_fiber_calls_add_shapes(self, state, mock_viewer, controller):
        """on_committed('fiber') should call viewer.add_shapes with fiber line data."""
        result = mock.MagicMock()
        fiber = mock.MagicMock()
        fiber.get_position.return_value = [50.0, 80.0]
        fiber.orientation = 45.0
        fiber.length = 20.0
        fiber.tacs_type = None
        result.fibers = [fiber]
        state.fiber_results["img_1"] = result

        controller.on_committed("img_1", "fiber")

        mock_viewer.add_shapes.assert_called_once()
        call_kwargs = mock_viewer.add_shapes.call_args
        name = call_kwargs[1].get("name") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]["name"]
        assert "img_1" in name
        assert "Fibers" in name

    def test_on_committed_curvealign_calls_add_points(self, state, mock_viewer, controller):
        """on_committed('curvealign') should call viewer.add_points."""
        result = mock.MagicMock()
        result.fiber_features_df = pd.DataFrame({
            "center_row": [10.0, 20.0],
            "center_col": [5.0, 15.0],
            "angle": [30.0, 60.0],
        })
        result.in_curvs_flag = None
        state.curvealign_pipeline_results["img_2"] = result

        controller.on_committed("img_2", "curvealign")

        mock_viewer.add_points.assert_called()

    def test_on_committed_unknown_type_does_nothing(self, state, mock_viewer, controller):
        """Unknown obj_type should not crash or call viewer methods."""
        controller.on_committed("img_3", "unknown_type")
        mock_viewer.add_shapes.assert_not_called()
        mock_viewer.add_points.assert_not_called()

    def test_classify_df_tacs_outside_when_no_angles(self, controller):
        """If angle column is missing, all fibers should be 'outside'."""
        result = mock.MagicMock()
        result.in_curvs_flag = None
        df = pd.DataFrame({"center_row": [1.0], "center_col": [2.0]})
        labels = controller._classify_df_tacs(df, result)
        assert labels[0] == "outside"

    def test_classify_df_tacs_high_angle_is_tacs3(self, controller):
        """angle_to_boundary_tangent ≥ 60° → TACS-3."""
        result = mock.MagicMock()
        result.in_curvs_flag = [True]
        df = pd.DataFrame({
            "center_row": [1.0],
            "center_col": [2.0],
            "angle_to_boundary_tangent": [75.0],
        })
        labels = controller._classify_df_tacs(df, result)
        assert labels[0] == "TACS-3"

    def test_classify_df_tacs_low_angle_is_tacs2(self, controller):
        """angle_to_boundary_tangent ≤ 30° → TACS-2."""
        result = mock.MagicMock()
        result.in_curvs_flag = [True]
        df = pd.DataFrame({
            "center_row": [1.0],
            "center_col": [2.0],
            "angle_to_boundary_tangent": [15.0],
        })
        labels = controller._classify_df_tacs(df, result)
        assert labels[0] == "TACS-2"

    def test_on_image_selected_hides_other_layers(self, state, mock_viewer, controller):
        """on_image_selected should show layers matching image_id and hide others.

        Visibility is applied via QTimer.singleShot(0) to avoid vispy recursion;
        we patch QTimer to execute the callback synchronously in tests.
        """
        layer_a = mock.MagicMock()
        layer_a.name = "img_A :: Fibers :: all"
        layer_raw = mock.MagicMock()
        layer_raw.name = "img_A"           # raw image layer — must stay visible
        layer_b = mock.MagicMock()
        layer_b.name = "img_B :: Fibers :: all"
        mock_viewer.layers = [layer_a, layer_raw, layer_b]

        # Patch QTimer.singleShot to run the callback immediately in tests
        with mock.patch(
            "napari_tme_quant.controllers.visualization_controller.QTimer"
        ) as mock_timer:
            mock_timer.singleShot.side_effect = lambda _ms, fn: fn()
            controller.on_image_selected("img_A")

        assert layer_a.visible is True
        assert layer_raw.visible is True   # raw image layer shown for selected image
        assert layer_b.visible is False
