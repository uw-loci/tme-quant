"""Tests for ROIController signal/state contract."""

import unittest.mock as mock

import pytest


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
    from napari_tme_quant.controllers.roi_controller import ROIController
    return ROIController(state, mock_viewer)


class TestROIControllerSignals:

    def test_connect_roi_drawn_registers_listener(self, controller):
        received = []
        controller.connect_roi_drawn(received.append)
        assert len(controller._on_roi_drawn) == 1

    def test_connect_roi_changed_registers_listener(self, controller):
        received = []
        controller.connect_roi_changed(received.append)
        assert len(controller._on_roi_changed) == 1

    def test_connect_multiple_roi_drawn_listeners(self, controller):
        a, b = [], []
        controller.connect_roi_drawn(a.append)
        controller.connect_roi_drawn(b.append)
        assert len(controller._on_roi_drawn) == 2

    def test_connect_multiple_roi_changed_listeners(self, controller):
        a, b = [], []
        controller.connect_roi_changed(a.append)
        controller.connect_roi_changed(b.append)
        assert len(controller._on_roi_changed) == 2


class TestROIControllerActions:

    def test_refresh_from_analysis_does_not_crash(self, controller):
        controller.refresh_from_analysis()

    def test_apply_to_all_images_does_not_crash(self, controller):
        controller.apply_to_all_images("roi_001")

    def test_refresh_from_analysis_does_not_modify_state(self, state, controller):
        controller.refresh_from_analysis()
        assert state.active_image_id is None
        assert state.fiber_results == {}

    def test_apply_to_all_images_arbitrary_id_does_not_crash(self, controller):
        controller.apply_to_all_images("")
        controller.apply_to_all_images("nonexistent_roi_xyz")


class TestROIControllerDebounce:

    def test_debounce_constant_is_300ms(self):
        from napari_tme_quant.controllers.roi_controller import ROIController
        assert ROIController._DEBOUNCE_MS == 300

    def test_controller_holds_viewer_reference(self, controller, mock_viewer):
        assert controller._viewer is mock_viewer
