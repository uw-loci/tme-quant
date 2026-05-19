"""Tests for AnalysisController fiber path."""

import pytest
import unittest.mock as mock
import numpy as np


@pytest.fixture
def state():
    from napari_tme_quant.controllers.state import PluginState
    return PluginState()


@pytest.fixture
def controller(state):
    from napari_tme_quant.controllers.analysis_controller import AnalysisController
    return AnalysisController(state)


class TestAnalysisControllerFiberPath:

    def test_ctfire_result_stored_in_state(self, state, controller):
        """_on_ctfire_result must store result in PluginState.fiber_results."""
        fake_result = mock.MagicMock()
        fake_result.fibers = []
        controller._on_ctfire_result("test_image", fake_result)
        assert state.fiber_results.get("test_image") is fake_result

    def test_curvealign_result_stored_in_state(self, state, controller):
        """_on_curvealign_result must store result in curvealign_pipeline_results."""
        fake_result = mock.MagicMock()
        fake_result.fiber_structure = None
        controller._on_curvealign_result("img_001", fake_result)
        assert state.curvealign_pipeline_results.get("img_001") is fake_result

    def test_curvealign_none_result_not_stored(self, state, controller):
        """None result from pipeline must not be stored."""
        controller._on_curvealign_result("img_002", None)
        assert "img_002" not in state.curvealign_pipeline_results

    def test_analysis_complete_signal_fires(self, state, controller):
        received = []
        controller.connect_analysis_complete(lambda step, iid, r: received.append((step, iid)))
        fake = mock.MagicMock()
        fake.fibers = []
        controller._on_ctfire_result("img_x", fake)
        assert received == [("fiber", "img_x")]

    def test_committed_signal_fires_on_commit(self, state, controller):
        # Store a fake result first
        fake = mock.MagicMock()
        fake.fibers = [mock.MagicMock()]
        state.fiber_results["img_y"] = fake

        committed = []
        controller.connect_committed_to_hierarchy(
            lambda iid, otype: committed.append((iid, otype))
        )

        with mock.patch.object(state.hierarchy, "attach_fiber_result"):
            controller.commit_fiber_result("img_y", method="ctfire")

        assert committed == [("img_y", "fiber")]

    def test_commit_with_no_result_does_not_crash(self, state, controller):
        """If there is no result in state, commit should just log a warning."""
        controller.commit_fiber_result("nonexistent_id", method="ctfire")

    def test_get_or_create_image_entry_creates_entry(self, state, controller):
        entry = controller._get_or_create_image_entry("brand_new")
        assert entry is not None
        assert entry.object_id == "brand_new"

    def test_get_or_create_image_entry_returns_existing(self, state, controller):
        e1 = controller._get_or_create_image_entry("existing")
        e2 = controller._get_or_create_image_entry("existing")
        assert e1 is e2
