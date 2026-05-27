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


class TestAnalysisControllerAbortInvalidate:

    def test_abort_emits_analysis_aborted(self, state, controller):
        """abort() must emit analysis_aborted with the tracked step/image_id."""
        aborted = []
        controller.connect_analysis_aborted(lambda s, i: aborted.append((s, i)))
        controller._active_worker = mock.MagicMock()
        controller._active_step = "curvealign"
        controller._active_image_for_abort = "img_abc"
        controller.abort()
        assert aborted == [("curvealign", "img_abc")]
        assert controller._active_worker is None

    def test_abort_noop_when_no_worker(self, state, controller):
        """abort() must not crash when no worker is running."""
        controller.abort()  # should not raise

    def test_invalidate_result_curvealign_clears_state(self, state, controller):
        """invalidate_result removes the cached curvealign result from state."""
        state.curvealign_pipeline_results["img1"] = mock.MagicMock()
        controller.invalidate_result("img1", "curvealign")
        assert "img1" not in state.curvealign_pipeline_results

    def test_invalidate_result_ctfire_clears_state(self, state, controller):
        """invalidate_result removes the cached fiber result from state."""
        state.fiber_results["img2"] = mock.MagicMock()
        controller.invalidate_result("img2", "ctfire")
        assert "img2" not in state.fiber_results

    def test_invalidate_result_emits_aborted(self, state, controller):
        """invalidate_result must emit analysis_aborted for the image."""
        aborted = []
        controller.connect_analysis_aborted(lambda s, i: aborted.append((s, i)))
        controller.invalidate_result("img1", "curvealign")
        assert any(i == "img1" for _, i in aborted)

    def test_invalidate_all_clears_all_results(self, state, controller):
        """invalidate_all must clear both curvealign and fiber results."""
        state.curvealign_pipeline_results["a"] = mock.MagicMock()
        state.fiber_results["b"] = mock.MagicMock()
        controller.invalidate_all()
        assert "a" not in state.curvealign_pipeline_results
        assert "b" not in state.fiber_results

    def test_notify_result_loaded_fires_analysis_complete(self, state, controller):
        """notify_result_loaded must fire analysis_complete so widgets update."""
        received = []
        controller.connect_analysis_complete(lambda s, i, r: received.append((s, i)))
        fake = mock.MagicMock()
        controller.notify_result_loaded("curvealign", "img_x", fake)
        assert received == [("curvealign", "img_x")]
