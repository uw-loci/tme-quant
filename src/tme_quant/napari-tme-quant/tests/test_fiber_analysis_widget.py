"""Tests for FiberAnalysisWidget."""

import pytest
import unittest.mock as mock

pytest.importorskip("qtpy", reason="qtpy required for widget tests")


@pytest.fixture
def widget(qtbot):
    from napari_tme_quant.widgets.fiber_analysis_widget import FiberAnalysisWidget
    w = FiberAnalysisWidget(napari_viewer=None)
    qtbot.addWidget(w)
    return w


class TestFiberAnalysisWidget:

    def test_instantiates(self, widget):
        assert widget is not None

    def test_default_method_is_ctfire(self, widget):
        assert widget._method_combo.currentIndex() == 0
        assert widget._ctfire_group.isVisible()
        assert not widget._curvealign_group.isVisible()

    def test_method_switch_shows_curvealign(self, widget):
        widget._method_combo.setCurrentIndex(1)
        assert not widget._ctfire_group.isVisible()
        assert widget._curvealign_group.isVisible()

    def test_method_switch_back_shows_ctfire(self, widget):
        widget._method_combo.setCurrentIndex(1)
        widget._method_combo.setCurrentIndex(0)
        assert widget._ctfire_group.isVisible()
        assert not widget._curvealign_group.isVisible()

    def test_ctfire_commit_btn_disabled_initially(self, widget):
        assert not widget._ctfire_commit_btn.isEnabled()

    def test_curvealign_commit_btn_disabled_initially(self, widget):
        assert not widget._ca_commit_btn.isEnabled()

    def test_precomputed_checkbox_enables_load_btn(self, widget):
        assert not widget._ca_load_btn.isEnabled()
        widget._ca_precomputed.setChecked(True)
        assert widget._ca_load_btn.isEnabled()
        widget._ca_precomputed.setChecked(False)
        assert not widget._ca_load_btn.isEnabled()

    def test_boundary_checkbox_shows_opts(self, widget):
        assert not widget._ca_boundary_opts.isVisible()
        widget._ca_boundary.setChecked(True)
        assert widget._ca_boundary_opts.isVisible()
        widget._ca_boundary.setChecked(False)
        assert not widget._ca_boundary_opts.isVisible()

    def test_get_ctfire_params(self, widget):
        widget._ctfire_threshold.setValue(0.15)
        widget._ctfire_pixel_size.setValue(0.5)
        params = widget.get_ctfire_params()
        assert params.ctfire_threshold == pytest.approx(0.15)
        assert params.pixel_size == pytest.approx(0.5)

    def test_get_curvealign_params_no_boundary(self, widget):
        kwargs = widget.get_curvealign_params()
        assert kwargs["tif_boundary"] == 0
        assert kwargs["boundary_img"] is None

    def test_get_curvealign_params_with_boundary(self, widget):
        widget._ca_boundary.setChecked(True)
        kwargs = widget.get_curvealign_params()
        assert kwargs["tif_boundary"] == 1
        assert "boundary_img" not in kwargs

    def test_ctfire_advanced_dialog_opens(self, widget, qtbot):
        widget._open_ctfire_advanced()
        assert widget._advanced_ctfire_dialog is not None

    def test_curvealign_advanced_dialog_opens(self, widget, qtbot):
        widget._open_curvealign_advanced()
        assert widget._advanced_curvealign_dialog is not None

    def test_on_ctfire_complete_updates_ui(self, widget):
        widget.on_ctfire_complete()
        assert "memory" in widget._ctfire_status.text().lower() or "●" in widget._ctfire_status.text()
        assert widget._ctfire_commit_btn.isEnabled()
        assert widget._ctfire_run_btn.isEnabled()

    def test_on_curvealign_complete_updates_ui(self, widget):
        widget.on_curvealign_complete()
        assert "●" in widget._ca_status.text()
        assert widget._ca_commit_btn.isEnabled()
        assert widget._ca_run_btn.isEnabled()
