"""TMEQuantDockWidget — top-level QTabWidget container for the plugin."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from qtpy.QtCore import Qt


def _placeholder(name: str) -> QWidget:
    """Temporary placeholder for tabs not yet implemented."""
    w = QWidget()
    lbl = QLabel(f"{name}\n(not yet implemented)")
    lbl.setAlignment(Qt.AlignCenter)
    layout = QVBoxLayout(w)
    layout.addWidget(lbl)
    return w


class TMEQuantDockWidget(QWidget):
    """Main plugin container: 6-tab workflow bar (Project → Log).

    All 11 sub-widgets are independently registerable via napari.yaml.
    This container assembles them into the standard workflow tab order.
    """

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Tab 1 — Project
        self._project_tab = self._load_project_widget()
        self._tabs.addTab(self._project_tab, "Project")

        # Tab 2 — Image
        self._image_tab = self._load_image_widget()
        self._tabs.addTab(self._image_tab, "Image")

        # Tab 3 — ROI Manager
        self._roi_tab = self._load_roi_manager_widget()
        self._tabs.addTab(self._roi_tab, "ROI Manager")

        # Tab 4 — Analysis (inner QTabWidget)
        self._analysis_tabs = self._build_analysis_tabs()
        self._tabs.addTab(self._analysis_tabs, "Analysis")

        # Tab 5 — Results (inner QTabWidget)
        self._results_tabs = self._build_results_tabs()
        self._tabs.addTab(self._results_tabs, "Results")

        # Tab 6 — Log
        self._log_tab = self._load_log_widget()
        self._tabs.addTab(self._log_tab, "Log")

    # ── Analysis sub-tabs ──────────────────────────────────────────────────────

    def _build_analysis_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._load_registration_widget(), "Registration")
        tabs.addTab(self._load_fiber_analysis_widget(), "Fiber")
        tabs.addTab(self._load_cell_analysis_widget(), "Cell")
        tabs.addTab(self._load_tme_pipeline_widget(), "TME")
        return tabs

    # ── Results sub-tabs ───────────────────────────────────────────────────────

    def _build_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._load_measurements_widget(), "Measurements")
        tabs.addTab(self._load_visualization_widget(), "Visualization")
        tabs.addTab(self._load_io_widget(), "I/O")
        return tabs

    # ── Widget loaders (try real widget; fall back to placeholder) ─────────────

    def _load_project_widget(self) -> QWidget:
        try:
            from .widgets.project_widget import ProjectWidget
            return ProjectWidget(self._viewer)
        except ImportError:
            return _placeholder("Project")

    def _load_image_widget(self) -> QWidget:
        try:
            from .widgets.image_widget import ImageWidget
            return ImageWidget(self._viewer)
        except ImportError:
            return _placeholder("Image")

    def _load_roi_manager_widget(self) -> QWidget:
        try:
            from .widgets.roi_manager_widget import ROIManagerWidget
            return ROIManagerWidget(self._viewer)
        except ImportError:
            return _placeholder("ROI Manager")

    def _load_registration_widget(self) -> QWidget:
        try:
            from .widgets.registration_widget import RegistrationWidget
            return RegistrationWidget(self._viewer)
        except ImportError:
            return _placeholder("Registration")

    def _load_fiber_analysis_widget(self) -> QWidget:
        try:
            from .widgets.fiber_analysis_widget import FiberAnalysisWidget
            return FiberAnalysisWidget(self._viewer)
        except ImportError:
            return _placeholder("Fiber Analysis")

    def _load_cell_analysis_widget(self) -> QWidget:
        try:
            from .widgets.cell_analysis_widget import CellAnalysisWidget
            return CellAnalysisWidget(self._viewer)
        except ImportError:
            return _placeholder("Cell Analysis")

    def _load_tme_pipeline_widget(self) -> QWidget:
        try:
            from .widgets.tme_pipeline_widget import TMEPipelineWidget
            return TMEPipelineWidget(self._viewer)
        except ImportError:
            return _placeholder("TME Pipelines")

    def _load_measurements_widget(self) -> QWidget:
        try:
            from .widgets.measurements_widget import MeasurementsWidget
            return MeasurementsWidget(self._viewer)
        except ImportError:
            return _placeholder("Measurements")

    def _load_visualization_widget(self) -> QWidget:
        try:
            from .widgets.visualization_widget import VisualizationWidget
            return VisualizationWidget(self._viewer)
        except ImportError:
            return _placeholder("Visualization")

    def _load_io_widget(self) -> QWidget:
        try:
            from .widgets.io_widget import IOWidget
            return IOWidget(self._viewer)
        except ImportError:
            return _placeholder("I/O")

    def _load_log_widget(self) -> QWidget:
        try:
            from .widgets.log_widget import LogWidget
            return LogWidget(self._viewer)
        except ImportError:
            return _placeholder("Log")
