"""TMEQuantDockWidget — top-level QTabWidget container for the plugin."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget


def _placeholder(name: str) -> QWidget:
    w = QWidget()
    lbl = QLabel(f"{name}\n(not yet implemented)")
    lbl.setAlignment(Qt.AlignCenter)
    QVBoxLayout(w).addWidget(lbl)
    return w


class TMEQuantDockWidget(QWidget):
    """Main plugin container: 6-tab workflow bar (Project → Log).

    All 11 sub-widgets are independently registerable via napari.yaml.
    This container assembles them into the standard workflow tab order and
    wires all controllers together at the end of __init__.
    """

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer

        # Widget references (set by loader methods below)
        self._project_widget  = None
        self._image_widget    = None
        self._tme_widget      = None
        self._viz_widget      = None
        self._log_widget      = None

        # Controller references (set by _wire_controllers)
        self._state           = None
        self._proj_ctrl       = None
        self._analysis_ctrl   = None
        self._viz_ctrl        = None
        self._log_ctrl        = None

        self._build_ui()
        self._wire_controllers()

    # ── Tab assembly ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._load_project_widget(),  "Project")
        self._tabs.addTab(self._load_image_widget(),    "Image")
        self._tabs.addTab(self._load_roi_manager_widget(), "ROI Manager")
        self._tabs.addTab(self._build_analysis_tabs(),  "Analysis")
        self._tabs.addTab(self._build_results_tabs(),   "Results")
        self._tabs.addTab(self._load_log_widget(),      "Log")

    def _build_analysis_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._load_registration_widget(),    "Registration")
        tabs.addTab(self._load_fiber_analysis_widget(),  "Fiber")
        tabs.addTab(self._load_cell_analysis_widget(),   "Cell")
        tabs.addTab(self._load_tme_pipeline_widget(),    "TME")
        return tabs

    def _build_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._load_measurements_widget(),    "Measurements")
        tabs.addTab(self._load_visualization_widget(),   "Visualization")
        tabs.addTab(self._load_io_widget(),              "I/O")
        return tabs

    # ── Controller wiring ──────────────────────────────────────────────────────

    def _wire_controllers(self) -> None:
        """Instantiate all controllers and connect signals to widgets."""
        try:
            from .controllers.state import PluginState
            from .controllers.project_controller import ProjectController
            from .controllers.analysis_controller import AnalysisController
            from .controllers.visualization_controller import VisualizationController
            from .controllers.log_controller import LogController
        except ImportError:
            return  # headless / partial install — skip wiring

        self._state         = PluginState()
        self._log_ctrl      = LogController(self._state, self._log_widget)
        self._proj_ctrl     = ProjectController(self._state)
        self._analysis_ctrl = AnalysisController(self._state, self._log_ctrl)
        self._viz_ctrl      = VisualizationController(self._state, self._viewer)

        # Inject controllers into widgets that need them
        if self._project_widget and hasattr(self._project_widget, "set_controller"):
            self._project_widget.set_controller(self._proj_ctrl)
        if self._image_widget and hasattr(self._image_widget, "set_controller"):
            self._image_widget.set_controller(self._proj_ctrl)
        if self._tme_widget and hasattr(self._tme_widget, "set_controller"):
            self._tme_widget.set_controller(self._analysis_ctrl)
        if self._tme_widget and hasattr(self._tme_widget, "set_project_controller"):
            self._tme_widget.set_project_controller(self._proj_ctrl)
        if self._viz_widget and hasattr(self._viz_widget, "set_controller"):
            self._viz_widget.set_controller(self._viz_ctrl)

        # image_selected → layer visibility + all sub-widgets
        def _on_image_selected(image_id: str) -> None:
            self._viz_ctrl.on_image_selected(image_id)
            if self._viz_widget and hasattr(self._viz_widget, "set_active_image"):
                self._viz_widget.set_active_image(image_id)
            if self._tme_widget and hasattr(self._tme_widget, "set_active_image"):
                self._tme_widget.set_active_image(image_id)
            if self._image_widget and hasattr(self._image_widget, "on_image_selected"):
                self._image_widget.on_image_selected(image_id)
            if self._project_widget and hasattr(self._project_widget, "on_image_selected"):
                self._project_widget.on_image_selected(image_id)

        self._proj_ctrl.connect_image_selected(_on_image_selected)

        # analysis_complete → status chips in TME widget + Image tab + Project table + viz
        def _on_complete(step: str, image_id: str, result) -> None:
            if self._tme_widget and step == "curvealign":
                if hasattr(self._tme_widget, "on_curvealign_complete"):
                    self._tme_widget.on_curvealign_complete()
            if self._viz_widget and step == "curvealign":
                if hasattr(self._viz_widget, "on_analysis_complete"):
                    self._viz_widget.on_analysis_complete(image_id, result)
            if self._image_widget and hasattr(self._image_widget, "on_analysis_complete"):
                self._image_widget.on_analysis_complete(step, image_id)
            if self._project_widget and hasattr(self._project_widget, "on_analysis_complete"):
                self._project_widget.on_analysis_complete(step, image_id)

        self._analysis_ctrl.connect_analysis_complete(_on_complete)

        # batch_progress → TME widget progress bar
        def _on_progress(step: int, total: int, msg: str) -> None:
            if self._tme_widget and hasattr(self._tme_widget, "update_progress"):
                self._tme_widget.update_progress(step, total, msg)

        self._analysis_ctrl.connect_batch_progress(_on_progress)

        # committed_to_hierarchy → napari layer creation + TACS View panel
        def _on_committed(image_id: str, obj_type: str) -> None:
            self._viz_ctrl.on_committed(image_id, obj_type)
            if obj_type == "curvealign" and self._viz_widget:
                result = self._state.curvealign_pipeline_results.get(image_id)
                if result is not None and hasattr(self._viz_widget, "on_curvealign_committed"):
                    self._viz_widget.on_curvealign_committed(image_id, result)

        self._analysis_ctrl.connect_committed_to_hierarchy(_on_committed)

        # image_type_changed → TME widget fiber selector refresh
        def _on_type_changed(image_id: str, image_type) -> None:
            if self._tme_widget and hasattr(self._tme_widget, "on_image_type_changed"):
                self._tme_widget.on_image_type_changed(image_id, image_type)

        self._proj_ctrl.connect_type_changed(_on_type_changed)

    # ── Widget loaders ─────────────────────────────────────────────────────────

    def _load_project_widget(self) -> QWidget:
        try:
            from .widgets.project_widget import ProjectWidget
            self._project_widget = ProjectWidget(self._viewer)
            return self._project_widget
        except ImportError:
            return _placeholder("Project")

    def _load_image_widget(self) -> QWidget:
        try:
            from .widgets.image_widget import ImageWidget
            self._image_widget = ImageWidget(self._viewer)
            return self._image_widget
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
            self._tme_widget = TMEPipelineWidget(self._viewer)
            return self._tme_widget
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
            self._viz_widget = VisualizationWidget(self._viewer)
            return self._viz_widget
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
            self._log_widget = LogWidget(self._viewer)
            return self._log_widget
        except ImportError:
            return _placeholder("Log")
