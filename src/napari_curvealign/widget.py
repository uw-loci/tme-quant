import os
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QListWidget,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QListWidgetItem,
    QAbstractItemView,
    QMenu,
    QInputDialog,
    QMessageBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QAbstractScrollArea,
    QHeaderView,
    QTabWidget,
    QScrollArea,
    QFrame,
    QSizePolicy,
)
from qtpy import QtCore
from qtpy.QtCore import Qt, QAbstractTableModel
from qtpy.QtGui import QColor
from skimage.io import imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import new modules
from .preprocessing import (
    PreprocessingOptions, ThresholdMethod,
    load_image_with_bioformats, preprocess_image
)
from .roi_manager import ROIManager, ROIShape, ROIAnalysisMethod
from .fiji_bridge import get_fiji_bridge
from .segmentation import (
    SegmentationMethod, SegmentationOptions,
    segment_image, masks_to_roi_data,
    check_available_methods, get_recommended_parameters
)

if TYPE_CHECKING:
    import napari.viewer

class BoundaryType(Enum):
    NO_BOUNDARY = "No boundary"
    TIFF_BOUNDARY = "Tiff boundary"


class LogTab(Enum):
    SUMMARY = 0
    OPTIONS = 1
    LOG = 2
    FIJI = 3
    ROIS = 4

class AdvancedParametersDialog(QDialog):
    """Advanced parameters dialog with getter method"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")
        
        layout = QVBoxLayout()
        
        # Create advanced parameters widgets
        self.param1 = QDoubleSpinBox()
        self.param1.setRange(0.0, 1.0)
        self.param1.setSingleStep(0.01)
        self.param1.setValue(0.1)
        param1_layout = QHBoxLayout()
        param1_layout.addWidget(QLabel("Advanced Param 1:"))
        param1_layout.addWidget(self.param1)
        layout.addLayout(param1_layout)
        
        self.param2 = QDoubleSpinBox()
        self.param2.setRange(0.0, 100.0)
        self.param2.setSingleStep(1.0)
        self.param2.setValue(5.0)
        param2_layout = QHBoxLayout()
        param2_layout.addWidget(QLabel("Advanced Param 2:"))
        param2_layout.addWidget(self.param2)
        layout.addLayout(param2_layout)
        
        self.param3 = QSpinBox()
        self.param3.setRange(1, 100)
        self.param3.setValue(10)
        param3_layout = QHBoxLayout()
        param3_layout.addWidget(QLabel("Iterations:"))
        param3_layout.addWidget(self.param3)
        layout.addLayout(param3_layout)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)


class ROIMetricsDialog(QDialog):
    """Dialog presenting ROI measurement statistics and histogram."""
    def __init__(self, metrics: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"ROI {metrics.get('roi_id')} Measurements")
        self.metrics = metrics
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        rows = [
            ("Area (px)", f"{metrics.get('area_px', 0):.2f}"),
            ("Perimeter (px)", f"{metrics.get('perimeter_px', 0):.2f}"),
            ("Centroid (x, y)", f"{metrics.get('centroid', [0, 0])}"),
            ("Bounding Box", f"{metrics.get('bbox')}"),
            ("Eccentricity", f"{metrics.get('eccentricity', 0):.3f}"),
            ("Orientation (deg)", f"{metrics.get('orientation_deg', 0):.2f}"),
            ("Mean Intensity", f"{metrics.get('mean_intensity', 0):.3f}"),
            ("Median Intensity", f"{metrics.get('median_intensity', 0):.3f}"),
            ("Std. Dev.", f"{metrics.get('std_intensity', 0):.3f}"),
        ]
        self.table.setColumnCount(2)
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        for r, (name, value) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(name)))
            self.table.setItem(r, 1, QTableWidgetItem(str(value)))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        hist = metrics.get("histogram")
        if hist:
            fig = Figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            centers = hist["bins"][:-1]
            width = centers[1] - centers[0] if len(centers) > 1 else 0.05
            ax.bar(centers, hist["counts"], width=width, color="#4a90e2")
            ax.set_title("Intensity Distribution")
            ax.set_xlabel("Normalized Intensity")
            ax.set_ylabel("Count")
            self.canvas = FigureCanvas(fig)
            layout.addWidget(self.canvas)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        export_btn = QPushButton("Export CSV")
        button_box.addButton(export_btn, QDialogButtonBox.ActionRole)
        button_box.rejected.connect(self.reject)
        export_btn.clicked.connect(self._export_metrics)
        layout.addWidget(button_box)

    def _export_metrics(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI Metrics",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        rows = [(k, v) for k, v in self.metrics.items() if k != "histogram"]
        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        df.to_csv(path, index=False)
    
    def get_parameters(self):
        """Return advanced parameters as a dictionary"""
        return {
            "advanced_param1": self.param1.value(),
            "advanced_param2": self.param2.value(),
            "iterations": self.param3.value()
        }

class ResultsTableModel(QAbstractTableModel):
    """Table model for displaying measurement results"""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = list(data.columns) if data is not None else []
    
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._headers)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        if role == Qt.DisplayRole:
            return str(self._data.iloc[row, col])
        
        if role == Qt.BackgroundRole and row % 2 == 0:
            return QColor(240, 240, 240)
            
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
            
        if orientation == Qt.Horizontal:
            return self._headers[section]
        else:
            return str(section + 1)

class ResultsDialog(QDialog):
    """Dialog to display measurement results in a table"""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        # Create table view
        self.table_view = QTableView()
        self.table_view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Create model
        self.model = ResultsTableModel(data)
        self.table_view.setModel(self.model)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        
        # Add to layout
        layout.addWidget(self.table_view)
        layout.addWidget(button_box)
        
        self.setLayout(layout)


class MetricsDialog(QDialog):
    """Dialog for displaying ROI measurements with export support."""

    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Measurements")
        self.df = data

        layout = QVBoxLayout()
        self.table_view = QTableView()
        self.table_view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model = ResultsTableModel(data)
        self.table_view.setModel(self.model)
        layout.addWidget(self.table_view)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        export_btn = button_box.addButton("Export CSV", QDialogButtonBox.ActionCommandRole)
        button_box.rejected.connect(self.reject)
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Measurements", "", "CSV Files (*.csv)")
        if path:
            self.df.to_csv(path, index=False)

class CurveAlignWidget(QWidget):
    """Main CurveAlign widget implemented with pure Qt"""
    def __init__(self, viewer: "napari.viewer.Viewer" = None, parent=None):
        super().__init__(parent)
        self._viewer = viewer  # Store viewer reference
        self.image_paths = []
        self.image_layers = {}  # Store image layers by filename
        self.current_image_label: Optional[str] = None
        self._last_segmentation_by_image: Dict[str, np.ndarray] = {}
        self._seg_layers_by_image: Dict[str, List[Any]] = defaultdict(list)
        self.ignore_layer_events = False  # Flag to prevent event recursion
        self.ignore_selection_events = False  # Flag for selection events
        self.results_viewer = None  # Separate viewer for results
        self.results_layers = {}  # Store results layers by name
        self.advanced_params = {  # Default values for advanced parameters
            "advanced_param1": 0.1,
            "advanced_param2": 5.0,
            "iterations": 10
        }
        
        # Initialize ROI Manager
        # Initialize Fiji bridge
        self.fiji_bridge = get_fiji_bridge()
        self.roi_manager = ROIManager(viewer=self._viewer, overlay_callback=self._handle_roi_overlay)
        if self._viewer is not None:
            self.roi_manager.set_viewer(self._viewer)
        
        # Main layout with scrollable tabs so the dock can compress
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.tab_widget)
        scroll_container.setLayout(scroll_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidget(scroll_container)
        # Encourage the widget to fit comfortably in ~1/3 screen width before scrolling
        self.scroll_area.setMinimumWidth(520)
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        scroll_container.setMinimumWidth(520)
        main_layout.addWidget(self.scroll_area)
        
        # Create tabs
        self.main_tab = QWidget()
        self.preprocessing_tab = QWidget()
        self.segmentation_tab = QWidget()
        self.roi_tab = QWidget()
        self.post_tab = QWidget()
        
        self.tab_widget.addTab(self.main_tab, "Main")
        self.tab_widget.addTab(self.preprocessing_tab, "Preprocessing")
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.roi_tab, "ROI Manager")
        self.tab_widget.addTab(self.post_tab, "Post-Processing")
        
        # Setup main tab
        main_tab_layout = QVBoxLayout()
        
        # Open button
        self.open_btn = QPushButton("Open")
        main_tab_layout.addWidget(self.open_btn)
        
        # Image list
        main_tab_layout.addWidget(QLabel("Selected Images:"))
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setMinimumHeight(100)
        main_tab_layout.addWidget(self.image_list)
        
        # Connect selection change
        self.image_list.itemSelectionChanged.connect(self.on_image_selected)
        
        # Analysis mode (Curvelets vs CT-FIRE)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Analysis mode:"))
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["Curvelets", "CT-FIRE", "Both"])
        self.analysis_mode_combo.setCurrentText("Curvelets")
        mode_layout.addWidget(self.analysis_mode_combo)
        main_tab_layout.addLayout(mode_layout)
        
        # Boundary type
        boundary_layout = QHBoxLayout()
        boundary_layout.addWidget(QLabel("Boundary type:"))
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems([bt.value for bt in BoundaryType])
        boundary_layout.addWidget(self.boundary_combo)
        main_tab_layout.addLayout(boundary_layout)
        
        # Parameters panel
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        # Curvelets threshold
        curve_layout = QHBoxLayout()
        curve_layout.addWidget(QLabel("Curvelets threshold:"))
        self.curve_threshold = QDoubleSpinBox()
        self.curve_threshold.setRange(0.0, 1.0)
        self.curve_threshold.setSingleStep(0.01)
        self.curve_threshold.setValue(0.001)  # Default from MATLAB
        curve_layout.addWidget(self.curve_threshold)
        params_layout.addLayout(curve_layout)
        
        # Distance to boundary
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance to boundary:"))
        self.distance_boundary = QSpinBox()
        self.distance_boundary.setRange(0, 100)
        self.distance_boundary.setValue(10)
        distance_layout.addWidget(self.distance_boundary)
        params_layout.addLayout(distance_layout)
        
        params_group.setLayout(params_layout)
        main_tab_layout.addWidget(params_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.histograms_cb = QCheckBox("Histograms")
        self.histograms_cb.setChecked(True)
        output_layout.addWidget(self.histograms_cb)
        
        self.boundary_association_cb = QCheckBox("Boundary association")
        self.boundary_association_cb.setChecked(True)
        output_layout.addWidget(self.boundary_association_cb)
        
        self.overlay_heatmap_cb = QCheckBox("Overlay and Heatmap")
        self.overlay_heatmap_cb.setChecked(True)
        output_layout.addWidget(self.overlay_heatmap_cb)
        
        output_group.setLayout(output_layout)
        main_tab_layout.addWidget(output_group)
        
        # Button row
        button_layout = QHBoxLayout()
        
        self.advanced_btn = QPushButton("Advanced")
        button_layout.addWidget(self.advanced_btn)
        
        self.run_btn = QPushButton("Run")
        button_layout.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("Reset")
        button_layout.addWidget(self.reset_btn)
        
        main_tab_layout.addLayout(button_layout)
        main_tab_layout.addStretch()
        self.main_tab.setLayout(main_tab_layout)
        
        # Setup preprocessing tab
        self._setup_preprocessing_tab()
        
        # Setup segmentation tab
        self._setup_segmentation_tab()
        
        # Setup ROI tab
        self._setup_roi_tab()
        
        self.setLayout(main_layout)
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        
        # Connect signals
        self.open_btn.clicked.connect(self.open_images)
        self.advanced_btn.clicked.connect(self.show_advanced)
        self.run_btn.clicked.connect(self.run_analysis)
        self.reset_btn.clicked.connect(self.reset_parameters)
    
    @property
    def viewer(self):
        """Get the current napari viewer instance"""
        if self._viewer is None:
            # Try to get the current viewer
            from napari import current_viewer
            self._viewer = current_viewer()
            
            # If still not available, create a new one
            if self._viewer is None:
                self._viewer = napari.Viewer()
                print("Created a new napari viewer")
            
            # Update ROI manager viewer
            self.roi_manager.set_viewer(self._viewer)
            
            # Connect viewer events now that we have a viewer
            self.connect_viewer_events()
        return self._viewer
    
    @property
    def current_image_shape(self):
        """Get current image shape from viewer."""
        if self.viewer and len(self.viewer.layers) > 0:
            layer = self.viewer.layers[0]
            if hasattr(layer, 'data'):
                return layer.data.shape[:2]
        return None
    
    def connect_viewer_events(self):
        """Connect to viewer events for synchronization"""
        # Connect to layer events
        self.viewer.layers.events.inserted.connect(self.on_layer_added)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)
        self.viewer.layers.selection.events.active.connect(self.on_active_layer_changed)
    
    def disconnect_viewer_events(self):
        """Disconnect from viewer events"""
        try:
            self.viewer.layers.events.inserted.disconnect(self.on_layer_added)
            self.viewer.layers.events.removed.disconnect(self.on_layer_removed)
            self.viewer.layers.selection.events.active.disconnect(self.on_active_layer_changed)
        except TypeError:
            pass  # If not connected, ignore

    def open_images(self):
        """Open image files and add to list"""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Image files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        )
        if paths:
            self.image_paths = paths
            self.image_list.clear()
            self.image_layers = {}
            
            # Add images to list widget and viewer
            for path in paths:
                filename = os.path.basename(path)
                try:
                    # Read image data using skimage
                    raw_image = imread(path)
                    image_data = raw_image
                    added_rgb_layer = None

                    # Handle common multi-dimensional cases
                    if raw_image.ndim == 3 and raw_image.shape[-1] in (3, 4):
                        # Optionally display the original RGB image
                        try:
                            added_rgb_layer = self.viewer.add_image(
                                raw_image,
                                name=f"{filename} (RGB)",
                                rgb=True,
                                blending="translucent",
                                opacity=0.85,
                            )
                            added_rgb_layer.visible = True
                        except Exception as exc:
                            print(f"Unable to display RGB layer for {filename}: {exc}")

                        # Convert RGB/RGBA to grayscale for analysis
                        rgb = raw_image[..., :3].astype(np.float32)
                        image_data = (
                            0.2125 * rgb[..., 0]
                            + 0.7154 * rgb[..., 1]
                            + 0.0721 * rgb[..., 2]
                        )
                    elif raw_image.ndim > 2 and raw_image.shape[0] > 1:
                        # Multi-page/stacked TIFF: take first plane
                        image_data = raw_image[0]
                    else:
                        # Remove singleton axes if any
                        image_data = np.squeeze(raw_image)
                    
                    # Add grayscale image (analysis layer) to viewer
                    layer = self.viewer.add_image(image_data, name=filename, visible=False)
                    
                    # Store layer reference with metadata
                    layer.metadata['curvealign_path'] = path
                    if added_rgb_layer is not None:
                        added_rgb_layer.metadata['curvealign_rgb_of'] = filename
                    
                    # Store layer reference
                    self.image_layers[filename] = layer
                    
                    # Add to list widget
                    item = QListWidgetItem(filename)
                    item.setData(Qt.UserRole, path)  # Store full path
                    item.setData(Qt.UserRole + 1, bool(added_rgb_layer))
                    self.image_list.addItem(item)
                    
                    # Initially hide all images except the first one
                    if len(self.image_layers) > 1:
                        layer.visible = False
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
            
            # Select first image by default
            if self.image_list.count() > 0:
                self.image_list.setCurrentRow(0)
                self._show_selected_image()

    def on_image_selected(self):
        """Handle image selection change in widget"""
        if self.ignore_selection_events:
            return
            
        selected_items = self.image_list.selectedItems()
        if selected_items:
            self._show_selected_image()

    def _active_image_label(self) -> Optional[str]:
        """Return the filename/label for the currently selected image."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].text()

    def _selected_image_layer(self) -> Optional[Any]:
        """Return the image layer corresponding to the image list selection."""
        label = self._active_image_label()
        if label and label in self.image_layers:
            return self.image_layers[label]
        return None

    def _set_active_image_context(self, layer: Optional[Any] = None):
        """Sync ROI manager with the active image label and shape."""
        label = self._active_image_label()
        if layer is None and self.viewer:
            layer = self.viewer.layers.selection.active
        shape = None
        image_layer = self._selected_image_layer()
        if image_layer is not None and hasattr(image_layer, "data"):
            try:
                data = np.asarray(image_layer.data)
                if data.ndim >= 2:
                    shape = data.shape[:2]
            except Exception:
                shape = None
        elif layer is not None and hasattr(layer, "data"):
            try:
                data = np.asarray(layer.data)
                if data.ndim >= 2:
                    shape = data.shape[:2]
            except Exception:
                shape = None
        self.current_image_label = label
        self.roi_manager.set_active_image(label, shape)
        # Keep shapes layer and lists in sync with the active image
        self.roi_manager._update_shapes_layer()
        self._update_roi_list()

    def _show_selected_image(self):
        """Display the currently selected image (and linked RGB layer) in the viewer."""
        selected_items = self.image_list.selectedItems()
        if not selected_items or not self.viewer:
            return
        filename = selected_items[0].text()
        layer = self.image_layers.get(filename)
        if layer is None:
            return
        self.ignore_selection_events = True
        try:
            for layer_name, lyr in self.image_layers.items():
                is_active = layer_name == filename
                lyr.visible = is_active
                rgb_label = f"{layer_name} (RGB)"
                if rgb_label in self.viewer.layers:
                    self.viewer.layers[rgb_label].visible = is_active
            self.viewer.layers.selection.select_only(layer)
            layer.visible = True
            self.viewer.reset_view()
            if self.viewer.dims.ndim > 2:
                steps = list(self.viewer.dims.current_step)
                for i in range(len(steps) - 2):
                    steps[i] = 0
                self.viewer.dims.current_step = tuple(steps)
            # Sync ROI manager to the active image
            self._set_active_image_context(layer)
        finally:
            self.ignore_selection_events = False

    def _handle_roi_overlay(self, payload: Optional[Dict[str, Any]] = None):
        """Render a simple overlay layer for analyzed ROIs."""
        if not payload or not self.viewer:
            return
        roi_id = payload.get("roi_id")
        mask = payload.get("mask")
        method = payload.get("method", "analysis")
        if roi_id is None or mask is None:
            return

        overlay = mask.astype(float)
        layer_name = f"ROI_{roi_id}_{method}_overlay"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[layer_name])
        self.viewer.add_image(
            overlay,
            name=layer_name,
            blending="additive",
            opacity=0.4,
            colormap="cividis" if method == "curvelets" else "magma",
        )

    def on_active_layer_changed(self, event):
        """Handle active layer change in napari viewer"""
        if self.ignore_selection_events:
            return
            
        active_layer = self.viewer.layers.selection.active
        if active_layer is not None:
            # Check if this is one of our layers
            path = active_layer.metadata.get('curvealign_path', None)
            if path:
                filename = os.path.basename(path)
                
                # Find the corresponding item in the list
                for index in range(self.image_list.count()):
                    item = self.image_list.item(index)
                    if item.text() == filename:
                        # Temporarily ignore widget selection events
                        self.ignore_selection_events = True
                        self.image_list.setCurrentItem(item)
                        
                        # Hide all other images
                        for layer_name, layer in self.image_layers.items():
                            layer.visible = (layer_name == filename)
                            rgb_label = f"{layer_name} (RGB)"
                            if rgb_label in self.viewer.layers:
                                self.viewer.layers[rgb_label].visible = (layer_name == filename)
                        
                        self.ignore_selection_events = False
                        self._set_active_image_context(active_layer)
                        break

    def on_layer_added(self, event):
        """Handle layer added to viewer"""
        layer = event.value
        if 'curvealign_path' in layer.metadata:
            # This is one of our layers
            path = layer.metadata['curvealign_path']
            filename = os.path.basename(path)
            
            # If not already in our list, add it
            if filename not in self.image_layers:
                self.image_layers[filename] = layer
                
                # Add to list widget
                item = QListWidgetItem(filename)
                item.setData(Qt.UserRole, path)
                self.image_list.addItem(item)
                
                # Hide the new layer by default
                layer.visible = False

    def on_layer_removed(self, event):
        """Handle layer removed from viewer"""
        layer = event.value
        if 'curvealign_path' in layer.metadata:
            # This is one of our layers
            path = layer.metadata['curvealign_path']
            filename = os.path.basename(path)
            
            # Remove from our dictionary
            if filename in self.image_layers:
                del self.image_layers[filename]
                
                # Remove from list widget
                for index in range(self.image_list.count()):
                    item = self.image_list.item(index)
                    if item.text() == filename:
                        self.image_list.takeItem(index)
                        break

    def show_advanced(self):
        """Show advanced parameters dialog and save parameters"""
        dialog = AdvancedParametersDialog(self)
        # Set current values
        dialog.param1.setValue(self.advanced_params["advanced_param1"])
        dialog.param2.setValue(self.advanced_params["advanced_param2"])
        dialog.param3.setValue(self.advanced_params["iterations"])
        
        if dialog.exec_():
            # Save parameters if OK clicked
            self.advanced_params = dialog.get_parameters()

    def reset_parameters(self):
        """Reset parameters to default values"""
        self.analysis_mode_combo.setCurrentText("Curvelets")
        self.boundary_combo.setCurrentText(BoundaryType.NO_BOUNDARY.value)
        self.curve_threshold.setValue(0.001)
        self.distance_boundary.setValue(10)
        self.histograms_cb.setChecked(True)
        self.boundary_association_cb.setChecked(True)
        self.overlay_heatmap_cb.setChecked(True)
        self.image_paths = []
        self.image_list.clear()
        self.image_layers = {}
        
        # Reset advanced parameters to defaults
        self.advanced_params = {
            "advanced_param1": 0.1,
            "advanced_param2": 5.0,
            "iterations": 10
        }
        
        # Close results viewer if open
        if self.results_viewer:
            try:
                self.results_viewer.close()
            except:
                pass
            self.results_viewer = None
        self.results_layers = {}

    def run_analysis(self):
        """Run image analysis and display results"""
        if not self.image_paths:
            print("Please select images first!")
            return
        
        # Get selected image path
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            print("Please select an image to analyze!")
            return
            
        selected_path = selected_items[0].data(Qt.UserRole)
        selected_filename = selected_items[0].text()
        
        try:
            from .new_curv import run_analysis  # Import analysis function
        except ImportError:
            from new_curv import run_analysis
            
        # Get current boundary type
        boundary_type = self.boundary_combo.currentText()
        for bt in BoundaryType:
            if bt.value == boundary_type:
                boundary_type = bt
                break
        
        # Apply preprocessing if enabled
        try:
            from skimage.io import imread
            image_data = imread(selected_path)
            if image_data.ndim > 2:
                image_data = image_data[0]
            
            # Apply preprocessing based on tab settings
            preprocess_options = PreprocessingOptions(
                apply_tubeness=self.apply_tubeness_cb.isChecked(),
                tubeness_sigma=self.tubeness_sigma.value(),
                apply_frangi=self.apply_frangi_cb.isChecked(),
                frangi_sigma_range=(self.frangi_sigma_min.value(), self.frangi_sigma_max.value()),
                apply_threshold=self.apply_threshold_cb.isChecked(),
                threshold_method=ThresholdMethod(self.threshold_method.currentText()),
            )
            
            if (preprocess_options.apply_tubeness or 
                preprocess_options.apply_frangi or 
                preprocess_options.apply_threshold):
                image_data = preprocess_image(image_data, preprocess_options)
                # Update image layer if in viewer
                if self.viewer and selected_filename in self.image_layers:
                    self.image_layers[selected_filename].data = image_data
        except Exception as e:
            print(f"Preprocessing failed: {e}")
        
        # Get analysis mode
        mode_text = self.analysis_mode_combo.currentText()
        if mode_text == "CT-FIRE":
            analysis_mode = "ctfire"
        elif mode_text == "Both":
            analysis_mode = "both"  # Will need to handle this specially
        else:
            analysis_mode = "curvelets"
        
        # Run analysis - this should return two images and a DataFrame
        overlay_img, heatmap_img, measurements = run_analysis(
            image_path=selected_path,
            image_name=selected_filename,
            boundary_type=boundary_type,
            curve_threshold=self.curve_threshold.value(),
            distance_boundary=self.distance_boundary.value(),
            analysis_mode=analysis_mode,
            output_options={
                "histograms": self.histograms_cb.isChecked(),
                "boundary_association": self.boundary_association_cb.isChecked(),
                "overlay_heatmap": self.overlay_heatmap_cb.isChecked()
            },
            # Pass advanced parameters
            advanced_params=self.advanced_params
        )
        
        # Display results
        self.display_results(overlay_img, heatmap_img, measurements, selected_filename)

    def display_results(self, overlay_img: np.ndarray, heatmap_img: np.ndarray, 
                       measurements: pd.DataFrame, image_name: str):
        """Display analysis results using napari backend."""
        # Use the main viewer if available, otherwise create new one
        if self.viewer:
            target_viewer = self.viewer
        else:
            if self.results_viewer:
                try:
                    self.results_viewer.close()
                except:
                    pass
            target_viewer = napari.Viewer(title=f"CurveAlign Results - {image_name}")
            self.results_viewer = target_viewer
        
        # Try to use CurveAlign napari backend for better visualization
        try:
            from curvealign_py.visualization.backends.napari_backend import (
                display_analysis_result
            )
            # If we have the actual analysis result, use it
            # For now, use the overlay and heatmap approach
        except ImportError:
            pass
        
        # Add overlay image
        overlay_layer = target_viewer.add_image(
            overlay_img,
            name=f"{image_name} - Overlay",
            blending='additive',
            colormap='green',
            opacity=0.7
        )
        self.results_layers['overlay'] = overlay_layer
        
        # Add heatmap image (angle map)
        heatmap_layer = target_viewer.add_image(
            heatmap_img,
            name=f"{image_name} - Angle Map",
            opacity=0.7,
            colormap='hsv'  # HSV colormap is good for angle visualization
        )
        self.results_layers['heatmap'] = heatmap_layer
        
        # Reset view to fit images
        target_viewer.reset_view()
        
        # Show measurements in a table dialog
        if not measurements.empty:
            self.show_measurements_table(measurements, image_name)

    def show_measurements_table(self, measurements: pd.DataFrame, image_name: str):
        """Display measurements in a table dialog"""
        dialog = ResultsDialog(measurements, self)
        dialog.setWindowTitle(f"Measurements - {image_name}")
        dialog.exec_()
    
    def closeEvent(self, event):
        """Clean up when widget is closed"""
        self.disconnect_viewer_events()
        
        # Close results viewer if open
        if self.results_viewer:
            try:
                self.results_viewer.close()
            except:
                pass
        
        super().closeEvent(event)

    def _setup_preprocessing_tab(self):
        """Setup preprocessing tab with filter options."""
        layout = QVBoxLayout()
        
        # Bio-Formats import
        bio_group = QGroupBox("Bio-Formats Import")
        bio_layout = QVBoxLayout()
        self.use_bioformats_cb = QCheckBox("Use Bio-Formats for import")
        self.use_bioformats_cb.setChecked(False)
        bio_layout.addWidget(self.use_bioformats_cb)
        self.use_fiji_import_cb = QCheckBox("Use Fiji/ImageJ bridge")
        self.use_fiji_import_cb.setChecked(False)
        bio_layout.addWidget(self.use_fiji_import_cb)
        bio_group.setLayout(bio_layout)
        layout.addWidget(bio_group)
        
        # Tubeness filter
        tubeness_group = QGroupBox("Tubeness Filter")
        tubeness_layout = QVBoxLayout()
        self.apply_tubeness_cb = QCheckBox("Apply Tubeness")
        tubeness_layout.addWidget(self.apply_tubeness_cb)
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.tubeness_sigma = QDoubleSpinBox()
        self.tubeness_sigma.setRange(0.1, 10.0)
        self.tubeness_sigma.setSingleStep(0.1)
        self.tubeness_sigma.setValue(1.0)
        sigma_layout.addWidget(self.tubeness_sigma)
        tubeness_layout.addLayout(sigma_layout)
        tubeness_group.setLayout(tubeness_layout)
        layout.addWidget(tubeness_group)
        
        # Frangi filter
        frangi_group = QGroupBox("Frangi Filter")
        frangi_layout = QVBoxLayout()
        self.apply_frangi_cb = QCheckBox("Apply Frangi")
        frangi_layout.addWidget(self.apply_frangi_cb)
        sigma_range_layout = QHBoxLayout()
        sigma_range_layout.addWidget(QLabel("Sigma range:"))
        self.frangi_sigma_min = QDoubleSpinBox()
        self.frangi_sigma_min.setRange(0.1, 20.0)
        self.frangi_sigma_min.setValue(1.0)
        sigma_range_layout.addWidget(self.frangi_sigma_min)
        sigma_range_layout.addWidget(QLabel("to"))
        self.frangi_sigma_max = QDoubleSpinBox()
        self.frangi_sigma_max.setRange(0.1, 20.0)
        self.frangi_sigma_max.setValue(10.0)
        sigma_range_layout.addWidget(self.frangi_sigma_max)
        frangi_layout.addLayout(sigma_range_layout)
        frangi_group.setLayout(frangi_layout)
        layout.addWidget(frangi_group)
        
        # Thresholding
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QVBoxLayout()
        self.apply_threshold_cb = QCheckBox("Apply threshold")
        threshold_layout.addWidget(self.apply_threshold_cb)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.threshold_method = QComboBox()
        self.threshold_method.addItems([m.value for m in ThresholdMethod])
        method_layout.addWidget(self.threshold_method)
        threshold_layout.addLayout(method_layout)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        layout.addStretch()
        self.preprocessing_tab.setLayout(layout)
    
    def _setup_segmentation_tab(self):
        """Setup automated segmentation tab for ROI generation."""
        layout = QVBoxLayout()
        
        # Check available methods
        available_methods = check_available_methods()
        
        # Method selection
        method_group = QGroupBox("Segmentation Method")
        method_layout = QVBoxLayout()
        
        self.seg_method = QComboBox()
        self.seg_method.addItem("Threshold-based" + (" ✓" if available_methods['threshold'] else " ✗"))
        self.seg_method.addItem("Cellpose (Cytoplasm)" + (" ✓" if available_methods['cellpose'] else " ✗"))
        self.seg_method.addItem("Cellpose (Nuclei)" + (" ✓" if available_methods['cellpose'] else " ✗"))
        self.seg_method.addItem("StarDist (Nuclei)" + (" ✓" if available_methods['stardist'] else " ✗"))
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self.seg_method)
        
        # Add install instructions
        install_label = QLabel(
            "<small>Note: Install segmentation dependencies with:<br>"
            "<code>pip install -e '.[segmentation]'</code></small>"
        )
        install_label.setWordWrap(True)
        method_layout.addWidget(install_label)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Threshold options
        self.threshold_seg_group = QGroupBox("Threshold Options")
        threshold_seg_layout = QVBoxLayout()
        
        threshold_seg_layout.addWidget(QLabel("Threshold Method:"))
        self.seg_threshold_method = QComboBox()
        self.seg_threshold_method.addItems(["Otsu", "Triangle", "Isodata", "Mean", "Minimum"])
        threshold_seg_layout.addWidget(self.seg_threshold_method)
        
        threshold_seg_layout.addWidget(QLabel("Min Area (pixels):"))
        self.seg_min_area = QSpinBox()
        self.seg_min_area.setRange(10, 100000)
        self.seg_min_area.setValue(100)
        threshold_seg_layout.addWidget(self.seg_min_area)
        
        threshold_seg_layout.addWidget(QLabel("Max Area (pixels, 0=no limit):"))
        self.seg_max_area = QSpinBox()
        self.seg_max_area.setRange(0, 1000000)
        self.seg_max_area.setValue(0)
        threshold_seg_layout.addWidget(self.seg_max_area)
        
        self.seg_remove_border = QCheckBox("Remove border objects")
        self.seg_remove_border.setChecked(True)
        threshold_seg_layout.addWidget(self.seg_remove_border)
        
        self.threshold_seg_group.setLayout(threshold_seg_layout)
        layout.addWidget(self.threshold_seg_group)
        
        # Cellpose options
        self.cellpose_group = QGroupBox("Cellpose Options")
        cellpose_layout = QVBoxLayout()
        
        cellpose_layout.addWidget(QLabel("Cell Diameter (pixels, 0=auto):"))
        self.cellpose_diameter = QDoubleSpinBox()
        self.cellpose_diameter.setRange(0, 500)
        self.cellpose_diameter.setValue(30.0)
        self.cellpose_diameter.setSingleStep(5.0)
        cellpose_layout.addWidget(self.cellpose_diameter)
        
        cellpose_layout.addWidget(QLabel("Flow Threshold:"))
        self.cellpose_flow_thresh = QDoubleSpinBox()
        self.cellpose_flow_thresh.setRange(0, 3)
        self.cellpose_flow_thresh.setValue(0.4)
        self.cellpose_flow_thresh.setSingleStep(0.1)
        cellpose_layout.addWidget(self.cellpose_flow_thresh)
        
        cellpose_layout.addWidget(QLabel("Cellprob Threshold:"))
        self.cellpose_cellprob_thresh = QDoubleSpinBox()
        self.cellpose_cellprob_thresh.setRange(-6, 6)
        self.cellpose_cellprob_thresh.setValue(0.0)
        self.cellpose_cellprob_thresh.setSingleStep(0.1)
        cellpose_layout.addWidget(self.cellpose_cellprob_thresh)
        
        self.cellpose_group.setLayout(cellpose_layout)
        self.cellpose_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.cellpose_group)
        
        # StarDist options
        self.stardist_group = QGroupBox("StarDist Options")
        stardist_layout = QVBoxLayout()
        
        stardist_layout.addWidget(QLabel("Probability Threshold:"))
        self.stardist_prob_thresh = QDoubleSpinBox()
        self.stardist_prob_thresh.setRange(0, 1)
        self.stardist_prob_thresh.setValue(0.5)
        self.stardist_prob_thresh.setSingleStep(0.05)
        stardist_layout.addWidget(self.stardist_prob_thresh)
        
        stardist_layout.addWidget(QLabel("NMS Threshold:"))
        self.stardist_nms_thresh = QDoubleSpinBox()
        self.stardist_nms_thresh.setRange(0, 1)
        self.stardist_nms_thresh.setValue(0.4)
        self.stardist_nms_thresh.setSingleStep(0.05)
        stardist_layout.addWidget(self.stardist_nms_thresh)
        
        self.stardist_group.setLayout(stardist_layout)
        self.stardist_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.stardist_group)
        
        # Post-processing options
        post_group = QGroupBox("Post-Processing")
        post_layout = QVBoxLayout()
        
        self.seg_fill_holes = QCheckBox("Fill holes")
        self.seg_fill_holes.setChecked(True)
        post_layout.addWidget(self.seg_fill_holes)
        
        self.seg_smooth_contours = QCheckBox("Smooth contours")
        self.seg_smooth_contours.setChecked(True)
        post_layout.addWidget(self.seg_smooth_contours)
        
        post_group.setLayout(post_layout)
        layout.addWidget(post_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.segment_btn = QPushButton("Run Segmentation")
        self.segment_btn.clicked.connect(self._run_segmentation)
        button_layout.addWidget(self.segment_btn)
        
        # Preview action removed (duplicate of Run Segmentation)
        
        self.create_rois_btn = QPushButton("Create ROIs from Mask")
        self.create_rois_btn.clicked.connect(self._create_rois_from_segmentation)
        button_layout.addWidget(self.create_rois_btn)
        # Segmentation layer visibility helpers
        self.show_active_layers_btn = QPushButton("Show Active Set")
        self.show_active_layers_btn.clicked.connect(self._show_active_image_layers)
        self.show_all_layers_btn = QPushButton("Show All Sets")
        self.show_all_layers_btn.clicked.connect(self._show_all_layers)
        button_layout.addWidget(self.show_active_layers_btn)
        button_layout.addWidget(self.show_all_layers_btn)
        self.seg_source_note = QLabel("Mask source: select a labels layer to use it directly.")
        self.seg_source_note.setStyleSheet("color: #aaa; font-size: 10pt;")
        layout.addWidget(self.seg_source_note)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        self.segmentation_tab.setLayout(layout)
        
        # Connect method change to show/hide options
        self.seg_method.currentIndexChanged.connect(self._on_seg_method_changed)
    
    def _on_seg_method_changed(self):
        """Show/hide segmentation options based on selected method."""
        method_text = self.seg_method.currentText().split(" ✓")[0].split(" ✗")[0]
        
        # Hide all groups
        self.threshold_seg_group.setVisible(False)
        self.cellpose_group.setVisible(False)
        self.stardist_group.setVisible(False)
        
        # Show relevant group
        if "Threshold" in method_text:
            self.threshold_seg_group.setVisible(True)
        elif "Cellpose" in method_text:
            self.cellpose_group.setVisible(True)
        elif "StarDist" in method_text:
            self.stardist_group.setVisible(True)
    
    def _run_segmentation(self):
        """Run segmentation on current image."""
        if not self._viewer or len(self._viewer.layers) == 0:
            print("No image loaded. Please open an image first.")
            return
        
        # Get current image (respect active selection / image list)
        image_layer = self._get_active_image_layer()
        if not hasattr(image_layer, 'data'):
            print("Selected layer is not an image.")
            return
        
        # Get segmentation method
        try:
            method_text = self.seg_method.currentText().split(" ✓")[0].split(" ✗")[0]
            print(f"Running {method_text} segmentation...")
            # Sync ROI context to this image before running
            self._set_active_image_context(image_layer)
            # Ensure the image layer is active so downstream tools know which image we’re on
            if self.viewer:
                try:
                    self.viewer.layers.selection.select_only(image_layer)
                except Exception:
                    pass
            labeled_mask = self._run_segmentation_on_layer(image_layer, add_labels=True)
            if labeled_mask is None:
                return
            n_objects = labeled_mask.max()
            print(f"Found {n_objects} objects")
                
        except Exception as e:
            print(f"Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _preview_segmentation(self):
        """Preview segmentation with current settings (same as run)."""
        self._run_segmentation()

    def _segment_image_data(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Run segmentation on a raw image array and return labeled mask."""
        if image is None:
            return None
        image = np.asarray(image)
        if image.ndim > 2:
            if image.shape[-1] in (3, 4):
                rgb = image[..., :3].astype(np.float32)
                image = (
                    0.2125 * rgb[..., 0]
                    + 0.7154 * rgb[..., 1]
                    + 0.0721 * rgb[..., 2]
                )
            else:
                image = image[0] if image.shape[0] < 10 else image

        method_text = self.seg_method.currentText().split(" ✓")[0].split(" ✗")[0]
        if "Threshold" in method_text:
            method = SegmentationMethod.THRESHOLD
        elif "Cellpose" in method_text and "Cytoplasm" in method_text:
            method = SegmentationMethod.CELLPOSE_CYTO
        elif "Cellpose" in method_text and "Nuclei" in method_text:
            method = SegmentationMethod.CELLPOSE_NUCLEI
        elif "StarDist" in method_text:
            method = SegmentationMethod.STARDIST
        else:
            print(f"Unknown method: {method_text}")
            return None

        max_area = self.seg_max_area.value() if self.seg_max_area.value() > 0 else None
        cellpose_diam = self.cellpose_diameter.value() if self.cellpose_diameter.value() > 0 else None
        options = SegmentationOptions(
            method=method,
            threshold_method=self.seg_threshold_method.currentText().lower(),
            min_area=self.seg_min_area.value(),
            max_area=max_area,
            remove_border_objects=self.seg_remove_border.isChecked(),
            cellpose_diameter=cellpose_diam,
            cellpose_flow_threshold=self.cellpose_flow_thresh.value(),
            cellpose_cellprob_threshold=self.cellpose_cellprob_thresh.value(),
            stardist_prob_thresh=self.stardist_prob_thresh.value(),
            stardist_nms_thresh=self.stardist_nms_thresh.value(),
            fill_holes=self.seg_fill_holes.isChecked(),
            smooth_contours=self.seg_smooth_contours.isChecked()
        )
        return segment_image(image, options)

    def _run_segmentation_on_layer(self, image_layer, add_labels: bool = True) -> Optional[np.ndarray]:
        """Run segmentation on a specific image layer."""
        if image_layer is None or not hasattr(image_layer, "data"):
            return None
        labeled_mask = self._segment_image_data(image_layer.data)
        if labeled_mask is None:
            return None
        if add_labels and self._viewer:
            method_text = self.seg_method.currentText().split(" ✓")[0].split(" ✗")[0]
            layer_name = f"{self._active_image_label() or image_layer.name}/seg/{method_text}"
            labels_layer = self._viewer.add_labels(
                labeled_mask,
                name=layer_name,
                opacity=0.5,
                metadata={"curvealign_parent": self._active_image_label() or image_layer.name}
            )
            label_key = self._active_image_label() or image_layer.name
            if label_key:
                self._seg_layers_by_image[label_key].append(labels_layer)
        self._last_segmentation = labeled_mask
        label_key = self._active_image_label() or image_layer.name
        if label_key:
            self._last_segmentation_by_image[label_key] = labeled_mask
        return labeled_mask
    
    def _create_rois_from_segmentation(self):
        """Convert last segmentation mask to ROIs."""
        try:
            # Prefer the currently selected labels layer; fall back to per-image cache, then last global
            mask = None
            active_layer = self.viewer.layers.selection.active if self.viewer else None
            if active_layer is not None and active_layer.__class__.__name__ == "Labels" and hasattr(active_layer, "data"):
                mask = np.asarray(active_layer.data)

            source_desc = None
            if mask is not None and active_layer is not None and active_layer.__class__.__name__ == "Labels":
                source_desc = f"selected labels layer: {active_layer.name}"

            if mask is None:
                label_key = self._active_image_label()
                if label_key and label_key in self._last_segmentation_by_image:
                    mask = self._last_segmentation_by_image[label_key]
                    source_desc = f"cached segmentation for image: {label_key}"

            if mask is None and hasattr(self, "_last_segmentation"):
                mask = self._last_segmentation
                source_desc = "last segmentation (global fallback)"

            if mask is None:
                print("No segmentation available. Select a labels layer or run segmentation first.")
                return

            print("Converting segmentation to ROIs...")
            if self.seg_source_note:
                self.seg_source_note.setText(f"Mask source: {source_desc or 'unknown'}")
            self.roi_manager.set_active_image(self._active_image_label(), mask.shape)
            roi_data_list = masks_to_roi_data(
                mask,
                min_area=self.seg_min_area.value(),
                simplify_tolerance=1.0
            )
            
            print(f"Created {len(roi_data_list)} ROIs")
            
            # Add to ROI Manager
            for roi_data in roi_data_list:
                coords_rc = np.asarray(roi_data['coordinates'], dtype=float)
                coords_xy = np.column_stack((coords_rc[:, 1], coords_rc[:, 0]))
                self.roi_manager.add_roi(
                    coordinates=coords_xy,
                    shape=ROIShape.POLYGON,
                    name=roi_data['name'],
                    annotation_type="tumor",
                    metadata={"source": "segmentation"}
                )

            # Register cell objects for annotation workflow
            self.roi_manager.set_image_shape(self._last_segmentation.shape)
            self.roi_manager.register_cell_objects(roi_data_list)
            self._update_object_list()
            
            # Update ROI list
            self._update_roi_list()
            
            # Switch to ROI Manager tab
            self.tab_widget.setCurrentWidget(self.roi_tab)
            
        except Exception as e:
            print(f"ROI creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_roi_tab(self):
        """Setup ROI Manager tab."""
        layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        
        # Workflow guide
        guide_frame = QFrame()
        guide_frame.setFrameStyle(QFrame.StyledPanel)
        guide_layout = QVBoxLayout()
        guide_label = QLabel(
            "<b>ROI Workflow:</b><br>"
            "1. Click shape button (Rectangle/Ellipse)<br>"
            "2. Draw on image, release mouse<br>"
            "3. ROI appears in list below<br>"
            "4. Repeat for more ROIs<br>"
            "5. Click 'Save All' to export"
        )
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; }")
        guide_layout.addWidget(guide_label)
        guide_frame.setLayout(guide_layout)
        left_panel.addWidget(guide_frame)
        
        # ROI creation buttons
        create_group = QGroupBox("Create ROI")
        create_layout = QVBoxLayout()
        self.create_rect_btn = QPushButton("Rectangle")
        self.create_ellipse_btn = QPushButton("Ellipse")
        self.create_polygon_btn = QPushButton("Polygon")
        self.create_freehand_btn = QPushButton("Freehand")
        create_layout.addWidget(self.create_rect_btn)
        create_layout.addWidget(self.create_ellipse_btn)
        create_layout.addWidget(self.create_polygon_btn)
        create_layout.addWidget(self.create_freehand_btn)
        create_group.setLayout(create_layout)
        left_panel.addWidget(create_group)
        
        # ROI management buttons
        manage_group = QGroupBox("Manage ROIs")
        manage_layout = QVBoxLayout()
        
        # Save/Load row with tooltips
        save_load_row = QHBoxLayout()
        self.save_roi_btn = QPushButton("Save Selected")
        self.save_roi_btn.setToolTip("Save selected ROIs to file (or all if none selected)")
        self.load_roi_btn = QPushButton("Load")
        self.load_roi_btn.setToolTip("Load ROIs from file")
        save_load_row.addWidget(self.save_roi_btn)
        save_load_row.addWidget(self.load_roi_btn)
        manage_layout.addLayout(save_load_row)
        
        # Quick save all button
        self.save_all_roi_btn = QPushButton("Save All ROIs (Quick)")
        self.save_all_roi_btn.setToolTip("Quickly save all ROIs to JSON format")
        manage_layout.addWidget(self.save_all_roi_btn)
        
        self.delete_roi_btn = QPushButton("Delete Selected")
        self.delete_roi_btn.setToolTip("Delete selected ROIs from the list below")
        self.rename_roi_btn = QPushButton("Rename ROI")
        self.combine_rois_btn = QPushButton("Combine ROIs")
        manage_layout.addWidget(self.delete_roi_btn)
        manage_layout.addWidget(self.rename_roi_btn)
        manage_layout.addWidget(self.combine_rois_btn)
        manage_group.setLayout(manage_layout)
        left_panel.addWidget(manage_group)
        
        # ROI analysis
        analysis_group = QGroupBox("ROI Analysis")
        analysis_layout = QVBoxLayout()
        self.analyze_roi_btn = QPushButton("Analyze Selected ROI")
        self.analyze_all_rois_btn = QPushButton("Analyze All ROIs")
        self.analyze_roi_ctfire_btn = QPushButton("Analyze ROI (CT-FIRE)")
        self.analyze_all_images_btn = QPushButton("Analyze All Images (batch)")
        self.batch_pipeline_btn = QPushButton("Batch Pipeline (seg→ROI→analysis→export)")
        self.measure_roi_btn = QPushButton("Measure / Stats")
        self.summary_stats_btn = QPushButton("Export Summary Statistics")
        self.roi_table_btn = QPushButton("Show ROI Table")
        self.ca_options_btn = QPushButton("CurveAlign Options")
        analysis_layout.addWidget(self.analyze_roi_btn)
        analysis_layout.addWidget(self.analyze_all_rois_btn)
        analysis_layout.addWidget(self.analyze_roi_ctfire_btn)
        analysis_layout.addWidget(self.analyze_all_images_btn)
        analysis_layout.addWidget(self.batch_pipeline_btn)
        analysis_layout.addWidget(self.measure_roi_btn)
        analysis_layout.addWidget(self.summary_stats_btn)
        analysis_layout.addWidget(self.roi_table_btn)
        analysis_layout.addWidget(self.ca_options_btn)
        analysis_group.setLayout(analysis_layout)
        left_panel.addWidget(analysis_group)
        
        # ROI list with prominent header
        roi_list_header = QLabel("<b>ROI List</b> (select here to delete/save)")
        roi_list_header.setStyleSheet("QLabel { padding: 5px; background-color: #2a2a2a; }")
        left_panel.addWidget(roi_list_header)
        self.roi_tabs = QTabWidget()
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.roi_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_list.customContextMenuRequested.connect(self._roi_list_context_menu)
        self.roi_list.itemSelectionChanged.connect(self._on_roi_list_selection)
        self.roi_tabs.addTab(self.roi_list, "List")
        self.roi_table_view = QTableWidget()
        self.roi_table_view.setColumnCount(6)
        self.roi_table_view.setHorizontalHeaderLabels(["ID", "Name", "Type", "Source", "Area", "Status"])
        self.roi_tabs.addTab(self.roi_table_view, "Table")
        left_panel.addWidget(self.roi_tabs)
        self.roi_details_group = self._build_roi_details_group()
        left_panel.addWidget(self.roi_details_group)
        left_panel.addStretch()
        
        # Annotations / Objects manager
        annotation_group = self._build_annotation_group()
        objects_group = self._build_object_group()
        right_panel = QVBoxLayout()
        right_panel.addWidget(annotation_group)
        right_panel.addWidget(objects_group)
        right_panel.addStretch()
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 1)
        self.roi_tab.setLayout(layout)
        
        # Post-Processing tab (histograms/graphs)
        post_layout = QVBoxLayout()
        post_layout.addWidget(QLabel("ROIs (current image)"))
        self.post_roi_list = QListWidget()
        self.post_roi_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.post_roi_list.itemSelectionChanged.connect(self._update_post_plots)
        post_layout.addWidget(self.post_roi_list)
        refresh_post_btn = QPushButton("Refresh ROIs")
        refresh_post_btn.clicked.connect(self._refresh_post_roi_list)
        post_layout.addWidget(refresh_post_btn)
        self.post_fig = Figure(figsize=(5, 3))
        self.post_canvas = FigureCanvas(self.post_fig)
        post_layout.addWidget(self.post_canvas)
        self.post_tab.setLayout(post_layout)
        
        # Connect signals
        self.create_rect_btn.clicked.connect(lambda: self._create_roi(ROIShape.RECTANGLE))
        self.create_ellipse_btn.clicked.connect(lambda: self._create_roi(ROIShape.ELLIPSE))
        self.create_polygon_btn.clicked.connect(lambda: self._create_roi(ROIShape.POLYGON))
        self.create_freehand_btn.clicked.connect(lambda: self._create_roi(ROIShape.FREEHAND))
        self.save_roi_btn.clicked.connect(self._save_roi)
        self.save_all_roi_btn.clicked.connect(self._save_all_rois_quick)
        self.load_roi_btn.clicked.connect(self._load_roi)
        self.delete_roi_btn.clicked.connect(self._delete_roi)
        self.rename_roi_btn.clicked.connect(self._rename_roi)
        self.combine_rois_btn.clicked.connect(self._combine_rois)
        self.analyze_roi_btn.clicked.connect(self._analyze_selected_roi)
        self.analyze_all_rois_btn.clicked.connect(self._analyze_all_rois)
        self.analyze_roi_ctfire_btn.clicked.connect(lambda: self._analyze_selected_roi(ctfire=True))
        self.analyze_all_images_btn.clicked.connect(self._batch_analyze_all_images)
        self.batch_pipeline_btn.clicked.connect(self._batch_pipeline_all_images)
        self.measure_roi_btn.clicked.connect(self._open_measurements)
        self.summary_stats_btn.clicked.connect(self._export_summary_statistics)
        self.roi_table_btn.clicked.connect(self._show_roi_table)
        self.ca_options_btn.clicked.connect(self._open_curvealign_options)
        
    def _build_annotation_group(self) -> QGroupBox:
        group = QGroupBox("Annotations (Advanced)")
        group.setToolTip("Advanced feature: Draw large boundary regions, then detect objects within them")
        layout = QVBoxLayout()
        
        # Add explanation label
        info_label = QLabel(
            "<i>For advanced workflows: Draw boundary regions (e.g., tumor areas), "
            "then detect cells/fibers within them.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 3px; }")
        layout.addWidget(info_label)
        
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shape"))
        self.annotation_shape_combo = QComboBox()
        self.annotation_shape_combo.addItem("Rectangle", ROIShape.RECTANGLE)
        self.annotation_shape_combo.addItem("Ellipse", ROIShape.ELLIPSE)
        self.annotation_shape_combo.addItem("Polygon", ROIShape.POLYGON)
        self.annotation_shape_combo.addItem("Freehand", ROIShape.FREEHAND)
        shape_row.addWidget(self.annotation_shape_combo)
        
        shape_row.addWidget(QLabel("Type"))
        self.annotation_type_combo = QComboBox()
        self.annotation_type_combo.addItem("Tumor", "tumor")
        self.annotation_type_combo.addItem("Custom", "custom_annotation")
        self.annotation_type_combo.addItem("Cell (computed)", "cell_computed")
        self.annotation_type_combo.addItem("Fiber (computed)", "fiber_computed")
        shape_row.addWidget(self.annotation_type_combo)
        layout.addLayout(shape_row)
        
        button_row = QHBoxLayout()
        self.draw_annotation_btn = QPushButton("Draw (d)")
        self.draw_annotation_btn.setShortcut("D")
        self.add_annotation_btn = QPushButton("Add (t)")
        self.add_annotation_btn.setShortcut("T")
        self.delete_annotation_btn = QPushButton("Delete (r)")
        self.delete_annotation_btn.setShortcut("R")
        button_row.addWidget(self.draw_annotation_btn)
        button_row.addWidget(self.add_annotation_btn)
        button_row.addWidget(self.delete_annotation_btn)
        layout.addLayout(button_row)
        
        detect_row = QHBoxLayout()
        self.detect_object_btn = QPushButton("Detect object")
        detect_row.addWidget(self.detect_object_btn)
        detect_row.addWidget(QLabel("Distance"))
        self.detect_distance_spin = QSpinBox()
        self.detect_distance_spin.setRange(0, 200)
        self.detect_distance_spin.setValue(self.roi_manager.detection_distance)
        detect_row.addWidget(self.detect_distance_spin)
        self.exclude_inside_checkbox = QCheckBox("Exclude interior")
        self.boundary_only_checkbox = QCheckBox("Boundary ring only")
        self.boundary_width_spin = QSpinBox()
        self.boundary_width_spin.setRange(1, 50)
        self.boundary_width_spin.setValue(5)
        detect_row.addWidget(self.exclude_inside_checkbox)
        detect_row.addWidget(self.boundary_only_checkbox)
        detect_row.addWidget(QLabel("Ring px"))
        detect_row.addWidget(self.boundary_width_spin)
        layout.addLayout(detect_row)
        
        layout.addWidget(QLabel("Annotations"))
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.annotation_list)
        
        group.setLayout(layout)
        
        # Connections
        self.draw_annotation_btn.clicked.connect(self._draw_annotation)
        self.add_annotation_btn.clicked.connect(self._add_annotation_from_shapes)
        self.delete_annotation_btn.clicked.connect(self._delete_annotation)
        self.detect_object_btn.clicked.connect(self._detect_objects_in_annotation)
        self.annotation_list.itemSelectionChanged.connect(self._on_annotation_selected)
        
        return group

    def _build_roi_details_group(self) -> QGroupBox:
        group = QGroupBox("ROI Details")
        layout = QVBoxLayout()
        self.roi_detail_labels = {}
        for label in ["Type", "Source", "Area", "Center", "Status", "Analysis"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}:"))
            value_label = QLabel("-")
            row.addWidget(value_label)
            layout.addLayout(row)
            self.roi_detail_labels[label.lower()] = value_label
        group.setLayout(layout)
        return group

    def _build_object_group(self) -> QGroupBox:
        group = QGroupBox("Objects")
        layout = QVBoxLayout()
        
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.object_filter_combo = QComboBox()
        self.object_filter_combo.addItems(["Cell", "Fiber", "Cell + Fiber", "All", "None"])
        self.object_filter_combo.setCurrentText("Cell + Fiber")
        filter_row.addWidget(self.object_filter_combo)
        layout.addLayout(filter_row)
        
        self.set_annotation_btn = QPushButton("Set Annotation")
        layout.addWidget(self.set_annotation_btn)
        
        layout.addWidget(QLabel("Objects"))
        self.object_list = QListWidget()
        self.object_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.object_list)
        
        group.setLayout(layout)
        
        self.object_filter_combo.currentIndexChanged.connect(self._on_object_filter_changed)
        self.object_list.itemSelectionChanged.connect(self._on_object_selected)
        self.set_annotation_btn.clicked.connect(self._set_objects_as_annotation)
        self._update_object_list()
        
        return group

    def _on_roi_list_selection(self):
        selected = self.roi_list.selectedItems()
        if not selected:
            self._update_roi_details(None)
            return
        roi_id = int(selected[0].text().split(":")[0])
        self._update_roi_details(roi_id)
        try:
            self.roi_manager.highlight_roi(roi_id)
        except Exception:
            pass

    def _roi_list_context_menu(self, pos):
        menu = QMenu(self.roi_list)
        rename_action = menu.addAction("Rename ROI")
        delete_action = menu.addAction("Delete ROI")
        analyze_action = menu.addAction("Analyze ROI")
        analyze_ctfire_action = menu.addAction("Analyze ROI (CT-FIRE)")
        action = menu.exec_(self.roi_list.mapToGlobal(pos))
        if not action:
            return
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        roi_id = int(selected[0].text().split(":")[0])
        if action == rename_action:
            self._rename_roi_dialog(roi_id)
        elif action == delete_action:
            self.roi_manager.delete_roi(roi_id)
            self._update_roi_list()
        elif action == analyze_action:
            self._analyze_roi_by_id(roi_id)
        elif action == analyze_ctfire_action:
            self._analyze_roi_by_id(roi_id, ctfire=True)

    def _rename_roi_dialog(self, roi_id: int):
        roi = self.roi_manager.get_roi(roi_id)
        if roi is None:
            return
        text, ok = QInputDialog.getText(self, "Rename ROI", "ROI name:", text=roi.name)
        if ok and text:
            roi.name = text
            self._update_roi_list()

    def _update_roi_details(self, roi_id: Optional[int]):
        if roi_id is None:
            for label in self.roi_detail_labels.values():
                label.setText("-")
            return
        summary = self.roi_manager.get_roi_summary(roi_id)
        if not summary:
            for label in self.roi_detail_labels.values():
                label.setText("-")
            return
        self.roi_detail_labels["type"].setText(summary["annotation_type"])
        self.roi_detail_labels["source"].setText(summary["source"])
        self.roi_detail_labels["area"].setText(f"{summary['area']:.0f} px")
        center = summary["center"]
        self.roi_detail_labels["center"].setText(f"{center[0]:.1f}, {center[1]:.1f}")
        status = "Analyzed" if summary["has_analysis"] else "Pending"
        self.roi_detail_labels["status"].setText(status)
        analysis_text = summary["analysis_method"] or "-"
        if summary["n_curvelets"]:
            analysis_text += f" ({summary['n_curvelets']} feats)"
        self.roi_detail_labels["analysis"].setText(analysis_text.strip())

    def _ensure_roi_viewer(self):
        """Make sure ROI manager is connected to the active viewer."""
        viewer = self.viewer
        if viewer is None:
            raise RuntimeError("Napari viewer is not available")
        if self.roi_manager.viewer is not viewer:
            self.roi_manager.set_viewer(viewer)
        shape = self.current_image_shape
        if shape and shape != self.roi_manager.current_image_shape:
            self.roi_manager.set_image_shape(shape)
        # Keep active image context in sync for ROI scoping
        self._set_active_image_context(viewer.layers.selection.active if viewer.layers else None)
        return viewer

    def _draw_annotation(self):
        """Activate drawing mode for the selected annotation shape."""
        try:
            self._ensure_roi_viewer()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot Draw", f"Unable to draw annotation: {exc}")
            return
        try:
            layer = self.roi_manager.create_shapes_layer()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Draw", f"Unable to draw annotation: {exc}")
            return
        
        shape = self.annotation_shape_combo.currentData()
        mode_map = {
            ROIShape.RECTANGLE: 'add_rectangle',
            ROIShape.ELLIPSE: 'add_ellipse',
            ROIShape.POLYGON: 'add_polygon',
            ROIShape.FREEHAND: 'add_path'
        }
        layer.mode = mode_map.get(shape, 'add_polygon')
        if self.viewer:
            self.viewer.layers.selection.active = layer
        
        # Show helpful message
        shape_name = shape.value if hasattr(shape, 'value') else str(shape)
        print(f"Draw mode active: {shape_name}. After drawing, click 'Add (t)' to save the ROI.")

    def _add_annotation_from_shapes(self):
        """Convert selected shapes into managed annotations."""
        try:
            self._ensure_roi_viewer()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Cannot Add", f"Unable to add annotation: {exc}")
            return
        
        annotation_type = self.annotation_type_combo.currentData()
        new_rois = self.roi_manager.add_rois_from_shapes(annotation_type=annotation_type)
        
        if not new_rois:
            QMessageBox.information(
                self, 
                "No Shapes", 
                "No shapes available to add as ROIs.\n\n"
                "Tip: First draw shapes using 'Draw (d)', then click 'Add (t)' to save them as ROIs."
            )
            return
        
        # Success feedback
        roi_names = ", ".join([roi.name for roi in new_rois])
        print(f"Added {len(new_rois)} ROI(s): {roi_names}")
        
        self._update_roi_list()
        self._update_annotation_list()

    def _delete_annotation(self):
        """Delete selected annotations."""
        items = self.annotation_list.selectedItems()
        if not items:
            return
        
        # Set sync flag to prevent shape callback from interfering
        if not hasattr(self, '_syncing_shapes'):
            self._syncing_shapes = False
        self._syncing_shapes = True
        
        for item in items:
            roi_id = item.data(Qt.UserRole)
            self.roi_manager.delete_roi(roi_id)
        
        self._update_roi_list()
        self._update_annotation_list()
        
        # Update shape count to match current state
        if self.roi_manager.shapes_layer is not None:
            self._last_shape_count = len(self.roi_manager.shapes_layer.data)
        
        self._syncing_shapes = False

    def _detect_objects_in_annotation(self):
        """Detect objects that fall within the selected annotation."""
        roi_id = self._selected_annotation_id()
        if roi_id is None:
            print("Select an annotation before detecting objects.")
            return
        distance = self.detect_distance_spin.value()
        types = self._object_filter_types()
        if not types:
            types = ["cell", "fiber"]
        self.roi_manager.detection_distance = distance
        include_interior = not self.exclude_inside_checkbox.isChecked()
        boundary_only = self.boundary_only_checkbox.isChecked()
        boundary_width = self.boundary_width_spin.value()
        try:
            detected = self.roi_manager.detect_objects_in_roi(
                roi_id,
                object_types=types,
                distance=distance,
                include_interior=include_interior,
                include_boundary_ring=boundary_only,
                boundary_width=boundary_width
            )
        except ValueError as exc:
            print(f"Object detection failed: {exc}")
            return
        object_ids = []
        for objs in detected.values():
            object_ids.extend(obj.id for obj in objs)
        self._update_object_list(object_ids=object_ids)

    def _set_objects_as_annotation(self):
        """Convert selected objects to annotations."""
        items = self.object_list.selectedItems()
        if not items:
            return
        annotation_type = self.annotation_type_combo.currentData()
        for item in items:
            obj_id = item.data(Qt.UserRole)
            self.roi_manager.add_annotation_from_object(obj_id, annotation_type=annotation_type)
        self._update_roi_list()
        self._update_annotation_list()

    def _on_annotation_selected(self):
        """Highlight annotation within napari when selected."""
        roi_id = self._selected_annotation_id()
        if roi_id is None:
            return
        self.roi_manager.highlight_roi(roi_id)

    def _on_object_selected(self):
        """Highlight selected objects."""
        object_ids = [item.data(Qt.UserRole) for item in self.object_list.selectedItems()]
        if object_ids:
            self.roi_manager.highlight_objects(object_ids)

    def _show_active_image_layers(self):
        """Show only layers belonging to the active image (and hide others)."""
        label = self._active_image_label()
        if not self.viewer or not label:
            return
        parent_names = {label, f"{label} (RGB)"}
        for lyr in list(self.viewer.layers):
            parent = getattr(lyr, "metadata", {}).get("curvealign_parent")
            is_parent = lyr.name in parent_names
            is_child = parent in parent_names
            lyr.visible = is_parent or is_child

    def _show_all_layers(self):
        """Show all layers again."""
        if not self.viewer:
            return
        for lyr in list(self.viewer.layers):
            lyr.visible = True

    def _on_object_filter_changed(self):
        """Update object visibility based on filter dropdown."""
        types = self._object_filter_types()
        if types:
            self.roi_manager.set_object_display_filter(types)
        else:
            self.roi_manager.set_object_display_filter(())
        self._update_object_list()

    def _object_filter_types(self) -> List[str]:
        """Map filter dropdown selection to object type list."""
        mapping = {
            "Cell": ["cell"],
            "Fiber": ["fiber"],
            "Cell + Fiber": ["cell", "fiber"],
            "All": ["cell", "fiber"],
            "None": []
        }
        return mapping.get(self.object_filter_combo.currentText(), ["cell", "fiber"])

    def _selected_annotation_id(self) -> Optional[int]:
        """Return currently selected annotation ROI id."""
        items = self.annotation_list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.UserRole)

    def _update_annotation_list(self):
        """Refresh annotation list widget."""
        if not hasattr(self, "annotation_list"):
            return
        selected_id = self._selected_annotation_id()
        self.annotation_list.clear()
        for roi in self.roi_manager.get_rois_for_active_image():
            label = f"{roi.id}: {roi.name} [{roi.annotation_type}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, roi.id)
            self.annotation_list.addItem(item)
            if selected_id is not None and roi.id == selected_id:
                item.setSelected(True)
        if selected_id is not None:
            self.roi_manager.highlight_roi(selected_id)
        self._refresh_post_roi_list()

    def _update_object_list(self, object_ids: Optional[Sequence[int]] = None):
        """Refresh object list display."""
        if not hasattr(self, "object_list"):
            return
        self.object_list.clear()
        if object_ids:
            objects = [self.roi_manager.get_object(obj_id) for obj_id in object_ids]
            objects = [obj for obj in objects if obj]
        else:
            objects = self.roi_manager.get_objects(self._object_filter_types())
        for obj in objects:
            label = f"{obj.kind.title()} {obj.id}"
            if obj.area:
                label += f" ({int(obj.area)} px)"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, obj.id)
            self.object_list.addItem(item)
        self._refresh_post_roi_list()
    
    def _create_roi(self, shape: ROIShape):
        """Create ROI of specified shape."""
        try:
            self._ensure_roi_viewer()
        except RuntimeError as exc:
            print(f"Cannot create ROI: {exc}")
            return
        
        # Enable shape creation in napari
        layer = self.roi_manager.create_shapes_layer()
        mode_map = {
            ROIShape.RECTANGLE: "add_rectangle",
            ROIShape.ELLIPSE: "add_ellipse",
            ROIShape.POLYGON: "add_polygon",
            ROIShape.FREEHAND: "add_path",
        }
        mode = mode_map.get(shape, "add_polygon")
        layer.mode = mode
        if self.viewer:
            self.viewer.layers.selection.active = layer
        
        # Set up safe auto-sync callback that prevents recursion
        if not hasattr(self, '_last_shape_count'):
            self._last_shape_count = 0
        if not hasattr(self, '_syncing_shapes'):
            self._syncing_shapes = False
        
        if not hasattr(self, '_shape_callback_connected') or not self._shape_callback_connected:
            def on_data_change(event):
                # Prevent recursion when we're updating shapes programmatically
                if self._syncing_shapes:
                    return
                
                current_count = len(layer.data)
                count_diff = current_count - self._last_shape_count
                
                if count_diff > 0:
                    # Shapes were added - auto-add as ROIs
                    try:
                        self._syncing_shapes = True
                        annotation_type = self.annotation_type_combo.currentData() if hasattr(self, 'annotation_type_combo') else "custom_annotation"
                        indices_to_add = list(range(self._last_shape_count, current_count))
                        new_rois = self.roi_manager.add_rois_from_shapes(
                            indices=indices_to_add,
                            annotation_type=annotation_type
                        )
                        if new_rois:
                            self._update_roi_list()
                            print(f"Auto-added {len(new_rois)} ROI(s)")
                        # Detach cursor from drawing mode after successful add
                        if layer.mode.startswith("add_"):
                            layer.mode = "select"
                        self._last_shape_count = len(layer.data)
                    except Exception as e:
                        print(f"Auto-add failed: {e}")
                    finally:
                        self._syncing_shapes = False
                        
                elif count_diff < 0:
                    # Shapes were deleted - clear ROIs and rebuild from remaining shapes
                    print(f"Shapes deleted from layer. Rebuilding ROI list from remaining {current_count} shape(s)...")
                    try:
                        # Clear current ROIs for this image
                        self._syncing_shapes = True
                        current_rois = list(self.roi_manager.get_rois_for_active_image())
                        for roi in current_rois:
                            self.roi_manager.delete_roi(roi.id)
                        
                        # Rebuild ROIs from remaining shapes
                        if current_count > 0:
                            annotation_type = self.annotation_type_combo.currentData() if hasattr(self, 'annotation_type_combo') else "custom_annotation"
                            new_rois = self.roi_manager.add_rois_from_shapes(
                                indices=list(range(current_count)),
                                annotation_type=annotation_type
                            )
                            print(f"Rebuilt {len(new_rois)} ROI(s) from shapes")
                        
                        self._update_roi_list()
                        self._last_shape_count = len(layer.data)
                    except Exception as e:
                        print(f"Auto-sync after deletion failed: {e}")
                    finally:
                        self._syncing_shapes = False
            
            layer.events.data.connect(on_data_change)
            self._shape_callback_connected = True
        
        # Update the initial count
        self._last_shape_count = len(layer.data)
        
        print(f"Draw {shape.value} - ROI will be added automatically when you finish drawing")
    
    def _save_roi(self):
        """Save selected ROI(s) in multiple formats."""
        selected = self.roi_list.selectedItems()
        if not selected:
            # If nothing selected, offer to save all
            reply = QMessageBox.question(
                self,
                "No Selection",
                "No ROIs selected. Save all ROIs?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                selected = [self.roi_list.item(i) for i in range(self.roi_list.count())]
            else:
                return
        
        if not selected:
            QMessageBox.information(self, "No ROIs", "No ROIs available to save.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save ROI", "", 
            "JSON files (*.json);;"
            "Fiji/ImageJ ROI (*.roi *.zip);;"
            "StarDist ROI (*.roi *.zip);;"
            "Cellpose mask (*.npy);;"
            "QuPath annotations (*.geojson);;"
            "CSV files (*.csv);;"
            "TIFF mask (*.tif);;"
            "All files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Use stored UserRole data instead of parsing text (which has colons)
            roi_ids = [item.data(Qt.UserRole) for item in selected]
            
            # Determine format from filter or extension
            if "JSON" in selected_filter or file_path.endswith('.json'):
                self.roi_manager.save_rois(file_path, roi_ids, format='json')
                format_name = "JSON"
            elif "Fiji" in selected_filter or (file_path.endswith(('.roi', '.zip')) and "StarDist" not in selected_filter):
                self.roi_manager.save_rois(file_path, roi_ids, format='fiji')
                format_name = "Fiji/ImageJ"
            elif "StarDist" in selected_filter:
                self.roi_manager.save_rois(file_path, roi_ids, format='stardist')
                format_name = "StarDist"
            elif "Cellpose" in selected_filter or file_path.endswith('.npy'):
                self.roi_manager.save_rois(file_path, roi_ids, format='cellpose')
                format_name = "Cellpose"
            elif "QuPath" in selected_filter or file_path.endswith('.geojson'):
                self.roi_manager.save_rois(file_path, roi_ids, format='qupath')
                format_name = "QuPath"
            elif "CSV" in selected_filter or file_path.endswith('.csv'):
                self.roi_manager.save_rois(file_path, roi_ids, format='csv')
                format_name = "CSV"
            elif "TIFF" in selected_filter or file_path.endswith(('.tif', '.tiff')):
                self.roi_manager.save_rois(file_path, roi_ids, format='mask')
                format_name = "TIFF mask"
            else:
                self.roi_manager.save_rois(file_path, roi_ids, format='auto')
                format_name = "auto-detected"
            
            print(f"Saved {len(roi_ids)} ROI(s) to {file_path} ({format_name} format)")
            QMessageBox.information(
                self,
                "Save Successful",
                f"Successfully saved {len(roi_ids)} ROI(s) in {format_name} format."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save ROIs:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _save_all_rois_quick(self):
        """Quick save all ROIs to multiple formats."""
        all_roi_ids = self.roi_manager.get_all_roi_ids()
        
        if not all_roi_ids:
            QMessageBox.information(self, "No ROIs", "No ROIs available to save.")
            return
        
        # Suggest a filename based on current image
        default_name = "all_rois"
        if self.current_image_label:
            image_name = os.path.splitext(self.current_image_label)[0]
            default_name = f"{image_name}_rois"
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save All ROIs", 
            default_name,
            "JSON files (*.json);;"
            "Fiji/ImageJ ROI (*.roi *.zip);;"
            "StarDist ROI (*.roi *.zip);;"
            "Cellpose mask (*.npy);;"
            "QuPath annotations (*.geojson);;"
            "CSV files (*.csv);;"
            "TIFF mask (*.tif);;"
            "All files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Determine format from filter or extension
            if "JSON" in selected_filter or file_path.endswith('.json'):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='json')
                format_name = "JSON"
            elif "Fiji" in selected_filter or (file_path.endswith(('.roi', '.zip')) and "StarDist" not in selected_filter):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='fiji')
                format_name = "Fiji/ImageJ"
            elif "StarDist" in selected_filter:
                self.roi_manager.save_rois(file_path, all_roi_ids, format='stardist')
                format_name = "StarDist"
            elif "Cellpose" in selected_filter or file_path.endswith('.npy'):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='cellpose')
                format_name = "Cellpose"
            elif "QuPath" in selected_filter or file_path.endswith('.geojson'):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='qupath')
                format_name = "QuPath"
            elif "CSV" in selected_filter or file_path.endswith('.csv'):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='csv')
                format_name = "CSV"
            elif "TIFF" in selected_filter or file_path.endswith(('.tif', '.tiff')):
                self.roi_manager.save_rois(file_path, all_roi_ids, format='mask')
                format_name = "TIFF mask"
            else:
                self.roi_manager.save_rois(file_path, all_roi_ids, format='auto')
                format_name = "auto-detected"

            print(f"Saved {len(all_roi_ids)} ROI(s) to {file_path}")
            QMessageBox.information(
                self,
                "Save Successful",
                f"Successfully saved all {len(all_roi_ids)} ROI(s) to:\n{file_path}\n(Format: {format_name})"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save ROIs:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _load_roi(self):
        """Load ROI(s) from file in multiple formats."""
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self, "Load ROI", "", 
            "All supported formats (*.json *.roi *.zip *.npy *.geojson *.csv *.tif);;"
            "JSON files (*.json);;"
            "Fiji/ImageJ ROI (*.roi *.zip);;"
            "StarDist ROI (*.roi *.zip);;"
            "Cellpose mask (*.npy);;"
            "QuPath annotations (*.geojson);;"
            "CSV files (*.csv);;"
            "TIFF mask (*.tif);;"
            "All files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Set image shape if available
            shape = self.current_image_shape
            if shape:
                self.roi_manager.set_active_image(self._active_image_label(), shape)
            
            # Load based on format
            loaded_rois = []
            if "JSON" in selected_filter or file_path.endswith('.json'):
                loaded_rois = self.roi_manager.load_rois(file_path, format='json')
                format_name = "JSON"
            elif "Fiji" in selected_filter or (file_path.endswith(('.roi', '.zip')) and "StarDist" not in selected_filter):
                loaded_rois = self.roi_manager.load_rois(file_path, format='fiji')
                format_name = "Fiji/ImageJ"
            elif "StarDist" in selected_filter:
                loaded_rois = self.roi_manager.load_rois(file_path, format='stardist')
                format_name = "StarDist"
            elif "Cellpose" in selected_filter or file_path.endswith('.npy'):
                loaded_rois = self.roi_manager.load_rois(file_path, format='cellpose')
                format_name = "Cellpose"
            elif "QuPath" in selected_filter or file_path.endswith('.geojson'):
                loaded_rois = self.roi_manager.load_rois(file_path, format='qupath')
                format_name = "QuPath"
            elif "CSV" in selected_filter or file_path.endswith('.csv'):
                loaded_rois = self.roi_manager.load_rois(file_path, format='csv')
                format_name = "CSV"
            elif file_path.endswith(('.tif', '.tiff')):
                loaded_rois = self.roi_manager.load_rois(file_path, format='mask')
                format_name = "TIFF mask"
            else:
                loaded_rois = self.roi_manager.load_rois(file_path, format='auto')
                format_name = "auto-detected"
            
            self._update_roi_list()
            
            if loaded_rois:
                print(f"Loaded {len(loaded_rois)} ROI(s) from {file_path} ({format_name} format)")
                QMessageBox.information(
                    self,
                    "Load Successful",
                    f"Successfully loaded {len(loaded_rois)} ROI(s) from {format_name} format."
                )
            else:
                QMessageBox.warning(
                    self,
                    "No ROIs Loaded",
                    "No ROIs were found in the file."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Failed",
                f"Failed to load ROIs:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _delete_roi(self):
        """Delete selected ROI(s) from the main ROI list."""
        selected = self.roi_list.selectedItems()
        
        if not selected:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select ROI(s) from the 'ROI List' at the bottom of the left panel before deleting."
            )
            return
        
        # Confirm deletion
        roi_names = [item.text().split()[1] for item in selected if len(item.text().split()) > 1]
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(selected)} ROI(s)?\n{', '.join(roi_names[:5])}{'...' if len(roi_names) > 5 else ''}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Set sync flag to prevent shape callback from interfering
            if not hasattr(self, '_syncing_shapes'):
                self._syncing_shapes = False
            self._syncing_shapes = True
            
            for item in selected:
                roi_id = item.data(Qt.UserRole)
                self.roi_manager.delete_roi(roi_id)
            
            self._update_roi_list()
            
            # Update shape count to match current ROI count
            if self.roi_manager.shapes_layer is not None:
                self._last_shape_count = len(self.roi_manager.shapes_layer.data)
            
            self._syncing_shapes = False
            print(f"Deleted {len(selected)} ROI(s)")
    
    def _rename_roi(self):
        """Rename selected ROI."""
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        
        roi_id = int(selected[0].text().split(":")[0])
        # Would show dialog for new name
        # For now, just update list
        self._update_roi_list()
    
    def _combine_rois(self):
        """Combine selected ROIs."""
        selected = self.roi_list.selectedItems()
        if len(selected) < 2:
            return
        
        roi_ids = [item.data(Qt.UserRole) for item in selected]
        self.roi_manager.combine_rois(roi_ids)
        self._update_roi_list()
    
    def _analyze_selected_roi(self, ctfire: bool = False):
        """Analyze selected ROI."""
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        
        roi_id = int(selected[0].text().split(":")[0])
        self._analyze_roi_by_id(roi_id, ctfire=ctfire)

    def _analyze_roi_by_id(self, roi_id: int, ctfire: bool = False):
        if self.viewer and len(self.viewer.layers) > 0:
            image_layer = self.viewer.layers[0]
            if hasattr(image_layer, 'data'):
                image = self._prepare_analysis_image(image_layer.data)
                method = ROIAnalysisMethod.CTFIRE if ctfire else ROIAnalysisMethod.CURVELETS
                self.roi_manager.analyze_roi(roi_id, image, method=method)
                self._update_roi_list()
    
    def _analyze_all_rois(self):
        """Analyze all ROIs."""
        if self.viewer and len(self.viewer.layers) > 0:
            image_layer = self.viewer.layers[0]
            if hasattr(image_layer, 'data'):
                image = self._prepare_analysis_image(image_layer.data)
                for roi in self.roi_manager.get_rois_for_active_image():
                    self.roi_manager.analyze_roi(roi.id, image)
                self._update_roi_list()

    def _batch_analyze_all_images(self):
        """Analyze ROIs for each loaded image in batch mode."""
        if not self.image_layers:
            print("No images loaded for batch analysis.")
            return
        for label, layer in self.image_layers.items():
            if not hasattr(layer, "data"):
                continue
            data = self._prepare_analysis_image(layer.data)
            shape = data.shape[:2] if data is not None else None
            self.roi_manager.set_active_image(label, shape)
            rois = self.roi_manager.get_rois_for_label(label)
            if not rois:
                continue
            for roi in rois:
                self.roi_manager.analyze_roi(roi.id, data)
        self._update_roi_list()

    def _batch_pipeline_all_images(self):
        """Batch pipeline: segmentation (if needed) -> ROI -> analysis -> export."""
        if not self.image_layers:
            print("No images loaded for batch processing.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not output_dir:
            return
        all_tables = []
        for label, layer in self.image_layers.items():
            if not hasattr(layer, "data"):
                continue
            self.current_image_label = label
            self.roi_manager.set_active_image(label, layer.data.shape[:2])
            rois = self.roi_manager.get_rois_for_label(label)
            if not rois:
                mask = self._run_segmentation_on_layer(layer, add_labels=False)
                if mask is None:
                    continue
                roi_data_list = masks_to_roi_data(
                    mask,
                    min_area=self.seg_min_area.value(),
                    simplify_tolerance=1.0
                )
                for roi_data in roi_data_list:
                    coords_rc = np.asarray(roi_data['coordinates'], dtype=float)
                    coords_xy = np.column_stack((coords_rc[:, 1], coords_rc[:, 0]))
                    self.roi_manager.add_roi(
                        coordinates=coords_xy,
                        shape=ROIShape.POLYGON,
                        name=roi_data['name'],
                        annotation_type="tumor",
                        metadata={"source": "segmentation"}
                    )
                self.roi_manager.set_image_shape(mask.shape)
                self.roi_manager.register_cell_objects(roi_data_list)
                rois = self.roi_manager.get_rois_for_label(label)
            if not rois:
                continue

            data = self._prepare_analysis_image(layer.data)
            mode = self.analysis_mode_combo.currentText()
            for roi in rois:
                if mode == "CT-FIRE":
                    self.roi_manager.analyze_roi(roi.id, data, method=ROIAnalysisMethod.CTFIRE)
                elif mode == "Both":
                    self.roi_manager.analyze_roi(roi.id, data, method=ROIAnalysisMethod.CURVELETS)
                    self.roi_manager.analyze_roi(roi.id, data, method=ROIAnalysisMethod.CTFIRE)
                else:
                    self.roi_manager.analyze_roi(roi.id, data, method=ROIAnalysisMethod.CURVELETS)

            roi_ids = [roi.id for roi in rois]
            json_path = os.path.join(output_dir, f"{label}_rois.json")
            self.roi_manager.save_rois_json(json_path, roi_ids)

            df = self.roi_manager.get_analysis_table()
            df_img = df[df["Image Label"] == label]
            df_img.to_csv(os.path.join(output_dir, f"{label}_analysis.csv"), index=False)
            all_tables.append(df_img)

        if all_tables:
            combined = pd.concat(all_tables, ignore_index=True)
            combined.to_csv(os.path.join(output_dir, "batch_analysis.csv"), index=False)
        self._update_roi_list()

    def _open_curvealign_options(self):
        """Placeholder to expose CurveAlign/CT-FIRE options (to be expanded)."""
        QMessageBox.information(
            self,
            "CurveAlign Options",
            "CurveAlign/CT-FIRE option panel will surface MATLAB-equivalent parameters here."
        )

    def _prepare_analysis_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure analysis image is grayscale 2D."""
        prepared = np.asarray(image)
        if prepared.ndim > 2:
            if prepared.shape[-1] in (3, 4):
                rgb = prepared[..., :3].astype(np.float32)
                prepared = (
                    0.2125 * rgb[..., 0]
                    + 0.7154 * rgb[..., 1]
                    + 0.0721 * rgb[..., 2]
                )
            else:
                prepared = prepared[0]
        prepared = prepared.astype(np.float32, copy=False)
        max_val = np.max(prepared) if prepared.size else 0.0
        if max_val > 0:
            # Normalize to [0,1] to match MATLAB/curvealign expectations
            prepared = prepared / max_val
        return prepared
    
    def _show_roi_table(self):
        """Show ROI analysis table."""
        df = self.roi_manager.get_analysis_table()
        if not df.empty:
            dialog = ResultsDialog(df, self)
            dialog.setWindowTitle("ROI Analysis Results")
            dialog.exec_()
    
    def _refresh_post_roi_list(self):
        """Refresh post-processing ROI list."""
        if not hasattr(self, "post_roi_list"):
            return
        self.post_roi_list.clear()
        for roi in self.roi_manager.get_rois_for_active_image():
            item = QListWidgetItem(f"{roi.id}: {roi.name}")
            item.setData(Qt.UserRole, roi.id)
            self.post_roi_list.addItem(item)
        if self.post_roi_list.count() > 0:
            self.post_roi_list.setCurrentRow(0)

    def _update_post_plots(self):
        """Update histograms/graphs for the selected ROI."""
        if not hasattr(self, "post_roi_list") or not hasattr(self, "post_fig"):
            return
        items = self.post_roi_list.selectedItems()
        if not items:
            return
        roi_id = items[0].data(Qt.UserRole)
        if roi_id is None:
            return
        image = self._get_active_image_data(grayscale=True)
        if image is None:
            return
        metrics = self.roi_manager.get_metrics(roi_id)
        if metrics is None:
            try:
                metrics = self.roi_manager.measure_roi(roi_id, image, histogram_bins=32)
            except ValueError:
                metrics = None
        roi = self.roi_manager.get_roi(roi_id)
        angles = None
        if roi and roi.analysis_result:
            feats = roi.analysis_result.get("features")
            if feats:
                angles = []
                for entry in feats:
                    if isinstance(entry, dict):
                        angle = entry.get("orientation") or entry.get("angle") or entry.get("theta")
                        if angle is not None:
                            angles.append(float(angle))
        self.post_fig.clear()
        ax1 = self.post_fig.add_subplot(121)
        ax2 = self.post_fig.add_subplot(122)
        if metrics and metrics.get("histogram"):
            bins = metrics["histogram"]["bins"]
            counts = metrics["histogram"]["counts"]
            width = (bins[1] - bins[0]) if len(bins) > 1 else 0.05
            ax1.bar(bins[:-1], counts, width=width, color="#4a90e2")
            ax1.set_title("Intensity Histogram")
            ax1.set_xlabel("Normalized Intensity")
            ax1.set_ylabel("Count")
        else:
            ax1.text(0.5, 0.5, "No histogram", ha="center", va="center")
            ax1.set_axis_off()
        if angles:
            ax2.hist(angles, bins=30, color="#ff7f0e")
            ax2.set_title("Curvelet Angles")
            ax2.set_xlabel("Angle (deg)")
            ax2.set_ylabel("Count")
        else:
            ax2.text(0.5, 0.5, "No angle data", ha="center", va="center")
            ax2.set_axis_off()
        self.post_fig.tight_layout()
        self.post_canvas.draw_idle()

    def _selected_roi_ids(self) -> List[int]:
        """Get IDs of selected ROIs from the list widget."""
        items = self.roi_list.selectedItems()
        ids = []
        for item in items:
            roi_id = item.data(Qt.UserRole)
            if roi_id is not None:
                ids.append(roi_id)
        return ids

    def _get_active_image_data(self, grayscale: bool = False) -> Optional[np.ndarray]:
        layer = self._get_active_image_layer()
        if layer is None:
            return None
        data = np.asarray(layer.data)
        if grayscale and data.ndim > 2:
            if data.shape[-1] in (3, 4):
                rgb = data[..., :3].astype(np.float32)
                data = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
            else:
                data = np.mean(data, axis=0)
        return data

    def _get_active_image_layer(self):
        """Return the image layer corresponding to the selected item or active layer."""
        if not self.viewer:
            return None
        # First, honor the image list selection
        selected_image_layer = self._selected_image_layer()
        if selected_image_layer is not None:
            return selected_image_layer
        # Next, inspect active layer
        layer = self.viewer.layers.selection.active
        # If a labels layer is active, try to jump to its parent image
        if layer is not None and getattr(layer, "metadata", {}).get("curvealign_parent"):
            parent = layer.metadata.get("curvealign_parent")
            if parent in self.image_layers:
                return self.image_layers[parent]
        # If the active layer is an image, use it
        if layer is not None and hasattr(layer, "data"):
            return layer
        # Fallback: first image layer in the stack
        for lyr in self.viewer.layers:
            if hasattr(lyr, "data"):
                return lyr
        return None

    def _open_measurements(self):
        roi_ids = self._selected_roi_ids()
        if not roi_ids:
            roi_ids = self.roi_manager.get_all_roi_ids()
        if not roi_ids:
            QMessageBox.information(self, "Measurements", "No ROIs available.")
            return
        image = self._get_active_image_data(grayscale=True)
        try:
            df = self.roi_manager.get_metrics_dataframe(roi_ids, intensity_image=image)
        except ValueError as exc:
            QMessageBox.warning(self, "Measurements", str(exc))
            return
        if df.empty:
            QMessageBox.information(self, "Measurements", "No metrics computed for selection.")
            return
        dialog = MetricsDialog(df, self)
        dialog.exec_()
    
    def _export_summary_statistics(self):
        """Export summary statistics for selected or all ROIs."""
        roi_ids = self._selected_roi_ids()
        if not roi_ids:
            roi_ids = self.roi_manager.get_all_roi_ids()
        if not roi_ids:
            QMessageBox.information(self, "Export Statistics", "No ROIs available.")
            return
        
        # Get file path for export
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary Statistics", "", 
            "CSV files (*.csv);;All files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get intensity image if available
            image = self._get_active_image_data(grayscale=True)
            
            # Export summary statistics
            self.roi_manager.export_summary_statistics(
                file_path,
                roi_ids=roi_ids,
                include_morphology=True,
                include_fiber_metrics=True,
                group_by_annotation_type=True,
                intensity_image=image
            )
            
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Summary statistics exported to:\n{file_path}\n\n"
                f"Per-ROI data exported to:\n{os.path.splitext(file_path)[0]}_per_roi.csv"
            )
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Export Failed", 
                f"Failed to export summary statistics:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _update_roi_list(self):
        """Update ROI list widget and table."""
        self.roi_list.clear()
        table_rows = []
        for roi in self.roi_manager.get_rois_for_active_image():
            status = "✓" if roi.analysis_result else ""
            source = roi.metadata.get("source", "manual")
            text = f"{roi.id}: {roi.name} [{roi.annotation_type}|{source}] {status}"
            item = QListWidgetItem(text.strip())
            item.setData(Qt.UserRole, roi.id)
            if roi.analysis_result:
                item.setForeground(QColor("#3fb950"))
            self.roi_list.addItem(item)
            table_rows.append(
                [
                    str(roi.id),
                    roi.name,
                    roi.annotation_type,
                    source,
                    f"{int(roi.area) if roi.area else ''}",
                    "Analyzed" if roi.analysis_result else "Pending",
                ]
            )
        self._populate_roi_table(table_rows)
        self._update_annotation_list()
        self._refresh_post_roi_list()

    def _populate_roi_table(self, rows: List[List[str]]):
        if not hasattr(self, "roi_table_view"):
            return
        self.roi_table_view.setRowCount(len(rows))
        for r_index, row in enumerate(rows):
            for c_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                self.roi_table_view.setItem(r_index, c_index, item)
        self.roi_table_view.resizeColumnsToContents()

    def sizeHint(self):
        """
        Prefer a compact default size so the dock fits in narrower panels.
        This keeps the full UI visible until roughly 1/3 of a 16\" screen.
        """
        return QtCore.QSize(700, 900)

# Factory function to create the widget
def create_curve_align_widget(viewer: "napari.viewer.Viewer" = None):
    """Factory function to create the CurveAlign widget"""
    return CurveAlignWidget(viewer=viewer)