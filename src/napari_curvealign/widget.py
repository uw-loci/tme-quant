import os
from enum import Enum
from typing import List, Dict, TYPE_CHECKING, Optional

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QListWidget,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QFileDialog,
    QDialog, QDialogButtonBox, QListWidgetItem, QAbstractItemView,
    QTableView, QAbstractScrollArea, QHeaderView, QVBoxLayout
)
from qtpy.QtCore import Qt, QAbstractTableModel
from qtpy.QtGui import QColor
from skimage.io import imread

if TYPE_CHECKING:
    import napari.viewer

class BoundaryType(Enum):
    NO_BOUNDARY = "No boundary"
    TIFF_BOUNDARY = "Tiff boundary"

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

class CurveAlignWidget(QWidget):
    """Main CurveAlign widget implemented with pure Qt"""
    def __init__(self, viewer: "napari.viewer.Viewer" = None, parent=None):
        super().__init__(parent)
        self._viewer = viewer  # Store viewer reference
        self.image_paths = []
        self.image_layers = {}  # Store image layers by filename
        self.ignore_layer_events = False  # Flag to prevent event recursion
        self.ignore_selection_events = False  # Flag for selection events
        self.results_viewer = None  # Separate viewer for results
        self.results_layers = {}  # Store results layers by name
        self.advanced_params = {  # Default values for advanced parameters
            "advanced_param1": 0.1,
            "advanced_param2": 5.0,
            "iterations": 10
        }
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Open button
        self.open_btn = QPushButton("Open")
        main_layout.addWidget(self.open_btn)
        
        # Image list
        main_layout.addWidget(QLabel("Selected Images:"))
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setMinimumHeight(100)
        main_layout.addWidget(self.image_list)
        
        # Connect selection change
        self.image_list.itemSelectionChanged.connect(self.on_image_selected)
        
        # Boundary type
        boundary_layout = QHBoxLayout()
        boundary_layout.addWidget(QLabel("Boundary type:"))
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems([bt.value for bt in BoundaryType])
        boundary_layout.addWidget(self.boundary_combo)
        main_layout.addLayout(boundary_layout)
        
        # Parameters panel
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        # Curvelets threshold
        curve_layout = QHBoxLayout()
        curve_layout.addWidget(QLabel("Curvelets threshold:"))
        self.curve_threshold = QDoubleSpinBox()
        self.curve_threshold.setRange(0.0, 1.0)
        self.curve_threshold.setSingleStep(0.01)
        self.curve_threshold.setValue(0.5)
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
        main_layout.addWidget(params_group)
        
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
        main_layout.addWidget(output_group)
        
        # Button row
        button_layout = QHBoxLayout()
        
        self.advanced_btn = QPushButton("Advanced")
        button_layout.addWidget(self.advanced_btn)
        
        self.run_btn = QPushButton("Run")
        button_layout.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("Reset")
        button_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
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
            
            # Connect viewer events now that we have a viewer
            self.connect_viewer_events()
        return self._viewer
    
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
                    image_data = imread(path)
                    
                    # For multi-page TIFFs, take the first page
                    if image_data.ndim > 2 and image_data.shape[0] > 1:
                        image_data = image_data[0]
                    
                    # Add to viewer
                    layer = self.viewer.add_image(image_data, name=filename)
                    
                    # Store layer reference with metadata
                    layer.metadata['curvealign_path'] = path
                    
                    # Store layer reference
                    self.image_layers[filename] = layer
                    
                    # Add to list widget
                    item = QListWidgetItem(filename)
                    item.setData(Qt.UserRole, path)  # Store full path
                    self.image_list.addItem(item)
                    
                    # Initially hide all images except the first one
                    if len(self.image_layers) > 1:
                        layer.visible = False
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
            
            # Select first image by default
            if self.image_list.count() > 0:
                self.image_list.setCurrentRow(0)
                if self.image_list.currentItem().text() in self.image_layers:
                    layer = self.image_layers[self.image_list.currentItem().text()]
                    self.viewer.layers.selection.select_only(layer)
                    layer.visible = True
                    self.viewer.reset_view()

    def on_image_selected(self):
        """Handle image selection change in widget"""
        if self.ignore_selection_events:
            return
            
        selected_items = self.image_list.selectedItems()
        if selected_items:
            filename = selected_items[0].text()
            if filename in self.image_layers:
                # Temporarily ignore layer selection events
                self.ignore_selection_events = True
                
                # Hide all other images
                for layer_name, layer in self.image_layers.items():
                    layer.visible = (layer_name == filename)
                
                # Select and show the corresponding layer in the viewer
                layer = self.image_layers[filename]
                self.viewer.layers.selection.select_only(layer)
                layer.visible = True
                
                # Center and zoom to the selected image
                self.viewer.reset_view()
                
                # Reset slice position for multi-dimensional images
                self.viewer.dims.current_step = (0,) * (self.viewer.dims.ndim - 2)
                
                self.ignore_selection_events = False

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
                        
                        self.ignore_selection_events = False
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
        self.boundary_combo.setCurrentText(BoundaryType.NO_BOUNDARY.value)
        self.curve_threshold.setValue(0.5)
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
        
        # Run analysis - this should return two images and a DataFrame
        overlay_img, heatmap_img, measurements = run_analysis(
            image_path=selected_path,
            image_name=selected_filename,
            boundary_type=boundary_type,
            curve_threshold=self.curve_threshold.value(),
            distance_boundary=self.distance_boundary.value(),
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
        """Display analysis results in a new viewer and table"""
        # Close previous results viewer if open
        if self.results_viewer:
            try:
                self.results_viewer.close()
            except:
                pass
        
        # Create new viewer for results
        self.results_viewer = napari.Viewer(title=f"CurveAlign Results - {image_name}")
        
        # Add overlay image
        overlay_layer = self.results_viewer.add_image(
            overlay_img,
            name=f"{image_name} - Overlay",
            blending='additive',
            colormap='green'
        )
        self.results_layers['overlay'] = overlay_layer
        
        # Add heatmap image
        heatmap_layer = self.results_viewer.add_image(
            heatmap_img,
            name=f"{image_name} - Heatmap",
            opacity=0.7,
            colormap='viridis'
        )
        self.results_layers['heatmap'] = heatmap_layer
        
        # Reset view to fit images
        self.results_viewer.reset_view()
        
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

# Factory function to create the widget
def create_curve_align_widget(viewer: "napari.viewer.Viewer" = None):
    """Factory function to create the CurveAlign widget"""
    return CurveAlignWidget(viewer=viewer)