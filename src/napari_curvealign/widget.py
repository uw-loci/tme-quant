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
    QTableView, QAbstractScrollArea, QHeaderView, QVBoxLayout, QTabWidget
)
from qtpy.QtCore import Qt, QAbstractTableModel
from qtpy.QtGui import QColor
from skimage.io import imread

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
        
        # Initialize ROI Manager
        self.roi_manager = ROIManager(viewer=self._viewer)
        
        # Initialize Fiji bridge
        self.fiji_bridge = get_fiji_bridge()
        
        # Main layout with tabs
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.main_tab = QWidget()
        self.preprocessing_tab = QWidget()
        self.segmentation_tab = QWidget()
        self.roi_tab = QWidget()
        
        self.tab_widget.addTab(self.main_tab, "Main")
        self.tab_widget.addTab(self.preprocessing_tab, "Preprocessing")
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.roi_tab, "ROI Manager")
        
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
        
        self.preview_seg_btn = QPushButton("Preview")
        self.preview_seg_btn.clicked.connect(self._preview_segmentation)
        button_layout.addWidget(self.preview_seg_btn)
        
        self.create_rois_btn = QPushButton("Create ROIs from Mask")
        self.create_rois_btn.clicked.connect(self._create_rois_from_segmentation)
        button_layout.addWidget(self.create_rois_btn)
        
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
        
        # Get current image
        image_layer = self._viewer.layers[0]
        if not hasattr(image_layer, 'data'):
            print("Selected layer is not an image.")
            return
        
        image = image_layer.data
        if image.ndim > 2:
            image = image[0] if image.shape[0] < 10 else image  # Handle multi-channel
        
        # Get segmentation method
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
            return
        
        # Create options
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
        
        # Run segmentation
        try:
            print(f"Running {method_text} segmentation...")
            labeled_mask = segment_image(image, options)
            n_objects = labeled_mask.max()
            print(f"Found {n_objects} objects")
            
            # Add mask as labels layer
            if self._viewer:
                self._viewer.add_labels(
                    labeled_mask,
                    name=f"Segmentation - {method_text}",
                    opacity=0.5
                )
                # Store for ROI creation
                self._last_segmentation = labeled_mask
                
        except Exception as e:
            print(f"Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _preview_segmentation(self):
        """Preview segmentation with current settings (same as run)."""
        self._run_segmentation()
    
    def _create_rois_from_segmentation(self):
        """Convert last segmentation mask to ROIs."""
        if not hasattr(self, '_last_segmentation'):
            print("No segmentation available. Run segmentation first.")
            return
        
        try:
            print("Converting segmentation to ROIs...")
            roi_data_list = masks_to_roi_data(
                self._last_segmentation,
                min_area=self.seg_min_area.value(),
                simplify_tolerance=1.0
            )
            
            print(f"Created {len(roi_data_list)} ROIs")
            
            # Add to ROI Manager
            for roi_data in roi_data_list:
                self.roi_manager.add_roi(
                    coordinates=roi_data['coordinates'],
                    shape=ROIShape.POLYGON,
                    name=roi_data['name']
                )
            
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
        layout = QVBoxLayout()
        
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
        layout.addWidget(create_group)
        
        # ROI management buttons
        manage_group = QGroupBox("Manage ROIs")
        manage_layout = QVBoxLayout()
        self.save_roi_btn = QPushButton("Save ROI")
        self.load_roi_btn = QPushButton("Load ROI")
        self.delete_roi_btn = QPushButton("Delete ROI")
        self.rename_roi_btn = QPushButton("Rename ROI")
        self.combine_rois_btn = QPushButton("Combine ROIs")
        manage_layout.addWidget(self.save_roi_btn)
        manage_layout.addWidget(self.load_roi_btn)
        manage_layout.addWidget(self.delete_roi_btn)
        manage_layout.addWidget(self.rename_roi_btn)
        manage_layout.addWidget(self.combine_rois_btn)
        manage_group.setLayout(manage_layout)
        layout.addWidget(manage_group)
        
        # ROI analysis
        analysis_group = QGroupBox("ROI Analysis")
        analysis_layout = QVBoxLayout()
        self.analyze_roi_btn = QPushButton("Analyze Selected ROI")
        self.analyze_all_rois_btn = QPushButton("Analyze All ROIs")
        self.roi_table_btn = QPushButton("Show ROI Table")
        analysis_layout.addWidget(self.analyze_roi_btn)
        analysis_layout.addWidget(self.analyze_all_rois_btn)
        analysis_layout.addWidget(self.roi_table_btn)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # ROI list
        layout.addWidget(QLabel("ROIs:"))
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.roi_list)
        
        # Connect signals
        self.create_rect_btn.clicked.connect(lambda: self._create_roi(ROIShape.RECTANGLE))
        self.create_ellipse_btn.clicked.connect(lambda: self._create_roi(ROIShape.ELLIPSE))
        self.create_polygon_btn.clicked.connect(lambda: self._create_roi(ROIShape.POLYGON))
        self.create_freehand_btn.clicked.connect(lambda: self._create_roi(ROIShape.FREEHAND))
        self.save_roi_btn.clicked.connect(self._save_roi)
        self.load_roi_btn.clicked.connect(self._load_roi)
        self.delete_roi_btn.clicked.connect(self._delete_roi)
        self.rename_roi_btn.clicked.connect(self._rename_roi)
        self.combine_rois_btn.clicked.connect(self._combine_rois)
        self.analyze_roi_btn.clicked.connect(self._analyze_selected_roi)
        self.analyze_all_rois_btn.clicked.connect(self._analyze_all_rois)
        self.roi_table_btn.clicked.connect(self._show_roi_table)
        
        layout.addStretch()
        self.roi_tab.setLayout(layout)
    
    def _create_roi(self, shape: ROIShape):
        """Create ROI of specified shape."""
        if self.viewer is None:
            return
        
        # Enable shape creation in napari
        self.roi_manager.create_shapes_layer()
        self.roi_manager.shapes_layer.mode = 'add_rectangle' if shape == ROIShape.RECTANGLE else 'add_ellipse'
        # Note: Full implementation would need to handle different shape types
    
    def _save_roi(self):
        """Save selected ROI(s) in multiple formats."""
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save ROI", "", 
            "JSON files (*.json);;Fiji ROI (*.roi *.zip);;CSV files (*.csv);;TIFF mask (*.tif);;All files (*)"
        )
        if file_path:
            roi_ids = [int(item.text().split()[0]) for item in selected]
            
            # Determine format from filter or extension
            if "JSON" in selected_filter or file_path.endswith('.json'):
                self.roi_manager.save_rois(file_path, roi_ids, format='json')
            elif "Fiji" in selected_filter or file_path.endswith(('.roi', '.zip')):
                self.roi_manager.save_rois(file_path, roi_ids, format='fiji')
            elif "CSV" in selected_filter or file_path.endswith('.csv'):
                self.roi_manager.save_rois(file_path, roi_ids, format='csv')
            elif "TIFF" in selected_filter or file_path.endswith(('.tif', '.tiff')):
                self.roi_manager.save_rois(file_path, roi_ids, format='mask')
            else:
                self.roi_manager.save_rois(file_path, roi_ids, format='auto')
    
    def _load_roi(self):
        """Load ROI(s) from file in multiple formats."""
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self, "Load ROI", "", 
            "JSON files (*.json);;Fiji ROI (*.roi *.zip);;CSV files (*.csv);;TIFF mask (*.tif);;All files (*)"
        )
        if file_path:
            # Set image shape if available
            shape = self.current_image_shape
            if shape:
                self.roi_manager.set_image_shape(shape)
            
            # Load based on format
            if "JSON" in selected_filter or file_path.endswith('.json'):
                self.roi_manager.load_rois(file_path, format='json')
            elif "Fiji" in selected_filter or file_path.endswith(('.roi', '.zip')):
                self.roi_manager.load_rois(file_path, format='fiji')
            elif "CSV" in selected_filter or file_path.endswith('.csv'):
                self.roi_manager.load_rois(file_path, format='csv')
            elif file_path.endswith(('.tif', '.tiff')):
                self.roi_manager.load_rois(file_path, format='mask')
            else:
                self.roi_manager.load_rois(file_path, format='auto')
            
            self._update_roi_list()
    
    def _delete_roi(self):
        """Delete selected ROI(s)."""
        selected = self.roi_list.selectedItems()
        for item in selected:
            roi_id = int(item.text().split()[0])
            self.roi_manager.delete_roi(roi_id)
        self._update_roi_list()
    
    def _rename_roi(self):
        """Rename selected ROI."""
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        
        roi_id = int(selected[0].text().split()[0])
        # Would show dialog for new name
        # For now, just update list
        self._update_roi_list()
    
    def _combine_rois(self):
        """Combine selected ROIs."""
        selected = self.roi_list.selectedItems()
        if len(selected) < 2:
            return
        
        roi_ids = [int(item.text().split()[0]) for item in selected]
        self.roi_manager.combine_rois(roi_ids)
        self._update_roi_list()
    
    def _analyze_selected_roi(self):
        """Analyze selected ROI."""
        selected = self.roi_list.selectedItems()
        if not selected:
            return
        
        roi_id = int(selected[0].text().split()[0])
        # Get current image
        if self.viewer and len(self.viewer.layers) > 0:
            image_layer = self.viewer.layers[0]
            if hasattr(image_layer, 'data'):
                image = image_layer.data
                self.roi_manager.analyze_roi(roi_id, image)
                self._update_roi_list()
    
    def _analyze_all_rois(self):
        """Analyze all ROIs."""
        if self.viewer and len(self.viewer.layers) > 0:
            image_layer = self.viewer.layers[0]
            if hasattr(image_layer, 'data'):
                image = image_layer.data
                for roi in self.roi_manager.rois:
                    self.roi_manager.analyze_roi(roi.id, image)
                self._update_roi_list()
    
    def _show_roi_table(self):
        """Show ROI analysis table."""
        df = self.roi_manager.get_analysis_table()
        if not df.empty:
            dialog = ResultsDialog(df, self)
            dialog.setWindowTitle("ROI Analysis Results")
            dialog.exec_()
    
    def _update_roi_list(self):
        """Update ROI list widget."""
        self.roi_list.clear()
        for roi in self.roi_manager.rois:
            status = "✓" if roi.analysis_result else ""
            self.roi_list.addItem(f"{roi.id}: {roi.name} {status}")

# Factory function to create the widget
def create_curve_align_widget(viewer: "napari.viewer.Viewer" = None):
    """Factory function to create the CurveAlign widget"""
    return CurveAlignWidget(viewer=viewer)