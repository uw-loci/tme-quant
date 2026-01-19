import os
import sys
import tempfile

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image

# Try importing from different Qt bindings (PyQt6, PyQt5, PySide6, PySide2)
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QGroupBox, QFileDialog, QMessageBox,
        QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
    )
    from PyQt6.QtCore import Qt
    QT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
            QCheckBox, QComboBox, QGroupBox, QFileDialog, QMessageBox,
            QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
        )
        from PyQt5.QtCore import Qt
        QT_VERSION = 5
    except ImportError:
        try:
            from PySide6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                QCheckBox, QComboBox, QGroupBox, QFileDialog, QMessageBox,
                QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
            )
            from PySide6.QtCore import Qt
            QT_VERSION = 6
        except ImportError:
            from PySide2.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                QCheckBox, QComboBox, QGroupBox, QFileDialog, QMessageBox,
                QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
            )
            from PySide2.QtCore import Qt
            QT_VERSION = 5

from pycurvelets.process_image import process_image
from pycurvelets.models import (
    AdvancedAnalysisOptions,
    BoundaryParameters,
    FiberAnalysisParameters,
    ImageInputParameters,
    OutputControlParameters,
)


# ============================================================================
# Helper Functions for Display
# ============================================================================

def display_image_comparison(original_img, overlay_path, procmap_path=None):
    """
    Display original, overlay, and optionally procmap images side by side with linked axes.
    
    Parameters
    ----------
    original_img : ndarray
        Original image array
    overlay_path : str
        Path to the overlay image file
    procmap_path : str, optional
        Path to the procmap image file
    """
    try:
        overlay_img = Image.open(overlay_path)
        overlay_img = np.array(overlay_img)
        
        # Load procmap if available
        procmap_img = None
        if procmap_path and os.path.exists(procmap_path):
            procmap_img = Image.open(procmap_path)
            procmap_img = np.array(procmap_img)
        
        print(f"\nImage dimensions:")
        print(f"  Original: {original_img.shape}")
        print(f"  Overlay: {overlay_img.shape}")
        if procmap_img is not None:
            print(f"  Procmap: {procmap_img.shape}")
        
        # Calculate figure size based on image aspect ratio
        height, width = original_img.shape[:2]
        aspect_ratio = width / height
        fig_height = 6
        
        # Determine number of subplots
        n_plots = 3 if procmap_img is not None else 2
        fig_width = fig_height * aspect_ratio * n_plots + 1
        
        # Create figure with linked subplots
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height),
                                sharex=True, sharey=True)
        
        if n_plots == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
        # Display original image with proper extent to match pixel dimensions
        ax1.imshow(original_img, cmap='gray', extent=[0, width, height, 0],
                   aspect='equal', interpolation='nearest')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        ax1.grid(False)
        
        # Display overlay image with same extent
        ax2.imshow(overlay_img, extent=[0, width, height, 0],
                   aspect='equal', interpolation='nearest')
        ax2.set_title('Fiber Overlay', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('X (pixels)', fontsize=10)
        ax2.set_ylabel('Y (pixels)', fontsize=10)
        ax2.grid(False)
        
        # Display procmap image if available
        if procmap_img is not None:
            ax3.imshow(procmap_img, extent=[0, width, height, 0],
                       aspect='equal', interpolation='nearest')
            ax3.set_title('Processed Map (Heatmap)', fontsize=14, fontweight='bold', pad=10)
            ax3.set_xlabel('X (pixels)', fontsize=10)
            ax3.set_ylabel('Y (pixels)', fontsize=10)
            ax3.grid(False)
        
        # Adjust layout without tight_layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.15)
        plt.show(block=False)
        
        print(f"\n✓ Image comparison displayed successfully ({n_plots} images)")
        print("  (Use zoom/pan tools - all images are linked)")
        
    except Exception as e:
        print(f"\n✗ Error displaying image comparison: {e}")


def display_excel_tables(xlsx_path):
    """
    Display contents of Excel file as formatted tables in Qt windows with scrollbars.
    
    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file
    """
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(xlsx_path)
        
        print(f"\n{'=' * 80}")
        print(f"Excel File Contents: {os.path.basename(xlsx_path)}")
        print(f"{'=' * 80}")
        
        # Get screen width for sizing
        try:
            screen = QApplication.primaryScreen()
            screen_width = screen.size().width()
            screen_height = screen.size().height()
        except:
            screen_width = 1920
            screen_height = 1080
        
        table_windows = []  # Keep references to prevent garbage collection
        
        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            
            print(f"\n{'─' * 80}")
            print(f"Sheet: {sheet_name}")
            print(f"{'─' * 80}")
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            if len(df) > 0:
                # Create Qt table window
                table_window = QMainWindow()
                table_window.setWindowTitle(f"Fiber Features - {sheet_name}")
                
                # Set window size to 75% of screen width, 60% of height
                window_width = int(screen_width * 0.75)
                window_height = int(screen_height * 0.6)
                table_window.setGeometry(100 + sheet_idx * 50, 100 + sheet_idx * 50, window_width, window_height)
                
                # Create central widget and layout
                central_widget = QWidget()
                table_window.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)
                
                # Add title label
                title_label = QLabel(f"{sheet_name}\n{df.shape[0]} rows × {df.shape[1]} columns")
                title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
                layout.addWidget(title_label)
                
                # Create table widget
                table_widget = QTableWidget()
                table_widget.setRowCount(len(df))
                table_widget.setColumnCount(len(df.columns))
                
                # Set headers
                table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])
                table_widget.setVerticalHeaderLabels([str(idx) for idx in df.index])
                
                # Populate table
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        value = df.iloc[i, j]
                        # Format numeric values
                        if isinstance(value, (int, float, np.number)):
                            if isinstance(value, (int, np.integer)):
                                item_text = str(int(value))
                            else:
                                item_text = f"{value:.3f}"
                        else:
                            item_text = str(value)
                        
                        item = QTableWidgetItem(item_text)
                        table_widget.setItem(i, j, item)
                
                # Enable sorting
                table_widget.setSortingEnabled(True)
                
                # Set column resize mode - resize to contents initially
                header = table_widget.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents if QT_VERSION == 6 else QHeaderView.ResizeToContents)
                
                # Enable horizontal and vertical scrollbars
                table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT_VERSION == 6 else Qt.ScrollBarAsNeeded)
                table_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT_VERSION == 6 else Qt.ScrollBarAsNeeded)
                
                # Style alternating row colors
                table_widget.setAlternatingRowColors(True)
                table_widget.setStyleSheet("""
                    QTableWidget {
                        alternate-background-color: #f0f0f0;
                        background-color: white;
                    }
                    QHeaderView::section {
                        background-color: #4CAF50;
                        color: white;
                        font-weight: bold;
                        padding: 5px;
                        border: 1px solid #ccc;
                    }
                """)
                
                layout.addWidget(table_widget)
                
                # Add statistics label if numeric columns exist
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_text = f"Numeric columns: {len(numeric_cols)}"
                    if len(numeric_cols) > 0:
                        stats_text += f" | Mean of first numeric column: {df[numeric_cols[0]].mean():.2f}"
                    stats_label = QLabel(stats_text)
                    stats_label.setStyleSheet("font-style: italic; padding: 5px;")
                    layout.addWidget(stats_label)
                
                # Show the window
                table_window.show()
                table_windows.append(table_window)  # Keep reference
                
                print(f"  ✓ Displayed table with {df.shape[1]} columns (all visible with horizontal scrollbar)")
                
            else:
                print("\n(Empty sheet)")
        
        print(f"\n{'=' * 80}\n")
        print(f"✓ Excel file tables displayed in {len(table_windows)} Qt window(s) with scrollbars")
        
        # Store windows globally to prevent garbage collection
        if not hasattr(display_excel_tables, '_windows'):
            display_excel_tables._windows = []
        display_excel_tables._windows.extend(table_windows)
        
    except Exception as e:
        print(f"\n✗ Error displaying Excel file: {e}")
        import traceback
        traceback.print_exc()



# ============================================================================
# GUI Class
# ============================================================================

class ProcessImageGUI(QMainWindow):
    """GUI for setting process_image parameters."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Analysis - Parameter Configuration")
        self.setGeometry(100, 100, 750, 950)
        
        # Initialize parameter widgets
        self.init_widgets()
        self.create_widgets()
        
    def init_widgets(self):
        """Initialize all parameter widgets."""
        # Image selection
        self.img_path = QLineEdit()
        self.output_dir = QLineEdit()
        
        # Image parameters
        self.img_name = QLineEdit("image")
        self.slice_num = QSpinBox()
        self.slice_num.setRange(1, 1000)
        self.slice_num.setValue(1)
        self.num_sections = QSpinBox()
        self.num_sections.setRange(1, 1000)
        self.num_sections.setValue(1)
        
        # Fiber parameters
        self.fiber_mode = QComboBox()
        self.fiber_mode.addItems(['0 - Curvelet', '1 - FIRE v1', '2 - FIRE v2', '3 - FIRE v3'])
        self.keep = QDoubleSpinBox()
        self.keep.setRange(0.001, 1.0)
        self.keep.setSingleStep(0.01)
        self.keep.setDecimals(3)
        self.keep.setValue(0.01)
        self.fire_directory = QLineEdit()
        
        # Output parameters
        self.make_associations = QCheckBox("Make Associations")
        self.make_associations.setChecked(True)
        self.make_map = QCheckBox("Make Map (Heatmap)")
        self.make_map.setChecked(True)
        self.make_overlay = QCheckBox("Make Overlay")
        self.make_overlay.setChecked(True)
        self.make_feature_file = QCheckBox("Make Feature File")
        self.make_feature_file.setChecked(True)
        
        # Boundary parameters
        self.distance_threshold = QDoubleSpinBox()
        self.distance_threshold.setRange(1, 1000)
        self.distance_threshold.setValue(100.0)
        self.tif_boundary = QComboBox()
        self.tif_boundary.addItems(['0 - None', '1 - CSV Type 1', '2 - CSV Type 2', '3 - TIFF'])
        self.boundary_img_path = QLineEdit()
        self.boundary_img_path.setPlaceholderText("Required when Boundary Type = 3 - TIFF")
        
        # Advanced parameters
        self.exclude_fibers_in_mask = QSpinBox()
        self.exclude_fibers_in_mask.setRange(0, 1)
        self.exclude_fibers_in_mask.setValue(0)
        self.curvelets_group_radius = QDoubleSpinBox()
        self.curvelets_group_radius.setRange(1, 100)
        self.curvelets_group_radius.setValue(10.0)
        self.selected_scale = QSpinBox()
        self.selected_scale.setRange(1, 10)
        self.selected_scale.setValue(1)
        self.heatmap_std_filter = QSpinBox()
        self.heatmap_std_filter.setRange(1, 100)
        self.heatmap_std_filter.setValue(16)
        self.heatmap_square_filter = QSpinBox()
        self.heatmap_square_filter.setRange(1, 100)
        self.heatmap_square_filter.setValue(12)
        self.heatmap_gaussian_sigma = QDoubleSpinBox()
        self.heatmap_gaussian_sigma.setRange(0.1, 50)
        self.heatmap_gaussian_sigma.setSingleStep(0.5)
        self.heatmap_gaussian_sigma.setValue(4.0)
        self.minimum_nearest_fibers = QSpinBox()
        self.minimum_nearest_fibers.setRange(1, 50)
        self.minimum_nearest_fibers.setValue(2)
        self.minimum_box_size = QSpinBox()
        self.minimum_box_size.setRange(1, 256)
        self.minimum_box_size.setValue(32)
        self.fiber_midpoint_estimate = QSpinBox()
        self.fiber_midpoint_estimate.setRange(0, 1)
        self.fiber_midpoint_estimate.setValue(1)
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # === Image Selection ===
        group1 = QGroupBox("Image Selection")
        layout1 = QGridLayout()
        layout1.addWidget(QLabel("Image File:"), 0, 0)
        layout1.addWidget(self.img_path, 0, 1)
        browse_img_btn = QPushButton("Browse...")
        browse_img_btn.clicked.connect(self.browse_image)
        layout1.addWidget(browse_img_btn, 0, 2)
        layout1.addWidget(QLabel("Output Directory:"), 1, 0)
        layout1.addWidget(self.output_dir, 1, 1)
        browse_out_btn = QPushButton("Browse...")
        browse_out_btn.clicked.connect(self.browse_output)
        layout1.addWidget(browse_out_btn, 1, 2)
        group1.setLayout(layout1)
        scroll_layout.addWidget(group1)
        
        # === Image Parameters ===
        group2 = QGroupBox("Image Parameters")
        layout2 = QGridLayout()
        layout2.addWidget(QLabel("Image Name:"), 0, 0)
        layout2.addWidget(self.img_name, 0, 1)
        layout2.addWidget(QLabel("Slice Number:"), 1, 0)
        layout2.addWidget(self.slice_num, 1, 1)
        layout2.addWidget(QLabel("Number of Sections:"), 2, 0)
        layout2.addWidget(self.num_sections, 2, 1)
        group2.setLayout(layout2)
        scroll_layout.addWidget(group2)
        
        # === Fiber Analysis Parameters ===
        group3 = QGroupBox("Fiber Analysis Parameters")
        layout3 = QGridLayout()
        layout3.addWidget(QLabel("Fiber Mode:"), 0, 0)
        layout3.addWidget(self.fiber_mode, 0, 1)
        layout3.addWidget(QLabel("Keep (fraction):"), 1, 0)
        layout3.addWidget(self.keep, 1, 1)
        layout3.addWidget(QLabel("FIRE Directory:"), 2, 0)
        layout3.addWidget(self.fire_directory, 2, 1)
        group3.setLayout(layout3)
        scroll_layout.addWidget(group3)
        
        # === Output Control ===
        group4 = QGroupBox("Output Control")
        layout4 = QGridLayout()
        layout4.addWidget(self.make_associations, 0, 0)
        layout4.addWidget(self.make_map, 0, 1)
        layout4.addWidget(self.make_overlay, 1, 0)
        layout4.addWidget(self.make_feature_file, 1, 1)
        group4.setLayout(layout4)
        scroll_layout.addWidget(group4)
        
        # === Boundary Parameters ===
        group5 = QGroupBox("Boundary Parameters")
        layout5 = QGridLayout()
        layout5.addWidget(QLabel("Distance Threshold:"), 0, 0)
        layout5.addWidget(self.distance_threshold, 0, 1)
        layout5.addWidget(QLabel("Boundary Type:"), 1, 0)
        layout5.addWidget(self.tif_boundary, 1, 1)
        layout5.addWidget(QLabel("Boundary Image (TIFF):"), 2, 0)
        layout5.addWidget(self.boundary_img_path, 2, 1)
        self.browse_boundary_btn = QPushButton("Browse...")
        self.browse_boundary_btn.clicked.connect(self.browse_boundary_image)
        self.browse_boundary_btn.setEnabled(False)  # Disabled by default
        layout5.addWidget(self.browse_boundary_btn, 2, 2)
        group5.setLayout(layout5)
        scroll_layout.addWidget(group5)
        
        # Connect boundary type change to enable/disable browse button and auto-detect
        self.tif_boundary.currentIndexChanged.connect(self.on_boundary_type_changed)
        
        # === Advanced Options ===
        group6 = QGroupBox("Advanced Options")
        layout6 = QGridLayout()
        layout6.addWidget(QLabel("Exclude Fibers in Mask:"), 0, 0)
        layout6.addWidget(self.exclude_fibers_in_mask, 0, 1)
        layout6.addWidget(QLabel("Curvelets Group Radius:"), 1, 0)
        layout6.addWidget(self.curvelets_group_radius, 1, 1)
        layout6.addWidget(QLabel("Selected Scale:"), 2, 0)
        layout6.addWidget(self.selected_scale, 2, 1)
        layout6.addWidget(QLabel("Heatmap STD Filter:"), 3, 0)
        layout6.addWidget(self.heatmap_std_filter, 3, 1)
        layout6.addWidget(QLabel("Heatmap Square Filter:"), 4, 0)
        layout6.addWidget(self.heatmap_square_filter, 4, 1)
        layout6.addWidget(QLabel("Heatmap Gaussian Sigma:"), 5, 0)
        layout6.addWidget(self.heatmap_gaussian_sigma, 5, 1)
        layout6.addWidget(QLabel("Minimum Nearest Fibers:"), 6, 0)
        layout6.addWidget(self.minimum_nearest_fibers, 6, 1)
        layout6.addWidget(QLabel("Minimum Box Size:"), 7, 0)
        layout6.addWidget(self.minimum_box_size, 7, 1)
        layout6.addWidget(QLabel("Fiber Midpoint Estimate:"), 8, 0)
        layout6.addWidget(self.fiber_midpoint_estimate, 8, 1)
        group6.setLayout(layout6)
        scroll_layout.addWidget(group6)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.run_analysis)
        defaults_btn = QPushButton("Load Defaults")
        defaults_btn.clicked.connect(self.load_defaults)
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(run_btn)
        button_layout.addWidget(defaults_btn)
        button_layout.addWidget(exit_btn)
        scroll_layout.addLayout(button_layout)
        
        # Set up scroll area
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
    def browse_image(self):
        """Browse for image file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if filename:
            self.img_path.setText(filename)
            # Auto-set image name from filename
            basename = os.path.basename(filename)
            name_without_ext = os.path.splitext(basename)[0]
            self.img_name.setText(name_without_ext)
            
            # Set output directory to CA_Out subfolder in image directory
            img_dir = os.path.dirname(filename)
            ca_out_dir = os.path.join(img_dir, "CA_Out")
            
            # Create CA_Out directory if it doesn't exist
            if not os.path.exists(ca_out_dir):
                os.makedirs(ca_out_dir)
                print(f"Created output directory: {ca_out_dir}")
            else:
                print(f"Using existing output directory: {ca_out_dir}")
            
            self.output_dir.setText(ca_out_dir)
            
            # Auto-detect boundary image if boundary type is TIFF
            self.auto_detect_boundary_image()
    
    def browse_output(self):
        """Browse for output directory."""
        dirname = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirname:
            self.output_dir.setText(dirname)
    
    def browse_boundary_image(self):
        """Browse for boundary image file (TIFF)."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Boundary Image File",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        if filename:
            self.boundary_img_path.setText(filename)
    
    def on_boundary_type_changed(self, index):
        """Handle boundary type selection change."""
        is_tiff = (index == 3)  # 3 corresponds to "3 - TIFF"
        self.browse_boundary_btn.setEnabled(is_tiff)
        self.boundary_img_path.setEnabled(is_tiff)
        
        if is_tiff:
            # Auto-detect boundary image when switching to TIFF mode
            self.auto_detect_boundary_image()
        else:
            # Clear boundary image path when not in TIFF mode
            self.boundary_img_path.clear()
    
    def auto_detect_boundary_image(self):
        """Auto-detect boundary image based on image name."""
        # Only auto-detect if boundary type is TIFF
        if self.tif_boundary.currentIndex() != 3:
            return
        
        img_path = self.img_path.text()
        if not img_path:
            return
        
        # Get image directory and name
        img_dir = os.path.dirname(img_path)
        img_name = self.img_name.text()
        
        if not img_name:
            return
        
        # Construct boundary image path: CA_Boundary/mask_<img_name>.tiff
        boundary_dir = os.path.join(img_dir, "CA_Boundary")
        boundary_filename = f"mask_{img_name}.tiff"
        boundary_path = os.path.join(boundary_dir, boundary_filename)
        
        # Check if file exists
        if os.path.exists(boundary_path):
            self.boundary_img_path.setText(boundary_path)
            print(f"Auto-detected boundary image: {boundary_path}")
        else:
            # Try .tif extension as well
            boundary_filename_alt = f"mask_{img_name}.tif"
            boundary_path_alt = os.path.join(boundary_dir, boundary_filename_alt)
            if os.path.exists(boundary_path_alt):
                self.boundary_img_path.setText(boundary_path_alt)
                print(f"Auto-detected boundary image: {boundary_path_alt}")
            else:
                print(f"Boundary image not found at: {boundary_path} or {boundary_path_alt}")
    
    def load_defaults(self):
        """Load default parameter values."""
        self.fiber_mode.setCurrentIndex(0)
        self.keep.setValue(0.01)
        self.make_associations.setChecked(True)
        self.make_map.setChecked(True)
        self.make_overlay.setChecked(True)
        self.make_feature_file.setChecked(True)
        self.distance_threshold.setValue(100.0)
        self.tif_boundary.setCurrentIndex(0)
        self.exclude_fibers_in_mask.setValue(0)
        self.curvelets_group_radius.setValue(10.0)
        self.selected_scale.setValue(1)
        self.heatmap_std_filter.setValue(16)
        self.heatmap_square_filter.setValue(12)
        self.heatmap_gaussian_sigma.setValue(4.0)
        self.minimum_nearest_fibers.setValue(2)
        self.minimum_box_size.setValue(32)
        self.fiber_midpoint_estimate.setValue(1)
        QMessageBox.information(self, "Defaults Loaded", "Default parameters have been restored.")
    
    def run_analysis(self):
        """Run the fiber analysis with current parameters."""
        # Validate inputs
        if not self.img_path.text():
            QMessageBox.critical(self, "Error", "Please select an image file.")
            return
        
        if not os.path.exists(self.img_path.text()):
            QMessageBox.critical(self, "Error", f"Image file not found: {self.img_path.text()}")
            return
        
        try:
            # Load image
            img = plt.imread(self.img_path.text())
            
            # Convert to grayscale if the image has multiple channels
            if img.ndim == 3:
                # If RGB or RGBA, convert to grayscale
                if img.shape[2] == 3:
                    # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
                    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                elif img.shape[2] == 4:
                    # RGBA to grayscale (ignore alpha channel)
                    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                print(f"Converted image from {img.ndim}D to 2D grayscale")
            
            # Create parameter objects
            image_params = ImageInputParameters(
                img=img,
                img_name=self.img_name.text(),
                slice_num=self.slice_num.value(),
                num_sections=self.num_sections.value(),
            )
            
            fiber_params = FiberAnalysisParameters(
                fiber_mode=self.fiber_mode.currentIndex(),
                keep=self.keep.value(),
                fire_directory=self.fire_directory.text() if self.fire_directory.text() else None,
            )
            
            output_params = OutputControlParameters(
                output_directory=self.output_dir.text(),
                make_associations=self.make_associations.isChecked(),
                make_map=self.make_map.isChecked(),
                make_overlay=self.make_overlay.isChecked(),
                make_feature_file=self.make_feature_file.isChecked(),
            )
            
            # Load boundary image if tif_boundary=3 and path is provided
            boundary_img = None
            tif_boundary_val = self.tif_boundary.currentIndex()
            exclude_fibers_flag = self.exclude_fibers_in_mask.value()
            
            if tif_boundary_val == 3 and self.boundary_img_path.text():
                if os.path.exists(self.boundary_img_path.text()):
                    boundary_img = plt.imread(self.boundary_img_path.text())
                    # Convert boundary image to grayscale if needed
                    if boundary_img.ndim == 3:
                        if boundary_img.shape[2] == 3:
                            # RGB to grayscale
                            boundary_img = 0.299 * boundary_img[:, :, 0] + 0.587 * boundary_img[:, :, 1] + 0.114 * boundary_img[:, :, 2]
                        elif boundary_img.shape[2] == 4:
                            # RGBA to grayscale (ignore alpha)
                            boundary_img = 0.299 * boundary_img[:, :, 0] + 0.587 * boundary_img[:, :, 1] + 0.114 * boundary_img[:, :, 2]
                        print(f"Converted boundary image to 2D grayscale")
                    print(f"Loaded boundary image: {self.boundary_img_path.text()}")
                    # Automatically set exclude_fibers_in_mask_flag=1 when using TIFF boundary
                    exclude_fibers_flag = 1
                    print("Automatically set exclude_fibers_in_mask_flag=1 for TIFF boundary")
                else:
                    QMessageBox.warning(self, "Warning", f"Boundary image file not found: {self.boundary_img_path.text()}\nProceeding without boundary analysis.")
                    tif_boundary_val = 0  # Reset to no boundary
            elif tif_boundary_val == 3 and not self.boundary_img_path.text():
                QMessageBox.warning(self, "Warning", "Boundary Type is set to '3 - TIFF' but no boundary image file is selected.\nProceeding without boundary analysis.")
                tif_boundary_val = 0  # Reset to no boundary
            
            boundary_params = BoundaryParameters(
                coordinates=None,
                distance_threshold=self.distance_threshold.value(),
                tif_boundary=tif_boundary_val,
                boundary_img=boundary_img,
            )
            
            advanced_options = AdvancedAnalysisOptions(
                exclude_fibers_in_mask_flag=exclude_fibers_flag,
                curvelets_group_radius=self.curvelets_group_radius.value(),
                selected_scale=self.selected_scale.value(),
                heatmap_STD_filter_size=self.heatmap_std_filter.value(),
                heatmap_SQUARE_max_filter_size=self.heatmap_square_filter.value(),
                heatmap_GAUSSIAN_disc_filter_sigma=self.heatmap_gaussian_sigma.value(),
                minimum_nearest_fibers=self.minimum_nearest_fibers.value(),
                minimum_box_size=self.minimum_box_size.value(),
                fiber_midpoint_estimate=self.fiber_midpoint_estimate.value(),
                min_dist=[],
            )
            
            # Run analysis
            print("\n" + "=" * 60)
            print("Starting Analysis...")
            print("=" * 60)
            
            result = process_image(
                image_params=image_params,
                fiber_params=fiber_params,
                output_params=output_params,
                boundary_params=boundary_params,
                advanced_options=advanced_options,
            )
            
            # Display results
            display_results(img, self.output_dir.text(), result, img_name=self.img_name.text())
            
            QMessageBox.information(self, "Success", f"Analysis complete!\nResults saved to:\n{self.output_dir.text()}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")
            import traceback
            traceback.print_exc()


def display_results(img, output_directory, result, img_name=None):
    """
    Display analysis results.
    
    Parameters
    ----------
    img : ndarray
        Original image array
    output_directory : str
        Path to output directory
    result : dict
        Analysis result dictionary
    img_name : str, optional
        Base name of the input image (without extension) for strict file matching
    """
    # List output files
    print(f"\nOutput files in {output_directory}:")
    output_files = {}
    
    # Get all files in the directory
    all_files = os.listdir(output_directory) if os.path.exists(output_directory) else []
    
    # If img_name is provided, use it for strict matching
    if img_name:
        print(f"Looking for files matching image name: '{img_name}'")
        base_name = img_name.replace('.tif', '').replace('.tiff', '').replace('.png', '').replace('.jpg', '')
        
        for f in all_files:
            print(f"  - {f}")
            # Strict name checking: file must start with the image base name
            if f.startswith(base_name):
                if "overlay" in f and (f.endswith(".tiff") or f.endswith(".tif")):
                    output_files['overlay'] = os.path.join(output_directory, f)
                    print(f"    ✓ Matched as overlay for '{base_name}'")
                elif "procmap" in f and (f.endswith(".tiff") or f.endswith(".tif")):
                    output_files['procmap'] = os.path.join(output_directory, f)
                    print(f"    ✓ Matched as procmap for '{base_name}'")
                elif "FiberFeatures" in f and f.endswith(".xlsx"):
                    output_files['features_xlsx'] = os.path.join(output_directory, f)
                    print(f"    ✓ Matched as Excel features for '{base_name}'")
    else:
        # Fallback to old behavior if no img_name provided (search for any matching file)
        print("Warning: No image name provided, using loose file matching")
        for f in all_files:
            print(f"  - {f}")
            # Track specific output files
            if "overlay" in f and (f.endswith(".tiff") or f.endswith(".tif")):
                output_files['overlay'] = os.path.join(output_directory, f)
            elif "procmap" in f and (f.endswith(".tiff") or f.endswith(".tif")):
                output_files['procmap'] = os.path.join(output_directory, f)
            elif "FiberFeatures" in f and f.endswith(".xlsx"):
                output_files['features_xlsx'] = os.path.join(output_directory, f)
    
    print(f"\nDetected files for current image:")
    print(f"  Overlay: {output_files.get('overlay', 'Not found')}")
    print(f"  Procmap: {output_files.get('procmap', 'Not found')}")
    print(f"  Excel: {output_files.get('features_xlsx', 'Not found')}")
    
    # Display original, overlay, and procmap images side by side
    if 'overlay' in output_files:
        procmap_path = output_files.get('procmap', None)
        display_image_comparison(img, output_files['overlay'], procmap_path)
    else:
        print("\n⚠ Overlay image not found, skipping image comparison")
    
    # Display Excel file contents as tables
    if 'features_xlsx' in output_files:
        display_excel_tables(output_files['features_xlsx'])
    else:
        print("\n⚠ Excel file not found, skipping table display")
    
    # Print result summary
    if result and 'fib_feat_df' in result:
        fib_feat_df = result['fib_feat_df']
        print(f"\n✓ Detected {len(fib_feat_df)} fibers")


def main():
    """Launch GUI for parameter configuration."""
    app = QApplication(sys.argv)
    window = ProcessImageGUI()
    window.show()
    # Use exec() for Qt6, exec_() for Qt5
    sys.exit(app.exec() if QT_VERSION == 6 else app.exec_())


def main_cli():
    """Run analysis with default parameters (command-line mode)."""
    # Load the test image
    img_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "test_images", "real1.tif"
    )
    img = plt.imread(img_path)
    
    # Create a temporary output directory
    output_directory = tempfile.mkdtemp()
    print(f"Output directory: {output_directory}")
    
    # Create parameter objects using dataclasses
    image_params = ImageInputParameters(
        img=img,
        img_name="real1",  # Name without extension
        slice_num=1,
        num_sections=1,
    )
    
    fiber_params = FiberAnalysisParameters(
        fiber_mode=0,  # 0=curvelet, 1/2/3=FIRE variants
        keep=0.1,  # Fraction of curvelets to keep
        fire_directory=None,  # None to use curvelet mode
    )
    
    output_params = OutputControlParameters(
        output_directory=output_directory,
        make_associations=True,
        make_map=True,
        make_overlay=True,
        make_feature_file=True,
    )
    
    # Optional: boundary parameters (None for no boundary analysis)
    boundary_params = BoundaryParameters(
        coordinates=None,  # None or dict for no ROI constraints
        distance_threshold=100,  # Distance threshold for associations
        tif_boundary=0,  # 0=none, 1/2=CSV, 3=TIFF
        boundary_img=None,
    )
    
    # Optional: advanced options with custom settings
    advanced_options = AdvancedAnalysisOptions(
        exclude_fibers_in_mask_flag=0,  # Don't exclude fibers (no mask for tif_boundary=0)
        curvelets_group_radius=10,
        selected_scale=1,
        heatmap_STD_filter_size=16,
        heatmap_SQUARE_max_filter_size=12,
        heatmap_GAUSSIAN_disc_filter_sigma=4,
        minimum_nearest_fibers=2,
        minimum_box_size=32,
        fiber_midpoint_estimate=1,
        min_dist=[],
    )
    
    # Call process_image with new parameter structure
    result = process_image(
        image_params=image_params,
        fiber_params=fiber_params,
        output_params=output_params,
        boundary_params=boundary_params,
        advanced_options=advanced_options,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    if result is not None and 'fib_feat_df' in result:
        fib_feat_df = result['fib_feat_df']
        
        print(f"\nFiber Features:")
        print(f"  - Number of fibers: {len(fib_feat_df)}")
        if len(fib_feat_df) > 0:
            print(f"  - Columns: {list(fib_feat_df.columns)}")
            print(f"\nFirst 5 fibers:")
            print(fib_feat_df.head())
            
            # Display some statistics
            print(f"\nFiber Angle Statistics:")
            print(f"  - Mean: {fib_feat_df['fiber_absolute_angle'].mean():.2f}°")
            print(f"  - Std: {fib_feat_df['fiber_absolute_angle'].std():.2f}°")
            
            if 'alignment_mean' in fib_feat_df.columns:
                print(f"\nAlignment Statistics:")
                print(f"  - Mean: {fib_feat_df['alignment_mean'].mean():.4f}")
    else:
        print("No fiber features generated (make_feature_file may be disabled).")
    
    # Display results
    display_results(img, output_directory, result)


if __name__ == "__main__":
    # Check if GUI mode should be used (default)
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Run in CLI mode with defaults
        main_cli()
    else:
        # Run GUI mode
        main()
    
    print("\n" + "=" * 60)
    print("Press Ctrl+C or close windows to exit")
    print("=" * 60)
    plt.show()
