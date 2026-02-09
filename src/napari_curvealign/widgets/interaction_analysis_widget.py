# napari_curvealign/widgets/interaction_analysis_widget.py
# UI components for:
# - Interaction analysis parameters
# - Real-time visualization
# - Interaction network exploration
# - Export of interaction data
"""
Napari widget for interactive TME hierarchy and interaction analysis
"""
import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTreeWidget, QTreeWidgetItem, QSplitter, QLabel
)
from ..utils.napari_utils import create_layer_from_tme_object
from ...tme_quant.core.hierarchy import TMEHierarchy
from ...tme_quant.tme_analysis import CellFiberInteractionAnalyzer

class TMETreeWidget(QTreeWidget):
    """Tree widget for displaying TME hierarchy"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Name", "Type", "Properties"])
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
    
    def load_hierarchy(self, hierarchy: TMEHierarchy):
        """Load TME hierarchy into tree widget"""
        self.clear()
        self._add_hierarchy_node(hierarchy.root, self.invisibleRootItem())
        self.expandAll()
    
    def _add_hierarchy_node(self, tme_object, parent_item):
        """Add TME object to tree"""
        item = QTreeWidgetItem(parent_item)
        item.setText(0, tme_object.name)
        item.setText(1, tme_object.type.value)
        
        # Store reference to TME object
        item.setData(0, 0x100, tme_object)
        
        # Add children recursively
        for child in tme_object.children:
            self._add_hierarchy_node(child, item)
    
    def _on_item_double_clicked(self, item, column):
        """Handle item double-click to focus on object"""
        tme_object = item.data(0, 0x100)
        self.parent().object_selected.emit(tme_object)

class InteractionAnalysisWidget(QWidget):
    """Main widget for TME interaction analysis in Napari"""
    
    object_selected = None  # Signal for object selection
    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.hierarchy = None
        self.interaction_network = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        
        # Hierarchy tree
        self.tree_widget = TMETreeWidget(self)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load TME Sample")
        self.load_button.clicked.connect(self._load_tme_sample)
        
        self.analyze_button = QPushButton("Analyze Interactions")
        self.analyze_button.clicked.connect(self._analyze_interactions)
        self.analyze_button.setEnabled(False)
        
        self.visualize_button = QPushButton("Visualize Hierarchy")
        self.visualize_button.clicked.connect(self._visualize_hierarchy)
        self.visualize_button.setEnabled(False)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.visualize_button)
        
        # Status label
        self.status_label = QLabel("Ready")
        
        # Add widgets to layout
        layout.addWidget(self.tree_widget)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def _load_tme_sample(self):
        """Load TME sample and build hierarchy"""
        # This would typically load from file or database
        from .example_data import create_sample_tme_hierarchy
        
        tissue_sample, _ = create_sample_tme_hierarchy()
        self.hierarchy = TMEHierarchy(tissue_sample)
        
        # Update UI
        self.tree_widget.load_hierarchy(self.hierarchy)
        self.analyze_button.setEnabled(True)
        self.visualize_button.setEnabled(True)
        self.status_label.setText(f"Loaded: {tissue_sample.name}")
    
    def _analyze_interactions(self):
        """Analyze cell-fiber interactions"""
        if not self.hierarchy:
            return
        
        # Extract cells and fibers from hierarchy
        cells = self.hierarchy.get_objects_by_type(TMEType.CELL)
        fibers = self.hierarchy.get_objects_by_type(TMEType.FIBER)
        
        # Run interaction analysis
        analyzer = CellFiberInteractionAnalyzer()
        self.interaction_network = analyzer.detect_interactions(cells, fibers)
        
        # Display results
        metrics = self.interaction_network.calculate_network_metrics()
        self.status_label.setText(
            f"Found {metrics['total_interactions']} interactions "
            f"(Density: {metrics['interaction_density']:.3f})"
        )
        
        # Visualize interactions
        self._visualize_interactions()
    
    def _visualize_hierarchy(self):
        """Visualize TME hierarchy in Napari"""
        if not self.hierarchy:
            return
        
        # Clear existing layers
        self.viewer.layers.clear()
        
        # Create layers for each object type
        for obj_type in [TMEType.CELL, TMEType.FIBER, TMEType.VESSEL, TMEType.REGION]:
            objects = self.hierarchy.get_objects_by_type(obj_type)
            if objects:
                layer = create_layer_from_tme_object(objects, obj_type.value)
                self.viewer.add_layer(layer)
    
    def _visualize_interactions(self):
        """Visualize interaction network"""
        if not self.interaction_network:
            return
        
        # Create interaction visualization layer
        from ...tme_quant.visualization.interaction_visualization import (
            create_interaction_visualization
        )
        
        viz_layer = create_interaction_visualization(
            self.interaction_network,
            line_width=2,
            colormap='viridis'
        )
        
        self.viewer.add_layer(viz_layer)