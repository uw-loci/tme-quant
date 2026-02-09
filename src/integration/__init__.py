"""
Integration layer between existing bioimage tools and QuPath-like hierarchy.

Provides adapters, bridges, and facades for seamless integration.
"""

from .adapters import (
    ImageProcessingAdapter,
    SegmentationAdapter,
    MeasurementAdapter,
    VisualizationAdapter,
    ProjectAdapter,
)

from .bridges import (
    BioImageBridge,
    AnalysisBridge,
    VisualizationBridge,
    DataBridge,
)

from .facades import (
    UnifiedAnalysisFacade,
    ProjectManagementFacade,
    VisualizationFacade,
)

from .events import (
    IntegrationEvent,
    EventHandler,
    EventPublisher,
    EventSubscriber,
)

from .registry import (
    ServiceRegistry,
    PluginRegistry,
    AdapterRegistry,
)

__version__ = "0.1.0"
__all__ = [
    # Adapters
    'ImageProcessingAdapter',
    'SegmentationAdapter',
    'MeasurementAdapter',
    'VisualizationAdapter',
    'ProjectAdapter',
    
    # Bridges
    'BioImageBridge',
    'AnalysisBridge',
    'VisualizationBridge',
    'DataBridge',
    
    # Facades
    'UnifiedAnalysisFacade',
    'ProjectManagementFacade',
    'VisualizationFacade',
    
    # Events
    'IntegrationEvent',
    'EventHandler',
    'EventPublisher',
    'EventSubscriber',
    
    # Registry
    'ServiceRegistry',
    'PluginRegistry',
    'AdapterRegistry',
]