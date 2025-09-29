"""
CT-FIRE type system - organized by functional area.

This module provides all type definitions used in the CT-FIRE API,
organized into logical categories:
- core: Fundamental data structures (Fiber, FiberNetwork, FiberGraph)
- config: Configuration and option classes  
- results: Result and output structures
"""

# Core data structures
from .core import Fiber, FiberNetwork, FiberGraph

# Configuration classes
from .config import CTFireOptions

# Result structures  
from .results import CTFireResult, FiberMetrics

__all__ = [
    # Core types
    'Fiber', 'FiberNetwork', 'FiberGraph',
    # Configuration
    'CTFireOptions',
    # Results
    'CTFireResult', 'FiberMetrics'
]
