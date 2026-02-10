## Method Registry

### Location: `tme_quant/cell_analysis/methods/registry.py`

"""
Registry for segmentation and classification methods.

Allows dynamic registration and retrieval of analysis methods.
"""

from typing import Dict, Type
from ...core.tme_models.cell_model import SegmentationMode, ClassificationMode


class MethodRegistry:
    """
    Registry for cell analysis methods.
    
    Manages registration and retrieval of segmentation and classification methods.
    """
    
    def __init__(self):
        """Initialize registry."""
        self._segmentation_methods: Dict[SegmentationMode, Type] = {}
        self._classification_methods: Dict[ClassificationMode, Type] = {}
    
    # ============================================================
    # SEGMENTATION METHODS
    # ============================================================
    
    def register_segmentation_method(
        self,
        mode: SegmentationMode,
        method_class: Type
    ) -> None:
        """
        Register a segmentation method.
        
        Args:
            mode: Segmentation mode enum
            method_class: Method class (must inherit from BaseSegmentationMethod)
        """
        self._segmentation_methods[mode] = method_class
    
    def get_segmentation_method(
        self,
        mode: SegmentationMode
    ) -> Type:
        """
        Get segmentation method class by mode.
        
        Args:
            mode: Segmentation mode
            
        Returns:
            Method class
            
        Raises:
            ValueError: If mode not registered
        """
        if mode not in self._segmentation_methods:
            raise ValueError(
                f"Segmentation method {mode.value} not registered. "
                f"Available: {list(self._segmentation_methods.keys())}"
            )
        
        return self._segmentation_methods[mode]
    
    def list_segmentation_methods(self) -> list:
        """List all registered segmentation methods."""
        return list(self._segmentation_methods.keys())
    
    # ============================================================
    # CLASSIFICATION METHODS
    # ============================================================
    
    def register_classification_method(
        self,
        mode: ClassificationMode,
        method_class: Type
    ) -> None:
        """
        Register a classification method.
        
        Args:
            mode: Classification mode enum
            method_class: Method class (must inherit from BaseClassificationMethod)
        """
        self._classification_methods[mode] = method_class
    
    def get_classification_method(
        self,
        mode: ClassificationMode
    ) -> Type:
        """
        Get classification method class by mode.
        
        Args:
            mode: Classification mode
            
        Returns:
            Method class
            
        Raises:
            ValueError: If mode not registered
        """
        if mode not in self._classification_methods:
            raise ValueError(
                f"Classification method {mode.value} not registered. "
                f"Available: {list(self._classification_methods.keys())}"
            )
        
        return self._classification_methods[mode]
    
    def list_classification_methods(self) -> list:
        """List all registered classification methods."""
        return list(self._classification_methods.keys())


# Global registry instance
_global_registry = MethodRegistry()


def get_registry() -> MethodRegistry:
    """Get the global method registry."""
    return _global_registry