"""Registry for fiber analysis methods."""

from typing import Dict, Type


class MethodRegistry:
    """Registry mapping mode enums to method classes."""

    def __init__(self):
        self._orientation_methods: Dict = {}
        self._extraction_methods: Dict = {}

    def register_orientation_method(self, mode, method_class: Type):
        self._orientation_methods[mode] = method_class

    def get_orientation_method(self, mode) -> Type:
        if mode not in self._orientation_methods:
            raise ValueError(
                f"No orientation method registered for mode '{mode}'. "
                f"Available: {list(self._orientation_methods.keys())}"
            )
        return self._orientation_methods[mode]

    def register_extraction_method(self, mode, method_class: Type):
        self._extraction_methods[mode] = method_class

    def get_extraction_method(self, mode) -> Type:
        if mode not in self._extraction_methods:
            raise ValueError(
                f"No extraction method registered for mode '{mode}'. "
                f"Available: {list(self._extraction_methods.keys())}"
            )
        return self._extraction_methods[mode]
