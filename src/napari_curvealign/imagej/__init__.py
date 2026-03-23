"""ImageJ / Fiji integration via napari-imagej (optional dependency)."""

from .client import FijiBridge, get_fiji_bridge

__all__ = ["FijiBridge", "get_fiji_bridge"]
