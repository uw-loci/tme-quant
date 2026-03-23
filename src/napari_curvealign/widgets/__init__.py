"""CurveAlign napari UI subpackage.

Layout follows the pattern used by `napari-imagej`_ (multiple focused modules under
``widgets/`` instead of one enormous file).

.. _napari-imagej: https://github.com/imagej/napari-imagej/tree/main/src/napari_imagej/widgets
"""

from .curve_align_widget import CurveAlignWidget, create_curve_align_widget

__all__ = ["CurveAlignWidget", "create_curve_align_widget"]
