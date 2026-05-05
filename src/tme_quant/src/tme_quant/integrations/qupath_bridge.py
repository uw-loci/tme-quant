"""
QuPath integration — bidirectional bridge between tme_quant and QuPath.

Export direction: tme_quant → QuPath
    export_hierarchy_geojson()   write a FeatureCollection QuPath can import

Import direction: QuPath → tme_quant
    load_qupath_annotations()    read QuPath annotation GeoJSON into ROIObjects
    load_qupath_measurements()   read QuPath measurement CSV into a DataFrame

Not included: napari — napari calls INTO tme_quant (one-way dependency rule).
See CLAUDE.md "Core / Plugin separation" and "integrations/" section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from tme_quant.core.base_models import TMEType
from tme_quant.core.hierarchy import TMEHierarchy
from tme_quant.core.roi_manager import ROIManager, ROIObject
from tme_quant.core.tme_objects.fiber_objects import FiberObject
from tme_quant.core.tme_objects.cell_objects import CellObject


# ── Export ────────────────────────────────────────────────────────────────────

def export_hierarchy_geojson(
    hierarchy: TMEHierarchy,
    output_path: Union[str, Path],
    pixel_size: float = 1.0,
    include_fibers: bool = True,
    include_cells: bool = True,
    include_rois: bool = True,
) -> Path:
    """Export a TMEHierarchy to a QuPath-compatible GeoJSON FeatureCollection.

    Parameters
    ----------
    hierarchy : TMEHierarchy
    output_path : str or Path
        Destination ``.geojson`` file (parent directories created automatically).
    pixel_size : float
        µm/pixel for coordinate scaling. 1.0 = pixel coordinates.
    include_fibers : bool
        Include ``FiberObject`` detections (LineString / Point geometries).
    include_cells : bool
        Include ``CellObject`` detections (Point geometries).
    include_rois : bool
        Include ``ROIObject`` annotations (Polygon / LineString / Point geometries).

    Returns
    -------
    Path — absolute path to the written ``.geojson`` file.
    """
    features = []

    if include_rois:
        for obj in hierarchy.get_objects_by_type(TMEType.ANNOTATION):
            if isinstance(obj, ROIObject):
                features.append(obj.to_geojson_feature())

    if include_fibers:
        for obj in hierarchy.get_objects_by_type(TMEType.FIBER):
            if isinstance(obj, FiberObject):
                feat = obj.to_geojson_feature(pixel_size=pixel_size)
                if feat["geometry"] is not None:
                    features.append(feat)

    if include_cells:
        for obj in hierarchy.get_objects_by_type(TMEType.CELL):
            if isinstance(obj, CellObject):
                features.append(obj.to_geojson_feature(pixel_size=pixel_size))

    geojson = {"type": "FeatureCollection", "features": features}
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2)
    return out


# ── Import ────────────────────────────────────────────────────────────────────

def load_qupath_annotations(
    geojson_path: Union[str, Path],
    manager: Optional[ROIManager] = None,
    parent_node=None,
) -> List[ROIObject]:
    """Import QuPath annotation GeoJSON as ``ROIObject`` instances.

    Delegates to ``ROIManager.from_qupath_geojson()`` which already handles
    Polygon, MultiPolygon, Point, and LineString features, and maps
    ``classification.name`` to ``ROIObject.annotation_type``.

    Parameters
    ----------
    geojson_path : str or Path
        Path to a ``.geojson`` file exported from QuPath
        (right-click Annotations → Export as GeoJSON, or via Groovy script).
    manager : ROIManager or None
        Existing manager to add ROIs into.  A fresh ``ROIManager()`` is
        created when None.
    parent_node : TMEObject or None
        Hierarchy node to attach the created ROIs under (e.g. an
        ``ImageEntry``).  When None the ROIs are returned without being
        attached to any hierarchy.

    Returns
    -------
    list of ROIObject — the newly created annotation objects.
    """
    if manager is None:
        manager = ROIManager()
    rois = manager.from_qupath_geojson(geojson_path)
    if parent_node is not None:
        for roi in rois:
            parent_node.add_child(roi)
    return rois


def load_qupath_measurements(
    csv_path: Union[str, Path],
) -> pd.DataFrame:
    """Read a QuPath measurement CSV export into a tidy ``pd.DataFrame``.

    QuPath exports per-object measurements via
    *File > Export Measurements* or a Groovy ``exportMeasurements`` script.
    The standard column layout is::

        Image | Name | Class | Parent | ROI | <measurement columns…>

    Returns
    -------
    pd.DataFrame
        Index is ``Name`` (the QuPath object label, matching
        ``ROIObject.label``).  All measurement columns are float64.
        ``Class`` and ``Parent`` columns are kept as strings when present.
    """
    df = pd.read_csv(csv_path)
    # Normalise common QuPath column name variants across versions
    renames = {
        "Object ID":     "Name",
        "Object type":   "Class",
        "Classification": "Class",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "Name" in df.columns:
        df = df.set_index("Name")
    return df


__all__ = [
    "export_hierarchy_geojson",
    "load_qupath_annotations",
    "load_qupath_measurements",
]
