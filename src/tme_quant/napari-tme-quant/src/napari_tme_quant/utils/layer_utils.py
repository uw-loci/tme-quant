"""Layer name construction and parsing helpers.

All napari layer names in the plugin follow the convention:
    [image_id] :: [ObjectType] :: [detail]

e.g.  "SHG_001 :: Fibers :: all"
      "SHG_001 :: Fibers :: TACS-3"
      "HE_001 :: Cells :: StarDist"
      "SHG_001 :: ROI :: Tumor_Boundary_1"

PluginState.layer_map is the only source of truth for layer ↔ object mapping.
"""

from __future__ import annotations

LAYER_SEP = " :: "

# TACS type → napari color (per CLAUDE_NAPARI.md §TACS color coding)
TACS_COLORS: dict[str, str] = {
    "TACS-3": "red",
    "TACS-2": "limegreen",
    "TACS-1": "dodgerblue",
    "outside": "lightgray",
}


def make_layer_name(image_id: str, obj_type: str, detail: str = "all") -> str:
    """Build a canonical layer name.

    Parameters
    ----------
    image_id : str  e.g. "SHG_001"
    obj_type : str  e.g. "Fibers", "Cells", "ROI", "Associations"
    detail   : str  e.g. "all", "TACS-3", "StarDist"
    """
    return f"{image_id}{LAYER_SEP}{obj_type}{LAYER_SEP}{detail}"


def parse_layer_name(name: str) -> tuple[str, str, str] | None:
    """Parse a layer name into (image_id, obj_type, detail).

    Returns None if the name does not follow the convention.
    """
    parts = name.split(LAYER_SEP)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


def image_id_from_layer(name: str) -> str | None:
    """Extract image_id from a layer name, or None if not a plugin layer."""
    parsed = parse_layer_name(name)
    return parsed[0] if parsed else None


def layers_for_image(viewer, image_id: str) -> list:
    """Return all napari layers whose name starts with [image_id] ::."""
    prefix = f"{image_id}{LAYER_SEP}"
    return [layer for layer in viewer.layers if layer.name.startswith(prefix)]
