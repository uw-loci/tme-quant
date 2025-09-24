"""
Boundary data structure.

This module defines the Boundary class for representing analysis boundaries
as masks, polygons, or polygon collections.
"""

from typing import NamedTuple, Literal, Union, List, Optional, Tuple, Any
import numpy as np


class Boundary(NamedTuple):
    """
    Boundary definition for relative angle measurements.
    
    Attributes
    ----------
    kind : {"mask", "polygon", "polygons"}
        Type of boundary representation
    data : np.ndarray | Polygon | List[Polygon]
        Boundary data (binary mask, single polygon, or list of polygons)
    spacing_xy : Tuple[float, float], optional
        Physical spacing between pixels (x, y) in micrometers
    """
    kind: Literal["mask", "polygon", "polygons"]
    data: Union[np.ndarray, Any, List[Any]]  # TODO: Define Polygon type
    spacing_xy: Optional[Tuple[float, float]] = None
