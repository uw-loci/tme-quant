# tissue_model.py
"""
Comprehensive tissue sample and region models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from enum import Enum

from ..base_models import (
    TMEObject, TMEType, ObjectType, Geometry, GeometryType, TMEMetadata, Measurement,
)


class TissueZone(Enum):
    """Tissue microenvironment zones."""
    TUMOR_CORE = "tumor_core"
    TUMOR_INVASIVE_FRONT = "tumor_invasive_front"
    PERITUMORAL_STROMA = "peritumoral_stroma"
    DISTANT_STROMA = "distant_stroma"
    NORMAL_TISSUE = "normal_tissue"
    NECROTIC_REGION = "necrotic_region"
    HYPOXIC_REGION = "hypoxic_region"


class TissueRegion(TMEObject):
    """Annotated tissue region with zone classification."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi: Optional[Any] = None,
        parent: Optional[TMEObject] = None,
        metadata: Optional[Dict[str, Any]] = None,
        geometry: Optional[Geometry] = None,
        zone_type: TissueZone = TissueZone.NORMAL_TISSUE,
        tissue_type: str = "",
        annotations: Optional[List[str]] = None,
        adjacent_regions: Optional[List["TissueRegion"]] = None,
        distance_to_tumor: float = float("inf"),
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.REGION,
            object_type=ObjectType.REGION,
            roi=roi,
            parent=parent,
            metadata=metadata,
        )
        self.geometry: Optional[Geometry] = geometry
        self.zone_type: TissueZone = zone_type
        self.tissue_type: str = tissue_type
        self.annotations: List[str] = annotations if annotations is not None else []
        self.adjacent_regions: List["TissueRegion"] = (
            adjacent_regions if adjacent_regions is not None else []
        )
        self.distance_to_tumor: float = distance_to_tumor


class TissueSample(TMEObject):
    """Comprehensive tissue sample — top-level container with QuPath-like hierarchy."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi: Optional[Any] = None,
        parent: Optional[TMEObject] = None,
        tme_metadata: Optional[TMEMetadata] = None,
        image_data: Optional[np.ndarray] = None,
        mask_data: Optional[np.ndarray] = None,
        annotations: Optional[List[TissueRegion]] = None,
        tumor: Optional[Any] = None,    # Tumor (avoid circular import)
        stroma: Optional[Any] = None,   # Stroma
        cells: Optional[list] = None,
        fibers: Optional[list] = None,
        vessels: Optional[list] = None,
        tissue_area: float = 0.0,
        tumor_stroma_ratio: float = 0.0,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.SAMPLE,
            object_type=ObjectType.REGION,
            roi=roi,
            parent=parent,
        )
        self.tme_metadata: Optional[TMEMetadata] = tme_metadata
        self.image_data: Optional[np.ndarray] = image_data
        self.mask_data: Optional[np.ndarray] = mask_data
        self.annotations: List[TissueRegion] = (
            annotations if annotations is not None else []
        )
        self.tumor = tumor
        self.stroma = stroma
        self.cells: list = cells if cells is not None else []
        self.fibers: list = fibers if fibers is not None else []
        self.vessels: list = vessels if vessels is not None else []
        self.tissue_area: float = tissue_area
        self.tumor_stroma_ratio: float = tumor_stroma_ratio

        # Wire all components into the hierarchy tree
        if self.tumor:
            self.add_child(self.tumor)
        if self.stroma:
            self.add_child(self.stroma)
        for ann in self.annotations:
            self.add_child(ann)
        for cell in self.cells:
            self.add_child(cell)
        for fiber in self.fibers:
            self.add_child(fiber)
        for vessel in self.vessels:
            self.add_child(vessel)

    # ------------------------------------------------------------------
    # TME metrics
    # ------------------------------------------------------------------

    def calculate_tme_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive TME metrics."""
        metrics: Dict[str, Any] = {
            "tissue_area": self.tissue_area,
            "tumor_stroma_ratio": self.tumor_stroma_ratio,
            "cell_density": self._calculate_cell_density(),
            "fiber_density": self._calculate_fiber_density(),
            "vessel_density": self._calculate_vessel_density(),
            "spatial_heterogeneity": self._calculate_spatial_heterogeneity(),
        }
        if self.tme_metadata:
            metrics["sample_id"] = self.tme_metadata.sample_id
        if self.tumor:
            metrics.update({
                "tumor_area": self.tumor.get_total_area(),
                "tumor_grade": self.tumor.dominant_grade.value,
                "necrosis_percentage": self._calculate_necrosis_percentage(),
                "invasion_front_length": self._calculate_invasion_front_length(),
            })
        return metrics

    def _calculate_cell_density(self) -> float:
        if self.tissue_area == 0:
            return 0.0
        return len(self.cells) / self.tissue_area

    def _calculate_fiber_density(self) -> float:
        if self.tissue_area == 0 or not self.fibers:
            return 0.0
        total_fiber_length = sum(
            getattr(f, "length", 0.0) for f in self.fibers
        )
        return total_fiber_length / self.tissue_area

    def _calculate_vessel_density(self) -> float:
        if self.tissue_area == 0 or not self.vessels:
            return 0.0
        total_vessel_area = sum(
            getattr(v, "lumen_area", 0.0) for v in self.vessels
        )
        return total_vessel_area / self.tissue_area

    def _calculate_spatial_heterogeneity(self) -> float:
        """Coefficient of variation of cell density across quadrants."""
        if not self.cells:
            return 0.0
        quadrants = self._divide_into_quadrants()
        densities = []
        for q in quadrants:
            if q.geometry is not None and q.geometry.area() > 0:
                count = self._count_cells_in_region(q)
                densities.append(count / q.geometry.area())
        if densities and np.mean(densities) > 0:
            return float(np.std(densities) / np.mean(densities))
        return 0.0

    def _divide_into_quadrants(self) -> List[TissueRegion]:
        bounds = self._get_tissue_bounds()
        if bounds is None:
            return []
        x_min, y_min, x_max, y_max = bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        quadrant_coords = [
            ([x_min, y_min], [x_mid, y_mid]),
            ([x_mid, y_min], [x_max, y_mid]),
            ([x_min, y_mid], [x_mid, y_max]),
            ([x_mid, y_mid], [x_max, y_max]),
        ]
        quadrants = []
        for i, (lo, hi) in enumerate(quadrant_coords):
            quadrants.append(TissueRegion(
                name=f"Quadrant_{i + 1}",
                geometry=Geometry(
                    type=GeometryType.RECTANGLE,
                    coordinates=np.array([lo, hi]),
                ),
            ))
        return quadrants

    def _count_cells_in_region(self, region: TissueRegion) -> int:
        if region.geometry is None or region.geometry.bounds is None:
            return 0
        b = region.geometry.bounds
        count = 0
        for cell in self.cells:
            cx, cy = getattr(cell, "centroid", (None, None))
            if cx is not None and b[0] <= cx <= b[3] and b[1] <= cy <= b[4]:
                count += 1
        return count

    def _get_tissue_bounds(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        if self.mask_data is not None:
            ys, xs = np.where(self.mask_data > 0)
            if len(xs) > 0:
                return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        if self.annotations:
            all_b = [
                a.geometry.bounds
                for a in self.annotations
                if a.geometry is not None and a.geometry.bounds is not None
            ]
            if all_b:
                return (
                    float(min(b[0] for b in all_b)),
                    float(min(b[1] for b in all_b)),
                    float(max(b[3] for b in all_b)),
                    float(max(b[4] for b in all_b)),
                )
        return None

    def _calculate_necrosis_percentage(self) -> float:
        if not self.tumor:
            return 0.0
        total = self.tumor.get_total_area()
        if total == 0:
            return 0.0
        necrotic = sum(
            r.geometry.area() * (r.necrosis_percentage / 100)
            for r in self.tumor.regions
            if r.geometry is not None
        )
        return (necrotic / total) * 100

    def _calculate_invasion_front_length(self) -> float:
        if not self.tumor:
            return 0.0
        return sum(
            r._calculate_perimeter()
            for r in self.tumor.get_invasion_front_regions()
        )


__all__ = ["TissueZone", "TissueRegion", "TissueSample"]