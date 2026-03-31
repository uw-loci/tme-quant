# tumor_model.py
"""
Tumor-related data models.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum
from ..base_models import (
    TMEObject, TMEType, ObjectType, Geometry, Classification,
    Measurement, GeometryType,
)


class TumorGrade(Enum):
    """Tumor grading classifications."""
    G1 = "Well differentiated"
    G2 = "Moderately differentiated"
    G3 = "Poorly differentiated"
    G4 = "Undifferentiated"
    GX = "Cannot be assessed"


class TumorRegion(TMEObject):
    """Represents a tumor region with spatial characteristics."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi: Optional[Any] = None,
        parent: Optional[TMEObject] = None,
        metadata: Optional[Dict[str, Any]] = None,
        geometry: Optional[Geometry] = None,
        grade: TumorGrade = TumorGrade.GX,
        necrosis_percentage: float = 0.0,
        proliferation_index: float = 0.0,
        invasion_front: bool = False,
        subregions: Optional[List["TumorRegion"]] = None,
        border_irregularity: float = 0.0,
        compactness: float = 0.0,
        distance_to_margin: float = 0.0,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.TUMOR_REGION,
            object_type=ObjectType.TUMOR_REGION,
            roi=roi,
            parent=parent,
            metadata=metadata,
        )
        self.geometry: Optional[Geometry] = geometry
        self.grade: TumorGrade = grade
        self.necrosis_percentage: float = necrosis_percentage
        self.proliferation_index: float = proliferation_index
        self.invasion_front: bool = invasion_front
        self.subregions: List["TumorRegion"] = (
            subregions if subregions is not None else []
        )
        self.border_irregularity: float = border_irregularity
        self.compactness: float = compactness
        self.distance_to_margin: float = distance_to_margin
        self.properties["is_tumor"] = True

    def calculate_morphometrics(self) -> Dict[str, float]:
        """Calculate tumor morphology metrics."""
        return {
            "area": self.geometry.area() if self.geometry else 0.0,
            "perimeter": self._calculate_perimeter(),
            "compactness": self.compactness,
            "border_irregularity": self.border_irregularity,
            "circularity": self._calculate_circularity(),
            "solidity": self._calculate_solidity(),
        }

    def _calculate_perimeter(self) -> float:
        if self.geometry is None:
            return 0.0
        if self.geometry.type == GeometryType.POLYGON:
            coords = self.geometry.coordinates
            if len(coords) > 1:
                perimeter = 0.0
                for i in range(len(coords)):
                    j = (i + 1) % len(coords)
                    perimeter += float(np.linalg.norm(coords[j] - coords[i]))
                return perimeter
        return 0.0

    def _calculate_circularity(self) -> float:
        if self.geometry is None:
            return 0.0
        area = self.geometry.area()
        perimeter = self._calculate_perimeter()
        if perimeter > 0:
            return (4 * np.pi * area) / (perimeter ** 2)
        return 0.0

    def _calculate_solidity(self) -> float:
        return 1.0  # Placeholder — requires convex hull implementation


class Tumor(TMEObject):
    """Comprehensive tumor representation."""

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        roi: Optional[Any] = None,
        parent: Optional[TMEObject] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regions: Optional[List[TumorRegion]] = None,
        dominant_grade: TumorGrade = TumorGrade.GX,
        spatial_distribution: str = "",
        molecular_subtype: str = "",
        tumor_stroma_ratio: float = 0.0,
        immune_infiltrate_density: float = 0.0,
        angiogenesis_index: float = 0.0,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.TUMOR,
            object_type=ObjectType.REGION,
            roi=roi,
            parent=parent,
            metadata=metadata,
        )
        self.regions: List[TumorRegion] = regions if regions is not None else []
        self.dominant_grade: TumorGrade = dominant_grade
        self.spatial_distribution: str = spatial_distribution
        self.molecular_subtype: str = molecular_subtype
        self.tumor_stroma_ratio: float = tumor_stroma_ratio
        self.immune_infiltrate_density: float = immune_infiltrate_density
        self.angiogenesis_index: float = angiogenesis_index
        self.properties["object_type"] = "tumor"

        # Wire regions into hierarchy
        for region in self.regions:
            self.add_child(region)

    def get_total_area(self) -> float:
        return sum(
            r.geometry.area() for r in self.regions if r.geometry is not None
        )

    def get_average_grade(self) -> float:
        if not self.regions:
            return 0.0
        total_area = self.get_total_area()
        if total_area == 0:
            return 0.0
        weighted_sum = 0.0
        for region in self.regions:
            if region.geometry is None:
                continue
            try:
                grade_value = float(region.grade.value.split()[0][1])
            except (IndexError, ValueError):
                continue
            weighted_sum += grade_value * region.geometry.area()
        return weighted_sum / total_area

    def get_invasion_front_regions(self) -> List[TumorRegion]:
        return [r for r in self.regions if r.invasion_front]


__all__ = ["TumorGrade", "TumorRegion", "Tumor"]