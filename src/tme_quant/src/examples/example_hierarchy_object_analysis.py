"""
TMEQuant Hierarchy — Object-Based Analysis and Query

This example demonstrates the TME object hierarchy in depth.
It builds a realistic multi-image project from scratch, populates it with
tumors, cells, fibers, and stroma, then shows every major pattern for
querying, filtering, traversing and annotating the tree.

No real images are needed — all objects are constructed programmatically
so the example runs without any external data files.

Structure of what we build
--------------------------

    TMEObject (project root)
    ├── ImageEntry  "image_001"  (SHG + H&E, breast biopsy A)
    │   ├── Tumor   "tumor_A1"
    │   │   ├── TumorRegion  "region_A1_core"
    │   │   │   ├── CellObject   "cell_A1_001" … "cell_A1_050"  (tumor cells)
    │   │   │   └── FiberObject  "fiber_A1_001" … "fiber_A1_020"  (TACS-3 fibers)
    │   │   └── TumorRegion  "region_A1_front"
    │   │       ├── CellObject   "cell_A1_051" … "cell_A1_080"  (mixed)
    │   │       └── FiberObject  "fiber_A1_021" … "fiber_A1_040"  (mixed TACS)
    │   └── StromaRegion "stroma_A1"
    │       ├── CellObject  "cell_A1_081" … "cell_A1_100"  (immune cells)
    │       └── FiberObject "fiber_A1_041" … "fiber_A1_060"  (TACS-2 fibers)
    └── ImageEntry  "image_002"  (SHG + H&E, breast biopsy B)
        └── Tumor   "tumor_B1"
            └── TumorRegion  "region_B1_core"
                ├── CellObject   "cell_B1_001" … "cell_B1_030"
                └── FiberObject  "fiber_B1_001" … "fiber_B1_020"

Topics covered
--------------
  1.  Building the hierarchy — project, images, tumors, regions, cells, fibers
  2.  Parent/child navigation — .parent, .children, .get_ancestors(), .depth()
  3.  Whole-tree traversal  — get_descendants(), iter_descendants()
  4.  Type-based filtering  — filter_by_type(tme_type=…) and filter_by_class(…)
                               Note: StromaRegion uses TMEType.REGION internally;
                               use filter_by_class(StromaRegion) not
                               filter_by_type(TMEType.STROMA) to find stroma nodes.
  5.  ID-based lookup       — find_by_id() and hierarchy.get_object()
  6.  Scoped subtree queries — "give me all fibers under tumor_A1 only"
  7.  Property queries       — filter on TACS type, cell type, straightness, …
  8.  Properties bag         — update_properties() / get_property() for ad-hoc
                               annotations without subclassing
  9.  Metadata bag           — set_metadata() for provenance tags
  10. Cross-image project query — find all TACS-3 fibers across the whole project
  11. Hierarchy-aware analysis — compute per-image and per-tumor statistics
  12. Spatial context flags  — in_tumor_core / in_stroma / at_invasive_front
  13. Tree modification      — detach(), add_child(), remove_child()
  14. Validation             — hierarchy.validate_hierarchy()
  15. Serialisation          — to_hierarchy_dict() / to_dict()
  16. QuPath export          — hierarchy.export_to_qupath()
"""

from __future__ import annotations

import random
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

# ── Core hierarchy ────────────────────────────────────────────────────────────
from tme_quant.core.base_models import (
    TMEObject, TMEType, ObjectType,
    Geometry, GeometryType,
)
from tme_quant.core.hierarchy import TMEHierarchy
from tme_quant.core.image_entry import ImageEntry

# ── Concrete model classes ────────────────────────────────────────────────────
from tme_quant.core.tme_models.cell_model import CellObject, CellType
from tme_quant.core.tme_models.fiber_model import FiberObject
from tme_quant.core.tme_models.tumor_model import Tumor, TumorRegion, TumorGrade
from tme_quant.core.tme_models.stroma_model import StromaRegion, ECMComponent

# ── TACS classifier ───────────────────────────────────────────────────────────
from tme_quant.fiber_analysis.tacs import (
    classify_fiber_tacs, get_tacs_color,
)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — BUILD THE HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────

def _make_polygon_geometry(cx: float, cy: float, rx: float, ry: float) -> Geometry:
    """Create an elliptical polygon Geometry centred at (cx, cy)."""
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    coords = np.column_stack([
        cx + rx * np.cos(angles),
        cy + ry * np.sin(angles),
    ])
    return Geometry(type=GeometryType.POLYGON, coordinates=coords)


def _make_cells(
    prefix: str,
    n: int,
    cx: float, cy: float,
    spread: float,
    cell_type: CellType,
    in_tumor: bool = False,
    in_stroma: bool = False,
    in_invasive_margin: bool = False,
) -> List[CellObject]:
    """Factory: produce *n* CellObject instances scattered around (cx, cy)."""
    cells = []
    for i in range(n):
        x = cx + np.random.uniform(-spread, spread)
        y = cy + np.random.uniform(-spread, spread)
        area = np.random.uniform(80, 180)
        cells.append(CellObject(
            object_id=f"{prefix}_{i+1:03d}",
            centroid=(x, y),
            area=area,
            perimeter=np.sqrt(area) * 4,
            circularity=np.random.uniform(0.5, 0.95),
            eccentricity=np.random.uniform(0.1, 0.6),
            cell_type=cell_type,
            cell_type_confidence=np.random.uniform(0.7, 0.99),
            in_tumor_region=in_tumor,
            in_invasive_margin=in_invasive_margin,
            distance_to_tumor_boundary=(
                np.random.uniform(0, 80) if in_invasive_margin else
                np.random.uniform(80, 500) if in_stroma else
                np.random.uniform(0, 30)
            ),
        ))
    return cells


def _make_fibers(
    prefix: str,
    n: int,
    cx: float, cy: float,
    spread: float,
    tacs_zone: bool,
    angle_min: float, angle_max: float,
    straightness_min: float = 0.6,
    straightness_max: float = 0.98,
) -> List[FiberObject]:
    """
    Factory: produce *n* FiberObject instances.

    angle_min/max are the boundary-tangent angles to assign.
    TACS classification is computed and stored at creation time.
    """
    fibers = []
    for i in range(n):
        x = cx + np.random.uniform(-spread, spread)
        y = cy + np.random.uniform(-spread, spread)
        angle = np.random.uniform(angle_min, angle_max)   # tangent angle
        straightness = np.random.uniform(straightness_min, straightness_max)
        length = np.random.uniform(15, 80)
        width  = np.random.uniform(0.5, 3.0)
        dist   = np.random.uniform(0, 90) if tacs_zone else np.random.uniform(110, 400)

        # Build a short 2-point centerline
        dx = np.cos(np.radians(angle)) * length / 2
        dy = np.sin(np.radians(angle)) * length / 2
        centerline = np.array([[x - dx, y - dy], [x + dx, y + dy]])

        # Tangent angle stored as relative_angle_to_boundary_tangent
        tacs_type = classify_fiber_tacs(
            angle_to_tangent=angle,
            straightness=straightness,
            distance_to_boundary=dist,
        )

        fibers.append(FiberObject(
            object_id=f"{prefix}_{i+1:03d}",
            centerline=centerline,
            length=length,
            width=width,
            angle=angle,
            straightness=straightness,
            curvature=1.0 - straightness,
            nearest_boundary_distance=dist,
            relative_angle_to_boundary_tangent=angle,
            relative_angle_to_boundary_normal=90.0 - angle,
            in_tumor_boundary=tacs_zone,
            in_stroma=not tacs_zone,
            at_invasive_front=(dist < 30),
            tacs_type=tacs_type,
            tacs_score=np.random.uniform(0.4, 0.9) if tacs_type else None,
        ))
    return fibers


def build_project() -> TMEObject:
    """
    Construct a two-image project hierarchy and return the root node.

    Returns
    -------
    TMEObject (project root) — the entry point for all hierarchy queries.
    """
    # ── Root ─────────────────────────────────────────────────────────────────
    root = TMEObject(
        object_id="project_breast_cohort",
        name="Breast Cancer Cohort — SHG/H&E",
        tme_type=TMEType.PROJECT,
    )
    root.set_metadata("cohort", "breast_invasive_ductal")
    root.set_metadata("institution", "Example Cancer Center")
    root.set_metadata("pixel_size_um", 0.5)

    # ── Image 1 ───────────────────────────────────────────────────────────────
    img1 = ImageEntry(
        object_id="image_001",
        name="Biopsy A — SHG/H&E",
        image_data=np.zeros((512, 512, 2), dtype=np.float32),  # synthetic 2-ch
        channel_names=["SHG", "HE"],
        pixel_size=(0.5, 0.5),
        modality="SHG+HE",
        parent=root,
    )
    img1.set_metadata("patient_id", "PT-001")
    img1.set_metadata("diagnosis", "IDC Grade 3")

    # ── Tumor A1 ──────────────────────────────────────────────────────────────
    tumor_A1 = Tumor(
        object_id="tumor_A1",
        name="Primary tumor — biopsy A",
        dominant_grade=TumorGrade.G3,
        tumor_stroma_ratio=0.62,
        parent=img1,
    )

    # Tumor core region
    region_core = TumorRegion(
        object_id="region_A1_core",
        name="Tumor core",
        geometry=_make_polygon_geometry(200, 200, 80, 60),
        grade=TumorGrade.G3,
        necrosis_percentage=12.0,
        invasion_front=False,
        parent=tumor_A1,
    )
    region_core.update_properties(zone="tumor_core", distance_from_margin=200)

    # Invasive front region
    region_front = TumorRegion(
        object_id="region_A1_front",
        name="Invasive front",
        geometry=_make_polygon_geometry(200, 200, 120, 100),
        grade=TumorGrade.G3,
        invasion_front=True,
        parent=tumor_A1,
    )
    region_front.update_properties(zone="invasive_front", distance_from_margin=20)

    # Add cells and fibers to core
    for cell in _make_cells(
        "cell_A1", 50, 200, 200, 70, CellType.TUMOR, in_tumor=True
    ):
        region_core.add_child(cell)

    for fiber in _make_fibers(
        "fiber_A1_core", 20, 200, 200, 70,
        tacs_zone=True, angle_min=62, angle_max=88,   # mostly TACS-3
        straightness_min=0.75,
    ):
        region_core.add_child(fiber)

    # Add cells and fibers to invasive front
    front_cell_types = [CellType.TUMOR] * 20 + [CellType.MACROPHAGE] * 10
    front_cells = _make_cells(
        "cell_A1_front", 30, 200, 200, 110, CellType.TUMOR,
        in_tumor=True, in_invasive_margin=True
    )
    for idx, cell in enumerate(front_cells):
        cell.cell_type = front_cell_types[idx % len(front_cell_types)]
        region_front.add_child(cell)

    for fiber in _make_fibers(
        "fiber_A1_front", 20, 200, 200, 110,
        tacs_zone=True, angle_min=0, angle_max=89,    # mixed TACS
        straightness_min=0.55,
    ):
        region_front.add_child(fiber)

    # ── Stroma A1 ─────────────────────────────────────────────────────────────
    stroma_A1 = StromaRegion(
        object_id="stroma_A1",
        name="Peri-tumoral stroma",
        geometry=_make_polygon_geometry(200, 200, 200, 180),
        ecm_composition={ECMComponent.COLLAGEN: 0.72, ECMComponent.FIBRONECTIN: 0.18},
        fibrosis_score=0.65,
        parent=img1,
    )
    stroma_A1.update_properties(zone="stroma")

    for cell in _make_cells(
        "cell_A1_stroma", 20, 200, 200, 190, CellType.T_CELL, in_stroma=True
    ):
        stroma_A1.add_child(cell)

    for fiber in _make_fibers(
        "fiber_A1_stroma", 20, 200, 200, 190,
        tacs_zone=False, angle_min=2, angle_max=28,   # mostly TACS-2 (parallel)
        straightness_min=0.80,
    ):
        stroma_A1.add_child(fiber)

    # ── Image 2 ───────────────────────────────────────────────────────────────
    img2 = ImageEntry(
        object_id="image_002",
        name="Biopsy B — SHG/H&E",
        image_data=np.zeros((512, 512, 2), dtype=np.float32),
        channel_names=["SHG", "HE"],
        pixel_size=(0.5, 0.5),
        modality="SHG+HE",
        parent=root,
    )
    img2.set_metadata("patient_id", "PT-002")
    img2.set_metadata("diagnosis", "IDC Grade 2")

    # Tumor B1 — smaller, lower grade
    tumor_B1 = Tumor(
        object_id="tumor_B1",
        name="Primary tumor — biopsy B",
        dominant_grade=TumorGrade.G2,
        tumor_stroma_ratio=0.45,
        parent=img2,
    )
    region_B1 = TumorRegion(
        object_id="region_B1_core",
        name="Tumor core B",
        geometry=_make_polygon_geometry(256, 256, 60, 50),
        grade=TumorGrade.G2,
        invasion_front=False,
        parent=tumor_B1,
    )
    for cell in _make_cells(
        "cell_B1", 30, 256, 256, 55, CellType.TUMOR, in_tumor=True
    ):
        region_B1.add_child(cell)

    for fiber in _make_fibers(
        "fiber_B1", 20, 256, 256, 55,
        tacs_zone=True, angle_min=30, angle_max=60,   # mostly TACS-1
        straightness_min=0.50, straightness_max=0.75,
    ):
        region_B1.add_child(fiber)

    return root


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HIERARCHY NAVIGATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def demo_navigation(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 2 — Parent/Child Navigation")
    print("═" * 70)

    # Pick a deep node
    img1 = root.find_by_id("image_001")
    tumor = root.find_by_id("tumor_A1")
    region = root.find_by_id("region_A1_core")
    fiber = root.find_by_id("fiber_A1_core_001")

    print(f"\n  fiber '{fiber.object_id}'")
    print(f"    .parent          → {fiber.parent.object_id!r}")
    print(f"    .parent.parent   → {fiber.parent.parent.object_id!r}")
    print(f"    .depth()         → {fiber.depth()}")
    print(f"    .get_root()      → {fiber.get_root().object_id!r}")

    ancestors = fiber.get_ancestors()
    print(f"    .get_ancestors() → {[a.object_id for a in ancestors]}")

    print(f"\n  region '{region.object_id}'")
    print(f"    direct children  → {len(region.children)} objects")
    type_counts = Counter(c.tme_type.value for c in region.children)
    for t, n in sorted(type_counts.items()):
        print(f"      {t}: {n}")

    print(f"\n  tumor '{tumor.object_id}'")
    print(f"    .get_children_of_type(TUMOR_REGION) → "
          f"{[r.object_id for r in tumor.get_children_of_type(TMEType.TUMOR_REGION)]}")


def demo_traversal(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 3 — Whole-Tree Traversal")
    print("═" * 70)

    all_objects = root.get_descendants(include_self=True)
    type_counts = Counter(o.tme_type.value for o in all_objects)

    print(f"\n  Total objects in hierarchy: {len(all_objects)}")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:20s}: {n:4d}")

    # iter_descendants is a lazy generator — useful for large trees
    n_fibers_lazy = sum(
        1 for o in root.iter_descendants()
        if o.tme_type == TMEType.FIBER
    )
    print(f"\n  Fibers counted via iter_descendants(): {n_fibers_lazy}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TYPE-BASED FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def demo_type_filtering(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 4 — Type-Based Filtering")
    print("═" * 70)

    # All fibers across both images
    all_fibers = root.filter_by_type(tme_type=TMEType.FIBER)
    print(f"\n  root.filter_by_type(FIBER)         → {len(all_fibers)} fibers")

    # All cells across both images
    all_cells = root.filter_by_type(tme_type=TMEType.CELL)
    print(f"  root.filter_by_type(CELL)          → {len(all_cells)} cells")

    # filter_by_class uses isinstance — works for any subclass
    fiber_objects = root.filter_by_class(FiberObject)
    cell_objects  = root.filter_by_class(CellObject)
    tumor_objects = root.filter_by_class(Tumor)
    print(f"\n  filter_by_class(FiberObject)       → {len(fiber_objects)}")
    print(f"  filter_by_class(CellObject)        → {len(cell_objects)}")
    print(f"  filter_by_class(Tumor)             → {len(tumor_objects)}")

    # Scoped: fibers only under image_001
    img1 = root.find_by_id("image_001")
    img1_fibers = img1.filter_by_type(tme_type=TMEType.FIBER)
    img2 = root.find_by_id("image_002")
    img2_fibers = img2.filter_by_type(tme_type=TMEType.FIBER)
    print(f"\n  Scoped to image_001 — fibers: {len(img1_fibers)}")
    print(f"  Scoped to image_002 — fibers: {len(img2_fibers)}")

    # Scoped: fibers only under tumor_A1 (excludes stroma fibers)
    tumor = root.find_by_id("tumor_A1")
    tumor_fibers = tumor.filter_by_type(tme_type=TMEType.FIBER)
    stroma = root.find_by_id("stroma_A1")
    stroma_fibers = stroma.filter_by_type(tme_type=TMEType.FIBER)
    print(f"  Scoped to tumor_A1  — fibers: {len(tumor_fibers)}")
    print(f"  Scoped to stroma_A1 — fibers: {len(stroma_fibers)}")

    # Combined type filter (dual enum)
    mixed = root.filter_by_type(
        tme_type=TMEType.FIBER, object_type=ObjectType.FIBER
    )
    print(f"\n  filter_by_type(tme_type=FIBER, object_type=FIBER) → {len(mixed)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ID-BASED LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def demo_id_lookup(root: TMEObject, hierarchy: TMEHierarchy) -> None:
    print("\n" + "═" * 70)
    print("SECTION 5 — ID-Based Lookup")
    print("═" * 70)

    # find_by_id on root descends the full tree
    obj = root.find_by_id("region_A1_front")
    print(f"\n  root.find_by_id('region_A1_front') → {obj!r}")

    # Scoped find: limit search to image_001 subtree
    img1 = root.find_by_id("image_001")
    found_in_img1   = img1.find_by_id("fiber_A1_core_005")
    not_in_img1     = img1.find_by_id("fiber_B1_001")   # lives in image_002
    print(f"\n  img1.find_by_id('fiber_A1_core_005') → {found_in_img1.object_id!r}")
    print(f"  img1.find_by_id('fiber_B1_001')      → {not_in_img1!r}  (not in subtree)")

    # TMEHierarchy.get_object always searches from root
    via_hierarchy = hierarchy.get_object("fiber_B1_001")
    print(f"\n  hierarchy.get_object('fiber_B1_001') → {via_hierarchy.object_id!r}")

    # hierarchy.get_children / get_descendants by id
    children = hierarchy.get_children("tumor_A1")
    print(f"\n  hierarchy.get_children('tumor_A1') → "
          f"{[c.object_id for c in children]}")

    descendants = hierarchy.get_descendants("region_A1_core")
    desc_types = Counter(d.tme_type.value for d in descendants)
    print(f"  hierarchy.get_descendants('region_A1_core') → "
          f"{dict(desc_types)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PROPERTY-BASED QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def demo_property_queries(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 6 — Property-Based Queries on Real Object Fields")
    print("═" * 70)

    all_fibers: List[FiberObject] = root.filter_by_class(FiberObject)
    all_cells:  List[CellObject]  = root.filter_by_class(CellObject)

    # ── TACS type distribution ────────────────────────────────────────────────
    tacs_counts = Counter(
        f.tacs_type for f in all_fibers if f.tacs_type is not None
    )
    total_classified = sum(tacs_counts.values())
    print(f"\n  TACS type distribution (all images, {len(all_fibers)} fibers):")
    for t in ['TACS-1', 'TACS-2', 'TACS-3', None]:
        n = tacs_counts.get(t, 0)
        label = t or "unclassified"
        pct = 100 * n / len(all_fibers) if all_fibers else 0
        print(f"    {label:15s}: {n:4d}  ({pct:.1f}%)")

    # ── High-risk TACS-3 fibers ───────────────────────────────────────────────
    tacs3_fibers = [f for f in all_fibers if f.tacs_type == 'TACS-3']
    print(f"\n  TACS-3 (invasive) fibers: {len(tacs3_fibers)}")

    # Scoped: TACS-3 under tumor_A1 only
    tumor_A1 = root.find_by_id("tumor_A1")
    tacs3_tumor_A1 = [
        f for f in tumor_A1.filter_by_class(FiberObject)
        if f.tacs_type == 'TACS-3'
    ]
    print(f"  TACS-3 under tumor_A1:    {len(tacs3_tumor_A1)}")

    # ── Straight fibers (straightness >= 0.8) ────────────────────────────────
    straight = [f for f in all_fibers if f.straightness >= 0.8]
    print(f"\n  Straight fibers (straightness ≥ 0.8): {len(straight)}")

    # Straight + TACS-3 (most invasive signature)
    invasive_straight = [f for f in straight if f.tacs_type == 'TACS-3']
    print(f"  Straight + TACS-3:                    {len(invasive_straight)}")

    # ── Fibers at the invasive front ──────────────────────────────────────────
    invasive_front_fibers = [
        f for f in all_fibers
        if getattr(f, 'at_invasive_front', False)
    ]
    print(f"\n  Fibers at invasive front: {len(invasive_front_fibers)}")

    # ── Cell type distribution ────────────────────────────────────────────────
    cell_types = Counter(
        c.cell_type.value for c in all_cells if c.cell_type is not None
    )
    print(f"\n  Cell type distribution ({len(all_cells)} total):")
    for ct, n in sorted(cell_types.items(), key=lambda x: -x[1]):
        print(f"    {ct:12s}: {n}")

    # ── Immune cells in tumor boundary regions ────────────────────────────────
    immune_types = {CellType.T_CELL, CellType.B_CELL,
                    CellType.MACROPHAGE, CellType.NK_CELL, CellType.IMMUNE}
    immune_in_tumor = [
        c for c in all_cells
        if c.cell_type in immune_types and c.in_tumor_region
    ]
    print(f"\n  Immune cells inside tumor region: {len(immune_in_tumor)}")

    # ── Cells with high confidence classification ─────────────────────────────
    high_conf = [
        c for c in all_cells
        if c.cell_type_confidence is not None and c.cell_type_confidence >= 0.90
    ]
    print(f"  Cells with classification confidence ≥ 0.90: {len(high_conf)}")

    # ── Cells within 50 µm of the tumor boundary ─────────────────────────────
    near_boundary = [
        c for c in all_cells
        if c.distance_to_tumor_boundary is not None
        and c.distance_to_tumor_boundary <= 50.0
    ]
    print(f"  Cells within 50 µm of tumor boundary: {len(near_boundary)}")

    # ── Long fibers (length > 50 µm) ─────────────────────────────────────────
    long_fibers = [f for f in all_fibers if f.length > 50]
    print(f"\n  Long fibers (length > 50 µm): {len(long_fibers)}")
    if long_fibers:
        print(f"  Mean length of long fibers:   "
              f"{np.mean([f.length for f in long_fibers]):.1f} µm")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PROPERTIES BAG AND METADATA
# ─────────────────────────────────────────────────────────────────────────────

def demo_properties_and_metadata(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 7 — Properties Bag and Metadata Annotations")
    print("═" * 70)

    # update_properties() stores arbitrary key-value pairs on any node.
    # This is the right pattern for ad-hoc analysis outputs that do not
    # require a new subclass.

    tumor = root.find_by_id("tumor_A1")
    tumor.update_properties(
        collagen_risk_score=0.78,
        tacs3_fraction=0.42,
        reviewer="Dr. Smith",
        reviewed_date="2025-03-01",
    )
    print(f"\n  Set properties on tumor_A1:")
    print(f"    collagen_risk_score = {tumor.get_property('collagen_risk_score')}")
    print(f"    tacs3_fraction      = {tumor.get_property('tacs3_fraction')}")
    print(f"    reviewer            = {tumor.get_property('reviewer')}")
    print(f"    missing_key         = {tumor.get_property('missing_key', default='N/A')}")

    # set_metadata() is for provenance / acquisition metadata.
    img2 = root.find_by_id("image_002")
    img2.set_metadata("scan_date",       "2025-01-15")
    img2.set_metadata("acquisition_mode","multiphoton_SHG")
    img2.set_metadata("exposure_time_ms", 80)
    print(f"\n  Metadata on image_002:")
    for k, v in img2.metadata.items():
        print(f"    {k}: {v}")

    # Annotate individual fibers with an analysis result
    region = root.find_by_id("region_A1_core")
    fibers = region.filter_by_class(FiberObject)
    for fiber in fibers[:5]:
        fiber.update_properties(
            mechanical_coupling=round(np.random.uniform(0.3, 0.9), 3),
            invasive_potential=round(np.random.uniform(0.4, 0.95), 3),
        )
    print(f"\n  Annotated first 5 fibers of region_A1_core with scores.")
    annotated = [
        f for f in fibers
        if f.get_property('mechanical_coupling') is not None
    ]
    print(f"  Fibers with mechanical_coupling set: {len(annotated)}")

    # Query annotated fibers by property value
    high_invasive = [
        f for f in fibers
        if f.get_property('invasive_potential', 0.0) >= 0.7
    ]
    print(f"  Fibers with invasive_potential ≥ 0.7: {len(high_invasive)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CROSS-IMAGE PROJECT-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def demo_project_analysis(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 8 — Cross-Image Project-Level Analysis")
    print("═" * 70)

    # Retrieve all ImageEntry nodes
    images = root.filter_by_class(ImageEntry)
    print(f"\n  Images in project: {len(images)}")

    per_image_stats = []

    for img in images:
        fibers = img.filter_by_class(FiberObject)
        cells  = img.filter_by_class(CellObject)
        tumors = img.filter_by_class(Tumor)

        tacs_counts = Counter(
            f.tacs_type for f in fibers if f.tacs_type is not None
        )
        n_classified = sum(tacs_counts.values())
        tacs3_frac = (
            tacs_counts.get('TACS-3', 0) / n_classified
            if n_classified > 0 else 0.0
        )
        mean_straight = (
            np.mean([f.straightness for f in fibers]) if fibers else np.nan
        )
        patient = img.metadata.get("patient_id", "?")
        diagnosis = img.metadata.get("diagnosis", "?")

        stat = {
            "image_id":       img.object_id,
            "patient_id":     patient,
            "diagnosis":      diagnosis,
            "n_fibers":       len(fibers),
            "n_cells":        len(cells),
            "n_tumors":       len(tumors),
            "tacs1":          tacs_counts.get('TACS-1', 0),
            "tacs2":          tacs_counts.get('TACS-2', 0),
            "tacs3":          tacs_counts.get('TACS-3', 0),
            "tacs3_fraction": round(tacs3_frac, 3),
            "mean_straightness": round(float(mean_straight), 3),
        }
        per_image_stats.append(stat)

    # Print table
    hdr = (
        f"  {'image_id':12s} {'patient':8s} {'diagnosis':20s} "
        f"{'fibers':>7s} {'cells':>6s} "
        f"{'TACS-1':>7s} {'TACS-2':>7s} {'TACS-3':>7s} "
        f"{'T3%':>6s} {'straight':>9s}"
    )
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for s in per_image_stats:
        print(
            f"  {s['image_id']:12s} {s['patient_id']:8s} {s['diagnosis']:20s} "
            f"{s['n_fibers']:7d} {s['n_cells']:6d} "
            f"{s['tacs1']:7d} {s['tacs2']:7d} {s['tacs3']:7d} "
            f"{s['tacs3_fraction']*100:5.1f}% {s['mean_straightness']:9.3f}"
        )

    # Cross-image: all TACS-3 fibers from any image
    all_tacs3 = [
        f for f in root.filter_by_class(FiberObject)
        if f.tacs_type == 'TACS-3'
    ]
    print(f"\n  All TACS-3 fibers across project: {len(all_tacs3)}")
    print(f"  Mean straightness of TACS-3:      "
          f"{np.mean([f.straightness for f in all_tacs3]):.3f}")

    # Per-tumor TACS-3 fractions (which tumors are most invasive?)
    print(f"\n  Per-tumor TACS-3 fraction:")
    for tumor in root.filter_by_class(Tumor):
        fibers = tumor.filter_by_class(FiberObject)
        n = sum(1 for f in fibers if f.tacs_type == 'TACS-3')
        frac = n / len(fibers) if fibers else 0.0
        grade = tumor.dominant_grade.value
        print(f"    {tumor.object_id:15s}  ({grade:25s})  "
              f"TACS-3: {n}/{len(fibers)}  ({frac*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — HIERARCHY-AWARE SPATIAL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def demo_spatial_context(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 9 — Spatial Context: Zone-Based Object Retrieval")
    print("═" * 70)

    # The hierarchy encodes spatial context via tme_type and properties.
    # We can ask "which cells are in which zone" purely from tree position.

    # Cells that are direct children of a TumorRegion with invasion_front=True
    invasive_front_cells = []
    for region in root.filter_by_class(TumorRegion):
        if region.invasion_front:
            invasive_front_cells.extend(region.filter_by_class(CellObject))

    ct = Counter(c.cell_type.value for c in invasive_front_cells
                 if c.cell_type is not None)
    print(f"\n  Cells at invasive front regions: {len(invasive_front_cells)}")
    print(f"  Cell types at invasive front:    {dict(ct)}")

    # Fibers under stroma nodes only.
    # StromaRegion.__init__ passes tme_type=TMEType.REGION (not TMEType.STROMA),
    # so filter_by_type(TMEType.STROMA) returns nothing.
    # filter_by_class(StromaRegion) is the correct query for StromaRegion nodes.
    stroma_fibers = []
    for node in root.filter_by_class(StromaRegion):
        stroma_fibers.extend(node.filter_by_class(FiberObject))
    tacs_st = Counter(f.tacs_type for f in stroma_fibers)
    print(f"\n  Fibers in stroma regions:        {len(stroma_fibers)}")
    print(f"  TACS types in stroma:            {dict(tacs_st)}")

    # Walk upward: for a given fiber, determine which image it belongs to
    fiber = root.find_by_id("fiber_A1_front_001")
    image_ancestor = next(
        (a for a in fiber.get_ancestors() if isinstance(a, ImageEntry)), None
    )
    tumor_ancestor = next(
        (a for a in fiber.get_ancestors() if isinstance(a, Tumor)), None
    )
    region_ancestor = next(
        (a for a in fiber.get_ancestors() if isinstance(a, TumorRegion)), None
    )
    print(f"\n  Upward walk from fiber_A1_front_001:")
    print(f"    TumorRegion ancestor: {region_ancestor.object_id!r}")
    print(f"    Tumor ancestor:       {tumor_ancestor.object_id!r}")
    print(f"    ImageEntry ancestor:  {image_ancestor.object_id!r}")

    # Cells whose nearest parent TumorRegion is the invasive front
    def parent_region(cell: TMEObject) -> str | None:
        for a in cell.get_ancestors():
            if isinstance(a, TumorRegion):
                return a.object_id
        return None

    cells_by_region: Dict[str, int] = defaultdict(int)
    for cell in root.filter_by_class(CellObject):
        r = parent_region(cell)
        if r:
            cells_by_region[r] += 1
    print(f"\n  Cell counts by parent TumorRegion:")
    for rid, n in sorted(cells_by_region.items()):
        print(f"    {rid}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — TREE MODIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def demo_tree_modification(root: TMEObject) -> None:
    print("\n" + "═" * 70)
    print("SECTION 10 — Tree Modification (add, detach, re-attach)")
    print("═" * 70)

    # Add a new CellObject at runtime — no schema change needed
    region_core = root.find_by_id("region_A1_core")
    n_before = len(region_core.filter_by_class(CellObject))

    new_cell = CellObject(
        object_id="cell_A1_new_runtime",
        centroid=(210.0, 185.0),
        area=140.0,
        cell_type=CellType.NEUTROPHIL,
        cell_type_confidence=0.83,
    )
    region_core.add_child(new_cell)
    n_after = len(region_core.filter_by_class(CellObject))
    print(f"\n  region_A1_core cells before: {n_before}  after add_child: {n_after}")

    # Confirm parent is set
    print(f"  new_cell.parent = {new_cell.parent.object_id!r}")

    # Detach — removes from parent but keeps object in memory
    new_cell.detach()
    n_detached = len(region_core.filter_by_class(CellObject))
    print(f"  After detach: region has {n_detached} cells")
    print(f"  new_cell.parent after detach: {new_cell.parent!r}")

    # Re-attach to a different region (re-assigning to invasive front)
    region_front = root.find_by_id("region_A1_front")
    region_front.add_child(new_cell)
    print(f"  Re-attached to {new_cell.parent.object_id!r}")

    # remove_child is equivalent to calling detach from the parent side
    region_front.remove_child(new_cell)
    print(f"  After remove_child: parent = {new_cell.parent!r}")

    # Move a whole subtree — detach a TumorRegion and re-attach to new tumor
    print(f"\n  Subtree move example:")
    tumor_A1 = root.find_by_id("tumor_A1")
    region_front_obj = root.find_by_id("region_A1_front")
    n_tumor_before = len(tumor_A1.filter_by_class(FiberObject))
    region_front_obj.detach()
    n_tumor_after = len(tumor_A1.filter_by_class(FiberObject))
    print(f"  tumor_A1 fibers before detaching front: {n_tumor_before}")
    print(f"  tumor_A1 fibers after  detaching front: {n_tumor_after}")
    # Re-attach so downstream sections still work
    tumor_A1.add_child(region_front_obj)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — VALIDATION AND SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def demo_validation_and_serialisation(
    root: TMEObject, hierarchy: TMEHierarchy
) -> None:
    print("\n" + "═" * 70)
    print("SECTION 11 — Validation and Serialisation")
    print("═" * 70)

    # Validate consistency (should be clean)
    issues = hierarchy.validate_hierarchy()
    print(f"\n  validate_hierarchy() issues: {issues or 'none — hierarchy is clean'}")

    # to_dict() — shallow, no children
    tumor = root.find_by_id("tumor_A1")
    d = tumor.to_dict()
    print(f"\n  tumor_A1.to_dict() keys: {list(d.keys())}")

    # to_hierarchy_dict() — recursive
    region = root.find_by_id("region_A1_core")
    tree_dict = region.to_hierarchy_dict()
    print(f"\n  region_A1_core.to_hierarchy_dict():")
    print(f"    object_id:  {tree_dict['object_id']}")
    print(f"    tme_type:   {tree_dict['tme_type']}")
    print(f"    n_children: {len(tree_dict['children'])}")
    child_types = Counter(c['tme_type'] for c in tree_dict['children'])
    print(f"    child types: {dict(child_types)}")

    # export_to_qupath — produces a nested dict compatible with QuPath JSON
    # (requires geometry objects with serialisable coordinates)
    try:
        qupath_tree = hierarchy.export_to_qupath()
        print(f"\n  hierarchy.export_to_qupath() root id: {qupath_tree['id']!r}")
        print(f"  Top-level children: {[c['id'] for c in qupath_tree['children']]}")
    except Exception as e:
        # Geometry coordinates not serialisable in this synthetic example;
        # in production use with real ImageEntry + Geometry objects this works.
        print(f"\n  hierarchy.export_to_qupath() skipped in synthetic demo ({type(e).__name__})")
        print(f"  (Use to_hierarchy_dict() for a geometry-independent export.)")

    # get_spatial_hierarchy — compact tree with bounds
    spatial = hierarchy.get_spatial_hierarchy()
    print(f"\n  get_spatial_hierarchy() root: {spatial['id']!r}, "
          f"children: {len(spatial['children'])}")

    # JSON round-trip of a single object dict
    fiber = root.find_by_id("fiber_A1_core_001")
    fiber_dict = fiber.to_dict()
    json_str = json.dumps(fiber_dict, default=str, indent=2)
    print(f"\n  fiber_A1_core_001.to_dict() → JSON ({len(json_str)} chars)")
    reloaded = json.loads(json_str)
    print(f"  Round-trip object_id: {reloaded['object_id']!r}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("TMEQuant — Hierarchy Object-Based Analysis and Query Demo")
    print("=" * 70)

    # Build and wrap in hierarchy manager
    root      = build_project()
    hierarchy = TMEHierarchy(root=root)

    all_objects = root.get_descendants(include_self=True)
    print(f"\nHierarchy built: {len(all_objects)} total objects")
    print(repr(hierarchy))

    # Run every demo section
    demo_navigation(root)
    demo_traversal(root)
    demo_type_filtering(root)
    demo_id_lookup(root, hierarchy)
    demo_property_queries(root)
    demo_properties_and_metadata(root)
    demo_project_analysis(root)
    demo_spatial_context(root)
    demo_tree_modification(root)
    demo_validation_and_serialisation(root, hierarchy)

    print("\n" + "=" * 70)
    print("Demo complete — all hierarchy patterns executed successfully.")
    print("=" * 70)
    print("""
Key take-aways
──────────────
  1. Every TME object (cell, fiber, tumor, stroma, image) is a first-class
     hierarchy node inheriting from TMEObject.  No separate manager is
     needed to move around the tree.

  2. filter_by_type() and filter_by_class() scope queries to any subtree.
     "All TACS-3 fibers under this tumor" is a one-liner.

  3. get_ancestors() / get_root() let you walk upward to determine which
     image, tumor, or region a given object belongs to — essential for
     generating per-region statistics without separate bookkeeping.

  4. update_properties() / set_metadata() attach arbitrary annotations to
     any node at runtime without subclassing.

  5. Tree modification (add_child / detach / remove_child) is safe — both
     sides of the parent–child link are always kept consistent.

  6. TMEHierarchy wraps the root and provides a manager-level API
     (get_object, add_object, validate_hierarchy, export_to_qupath) that
     mirrors the QuPath project model.
""")


if __name__ == "__main__":
    main()