"""Project-level IO for TMEQuant.

Functions
---------
save_project(project, output_dir, overwrite=False)
    Persist a TMEProject to a directory of JSON files.

load_project(project_dir, reload_images=False)
    Restore a TMEProject from a saved snapshot.

export_project_summary(project, output_dir, ...)
    Write human-readable CSV / Excel summaries with per-fiber measurements,
    TACS statistics, local spatial density, and a spatial grid breakdown.

Saved file layout (save_project)
---------------------------------
output_dir/
    project_manifest.json   – name, version, base_path, saved_at
    hierarchy.json          – full TMEObject tree (subclass fields included)
    images.json             – ImageEntry metadata (no pixel arrays)
    orientation_maps.json   – RegionOrientationMap records
    fiber_populations.json  – FiberPopulation records
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base_models import TMEObject, TMEType, ObjectType
from .hierarchy import TMEHierarchy
from .image_entry import ImageEntry
from .project import TMEProject
from .tme_objects.fiber_objects import (
    FiberObject,
    FiberPopulation,
    RegionOrientationMap,
)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class _NumpyJSONEncoder(json.JSONEncoder):
    """Extend JSONEncoder to handle numpy scalars, arrays, Path, and datetime."""

    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Type registry  (tme_type value → concrete class for hierarchy reconstruction)
# ---------------------------------------------------------------------------

_TYPE_REGISTRY: Dict[str, type] = {
    TMEType.FIBER.value: FiberObject,
    TMEType.ORIENTATION_MAP.value: RegionOrientationMap,
    TMEType.IMAGE.value: ImageEntry,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, cls=_NumpyJSONEncoder, indent=2)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _reconstruct_node(d: Dict[str, Any]) -> TMEObject:
    """Recursively rebuild a TMEObject subtree from a to_hierarchy_dict() snapshot."""
    tme_type_val = d.get("tme_type", "unknown")
    cls = _TYPE_REGISTRY.get(tme_type_val)

    if cls is FiberObject:
        node: TMEObject = FiberObject.from_dict(d)
    elif cls is RegionOrientationMap:
        node = RegionOrientationMap.from_dict(d)
    elif cls is ImageEntry:
        node = ImageEntry.from_dict(d)
    else:
        try:
            tme_type = TMEType(tme_type_val)
        except ValueError:
            tme_type = TMEType.UNKNOWN
        try:
            obj_type = ObjectType(d.get("object_type", "unknown"))
        except ValueError:
            obj_type = ObjectType.UNKNOWN
        node = TMEObject(
            object_id=d.get("object_id", ""),
            name=d.get("name", ""),
            tme_type=tme_type,
            object_type=obj_type,
            metadata=d.get("metadata") or {},
            properties=d.get("properties") or {},
        )

    for child_d in d.get("children", []):
        child = _reconstruct_node(child_d)
        node.add_child(child)

    return node


# ---------------------------------------------------------------------------
# save_project
# ---------------------------------------------------------------------------

def save_project(
    project: TMEProject,
    output_dir: Union[str, Path],
    overwrite: bool = False,
    save_arrays: bool = False,
) -> Path:
    """Save a TMEProject to a directory of JSON files.

    Parameters
    ----------
    project : TMEProject
    output_dir : str or Path
    overwrite : bool
        When False (default), raises FileExistsError if a snapshot already
        exists at *output_dir*.
    save_arrays : bool
        When True, write per-pixel orientation and coherency map arrays as
        ``.npy`` sidecar files under ``<output_dir>/arrays/``.  The JSON
        schema is unchanged; sidecars are resolved by naming convention
        ``arrays/{map_id}_orientation.npy`` / ``arrays/{map_id}_coherency.npy``.
        Use ``load_project(..., load_arrays=True)`` to restore them.

    Returns
    -------
    Path
        Absolute path to the output directory.
    """
    out = Path(output_dir).resolve()
    manifest_path = out / "project_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"A project snapshot already exists at '{out}'. "
            "Pass overwrite=True to replace it."
        )
    out.mkdir(parents=True, exist_ok=True)

    # 1. Manifest
    _write_json(manifest_path, {
        "name": project.name,
        "version": "0.2.0",
        "base_path": str(project.base_path) if project.base_path else None,
        "saved_at": datetime.now().isoformat(),
    })

    # 2. Full hierarchy (FiberObject and RegionOrientationMap override
    #    to_hierarchy_dict() to include their subclass-specific fields)
    hierarchy_data: Any = {}
    if project.hierarchy.root is not None:
        hierarchy_data = project.hierarchy.root.to_hierarchy_dict()
    _write_json(out / "hierarchy.json", hierarchy_data)

    # 3. Images – path + metadata only; pixel arrays are never serialised
    _write_json(out / "images.json", {
        img_id: entry.to_dict()
        for img_id, entry in project.images.items()
    })

    # 4. Orientation maps (to_dict() override includes orientation_result summary)
    _write_json(out / "orientation_maps.json", {
        map_id: om.to_dict()
        for map_id, om in project.orientation_maps.items()
    })

    # 5. Fiber populations
    _write_json(out / "fiber_populations.json", {
        pop_id: pop.to_dict()
        for pop_id, pop in project.fiber_populations.items()
    })

    # 6. Optional: orientation map pixel arrays as .npy sidecars
    if save_arrays:
        arrays_dir = out / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        for map_id, om in project.orientation_maps.items():
            result = getattr(om, "orientation_result", None)
            if result is None:
                continue
            arr = getattr(result, "orientation_map", None)
            if arr is not None and isinstance(arr, np.ndarray):
                np.save(arrays_dir / f"{map_id}_orientation.npy", arr)
            coh = getattr(result, "coherency_map", None)
            if coh is not None and isinstance(coh, np.ndarray):
                np.save(arrays_dir / f"{map_id}_coherency.npy", coh)

    return out


# ---------------------------------------------------------------------------
# load_project
# ---------------------------------------------------------------------------

def load_project(
    project_dir: Union[str, Path],
    reload_images: bool = False,
    load_arrays: bool = False,
) -> TMEProject:
    """Reconstruct a TMEProject from a saved snapshot directory.

    Parameters
    ----------
    project_dir : str or Path
        Directory written by save_project().
    reload_images : bool
        When True, attempt to reload pixel data for every ImageEntry whose
        stored path still exists on disk.
    load_arrays : bool
        When True and an ``arrays/`` subdirectory exists, restore orientation
        and coherency map arrays from the ``.npy`` sidecar files written by
        ``save_project(..., save_arrays=True)``.

    Returns
    -------
    TMEProject
    """
    d = Path(project_dir).resolve()
    if not (d / "project_manifest.json").exists():
        raise FileNotFoundError(
            f"No project snapshot found at '{d}'. "
            "Expected a 'project_manifest.json' file."
        )

    manifest = _read_json(d / "project_manifest.json")
    project = TMEProject(
        name=manifest["name"],
        base_path=manifest.get("base_path"),
    )

    # Hierarchy
    hierarchy_raw = _read_json(d / "hierarchy.json")
    if hierarchy_raw:
        root = _reconstruct_node(hierarchy_raw)
        project.hierarchy = TMEHierarchy(root=root)

    # Images
    images_raw = _read_json(d / "images.json")
    for img_id, img_d in images_raw.items():
        entry = ImageEntry.from_dict(img_d)
        project.images[img_id] = entry
        if reload_images and entry.path is not None:
            p = Path(entry.path)
            if p.exists():
                try:
                    entry.load()
                except Exception:
                    pass  # silently skip unreadable files

    # Orientation maps
    maps_raw = _read_json(d / "orientation_maps.json")
    for map_id, om_d in maps_raw.items():
        project.orientation_maps[map_id] = RegionOrientationMap.from_dict(om_d)

    # Fiber populations
    pops_raw = _read_json(d / "fiber_populations.json")
    for pop_id, pop_d in pops_raw.items():
        project.fiber_populations[pop_id] = FiberPopulation.from_dict(pop_d)

    # Optional: restore orientation map arrays from .npy sidecars
    if load_arrays:
        arrays_dir = d / "arrays"
        if arrays_dir.is_dir():
            for map_id, om in project.orientation_maps.items():
                result = getattr(om, "orientation_result", None)
                if result is None:
                    continue
                p_orient = arrays_dir / f"{map_id}_orientation.npy"
                if p_orient.exists():
                    result.orientation_map = np.load(p_orient)
                p_coh = arrays_dir / f"{map_id}_coherency.npy"
                if p_coh.exists():
                    result.coherency_map = np.load(p_coh)

    return project


# ---------------------------------------------------------------------------
# export_project_summary
# ---------------------------------------------------------------------------

def export_project_summary(
    project: TMEProject,
    output_dir: Union[str, Path],
    formats: List[str] = ("csv", "excel"),
    prefix: str = "project_summary",
    local_radius: float = 50.0,
    grid_bin_size: float = 100.0,
) -> Dict[str, str]:
    """Export human-readable CSV / Excel summaries of the project.

    Parameters
    ----------
    project : TMEProject
    output_dir : str or Path
    formats : list of {"csv", "excel"}
    prefix : str
        File-name prefix for all output files.
    local_radius : float
        Radius in µm for per-fiber local density / alignment calculation.
    grid_bin_size : float
        Spatial grid cell size in µm for the SpatialGrid sheet.

    Returns
    -------
    dict mapping table name → output file path.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}
    formats_lower = [f.lower() for f in formats]

    # ── collect all FiberObjects from hierarchy ──────────────────────
    _all = project.hierarchy.get_objects_by_type(TMEType.FIBER)
    all_fibers: List[FiberObject] = [f for f in _all if isinstance(f, FiberObject)]

    # ── ancestor-id walk ─────────────────────────────────────────────
    def _ancestor_ids(obj: TMEObject):
        image_id: Optional[str] = None
        region_id: Optional[str] = None
        current = obj.parent
        while current is not None:
            if image_id is None and current.tme_type == TMEType.IMAGE:
                image_id = current.object_id
            if region_id is None and current.tme_type in (
                TMEType.REGION, TMEType.TUMOR_REGION,
                TMEType.TUMOR, TMEType.ANNOTATION,
            ):
                region_id = current.object_id
            current = current.parent
        return image_id, region_id

    # ── group fibers by parent image ──────────────────────────────────
    # Fiber centerlines are in image-local pixel coordinates, so local
    # density and alignment must be computed within each image separately.
    fiber_image_ids: List[Optional[str]] = []
    fiber_region_ids: List[Optional[str]] = []
    for f in all_fibers:
        img_id, reg_id = _ancestor_ids(f)
        fiber_image_ids.append(img_id)
        fiber_region_ids.append(reg_id)

    # Build a map: image_id → list of (global_fiber_index, center_point)
    from collections import defaultdict
    image_fiber_map: Dict[Optional[str], List] = defaultdict(list)
    for i, f in enumerate(all_fibers):
        cp = f.center_point
        if cp is not None:
            image_fiber_map[fiber_image_ids[i]].append((i, cp))

    # ── compute local fiber density / alignment (per-image KD-tree) ───
    local_counts: List[Optional[int]] = [None] * len(all_fibers)
    local_densities: List[Optional[float]] = [None] * len(all_fibers)
    local_mean_angles: List[Optional[float]] = [None] * len(all_fibers)
    local_alignment_scores: List[Optional[float]] = [None] * len(all_fibers)
    area = math.pi * local_radius ** 2

    for img_id, valid_pairs in image_fiber_map.items():
        if len(valid_pairs) < 2:
            # single fiber in this image — density=0, no alignment
            if len(valid_pairs) == 1:
                gi = valid_pairs[0][0]
                local_counts[gi] = 0
                local_densities[gi] = 0.0
            continue
        idxs, pts = zip(*valid_pairs)
        pts_arr = np.array(pts, dtype=float)[:, :2]
        fiber_angles_arr = np.array(
            [all_fibers[i].angle for i in idxs], dtype=float
        )
        tree = cKDTree(pts_arr)
        neighbor_lists = tree.query_ball_tree(tree, r=local_radius)

        for k, global_idx in enumerate(idxs):
            neighbors = [n for n in neighbor_lists[k] if n != k]
            count = len(neighbors)
            local_counts[global_idx] = count
            local_densities[global_idx] = (count / area) * 1e6  # per mm²
            if neighbors:
                nb_angles_rad = np.radians(fiber_angles_arr[neighbors])
                mc = float(np.mean(np.cos(2 * nb_angles_rad)))
                ms = float(np.mean(np.sin(2 * nb_angles_rad)))
                local_mean_angles[global_idx] = float(
                    np.degrees(np.arctan2(ms, mc) / 2) % 180
                )
                local_alignment_scores[global_idx] = float(
                    np.sqrt(mc ** 2 + ms ** 2)
                )
            else:
                local_mean_angles[global_idx] = None
                local_alignment_scores[global_idx] = None

    # ── per-fiber rows ────────────────────────────────────────────────
    fiber_rows: List[Dict[str, Any]] = []
    for i, f in enumerate(all_fibers):
        row = f.to_dict()
        row["image_id"] = fiber_image_ids[i]
        row["region_id"] = fiber_region_ids[i]
        row["local_fiber_count"] = local_counts[i]
        row["local_fiber_density_per_mm2"] = local_densities[i]
        row["local_mean_angle"] = local_mean_angles[i]
        row["local_alignment_score"] = local_alignment_scores[i]
        fiber_rows.append(row)

    df_fibers = pd.DataFrame(fiber_rows)

    # ── spatial grid (per-image) ──────────────────────────────────────
    grid_rows: List[Dict[str, Any]] = []
    for img_id, valid_pairs in image_fiber_map.items():
        if not valid_pairs:
            continue
        _, pts_for_grid = zip(*valid_pairs)
        pts_g = np.array(pts_for_grid, dtype=float)[:, :2]
        xs, ys = pts_g[:, 0], pts_g[:, 1]
        x_bins = np.arange(xs.min(), xs.max() + grid_bin_size, grid_bin_size)
        y_bins = np.arange(ys.min(), ys.max() + grid_bin_size, grid_bin_size)
        bin_area = grid_bin_size ** 2

        for xi in range(len(x_bins) - 1):
            x0, x1 = x_bins[xi], x_bins[xi + 1]
            for yi in range(len(y_bins) - 1):
                y0, y1 = y_bins[yi], y_bins[yi + 1]
                mask = (
                    (pts_g[:, 0] >= x0) & (pts_g[:, 0] < x1) &
                    (pts_g[:, 1] >= y0) & (pts_g[:, 1] < y1)
                )
                local_idxs = np.where(mask)[0]
                if len(local_idxs) == 0:
                    continue
                global_idxs = [valid_pairs[k][0] for k in local_idxs]
                bin_fibers = [all_fibers[gi] for gi in global_idxs]
                angles_rad = np.radians([bf.angle for bf in bin_fibers])
                mc = float(np.mean(np.cos(2 * angles_rad)))
                ms = float(np.mean(np.sin(2 * angles_rad)))
                tacs_ctr = Counter(
                    bf.tacs_type for bf in bin_fibers if bf.tacs_type
                )
                dominant = tacs_ctr.most_common(1)[0][0] if tacs_ctr else None
                grid_rows.append({
                    "grid_x": float((x0 + x1) / 2),
                    "grid_y": float((y0 + y1) / 2),
                    "fiber_count": len(local_idxs),
                    "fiber_density_per_mm2": (len(local_idxs) / bin_area) * 1e6,
                    "mean_angle": float(
                        np.degrees(np.arctan2(ms, mc) / 2) % 180
                    ),
                    "alignment_score": float(np.sqrt(mc ** 2 + ms ** 2)),
                    "tacs1_count": tacs_ctr.get("TACS-1", 0),
                    "tacs2_count": tacs_ctr.get("TACS-2", 0),
                    "tacs3_count": tacs_ctr.get("TACS-3", 0),
                    "dominant_tacs_type": dominant,
                    "image_id": img_id,
                })

    df_grid = pd.DataFrame(grid_rows)

    # ── project-wide TACS summary ─────────────────────────────────────
    all_tacs = [f.tacs_type for f in all_fibers]
    in_zone = [t for t in all_tacs if t is not None]
    n_zone = len(in_zone)
    tacs_ctr_proj = Counter(in_zone)
    tacs1 = tacs_ctr_proj.get("TACS-1", 0)
    tacs2 = tacs_ctr_proj.get("TACS-2", 0)
    tacs3 = tacs_ctr_proj.get("TACS-3", 0)
    dominant_proj = tacs_ctr_proj.most_common(1)[0][0] if in_zone else None
    angles_for_stats = [
        f.relative_angle_to_boundary_tangent for f in all_fibers
        if f.relative_angle_to_boundary_tangent is not None
    ]
    dists_for_stats = [
        f.nearest_boundary_distance for f in all_fibers
        if f.nearest_boundary_distance is not None
    ]

    tacs_summary_row: Dict[str, Any] = {
        "total_fibers": len(all_fibers),
        "fibers_in_tacs_zone": n_zone,
        "tacs1_count": tacs1,
        "tacs2_count": tacs2,
        "tacs3_count": tacs3,
        "tacs1_ratio": tacs1 / n_zone if n_zone else None,
        "tacs2_ratio": tacs2 / n_zone if n_zone else None,
        "tacs3_ratio": tacs3 / n_zone if n_zone else None,
        "dominant_tacs_type": dominant_proj,
        "mean_angle_to_boundary_tangent": (
            float(np.mean(angles_for_stats)) if angles_for_stats else None
        ),
        "std_angle_to_boundary_tangent": (
            float(np.std(angles_for_stats)) if angles_for_stats else None
        ),
        "mean_nearest_boundary_distance": (
            float(np.mean(dists_for_stats)) if dists_for_stats else None
        ),
        "parallel_ratio": (
            sum(1 for a in angles_for_stats if a < 30) / len(angles_for_stats)
            if angles_for_stats else None
        ),
        "perpendicular_ratio": (
            sum(1 for a in angles_for_stats if a >= 60) / len(angles_for_stats)
            if angles_for_stats else None
        ),
        "fibers_in_tumor_boundary": sum(
            1 for f in all_fibers if f.in_tumor_boundary
        ),
        "fibers_in_tumor_core": sum(
            1 for f in all_fibers if f.in_tumor_core
        ),
    }
    df_tacs = pd.DataFrame([tacs_summary_row])

    # ── project overview ──────────────────────────────────────────────
    df_project = pd.DataFrame([{
        "project_name": project.name,
        "base_path": str(project.base_path) if project.base_path else None,
        "n_images": len(project.images),
        "n_fibers": len(all_fibers),
        "n_orientation_maps": len(project.orientation_maps),
        "n_fiber_populations": len(project.fiber_populations),
    }])

    # ── images table ──────────────────────────────────────────────────
    image_rows: List[Dict[str, Any]] = []
    for img_id, entry in project.images.items():
        n_fiber_ch = len(entry.filter_by_type(TMEType.FIBER))
        n_cell_ch = len(entry.filter_by_type(TMEType.CELL))
        image_rows.append({
            "image_id": img_id,
            "name": entry.name,
            "path": str(entry.path) if entry.path else None,
            "shape": str(entry.shape) if entry.shape else None,
            "pixel_size": str(entry.pixel_size) if entry.pixel_size else None,
            "magnification": entry.magnification,
            "modality": entry.modality,
            "n_fiber_children": n_fiber_ch,
            "n_cell_children": n_cell_ch,
        })
    df_images = pd.DataFrame(image_rows) if image_rows else pd.DataFrame()

    # ── fiber populations table ───────────────────────────────────────
    pop_rows: List[Dict[str, Any]] = []
    for pop_id, pop in project.fiber_populations.items():
        dist = pop.tacs_type_distribution or {}
        n_typed = sum(dist.values()) if dist else 0
        pop_rows.append({
            "population_id": pop_id,
            "region_id": pop.region_id,
            "region_type": pop.region_type,
            "count": pop.count,
            "mean_length": pop.mean_length,
            "mean_width": pop.mean_width,
            "mean_straightness": pop.mean_straightness,
            "mean_orientation": pop.mean_orientation,
            "alignment_score": pop.alignment_score,
            "dominant_tacs_type": pop.tacs_type,
            "tacs_score": pop.tacs_score,
            "tacs1_count": dist.get("TACS-1", 0),
            "tacs2_count": dist.get("TACS-2", 0),
            "tacs3_count": dist.get("TACS-3", 0),
            "tacs1_ratio": dist.get("TACS-1", 0) / n_typed if n_typed else None,
            "tacs2_ratio": dist.get("TACS-2", 0) / n_typed if n_typed else None,
            "tacs3_ratio": dist.get("TACS-3", 0) / n_typed if n_typed else None,
        })
    df_pops = pd.DataFrame(pop_rows) if pop_rows else pd.DataFrame()

    # ── orientation maps table ────────────────────────────────────────
    map_rows: List[Dict[str, Any]] = []
    for map_id, om in project.orientation_maps.items():
        map_rows.append({
            "map_id": map_id,
            "object_id": om.object_id,
            "name": om.name,
            "region_type": om.region_type,
            "mean_orientation": om.get_dominant_orientation(),
            "alignment_score": om.get_alignment_score(),
        })
    df_maps = pd.DataFrame(map_rows) if map_rows else pd.DataFrame()

    # ── assemble and write ────────────────────────────────────────────
    tables: Dict[str, pd.DataFrame] = {
        "project": df_project,
        "images": df_images,
        "fiber_populations": df_pops,
        "orientation_maps": df_maps,
        "tacs_summary": df_tacs,
        "fibers": df_fibers,
        "spatial_grid": df_grid,
    }

    if "csv" in formats_lower:
        for tname, df in tables.items():
            if df.empty:
                continue
            p = out / f"{prefix}_{tname}.csv"
            df.to_csv(p, index=False)
            exported[f"csv_{tname}"] = str(p)

    if "excel" in formats_lower:
        xlsx_path = out / f"{prefix}.xlsx"
        sheet_order: Dict[str, pd.DataFrame] = {
            "Project": df_project,
            "Images": df_images,
            "FiberPopulations": df_pops,
            "OrientationMaps": df_maps,
            "TACS_Summary": df_tacs,
            "Fibers": df_fibers,
            "SpatialGrid": df_grid,
        }
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in sheet_order.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        exported["excel"] = str(xlsx_path)

    return exported


__all__ = ["save_project", "load_project", "export_project_summary"]
