"""Compare Python BDcreationHE / HE2 intermediates to MATLAB dumps.

Usage (from tme-quant root, after dump_bdc_he2.m / dump_bdc_he.m)::

    .venv/bin/python tests/matlab_parity/analyze_he_dumps.py
    .venv/bin/python tests/matlab_parity/analyze_he_dumps.py --preproc
    .venv/bin/python tests/matlab_parity/analyze_he_dumps.py he2_test1 he_test2

Does not edit ``analyze_dumps.py`` (registration). Writes a local
``dumps/he_analysis_summary.json`` (gitignored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from skimage import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pycurvelets._he_bdc_annotation import (  # noqa: E402
    annotate_he,
    annotate_he2,
    matlab_fspecial_disk,
    matlab_histeq,
    matlab_im2bw,
    matlab_padarray,
)
from pycurvelets._he_bdc_reg1 import DEFAULT_KMEANS_SEED, matlab_strel_disk  # noqa: E402

DUMPS = Path(__file__).resolve().parent / "dumps"
FIXTURE = ROOT / "tests" / "test_for_shg_he_registration_BDcreation"
GOLDEN_DIR = FIXTURE / "SHG" / "CA_Boundary"

HE2_CASES = (
    ("he2_test1", FIXTURE / "HE" / "HE_registered_test1", "patient_001.tif", 1.5),
    ("he2_test2", FIXTURE / "HE" / "HE_registered_test2", "patient_001.tif", 2.0),
    ("he2_test3", FIXTURE / "HE" / "HE_registered_test3", "patient_001.tif", 3.0),
)
HE_CASES = (
    ("he_test1", FIXTURE / "HE" / "HE_registered_test1", "patient_001.tif", 1.5),
    ("he_test2", FIXTURE / "HE" / "HE_registered_test2", "patient_001.tif", 2.0),
    ("he_test3", FIXTURE / "HE" / "HE_registered_test3", "patient_001.tif", 3.0),
)

HE2_STEPS = (
    ("he_adjusted", "he_adjusted", "float"),
    ("BW_nuclei", "BW_nuclei", "bool"),
    ("maskednucleiImage", "maskednucleiImage", "float"),
    ("BW_collagen1", "BW_collagen1", "bool"),
    ("BW_nobackground", "BW_nobackground", "bool"),
    ("epith_cell_BW", "epith_cell_BW", "bool"),
    ("BWx", "BWx", "bool"),
    ("mask_image1", "mask_image1", "bool"),
    ("B", "B", "float"),
    ("mask_temp", "mask_temp", "float"),
    ("BDmask", "BDmask", "bool"),
)
HE_STEPS = (
    ("S", "S", "uint8"),
    ("k3", "k3", "uint8"),
    ("pixel_labels", "pixel_labels", "labels"),
    ("epith_cell_BW", "epith_cell_BW", "bool"),
    ("BWx", "BWx", "bool"),
    ("mask_image", "mask_image", "bool"),
    ("BDmask", "BDmask", "uint8"),
)


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).squeeze().astype(bool)


def _squeeze(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).squeeze()


def _report(name: str, kind: str, py: np.ndarray, ml: np.ndarray) -> dict[str, Any]:
    py_a = np.asarray(py)
    ml_a = np.asarray(ml)
    if kind == "bool":
        p = _as_bool(py_a)
        m = _as_bool(ml_a)
        n_diff = int(np.count_nonzero(p != m))
        return {"name": name, "kind": kind, "n_diff": n_diff, "shape_py": list(p.shape), "shape_ml": list(m.shape)}
    if kind == "uint8":
        p = np.asarray(py_a)
        m = np.asarray(ml_a)
        n_diff = int(np.count_nonzero(p != m))
        return {"name": name, "kind": kind, "n_diff": n_diff, "max_abs": float(np.max(np.abs(p.astype(np.int16) - m.astype(np.int16))))}
    if kind == "labels":
        # MATLAB labels are 1-based; Python debug stores 0-based.
        p = np.asarray(py_a).squeeze().astype(np.int64) + 1
        m = np.asarray(ml_a).squeeze().astype(np.int64)
        n_diff = int(np.count_nonzero(p != m))
        return {"name": name, "kind": kind, "n_diff": n_diff}
    p = np.asarray(py_a, dtype=np.float64)
    m = np.asarray(ml_a, dtype=np.float64)
    max_abs = float(np.max(np.abs(p - m))) if p.shape == m.shape else float("inf")
    return {
        "name": name,
        "kind": kind,
        "max_abs": max_abs,
        "shape_py": list(p.shape),
        "shape_ml": list(m.shape),
    }


def _load_images(case_dir: Path) -> dict[str, Any]:
    mat = sio.loadmat(str(case_dir / "images.mat"), squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def analyze_primitives() -> list[dict[str, Any]]:
    path = DUMPS / "annotation_primitives.mat"
    if not path.is_file():
        return [{"name": "annotation_primitives.mat", "missing": True}]
    m = sio.loadmat(str(path), squeeze_me=True)
    rows: list[dict[str, Any]] = []
    for r in (1, 2, 3, 4, 5, 7, 11, 14, 21):
        key = f"disk_r{r}"
        if key not in m:
            continue
        py = matlab_fspecial_disk(float(r))
        rows.append(_report(key, "float", py, m[key]))
    if "disk_r10p5" in m:
        rows.append(_report("disk_r10p5", "float", matlab_fspecial_disk(10.5), m["disk_r10p5"]))
    for r in (1, 2, 3, 4, 5, 7, 11):
        key = f"strel_r{r}"
        if key not in m:
            continue
        py = matlab_strel_disk(r)
        rows.append(_report(key, "bool", py, m[key]))
    if "pad_src" in m:
        py = matlab_padarray(m["pad_src"], 2, mode="symmetric")
        rows.append(_report("pad_sym", "uint8", py, m["pad_sym"]))
    if "histeq_rand_in" in m:
        py = matlab_histeq(m["histeq_rand_in"])
        rows.append(_report("histeq_rand_out", "uint8", py, m["histeq_rand_out"]))
    if "histeq_u8_in" in m:
        py = matlab_histeq(m["histeq_u8_in"])
        rows.append(_report("histeq_u8_out", "uint8", py, m["histeq_u8_out"]))
    if "histeq_ramp" in m:
        py = matlab_histeq(np.arange(256, dtype=np.uint8).reshape((16, 16), order="F"))
        rows.append(_report("histeq_ramp", "uint8", py, m["histeq_ramp"]))
    if "histeq_dbl_in" in m:
        py = matlab_histeq(np.asarray(m["histeq_dbl_in"], dtype=np.float64))
        rows.append(_report("histeq_dbl_out", "float", py, m["histeq_dbl_out"]))
    if "im2bw_in" in m:
        py = matlab_im2bw(np.asarray(m["im2bw_in"], dtype=np.float64), 0.5)
        rows.append(_report("im2bw_out", "bool", py, m["im2bw_out"]))
    return rows


def analyze_he2(case_id: str, he_dir: Path, fname: str, ppm: float, *, preproc: bool) -> dict[str, Any]:
    dump = DUMPS / case_id
    images = dump / "images.mat"
    he_path = he_dir / fname
    if not images.is_file() or not he_path.is_file():
        return {"case": case_id, "missing": True}
    ml = _load_images(dump)
    he = io.imread(str(he_path))
    _mask, dbg = annotate_he2(he, ppm)
    out: dict[str, Any] = {"case": case_id, "ppm": ppm, "steps": []}
    keys = HE2_STEPS if preproc else (("BDmask", "BDmask", "bool"),)
    for py_key, ml_key, kind in keys:
        if ml_key not in ml:
            out["steps"].append({"name": ml_key, "missing_ml": True})
            continue
        py_val = dbg[py_key]
        if kind == "bool" and py_key == "mask_image1":
            py_val = np.asarray(dbg["mask_image1"]) > 0
        out["steps"].append(_report(ml_key, kind, py_val, ml[ml_key]))
    golden = GOLDEN_DIR / f"BDcreationHE_{case_id.replace('he2_', '')}results_mask for patient_001.tif.tif"
    if golden.is_file():
        g = io.imread(str(golden)) > 0
        out["vs_golden_n_diff"] = int(np.count_nonzero(_as_bool(dbg["BDmask"]) != g))
    return out


def analyze_he(case_id: str, he_dir: Path, fname: str, ppm: float, *, preproc: bool) -> dict[str, Any]:
    dump = DUMPS / case_id
    images = dump / "images.mat"
    he_path = he_dir / fname
    if not images.is_file() or not he_path.is_file():
        return {"case": case_id, "missing": True}
    ml = _load_images(dump)
    he = io.imread(str(he_path))
    seed = int(ml["kmeansSeed"]) if "kmeansSeed" in ml else DEFAULT_KMEANS_SEED
    _mask, dbg = annotate_he(he, ppm, kmeans_seed=seed)
    out: dict[str, Any] = {"case": case_id, "ppm": ppm, "kmeans_seed": seed, "steps": []}
    if "blue_cluster_num" in ml:
        ml_blue = int(np.asarray(ml["blue_cluster_num"]).squeeze())
        py_blue = int(dbg["blue_cluster_num"][0]) + 1
        out["blue_cluster_num"] = {"python_1based": py_blue, "matlab": ml_blue, "match": py_blue == ml_blue}
    keys = HE_STEPS if preproc else (("BDmask", "BDmask", "uint8"),)
    for py_key, ml_key, kind in keys:
        if ml_key not in ml:
            out["steps"].append({"name": ml_key, "missing_ml": True})
            continue
        out["steps"].append(_report(ml_key, kind, dbg[py_key], ml[ml_key]))
    return out


def _print_case(row: dict[str, Any]) -> None:
    if row.get("missing"):
        print(f"  {row.get('case', '?')}: MISSING dump or fixture")
        return
    print(f"  {row.get('case')} ppm={row.get('ppm')}")
    if "blue_cluster_num" in row:
        b = row["blue_cluster_num"]
        print(f"    blue_cluster_num py={b['python_1based']} ml={b['matlab']} match={b['match']}")
    for step in row.get("steps", []):
        if step.get("missing_ml"):
            print(f"    {step['name']}: missing in MATLAB dump")
            continue
        extra = []
        if "n_diff" in step:
            extra.append(f"n_diff={step['n_diff']}")
        if "max_abs" in step:
            extra.append(f"max_abs={step['max_abs']:.3e}")
        print(f"    {step['name']}: " + " ".join(extra))
        if step.get("n_diff", 0) or (step.get("max_abs") or 0) > 1e-12:
            print(f"      FIRST DIVERGENT: {step['name']}")
    if "vs_golden_n_diff" in row:
        print(f"    vs committed golden n_diff={row['vs_golden_n_diff']}")


def main(argv: list[str]) -> int:
    preproc = "--preproc" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    summary: dict[str, Any] = {"primitives": analyze_primitives(), "he2": [], "he": []}
    print("primitives")
    for row in summary["primitives"]:
        if row.get("missing"):
            print("  annotation_primitives.mat missing")
            continue
        extra = []
        if "n_diff" in row:
            extra.append(f"n_diff={row['n_diff']}")
        if "max_abs" in row:
            extra.append(f"max_abs={row['max_abs']:.3e}")
        print(f"  {row['name']}: " + " ".join(extra))
        if row.get("n_diff", 0) or (row.get("max_abs") or 0) > 1e-12:
            print(f"    FIRST DIVERGENT: {row['name']}")
    print("HE2")
    for case_id, he_dir, fname, ppm in HE2_CASES:
        if wanted and case_id not in wanted:
            continue
        row = analyze_he2(case_id, he_dir, fname, ppm, preproc=preproc)
        summary["he2"].append(row)
        _print_case(row)
    print("HE")
    for case_id, he_dir, fname, ppm in HE_CASES:
        if wanted and case_id not in wanted:
            continue
        row = analyze_he(case_id, he_dir, fname, ppm, preproc=preproc)
        summary["he"].append(row)
        _print_case(row)
    out = DUMPS / "he_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
