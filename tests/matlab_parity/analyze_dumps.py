"""Step-0 analysis of MATLAB ``BDcreation_reg2`` dumps vs the Python port.

Usage (from tme-quant root, after ``dump_bdc_reg2.m`` has run)::

    .venv/bin/python tests/matlab_parity/analyze_dumps.py            # all cases
    .venv/bin/python tests/matlab_parity/analyze_dumps.py test1 test4  # subset
    .venv/bin/python tests/matlab_parity/analyze_dumps.py --no-engine  # skip ITK runs
    .venv/bin/python tests/matlab_parity/analyze_dumps.py --preproc    # step-by-step
                                                                         # mask parity

For every case this reports

1. **Preprocessing parity** - Dice between Python's ``he_moving`` collagen mask
   and MATLAB's dumped ``HEmoving``.
2. **Engine parity** - the ITK-v3 engine (``registration_method="matlab"``)
   run on MATLAB's *exact* ``HEmoving``/``fixedSHG`` doubles; both stage
   transforms are compared against ``tformSimilarity.T`` / ``tform.T`` in
   parameter space. Where ``optimization_trace.txt`` exists, the per-iteration
   metric values are compared too (first divergent iteration is reported).
3. **Same-basin check** - the engine run on *Python's* mask, compared to
   MATLAB's transform (tells whether small mask differences change the
   optimum).
4. **Ground truth** (patient_02 only) - GT affine recovered by SIFT+RANSAC
   between ``HE/patient_02_roiN.tif`` and ``patient_02_HE_original-roiN.tif``,
   mapped onto the working grid, with angle/scale/translation errors of the
   MATLAB and Python transforms, and Mattes MI (v3 metric) evaluated at the
   MATLAB, Python and GT transforms.
5. **Determinism** - ``test2`` vs ``test2_rerun`` MATLAB transforms.

Writes ``tests/matlab_parity/dumps/analysis_summary.json`` and, for patient_02,
``dumps/gt_affine_<case>.json``.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from skimage import io
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pycurvelets._he_bdc_common import (  # noqa: E402
    adjust_rgb_mean_std,
    decorrelation_stretch,
    matlab_rgb2gray,
    prepare_registration_pair,
)
from pycurvelets._itk_v3_matlab_engine import (  # noqa: E402
    OnePlusOneConfig,
    forward_0based_to_moving_to_fixed_1based,
    has_itk,
    matlab_T_to_A,
    matlab_imregtform_v3,
    moving_to_fixed_1based_to_forward_0based,
    register_bdcreation_reg2_matlab,
)
from pycurvelets.SHG_HE_registration import _build_he_moving  # noqa: E402

DUMPS = Path(__file__).resolve().parent / "dumps"
FIXTURE = ROOT / "tests" / "test_for_shg_he_registration_BDcreation"
P02 = FIXTURE / "new_test_datasets_tests4-5-6-7"

CASES = [
    ("test1", FIXTURE / "HE", "patient_001.tif", FIXTURE / "SHG", 1.5, None),
    ("test2", FIXTURE / "HE", "patient_001.tif", FIXTURE / "SHG", 2.0, None),
    ("test3", FIXTURE / "HE", "patient_001.tif", FIXTURE / "SHG", 3.0, None),
    ("test4", P02 / "HE", "patient_02_roi2.tif", P02 / "SHG", 2.6, "roi2"),
    ("test5", P02 / "HE", "patient_02_roi4.tif", P02 / "SHG", 1.5, "roi4"),
    ("test6", P02 / "HE", "patient_02_roi4.tif", P02 / "SHG", 2.6, "roi4"),
    ("test7", P02 / "HE", "patient_02_roi5.tif", P02 / "SHG", 2.6, "roi5"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    denom = a_b.sum() + b_b.sum()
    return float(2.0 * inter / denom) if denom else float("nan")


def _load_matlab_inputs(dump: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Exact doubles from images.mat if present, else the 8-bit tif dumps."""
    mat = dump / "images.mat"
    if mat.is_file():
        m = sio.loadmat(str(mat))
        he = np.ascontiguousarray(m["HEmoving"].astype(np.float64))
        shg = np.ascontiguousarray(m["fixedSHG"].astype(np.float64))
        return he, shg, "images.mat"
    he = io.imread(str(dump / "HEmoving.tif")).astype(np.float64)
    shg = io.imread(str(dump / "fixedSHG.tif")).astype(np.float64)
    if he.max() > 1:
        he /= he.max()
    if shg.max() > 1:
        shg /= 255.0
    return he, shg, "tif (8-bit quantised)"


def _python_pipeline(
    he_path: Path, shg_path: Path, ppm: float
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int], tuple[int, int]]:
    """Python preprocessing exactly as ``_shg_he_registration_core`` (hsv ECM)."""
    he = io.imread(str(he_path)).astype(np.float64) / 255.0
    shg = io.imread(str(shg_path)).astype(np.float64) / 255.0
    if shg.ndim == 3:
        shg = matlab_rgb2gray(shg)
    he_scaled, fixed_shg, pix = prepare_registration_pair(he, shg, float(ppm))
    he_adj = adjust_rgb_mean_std(he_scaled)
    he_dec = decorrelation_stretch(he_adj, tol=0.01)
    fixed = fixed_shg.astype(np.float64)
    if fixed.ndim == 3:
        fixed = matlab_rgb2gray(fixed)
    he_moving, _mode, _extras = _build_he_moving(he_scaled, he_adj, he_dec, pix, "hsv", 0)
    return he_moving.astype(np.float64), fixed, pix, he.shape[:2], shg.shape[:2]


def _decomp_A(A: np.ndarray) -> dict[str, float]:
    """Decompose a 3x3 moving->fixed matrix (column form) into scale/angle/shear/t."""
    A = np.asarray(A, dtype=np.float64)
    M = A[:2, :2]
    sx = float(np.hypot(M[0, 0], M[1, 0]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # QR-style: M = R(angle) @ [[sx, shear],[0, sy]]
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    R_inv = np.array([[c, s], [-s, c]])
    U = R_inv @ M
    return {
        "scale_x": sx,
        "scale_y": float(U[1, 1]),
        "shear": float(U[0, 1] / sx) if sx else float("nan"),
        "angle_deg": angle,
        "tx": float(A[0, 2]),
        "ty": float(A[1, 2]),
    }


def _param_error(A: np.ndarray, B: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    """Transform error between two moving->fixed 3x3 matrices.

    Reports matrix/translation max-abs differences, angle & scale differences,
    and the mean displacement (px) of the image corners + centre.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    da = _decomp_A(A)
    db = _decomp_A(B)
    h, w = shape
    pts = np.array(
        [[1, 1, 1], [w, 1, 1], [1, h, 1], [w, h, 1], [(w + 1) / 2, (h + 1) / 2, 1]],
        dtype=np.float64,
    ).T
    disp = (A @ pts - B @ pts)[:2]
    return {
        "max_abs_dM": float(np.abs(A[:2, :2] - B[:2, :2]).max()),
        "max_abs_dt": float(np.abs(A[:2, 2] - B[:2, 2]).max()),
        "d_angle_deg": float(da["angle_deg"] - db["angle_deg"]),
        "d_scale": float(da["scale_x"] - db["scale_x"]),
        "mean_corner_disp_px": float(np.linalg.norm(disp, axis=0).mean()),
    }


def _parse_matlab_trace(path: Path) -> dict[str, dict[int, list[float]]]:
    stages: dict[str, dict[int, list[float]]] = {}
    cur = None
    level = None
    for line in path.read_text().splitlines():
        m = re.match(r"### STAGE (\w+)", line)
        if m:
            cur = m.group(1)
            stages[cur] = {}
            continue
        m = re.match(r"Pyramid Level: (\d+)", line)
        if m and cur is not None:
            level = int(m.group(1))
            stages[cur][level] = []
            continue
        m = re.match(r"^\s*(\d+)\s+(-?[0-9.eE+-]+)\s*$", line)
        if m and cur is not None and level is not None:
            stages[cur][level].append(float(m.group(2)))
    return stages


def _compare_trace(py_trace: list[dict], level_starts: list[int], ml: dict[int, list[float]]) -> dict:
    py = [-x["metric"] for x in py_trace]
    bounds = level_starts + [len(py)]
    out: dict[str, Any] = {}
    for lv in range(len(bounds) - 1):
        seg = py[bounds[lv] : bounds[lv + 1]]
        mlseg = ml.get(lv + 1, [])
        n = min(len(seg), len(mlseg))
        # MATLAB prints 8 decimals
        first_div = next((i for i in range(n) if abs(seg[i] - mlseg[i]) > 6e-9), None)
        out[f"level{lv + 1}"] = {
            "n_py": len(seg),
            "n_matlab": len(mlseg),
            "first_divergent_iteration": first_div,
            "py_last": seg[-1] if seg else None,
            "matlab_last": mlseg[-1] if mlseg else None,
        }
    return out


def _mi_at(he: np.ndarray, shg: np.ndarray, A_m2f_1based: np.ndarray) -> float:
    """Mattes MI (v3 metric, MATLAB config, finest pyramid level) at a transform."""
    import itk

    from pycurvelets._itk_v3_matlab_engine import _build_transform, _to_itk_image

    IT = itk.Image[itk.D, 2]
    fixed = _to_itk_image(shg)
    moving = _to_itk_image(he)
    pf = itk.MultiResolutionPyramidImageFilter[IT, IT].New(Input=fixed, NumberOfLevels=3)
    pm = itk.MultiResolutionPyramidImageFilter[IT, IT].New(Input=moving, NumberOfLevels=3)
    pf.UpdateLargestPossibleRegion()
    pm.UpdateLargestPossibleRegion()
    f2m = np.linalg.inv(np.asarray(A_m2f_1based, dtype=np.float64))
    tr = _build_transform("affine", f2m[:2, :2], f2m[:2, 2], (0.0, 0.0))
    met = itk.MattesMutualInformationImageToImageMetric[IT, IT].New()
    met.SetFixedImage(pf.GetOutput(2))
    met.SetMovingImage(pm.GetOutput(2))
    met.SetTransform(tr)
    met.SetInterpolator(itk.LinearInterpolateImageFunction[IT, itk.D].New())
    met.SetFixedImageRegion(pf.GetOutput(2).GetBufferedRegion())
    met.SetNumberOfHistogramBins(50)
    met.UseAllPixelsOn()
    met.Initialize()
    return -float(met.GetValue(tr.GetParameters()))


def _recover_gt_affine(he_path: Path, gt_path: Path) -> dict | None:
    """Same-modality SIFT+RANSAC HE(input grid) -> HE_original (SHG grid), 0-based."""
    if not gt_path.is_file():
        return None
    src = io.imread(str(he_path))
    dst = io.imread(str(gt_path))

    def gray(a: np.ndarray) -> np.ndarray:
        if a.ndim == 3:
            return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
        return a.astype(np.float32)

    src_g, dst_g = gray(src), gray(dst)
    sift = SIFT()
    try:
        sift.detect_and_extract(src_g)
        kp1, desc1 = sift.keypoints, sift.descriptors
        sift.detect_and_extract(dst_g)
        kp2, desc2 = sift.keypoints, sift.descriptors
    except Exception as exc:  # pragma: no cover
        return {"error": f"SIFT failed: {exc}"}

    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        return {"error": "too few SIFT descriptors"}
    matches = match_descriptors(desc1, desc2, max_ratio=0.8, cross_check=True)
    if matches.shape[0] < 6:
        return {"error": f"too few matches ({matches.shape[0]})"}
    src_pts = kp1[matches[:, 0]][:, ::-1]
    dst_pts = kp2[matches[:, 1]][:, ::-1]
    model, inliers = ransac(
        (src_pts, dst_pts), AffineTransform, min_samples=3, residual_threshold=3.0,
        max_trials=5000, rng=0,
    )
    if model is None:
        return {"error": "RANSAC failed"}
    P = np.asarray(model.params, dtype=np.float64)
    n_in = int(inliers.sum()) if inliers is not None else 0
    resid = model.residuals(src_pts[inliers], dst_pts[inliers]) if n_in else np.array([np.nan])
    return {
        "forward_2x3_input_grid_0based": P[:2, :].tolist(),
        "n_matches": int(matches.shape[0]),
        "n_inliers": n_in,
        "inlier_rms_px": float(np.sqrt(np.mean(resid**2))),
        "src_shape": list(src.shape[:2]),
        "dst_shape": list(dst.shape[:2]),
    }


def _gt_to_working_grid(
    gt_fwd_0based: np.ndarray,
    he_in_shape: tuple[int, int],
    shg_in_shape: tuple[int, int],
    work_shape: tuple[int, int],
) -> np.ndarray:
    """
    Map a GT forward affine (HE input grid -> HE_original/SHG grid, 0-based
    pixel centres) onto the working grid where both images are resized to
    ``work_shape`` (imresize-style mapping ``x' = (x + 0.5) * s - 0.5``).
    Returns the 0-based forward 2x3 on the working grid.
    """
    G = np.eye(3)
    G[:2, :] = np.asarray(gt_fwd_0based, dtype=np.float64)

    def scale_map(src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
        sy = dst_shape[0] / src_shape[0]
        sx = dst_shape[1] / src_shape[1]
        S = np.eye(3)
        S[0, 0] = sx
        S[0, 2] = 0.5 * sx - 0.5
        S[1, 1] = sy
        S[1, 2] = 0.5 * sy - 0.5
        return S

    S_he = scale_map(he_in_shape, work_shape)
    S_shg = scale_map(shg_in_shape, work_shape)
    F = S_shg @ G @ np.linalg.inv(S_he)
    return F[:2, :]


# ---------------------------------------------------------------------------
# preprocessing step-by-step parity (needs intermediates.mat)
# ---------------------------------------------------------------------------


def compare_preprocessing(case_id: str, he_dir: Path, he_file: str, shg_dir: Path, ppm: float) -> dict[str, Any]:
    """Compare every MATLAB preprocessing intermediate with the Python equivalent.

    Returns ``{step: {"max_abs_diff" | "dice" | "n_diff_px": ...}}`` in pipeline
    order so the *first* divergent step is obvious.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage import morphology

    from pycurvelets._he_bdc_common import (
        disk_se,
        gaussian_filter_matlab_like,
        make_collagen_mask,
        make_nuclei_mask,
        matlab_graythresh,
        matlab_rgb2hsv,
        remove_small_components,
    )

    dump = DUMPS / case_id
    mat = dump / "intermediates.mat"
    if not mat.is_file():
        return {"status": "missing intermediates.mat"}
    m = sio.loadmat(str(mat))

    he = io.imread(str(he_dir / he_file)).astype(np.float64) / 255.0
    shg = io.imread(str(shg_dir / he_file)).astype(np.float64) / 255.0
    if shg.ndim == 3:
        shg = matlab_rgb2gray(shg)
    he_scaled, _fixed, pix = prepare_registration_pair(he, shg, float(ppm))
    he_adj = adjust_rgb_mean_std(he_scaled)

    out: dict[str, Any] = {}

    def cmp_float(name: str, py: np.ndarray, ml: np.ndarray) -> None:
        if py.shape != ml.shape:
            out[name] = {"shape_py": list(py.shape), "shape_ml": list(ml.shape)}
        else:
            out[name] = {"max_abs_diff": float(np.abs(py - ml).max())}

    def cmp_mask(name: str, py: np.ndarray, ml: np.ndarray) -> None:
        py_b = np.asarray(py).astype(bool)
        ml_b = np.asarray(ml).astype(bool)
        if py_b.shape != ml_b.shape:
            out[name] = {"shape_py": list(py_b.shape), "shape_ml": list(ml_b.shape)}
        else:
            out[name] = {"dice": _dice(py_b, ml_b), "n_diff_px": int((py_b != ml_b).sum())}

    cmp_float("RGB (imresize HE)", he_scaled, m["RGB"])
    out["HIGH_IN"] = {
        "matlab": [float(m["HIGH_IN_r"].item()), float(m["HIGH_IN_g"].item()), float(m["HIGH_IN_b"].item())],
    }
    cmp_float("HEdata (imadjust)", he_adj, m["HEdata"])

    hsv = matlab_rgb2hsv(he_adj)
    sat_thresh = matlab_graythresh(hsv[..., 1])
    out["channel2Min (graythresh S)"] = {
        "python": sat_thresh, "matlab": float(m["channel2Min"].item()),
        "matlab_collagen": float(m["channel2Min_c"].item()),
    }
    nuclei_raw = (
        (hsv[..., 0] >= 0.500) & (hsv[..., 0] <= 0.790)
        & (hsv[..., 1] >= sat_thresh) & (hsv[..., 1] <= 1.0)
    )
    nuclei_raw = remove_small_components(nuclei_raw, 150)
    cmp_mask("BW (nuclei raw + bwareaopen150)", nuclei_raw, m["BW"])
    bw_nuclei, masked = make_nuclei_mask(he_adj, pix)
    cmp_mask("BW_nuclei (imopen)", bw_nuclei, m["BW_nuclei"])
    bw_col, _, _ = make_collagen_mask(he_adj, pix, enhanced_postprocessing=False)
    cmp_mask("BW_collagen", bw_col, m["BW_collagen"])

    gray_nuclei = matlab_rgb2gray(masked)
    cmp_float("gray_nuclei (rgb2gray masked)", gray_nuclei, m["gray_nuclei"])
    ksize = max(1, int(np.floor(pix)))
    nuclei_filtered = gaussian_filter_matlab_like(gray_nuclei, sigma=0.5, kernel_size=ksize, boundary="zero")
    cmp_float("nuclei_filtered (imfilter gaussian)", nuclei_filtered, m["nuclei_filtered"])
    bw_n2 = nuclei_filtered > 0.001
    cmp_mask("BW_nuclei2 (im2bw 0.001)", bw_n2, m["BW_nuclei2"])
    bw_disc = remove_small_components(bw_n2, int(np.ceil(50.0 * pix**2)))
    cmp_mask("BW_nuclei_discard (bwareaopen)", bw_disc, m["BW_nuclei_discard"])
    bw_dil = morphology.dilation(bw_disc, disk_se(np.floor(pix)))
    cmp_mask("BW_nuclei_dilated (imdilate)", bw_dil, m["BW_nuclei_dilated"])
    bw_fill = binary_fill_holes(bw_dil)
    cmp_mask("BW_nuclei_filled (imfill holes)", bw_fill, m["BW_nuclei_filled"])
    he_col = bw_col & (~bw_fill)
    cmp_mask("HE_collagen_BW", he_col, m["HE_collagen_BW"])
    bw_discard = remove_small_components(he_col, int(np.ceil(pix**2)))
    cmp_mask("BW_discard / HEmoving", bw_discard, m["BW_discard"])
    return out


def print_preprocessing_report(cases: list) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for case_id, he_dir, he_file, shg_dir, ppm, _roi in cases:
        res = compare_preprocessing(case_id, he_dir, he_file, shg_dir, ppm)
        report[case_id] = res
        print(f"\n== {case_id} (ppm={ppm}) preprocessing step parity ==")
        for step, v in res.items():
            if "max_abs_diff" in v:
                flag = "" if v["max_abs_diff"] < 1e-9 else "   <-- differs"
                print(f"  {step:40s} max|diff|={v['max_abs_diff']:.3e}{flag}")
            elif "dice" in v:
                flag = "" if v["n_diff_px"] == 0 else "   <-- differs"
                print(f"  {step:40s} dice={v['dice']:.5f} n_diff={v['n_diff_px']}{flag}")
            else:
                print(f"  {step:40s} {v}")
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    run_engine = "--no-engine" not in argv
    selected = [a for a in argv if not a.startswith("--")]
    cases = [c for c in CASES if not selected or c[0] in selected]
    if "--preproc" in argv:
        report = print_preprocessing_report(cases)
        out = DUMPS / "preprocessing_parity.json"
        out.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nWrote {out}")
        return
    if run_engine and not has_itk():
        print("itk not importable -> engine checks disabled (--no-engine).")
        run_engine = False

    results: dict[str, Any] = {"cases": {}, "determinism": {}}

    t2 = DUMPS / "test2" / "tform_affine.txt"
    t2r = DUMPS / "test2_rerun" / "tform_affine.txt"
    if t2.is_file() and t2r.is_file():
        a, b = np.loadtxt(t2), np.loadtxt(t2r)
        results["determinism"] = {
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "identical": bool(np.array_equal(a, b)),
        }
        print(
            f"MATLAB determinism test2 vs rerun: max|dT|={results['determinism']['max_abs_diff']:.3e} "
            f"identical={results['determinism']['identical']}"
        )

    hdr = (
        f"{'case':6s} {'src':10s} {'dice':>6s} {'cov_ml':>6s} {'cov_py':>6s} | "
        f"{'ML ang':>7s} {'ML s':>6s} {'ML tx':>7s} {'ML ty':>7s} | "
        f"{'PY(mlmask) max|dA|':>19s} {'trace':>8s} | {'PY(pymask) ang':>14s} {'corner px':>9s} | "
        f"{'GT ang':>7s} {'ML-GT px':>8s} {'PY-GT px':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for case_id, he_dir, he_file, shg_dir, ppm, roi in cases:
        dump = DUMPS / case_id
        entry: dict[str, Any] = {"ppm": ppm}
        if not (dump / "tform_affine.txt").is_file():
            entry["status"] = "missing_dump"
            results["cases"][case_id] = entry
            print(f"{case_id:6s} MISSING dump")
            continue

        ml_he, ml_shg, src = _load_matlab_inputs(dump)
        py_he, py_shg, pix, he_in_shape, shg_in_shape = _python_pipeline(
            he_dir / he_file, shg_dir / he_file, ppm
        )
        entry["inputs_source"] = src
        entry["shape"] = list(ml_he.shape)
        entry["fixed_shg_max_abs_diff"] = (
            float(np.abs(py_shg - ml_shg).max()) if py_shg.shape == ml_shg.shape else None
        )
        dice = _dice(ml_he > 0.5, py_he > 0.5) if py_he.shape == ml_he.shape else float("nan")
        entry["dice_he_moving"] = dice
        entry["matlab_mask_coverage"] = float((ml_he > 0.5).mean())
        entry["python_mask_coverage"] = float((py_he > 0.5).mean())

        A_sim_ml = matlab_T_to_A(np.loadtxt(dump / "tform_similarity.txt"))
        A_aff_ml = matlab_T_to_A(np.loadtxt(dump / "tform_affine.txt"))
        entry["matlab_similarity"] = _decomp_A(A_sim_ml)
        entry["matlab_affine"] = _decomp_A(A_aff_ml)
        entry["matlab_affine_forward_2x3_0based"] = moving_to_fixed_1based_to_forward_0based(
            A_aff_ml
        ).tolist()
        dm = entry["matlab_affine"]

        parity_str = f"{'skipped':>19s}"
        trace_str = f"{'-':>8s}"
        basin_str = f"{'skipped':>14s} {'-':>9s}"
        A_aff_py_pymask: np.ndarray | None = None
        if run_engine:
            # (2) engine on MATLAB's exact inputs
            t0 = time.time()
            cfg = OnePlusOneConfig(initial_radius=6.25e-3 / 3.5, maximum_iterations=700)
            sim = matlab_imregtform_v3(ml_he, ml_shg, "similarity", cfg)
            aff = matlab_imregtform_v3(
                ml_he, ml_shg, "affine", cfg, initial_fixed_to_moving_3x3=sim.fixed_to_moving_3x3
            )
            e_sim = _param_error(sim.moving_to_fixed_1based_3x3, A_sim_ml, ml_he.shape)
            e_aff = _param_error(aff.moving_to_fixed_1based_3x3, A_aff_ml, ml_he.shape)
            entry["engine_on_matlab_inputs"] = {
                "seconds": time.time() - t0,
                "similarity": _decomp_A(sim.moving_to_fixed_1based_3x3),
                "affine": _decomp_A(aff.moving_to_fixed_1based_3x3),
                "similarity_error_vs_matlab": e_sim,
                "affine_error_vs_matlab": e_aff,
                "sim_stop": sim.stop_condition.strip(),
                "aff_stop": aff.stop_condition.strip(),
            }
            parity_str = f"{max(e_aff['max_abs_dM'], e_aff['max_abs_dt']):19.3e}"
            trace_path = dump / "optimization_trace.txt"
            if trace_path.is_file():
                ml_trace = _parse_matlab_trace(trace_path)
                cmp = {
                    "similarity": _compare_trace(sim.trace, sim.level_starts, ml_trace.get("similarity", {})),
                    "affine": _compare_trace(aff.trace, aff.level_starts, ml_trace.get("affine", {})),
                }
                entry["trace_comparison"] = cmp
                divs = [
                    v["first_divergent_iteration"]
                    for st in cmp.values()
                    for v in st.values()
                ]
                trace_str = f"{'exact':>8s}" if all(d is None for d in divs) else f"{'DIVERGES':>8s}"

            # (3) engine on Python's mask
            if py_he.shape == ml_he.shape:
                fwd_py, dbg = register_bdcreation_reg2_matlab(py_he, py_shg)
                A_aff_py_pymask = np.asarray(dbg["aff_matlab_tform_A"])
                e_basin = _param_error(A_aff_py_pymask, A_aff_ml, ml_he.shape)
                entry["engine_on_python_mask"] = {
                    "affine": _decomp_A(A_aff_py_pymask),
                    "affine_error_vs_matlab": e_basin,
                    "forward_2x3_0based": fwd_py.tolist(),
                }
                basin_str = (
                    f"{entry['engine_on_python_mask']['affine']['angle_deg']:14.3f} "
                    f"{e_basin['mean_corner_disp_px']:9.2f}"
                )

        # (4) ground truth
        gt_str = f"{'n/a':>7s} {'-':>8s} {'-':>8s}"
        if roi is not None:
            gt_path = P02 / f"patient_02_HE_original-{roi}.tif"
            gt_info = _recover_gt_affine(he_dir / he_file, gt_path)
            if isinstance(gt_info, dict) and "forward_2x3_input_grid_0based" in gt_info:
                gt_work = _gt_to_working_grid(
                    np.asarray(gt_info["forward_2x3_input_grid_0based"]),
                    he_in_shape, shg_in_shape, tuple(ml_he.shape),
                )
                A_gt = forward_0based_to_moving_to_fixed_1based(gt_work)
                gt_info["forward_2x3_working_grid_0based"] = gt_work.tolist()
                gt_info["working_grid_params"] = _decomp_A(A_gt)
                gt_info["matlab_error_vs_gt"] = _param_error(A_aff_ml, A_gt, ml_he.shape)
                if A_aff_py_pymask is not None:
                    gt_info["python_matlab_engine_error_vs_gt"] = _param_error(
                        A_aff_py_pymask, A_gt, ml_he.shape
                    )
                if run_engine:
                    gt_info["mattes_mi_at"] = {
                        "identity": _mi_at(ml_he, ml_shg, np.eye(3)),
                        "matlab_tform": _mi_at(ml_he, ml_shg, A_aff_ml),
                        "gt_tform": _mi_at(ml_he, ml_shg, A_gt),
                    }
                    if A_aff_py_pymask is not None:
                        gt_info["mattes_mi_at"]["python_tform_on_matlab_mask"] = _mi_at(
                            ml_he, ml_shg, A_aff_py_pymask
                        )
                py_gt = gt_info.get("python_matlab_engine_error_vs_gt", {}).get(
                    "mean_corner_disp_px", float("nan")
                )
                gt_str = (
                    f"{gt_info['working_grid_params']['angle_deg']:7.2f} "
                    f"{gt_info['matlab_error_vs_gt']['mean_corner_disp_px']:8.1f} "
                    f"{py_gt:8.1f}"
                )
                (DUMPS / f"gt_affine_{case_id}.json").write_text(json.dumps(gt_info, indent=2))
            entry["gt_affine"] = gt_info

        results["cases"][case_id] = entry
        print(
            f"{case_id:6s} {src[:10]:10s} {dice:6.3f} {entry['matlab_mask_coverage']:6.3f} "
            f"{entry['python_mask_coverage']:6.3f} | "
            f"{dm['angle_deg']:7.2f} {dm['scale_x']:6.3f} {dm['tx']:7.2f} {dm['ty']:7.2f} | "
            f"{parity_str} {trace_str} | {basin_str} | {gt_str}",
            flush=True,
        )

    out_json = DUMPS / "analysis_summary.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
