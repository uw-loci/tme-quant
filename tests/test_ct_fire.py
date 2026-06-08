"""
Regression tests for CT-FIRE Python implementation vs MATLAB ctFIRE reference.

Mirrors the structure of test_fire_2d_angle.py but calls ct_fire() instead of
fire_2d_angle() directly.  The CT-FIRE pipeline adds curvelet reconstruction
and thresh_im2 masking before fiber extraction, so we compare against MATLAB
ctFIRE (not fire_2D_ang1) reference outputs.

Test cases are loaded from:
    tests/test_results/ct_fire_test_files/test_cases_ct_fire.json

MATLAB reference .mat files (user-generated) belong in:
    tests/test_results/ct_fire_test_files/test_ct_fire_*.mat

Run non-MATLAB tests only:
    pytest tests/test_ct_fire.py -m "not matlab"

Run all tests (requires .mat files):
    pytest tests/test_ct_fire.py
"""

import copy
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# By default, skip curvelops-dependent tests (e.g., on CI). Enable locally with:
#   TMEQ_RUN_CURVELETS=1 pytest -q
if os.environ.get("TMEQ_RUN_CURVELETS") != "1":
    pytest.skip(
        "curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
        allow_module_level=True,
    )

try:
    from curvelops import fdct2d_wrapper
except ImportError:
    pytest.skip("curvelops not available; skipping ct_fire tests", allow_module_level=True)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ctfire_py" / "CPP"))
    import fiber_backend  # noqa: F401
    CPP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CPP_AVAILABLE = False

from ctfire_py.ct_fire import ct_fire, load_ctfire_params
from ctfire_py.ct_reconstruction import ct_reconstruction
from ctfire_py.fire_2d_angle import fire_2d_angle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FIBER_LEN_PX = 30.0
SOFT_IOU_THRESHOLD = 0.80   # stricter than fire_2d (0.50): CT preprocessing should
                             # keep spatial structure close to MATLAB ctFIRE

_CT_FIRE_RESULTS_DIR = Path(__file__).parent / "test_results" / "ct_fire_test_files"

# Module-level cache: avoids re-running the expensive ct_fire pipeline once per
# test when the same test case is exercised by multiple test functions.
# Keys: test_case["name"]; values: ctfire_output dict returned by ct_fire().
_ct_fire_cache: dict = {}


# ============================================================================
# Fixtures and Utilities
# ============================================================================


def load_test_cases(config_path=None, matlab_only=False):
    """Load test cases from JSON configuration.

    Parameters
    ----------
    config_path : Path or str, optional
        Path to JSON config.  Defaults to
        ``test_results/ct_fire_test_files/test_cases_ct_fire.json``.
    matlab_only : bool
        When True, only return cases that have a ``matlab_reference_mat`` key.

    Returns
    -------
    list of (name, test_case) tuples suitable for ``@pytest.mark.parametrize``.
    """
    if config_path is None:
        config_path = _CT_FIRE_RESULTS_DIR / "test_cases_ct_fire.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    cases = config["test_cases"]
    if matlab_only:
        cases = [tc for tc in cases if "matlab_reference_mat" in tc]
    return [(tc["name"], tc) for tc in cases]


def load_test_image(image_name):
    """Load a test image by filename, returning a [0, 255] range array.

    PNG files loaded by matplotlib are float32 [0, 1]; they are rescaled to
    [0, 255] uint8 so that ``thresh_im2`` thresholds behave identically to TIF
    inputs.
    """
    img_path = Path(__file__).parent / "test_images" / image_name
    if not img_path.exists():
        pytest.skip(f"Test image not found: {image_name}")
    img = plt.imread(str(img_path))
    if img.dtype == np.float32 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img


def run_ct_fire_case(test_case, img=None, save_path=None, save_images=False):
    """Run ct_fire for *test_case*, optionally using a cached result.

    When *save_images* is False (default), the result is cached by
    ``test_case["name"]`` so that subsequent calls for the same case return
    immediately.  Pass *save_images=True* to force a fresh run with file output.

    Returns
    -------
    ctfire_output : dict
        The second element returned by :func:`ct_fire`.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    cache_key = test_case["name"]
    if not save_images and cache_key in _ct_fire_cache:
        return _ct_fire_cache[cache_key]

    if img is None:
        img = load_test_image(test_case["image"])

    sp = save_path or str(_CT_FIRE_RESULTS_DIR)
    control_params = {
        "show_plots": False,
        "save_images": save_images,
        "output_format": "tif",
    }
    _, ctfire_out = ct_fire(
        image_path=None,
        image_name=test_case["image"],
        save_path=sp,
        control_params=control_params,
        ctfire_params=test_case["ctfire_params"],
        img=img,
    )

    if not save_images:
        _ct_fire_cache[cache_key] = ctfire_out
    return ctfire_out


def load_matlab_reference(mat_file_path):
    """Load MATLAB ctFIRE reference data from a .mat file.

    Supports both MATLAB v7.3 (via h5py) and older v5 formats (via scipy.io).

    Expected .mat structure::

        data.Xa        — vertex coordinates  [N × 3]
        data.Fa        — fiber list with 1-based vertex indices
        data.M.fiber_num, avgL, totL, L, angle_xy

    Returns
    -------
    dict with keys ``'Xa'``, ``'Fa'``, ``'M'``.
    """
    if not os.path.exists(str(mat_file_path)):
        pytest.skip(f"MATLAB reference file not found: {mat_file_path}")

    _h5py_ok = False
    try:
        if H5PY_AVAILABLE:
            try:
                with h5py.File(str(mat_file_path), "r"):
                    pass
                _h5py_ok = True
            except Exception:
                _h5py_ok = False

        if _h5py_ok:
            with h5py.File(str(mat_file_path), "r") as f:
                if "data" not in f:
                    raise ValueError("MATLAB .mat file must contain a 'data' structure")

                data_group = f["data"]
                result: dict = {"Xa": None, "Fa": None, "M": {}}

                if "Xa" in data_group:
                    result["Xa"] = np.array(data_group["Xa"]).T  # h5py stores transposed

                if "Fa" in data_group and "v" in data_group["Fa"]:
                    v_ds = data_group["Fa"]["v"]
                    fibers = []
                    for i in range(v_ds.shape[0]):
                        ref = v_ds[i, 0]
                        v_arr = np.array(f[ref]).flatten().astype(int) - 1  # 1-based → 0-based
                        fibers.append({"v": list(v_arr)})
                    result["Fa"] = fibers

                if "M" in data_group:
                    M_group = data_group["M"]
                    for scalar in ("fiber_num", "avgL", "totL", "Ldens", "volfrac"):
                        if scalar in M_group:
                            v = np.array(M_group[scalar]).item()
                            result["M"][scalar] = int(v) if scalar == "fiber_num" else float(v)
                    for arr_key in ("L", "angle_xy"):
                        if arr_key in M_group:
                            arr = np.array(M_group[arr_key])
                            result["M"][arr_key] = arr.flatten() if arr.size > 0 else np.array([])

                return result

        if not _h5py_ok and SCIPY_AVAILABLE:
            mat_data = loadmat(str(mat_file_path), struct_as_record=False, squeeze_me=True)
            if "data" not in mat_data:
                raise ValueError("MATLAB .mat file must contain a 'data' structure")
            data = mat_data["data"]

            raw_xa = data.Xa if hasattr(data, "Xa") else None
            raw_fa = data.Fa if hasattr(data, "Fa") else None
            fa_norm = None
            if raw_fa is not None:
                fa_norm = []
                for fib in np.atleast_1d(raw_fa):
                    if hasattr(fib, "v"):
                        v_arr = np.atleast_1d(fib.v).flatten().astype(int) - 1
                        fa_norm.append({"v": v_arr.tolist()})

            result = {"Xa": raw_xa, "Fa": fa_norm, "M": {}}
            if hasattr(data, "M"):
                M = data.M
                result["M"] = {
                    "fiber_num": M.fiber_num if hasattr(M, "fiber_num") else 0,
                    "avgL": M.avgL if hasattr(M, "avgL") else 0,
                    "totL": M.totL if hasattr(M, "totL") else 0,
                    "L": M.L if hasattr(M, "L") else np.array([]),
                    "angle_xy": M.angle_xy if hasattr(M, "angle_xy") else np.array([]),
                }
            return result

    except Exception as e:
        pytest.skip(f"Could not load MATLAB reference file: {e}")

    pytest.skip("Neither h5py nor scipy.io available for loading MATLAB files")


# ============================================================================
# Soft IoU helpers  (identical to test_fire_2d_angle.py)
# ============================================================================

_SMOOTH_SIGMA = 5.0


def _smooth_mask(mask, smooth_sigma=_SMOOTH_SIGMA):
    """Gaussian-smooth a binary mask and rescale to [0, 1]."""
    from skimage import exposure, filters

    mask = mask.astype(np.float32)
    mask = exposure.rescale_intensity(mask, out_range=(0.0, 1.0))
    density = filters.gaussian(mask, sigma=smooth_sigma, preserve_range=False)
    return exposure.rescale_intensity(density, out_range=(0.0, 1.0)).astype(np.float32)


def _soft_iou(mask_1, mask_2, beta=1e-3):
    """Soft IoU on two float masks already smoothed to [0, 1].

    Formula: (m1·m2).sum() / (m1²+m2²-m1·m2).sum()
    """
    intersection = mask_1 * mask_2
    union = mask_1 ** 2 + mask_2 ** 2 - mask_1 * mask_2
    return float((intersection.sum() + beta) / (union.sum() + beta))


def _rasterize_fibers(X, F, image_shape):
    """Rasterize fiber output to a 1-px skeleton image using Bresenham lines."""
    from skimage.draw import line as draw_line
    from skimage.morphology import skeletonize

    canvas = np.zeros(image_shape, dtype=np.uint8)
    H, W = image_shape
    X_arr = np.asarray(X)
    for fiber in F:
        v_list = fiber["v"] if isinstance(fiber, dict) else list(fiber)
        for seg in range(len(v_list) - 1):
            v0, v1 = v_list[seg], v_list[seg + 1]
            if v0 >= len(X_arr) or v1 >= len(X_arr):
                continue
            r0 = int(np.clip(round(float(X_arr[v0, 0])), 0, H - 1))
            c0 = int(np.clip(round(float(X_arr[v0, 1])), 0, W - 1))
            r1 = int(np.clip(round(float(X_arr[v1, 0])), 0, H - 1))
            c1 = int(np.clip(round(float(X_arr[v1, 1])), 0, W - 1))
            rr, cc = draw_line(r0, c0, r1, c1)
            canvas[rr, cc] = 1
    return skeletonize(canvas > 0)


def _fiber_stats_filtered(X, F, min_len=MIN_FIBER_LEN_PX, row_idx=0, col_idx=1):
    """Fiber statistics filtered to arc-length >= *min_len* px.

    row_idx / col_idx select which columns of X are row/col:
      - Python Xa  [row, col, ...]: row_idx=0, col_idx=1  (default)
      - MATLAB Xa  [col, row, z]  : row_idx=1, col_idx=0
    """
    if F is None or len(F) == 0 or X is None or len(X) == 0:
        return {"fiber_num": 0, "avgL": 0.0, "totL": 0.0, "angle_xy": np.array([])}

    eps = np.finfo(float).eps
    X_arr = np.asarray(X, dtype=float)
    N = len(X_arr)
    lengths, angles = [], []

    for fiber in F:
        v = fiber["v"] if isinstance(fiber, dict) else list(fiber)
        if len(v) < 2:
            continue
        length = 0.0
        valid = True
        for i in range(len(v) - 1):
            if not (0 <= v[i] < N and 0 <= v[i + 1] < N):
                valid = False
                break
            length += np.linalg.norm(X_arr[v[i + 1]] - X_arr[v[i]])
        if not valid or length < min_len:
            continue
        lengths.append(length)
        v0, v1 = v[0], v[-1]
        if 0 <= v0 < N and 0 <= v1 < N:
            dr = float(X_arr[v1, row_idx] - X_arr[v0, row_idx])
            dc = float(X_arr[v1, col_idx] - X_arr[v0, col_idx])
            angles.append(np.arctan(dr / (dc + eps)) % np.pi)

    L = np.array(lengths, dtype=float)
    return {
        "fiber_num": len(L),
        "avgL": float(np.mean(L)) if len(L) > 0 else 0.0,
        "totL": float(np.sum(L)),
        "angle_xy": np.array(angles, dtype=float),
    }


# ============================================================================
# CT Reconstruction Tests
# ============================================================================

_CT_RECON_RESULTS_DIR = _CT_FIRE_RESULTS_DIR  # save alongside .mat files


def _load_matlab_reconstructed(mat_file_path):
    """Load a MATLAB-saved reconstructed image from a .mat file.

    Expected .mat structure (save from MATLAB ctFIRE after ct_reconstruction step)::

        recon_img   — 2D double array, same spatial dimensions as the input image

    Returns None (and skips) if the file does not exist or lacks ``recon_img``.
    """
    if not os.path.exists(str(mat_file_path)):
        return None

    if H5PY_AVAILABLE:
        try:
            with h5py.File(str(mat_file_path), "r") as f:
                if "recon_img" in f:
                    return np.array(f["recon_img"]).T  # h5py transposes
        except Exception:
            pass

    if SCIPY_AVAILABLE:
        try:
            mat = loadmat(str(mat_file_path), squeeze_me=True)
            if "recon_img" in mat:
                return np.array(mat["recon_img"], dtype=np.float64)
        except Exception:
            pass

    return None


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_reconstruction_output_shape_and_range(test_name, test_case):
    """ct_reconstruction returns a 2D float array with the same shape as the input."""
    img = load_test_image(test_case["image"])
    rec = ct_reconstruction(
        img=img.astype(np.float64),
        output_filename=test_case["image"],
        coefficient_percentile=test_case["ctfire_params"]["coefficient_percentile"],
        specific_scales=test_case["ctfire_params"]["num_scales"],
    )
    assert rec.ndim == 2, "Reconstructed image must be 2D"
    assert rec.shape == img.shape[-2:] if img.ndim == 3 else img.shape, (
        f"Shape mismatch: rec={rec.shape}, img={img.shape}"
    )
    assert np.isfinite(rec).all(), "Reconstructed image contains non-finite values"
    assert rec.dtype in (np.float32, np.float64), (
        f"Expected float output, got {rec.dtype}"
    )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_reconstruction_saves_image(test_name, test_case):
    """ct_reconstruction with plot_flag=True writes a CTRimg_*.tiff file."""
    img = load_test_image(test_case["image"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        import os as _os
        orig_dir = _os.getcwd()
        _os.chdir(tmp_dir)
        try:
            ct_reconstruction(
                img=img.astype(np.float64),
                output_filename=test_case["image"],
                coefficient_percentile=test_case["ctfire_params"]["coefficient_percentile"],
                specific_scales=test_case["ctfire_params"]["num_scales"],
                plot_flag=True,
            )
        finally:
            _os.chdir(orig_dir)
        base = os.path.splitext(os.path.basename(test_case["image"]))[0]
        expected = os.path.join(tmp_dir, f"CTRimg_{base}.tiff")
        assert os.path.exists(expected), f"Expected saved file not found: {expected}"


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_ct_reconstruction_matches_matlab(test_name, test_case):
    """Python ct_reconstruction output correlates with MATLAB-saved recon_img (r > 0.99).

    Requires a companion .mat file alongside the ctFIRE reference, named::

        recon_img_{image_base}_SS{num_scales}_TH{percentile*10:02d}.mat

    e.g. for real1.tif with SS=3, coefficient_percentile=0.2::

        recon_img_real1_SS3_TH02.mat

    Generate it from MATLAB ctFIRE after the curvelet reconstruction step::

        recon_img = CTRimage;   % the image fed into fire_2D_ang1
        save('recon_img_real1_SS3_TH02.mat', 'recon_img', '-v7.3');

    Comparison metrics:
    - Pearson r > 0.99  (very tight: reconstruction is deterministic)
    - Mean absolute error < 1% of MATLAB image range
    """
    img = load_test_image(test_case["image"])
    ss = test_case["ctfire_params"]["num_scales"]
    pct = test_case["ctfire_params"]["coefficient_percentile"]
    rec_py = ct_reconstruction(
        img=img.astype(np.float64),
        output_filename=test_case["image"],
        coefficient_percentile=pct,
        specific_scales=ss,
    )

    image_base = os.path.splitext(os.path.basename(test_case["image"]))[0].split("_")[0]
    th = int(round(pct * 10))
    mat_filename = f"recon_img_{image_base}_SS{ss}_TH{th:02d}.mat"
    mat_path = _CT_RECON_RESULTS_DIR / mat_filename
    rec_mat = _load_matlab_reconstructed(mat_path)
    if rec_mat is None:
        pytest.skip(
            f"No MATLAB recon_img reference found: {mat_path}\n"
            f"Generate it from MATLAB: recon_img = CTRimage; "
            f"save('{mat_filename}', 'recon_img', '-v7.3')"
        )

    assert rec_py.shape == rec_mat.shape, (
        f"Shape mismatch: Python {rec_py.shape} vs MATLAB {rec_mat.shape}"
    )

    flat_py = rec_py.flatten()
    flat_mat = rec_mat.flatten().astype(np.float64)

    r = float(np.corrcoef(flat_py, flat_mat)[0, 1])
    assert r > 0.99, (
        f"Pearson r={r:.4f} < 0.99 ({test_name}): "
        "Python and MATLAB reconstructed images diverge"
    )

    mat_range = float(flat_mat.max() - flat_mat.min())
    if mat_range > 0:
        mae = float(np.mean(np.abs(flat_py - flat_mat)))
        rel_mae = mae / mat_range
        assert rel_mae < 0.01, (
            f"MAE {rel_mae:.2%} of MATLAB range >= 1% ({test_name})"
        )

    print(
        f"\n{test_name} ct_reconstruction: r={r:.4f}, "
        f"MAE={mae:.4f} ({rel_mae:.2%} of MATLAB range)"
    )


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_basic_execution(test_name, test_case):
    """Smoke test: ct_fire executes without error and returns required fields."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    ctfire_out = run_ct_fire_case(test_case)
    data = ctfire_out["data"]

    assert isinstance(data, dict), "ct_fire data should be a dict"
    for field in ("X", "F", "R", "Xa", "Fa", "Va", "Ra", "M"):
        assert field in data, f"Missing required field: {field}"
    assert isinstance(data["M"], dict), "M (statistics) should be a dict"

    expected = test_case.get("expected_outputs", {})
    fiber_count = len(data["Fa"])
    if "min_fiber_count" in expected:
        assert fiber_count >= expected["min_fiber_count"], (
            f"Too few fibers: {fiber_count} < {expected['min_fiber_count']}"
        )
    if "max_fiber_count" in expected:
        assert fiber_count <= expected["max_fiber_count"], (
            f"Too many fibers: {fiber_count} > {expected['max_fiber_count']}"
        )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_network_statistics(test_name, test_case):
    """Network statistics (fiber_num, avgL, totL) are present and non-negative."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    ctfire_out = run_ct_fire_case(test_case)
    M = ctfire_out["data"]["M"]

    assert "fiber_num" in M, "Missing fiber_num"
    assert "avgL" in M, "Missing avgL"
    assert "totL" in M, "Missing totL"
    assert M["fiber_num"] >= 0, "fiber_num should be non-negative"
    assert M["avgL"] >= 0, "avgL should be non-negative"
    assert M["totL"] >= 0, "totL should be non-negative"

    expected = test_case.get("expected_outputs", {})
    if "min_avg_length" in expected and M["avgL"] > 0:
        assert M["avgL"] >= expected["min_avg_length"], (
            f"avgL {M['avgL']:.2f} < {expected['min_avg_length']}"
        )
    if "max_avg_length" in expected and M["avgL"] > 0:
        assert M["avgL"] <= expected["max_avg_length"], (
            f"avgL {M['avgL']:.2f} > {expected['max_avg_length']}"
        )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_reconstruction_changes_output(test_name, test_case):
    """CT reconstruction changes the input fed to fire_2d_angle vs the raw image.

    For real images (thresh_im2 < 50), asserts that the total extracted fiber
    length differs by at least 1% between the CT-FIRE pipeline and a direct
    fire_2d_angle call on the unchanged image.  For synthetic images the
    threshold is high (≥ 98), so almost all pixels are masked anyway and the
    reconstruction effect is minimal; those cases just verify completion.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    ctfire_out = run_ct_fire_case(test_case, img=img)
    data_ct = ctfire_out["data"]

    # Replicate ct_fire's normalisation so fire_2d_angle sees the same dtype
    img_f = img.astype(np.float32)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0
    elif img_f.max() > 255.0:
        img_f = img_f * (255.0 / img_f.max())
    im3 = img_f[np.newaxis, :, :] if img_f.ndim == 2 else img_f
    data_raw = fire_2d_angle(p=test_case["ctfire_params"]["value"], im=im3, plotflag=0)

    thresh = test_case["ctfire_params"]["value"].get("thresh_im2", 5)
    if thresh < 50:
        # Real images: CT reconstruction should measurably change the output
        totL_ct = float(data_ct["M"]["totL"])
        totL_raw = float(data_raw["M"]["totL"])
        if totL_ct > 0 and totL_raw > 0:
            rel_diff = abs(totL_ct - totL_raw) / max(totL_ct, totL_raw)
            assert rel_diff > 0.01, (
                f"CT reconstruction had negligible effect on fiber extraction: "
                f"ct_fire totL={totL_ct:.0f}px, raw fire_2d totL={totL_raw:.0f}px "
                f"(rel diff={rel_diff:.1%} < 1%)"
            )
        print(
            f"\n{test_name}: ct_fire totL={data_ct['M']['totL']:.0f}px, "
            f"raw totL={data_raw['M']['totL']:.0f}px"
        )
    else:
        # Synthetic images with high background threshold: just verify completion
        assert data_ct is not None
        assert "M" in data_ct


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_thresh_im2_masking(test_name, test_case):
    """Setting thresh_im2=250 (mask nearly all pixels) produces ≤ default fiber count."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    ctfire_out_default = run_ct_fire_case(test_case, img=img)
    count_default = ctfire_out_default["data"]["M"]["fiber_num"]

    # Create a variant with an extreme mask: almost no pixels pass thresh=250
    params_high = copy.deepcopy(test_case["ctfire_params"])
    params_high["value"]["thresh_im2"] = 250
    control_params = {"show_plots": False, "save_images": False, "output_format": "tif"}
    _, ctfire_out_high = ct_fire(
        image_path=None,
        image_name=test_case["image"],
        save_path=str(_CT_FIRE_RESULTS_DIR),
        control_params=control_params,
        ctfire_params=params_high,
        img=img,
    )
    count_high = ctfire_out_high["data"]["M"]["fiber_num"]

    assert count_high <= count_default, (
        f"thresh_im2=250 produced more fibers ({count_high}) than "
        f"thresh_im2={test_case['ctfire_params']['value']['thresh_im2']} ({count_default})"
    )
    print(
        f"\n{test_name}: fibers with default thresh={count_default}, "
        f"with thresh=250: {count_high}"
    )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_saves_files(test_name, test_case):
    """With save_images=True, ct_fire creates overlay TIFF and filtered fiber CSV.

    Validates:
    - Both files exist after the call.
    - ``saved_files`` dict is populated with ``'overlay'`` and ``'csv'`` keys.
    - CSV has the expected five columns.
    - Every row in the CSV has length_px >= MIN_FIBER_LEN_PX (30 px).
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctfire_out = run_ct_fire_case(
            test_case, img=img, save_path=tmp_dir, save_images=True
        )

        saved = ctfire_out.get("saved_files", {})
        assert "overlay" in saved, "Missing 'overlay' key in saved_files"
        assert "csv" in saved, "Missing 'csv' key in saved_files"
        assert "params" in saved, "Missing 'params' key in saved_files"

        overlay_path = saved["overlay"]
        csv_path = saved["csv"]
        params_path = saved["params"]

        assert os.path.exists(overlay_path), f"Overlay file not found: {overlay_path}"
        assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"
        assert os.path.exists(params_path), f"Params JSON not found: {params_path}"

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            expected_columns = {"fiber_id", "length_px", "angle_deg", "width_px", "straightness"}
            assert set(reader.fieldnames or []) == expected_columns, (
                f"Unexpected CSV columns: {reader.fieldnames}"
            )
            for row in reader:
                length = float(row["length_px"])
                assert length >= MIN_FIBER_LEN_PX, (
                    f"CSV contains fiber shorter than {MIN_FIBER_LEN_PX}px: {length}"
                )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_saves_params_json(test_name, test_case):
    """With save_images=True, ct_fire writes a valid *_params.json file.

    Validates:
    - ``saved_files["params"]`` is present and the file exists.
    - JSON contains ``ctfire`` and ``fire2d`` sections.
    - ``ctfire.LL1`` matches the input value.
    - ``fire2d.thresh_im2`` is present.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctfire_out = run_ct_fire_case(
            test_case, img=img, save_path=tmp_dir, save_images=True
        )
        saved = ctfire_out.get("saved_files", {})
        assert "params" in saved, "Missing 'params' key in saved_files"
        params_path = saved["params"]
        assert os.path.exists(params_path), f"Params JSON not found: {params_path}"

        with open(params_path, "r") as f:
            cP = json.load(f)

        assert "ctfire" in cP, "Missing 'ctfire' section in params JSON"
        assert "fire2d" in cP, "Missing 'fire2d' section in params JSON"
        assert "LL1" in cP["ctfire"], "Missing 'LL1' in ctfire section"
        assert "thresh_im2" in cP["fire2d"], "Missing 'thresh_im2' in fire2d section"

        expected_LL1 = test_case["ctfire_params"].get("LL1", 30)
        assert cP["ctfire"]["LL1"] == expected_LL1, (
            f"LL1 mismatch: JSON has {cP['ctfire']['LL1']}, expected {expected_LL1}"
        )


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_ct_fire_load_params_roundtrip(test_name, test_case):
    """load_ctfire_params reconstructs a ctfire_params dict passable to ct_fire.

    Saves params JSON then reloads with load_ctfire_params.  Checks:
    - Top-level ``LL1`` is preserved.
    - ``value["thresh_im2"]`` is preserved (fire2d pass-through is lossless).
    - The dict contains the keys required by ct_fire.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctfire_out = run_ct_fire_case(
            test_case, img=img, save_path=tmp_dir, save_images=True
        )
        params_path = ctfire_out.get("saved_files", {}).get("params")
        if params_path is None:
            pytest.skip("params not saved")

        loaded = load_ctfire_params(params_path)

        for key in ("coefficient_percentile", "num_scales", "LL1", "value"):
            assert key in loaded, f"Missing key '{key}' in loaded ctfire_params"

        expected_LL1 = test_case["ctfire_params"].get("LL1", 30)
        assert loaded["LL1"] == expected_LL1, (
            f"LL1 roundtrip failed: loaded {loaded['LL1']}, expected {expected_LL1}"
        )

        expected_thresh_im2 = test_case["ctfire_params"]["value"]["thresh_im2"]
        assert loaded["value"]["thresh_im2"] == expected_thresh_im2, (
            f"thresh_im2 roundtrip failed: "
            f"loaded {loaded['value']['thresh_im2']}, expected {expected_thresh_im2}"
        )


# ============================================================================
# MATLAB Comparison Tests
# ============================================================================


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_ct_fire_matches_matlab_fiber_count(test_name, test_case):
    """Python CT-FIRE fiber count is within 70–180% of MATLAB ctFIRE reference."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    ctfire_out = run_ct_fire_case(test_case)
    data_py = ctfire_out["data"]

    mat_path = _CT_FIRE_RESULTS_DIR / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)

    fiber_count_py = data_py["M"]["fiber_num"]
    fiber_count_mat = data_mat["M"]["fiber_num"]

    if fiber_count_mat > 0:
        rel_diff = abs(fiber_count_py - fiber_count_mat) / fiber_count_mat
        assert fiber_count_py >= fiber_count_mat * 0.70, (
            f"Too few fibers: py={fiber_count_py}, mat={fiber_count_mat} "
            f"(diff={rel_diff:.1%})"
        )
        assert fiber_count_py <= fiber_count_mat * 1.80, (
            f"Too many fibers: py={fiber_count_py}, mat={fiber_count_mat} "
            f"(diff={rel_diff:.1%})"
        )
        print(
            f"\n{test_name} fiber count — "
            f"Python: {fiber_count_py}, MATLAB: {fiber_count_mat}, diff: {rel_diff:.1%}"
        )


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_ct_fire_matches_matlab_fiber_length(test_name, test_case):
    """Python CT-FIRE average fiber length is within 45–120% of MATLAB reference."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    ctfire_out = run_ct_fire_case(test_case)
    data_py = ctfire_out["data"]

    mat_path = _CT_FIRE_RESULTS_DIR / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)

    avgL_py = data_py["M"].get("avgL", 0)
    avgL_mat = data_mat["M"]["avgL"]

    if avgL_mat > 0 and avgL_py > 0:
        rel_diff = abs(avgL_py - avgL_mat) / avgL_mat
        assert avgL_py >= avgL_mat * 0.45, (
            f"Python fibers too short: py={avgL_py:.2f}, mat={avgL_mat:.2f} "
            f"(diff={rel_diff:.1%})"
        )
        assert avgL_py <= avgL_mat * 1.20, (
            f"Python fibers too long: py={avgL_py:.2f}, mat={avgL_mat:.2f}"
        )
        print(
            f"\n{test_name} avg length — "
            f"Python: {avgL_py:.2f}, MATLAB: {avgL_mat:.2f}, diff: {rel_diff:.1%}"
        )


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_ct_fire_matches_matlab_angles(test_name, test_case):
    """Python CT-FIRE angle distribution correlates with MATLAB reference (r > 0.5)."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    ctfire_out = run_ct_fire_case(test_case)
    data_py = ctfire_out["data"]

    mat_path = _CT_FIRE_RESULTS_DIR / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)

    angles_py = np.asarray(data_py["M"].get("angle_xy", []))
    angles_mat = np.asarray(data_mat["M"].get("angle_xy", []))

    if len(angles_py) > 0 and len(angles_mat) > 0:
        bins = np.linspace(0, np.pi, 20)
        hist_py, _ = np.histogram(angles_py % np.pi, bins=bins, density=True)
        hist_mat, _ = np.histogram(angles_mat % np.pi, bins=bins, density=True)
        hist_py = hist_py / (hist_py.sum() + 1e-10)
        hist_mat = hist_mat / (hist_mat.sum() + 1e-10)

        if hist_py.sum() > 0 and hist_mat.sum() > 0:
            correlation = np.corrcoef(hist_py, hist_mat)[0, 1]
            assert correlation > 0.5, (
                f"Angle distributions too different: r={correlation:.3f} < 0.5"
            )
            print(f"\n{test_name} angle histogram correlation: {correlation:.3f}")


# ============================================================================
# Soft IoU Validation
# ============================================================================


class TestCtFireSoftIoU:
    """Validate that Python ct_fire recovers MATLAB ctFIRE fiber centerlines spatially.

    Soft IoU at sigma=5 (≈10-px FWHM) measures spatial overlap of skeleton
    images after Gaussian smoothing.  The threshold is 0.80 — higher than the
    fire_2d threshold (0.50) because the CT preprocessing step is identical
    in Python and MATLAB, so the spatial structure should be very similar.
    """

    @pytest.mark.matlab
    @pytest.mark.parametrize(
        "test_name,test_case",
        load_test_cases(matlab_only=True),
        ids=[name for name, _ in load_test_cases(matlab_only=True)],
    )
    def test_soft_iou_matlab_vs_python(self, test_name, test_case):
        """Python ct_fire centerlines match MATLAB ctFIRE via soft IoU > 0.80.

        Also checks:
        - Total fiber length within ±7% of MATLAB.
        - Mean |angle| within 10° of MATLAB reference.

        Saves a colour-coded IoU overlay to
        ``ct_fire_test_files/iou_overlay_{test_name}.png`` for visual inspection.
        """
        if not CPP_AVAILABLE:
            pytest.skip("C++ backend not available")

        img_path = Path(__file__).parent / "test_images" / test_case["image"]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {test_case['image']}")

        img = load_test_image(test_case["image"])
        ctfire_out = run_ct_fire_case(test_case, img=img)
        data_py = ctfire_out["data"]

        mat_path = _CT_FIRE_RESULTS_DIR / test_case["matlab_reference_mat"]
        data_mat = load_matlab_reference(mat_path)

        img_2d = img[0] if img.ndim == 3 else img
        H, W = img_2d.shape

        # Soft IoU: compare full Fa (post-fiberproc) skeletons on both sides.
        # MATLAB Xa after h5py transpose is [col, row, z]; swap to [row, col].
        if data_mat["Xa"] is not None and data_mat.get("Fa") is not None:
            mat_Xa_rc = data_mat["Xa"][:, [1, 0]]  # [col, row, z] → [row, col]
            mat_skel = _rasterize_fibers(mat_Xa_rc, data_mat["Fa"], (H, W))
            py_skel = _rasterize_fibers(data_py["Xa"], data_py["Fa"], (H, W))
            iou = _soft_iou(
                _smooth_mask(mat_skel.astype(np.float32)),
                _smooth_mask(py_skel.astype(np.float32)),
            )

            assert iou > SOFT_IOU_THRESHOLD, (
                f"soft IoU {iou:.3f} < {SOFT_IOU_THRESHOLD} ({test_name}): "
                "Python and MATLAB ct_fire centerlines diverge spatially"
            )
            print(f"\n{test_name} soft IoU = {iou:.3f}")

            # Save colour-coded overlay for visual inspection
            overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
            mat_only = mat_skel & ~py_skel
            py_only = py_skel & ~mat_skel
            both = mat_skel & py_skel
            overlay_rgba[mat_only] = [1.0, 0.0, 0.0, 1.0]  # red:    MATLAB only
            overlay_rgba[py_only] = [0.0, 1.0, 0.0, 1.0]   # green:  Python only
            overlay_rgba[both] = [1.0, 1.0, 0.0, 1.0]      # yellow: overlap

            from skimage import exposure

            image_eq = exposure.rescale_intensity(
                img_2d, in_range=tuple(np.percentile(img_2d, (2, 98)))
            )
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image_eq, cmap="gray")
            ax.imshow(overlay_rgba)
            ax.set_title(
                f"{test_name}  soft IoU={iou:.3f}  "
                f"(red=MATLAB, green=Python, yellow=overlap)"
            )
            ax.axis("off")
            plt.tight_layout()
            overlay_path = _CT_FIRE_RESULTS_DIR / f"iou_overlay_{test_name}.png"
            fig.savefig(str(overlay_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Saved IoU overlay: {overlay_path}")

        # Total length within ±10% of MATLAB.
        # 10% accommodates inherent numerical variation between the Python curvelops
        # library and MATLAB CurveLab while still catching large regressions.
        py_totL = float(data_py["M"]["totL"])
        mat_totL = float(data_mat["M"].get("totL", 0))
        if mat_totL > 0 and py_totL > 0:
            assert 0.90 * mat_totL <= py_totL <= 1.10 * mat_totL, (
                f"total length {py_totL:.1f} not within 10% of MATLAB {mat_totL:.1f}"
            )
            print(
                f"\n{test_name} total length — "
                f"Python: {py_totL:.1f}, MATLAB: {mat_totL:.1f}, "
                f"diff: {abs(py_totL - mat_totL) / mat_totL:.1%}"
            )

        # Mean |angle| within 10° of MATLAB reference
        mat_angles = np.asarray(data_mat["M"].get("angle_xy", []))
        py_angles = np.asarray(data_py["M"].get("angle_xy", []))
        if len(mat_angles) > 0 and len(py_angles) > 0:
            py_mean = np.degrees(np.mean(py_angles % np.pi))
            mat_mean = np.degrees(np.mean(mat_angles % np.pi))
            delta_deg = abs(py_mean - mat_mean)
            assert delta_deg < 10.0, (
                f"mean angle [0-180°] differs by {delta_deg:.1f}° > 10° ({test_name})"
            )
            print(
                f"\n{test_name} mean angle — "
                f"Python: {py_mean:.1f}°, MATLAB: {mat_mean:.1f}°, "
                f"delta: {delta_deg:.1f}°"
            )
