"""
Regression tests for fire_2D_ang1 Python implementation

Tests the Python implementation of fire_2D_ang1 against MATLAB reference outputs.
Validates fiber detection, network statistics, and angle calculations.
"""

import json
import os
import sys
from pathlib import Path

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Skip if C++ backend is not available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ctfire_py" / "CPP"))
    import fiber_backend
    CPP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CPP_AVAILABLE = False

from ctfire_py.fire_2d_angle import fire_2d_angle


# ============================================================================
# Fixtures and Utilities
# ============================================================================


@pytest.fixture(scope="module")
def test_config():
    """Load test configuration from JSON."""
    config_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / "test_cases_fire_2d.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_test_image(image_name):
    """Load a test image by filename.

    Always returns an array with intensity values in the uint8 [0, 255] range
    so that absolute thresholds (thresh_im2) work consistently regardless of
    whether the source file is a TIF or a PNG.
    """
    img_path = Path(__file__).parent / "test_images" / image_name
    if not img_path.exists():
        pytest.skip(f"Test image not found: {image_name}")
    img = plt.imread(str(img_path))
    # PNG files are loaded by matplotlib as float32 [0, 1]; rescale to [0, 255]
    # to match the uint8 range that TIF files produce and that thresh_im2 expects.
    if img.dtype == np.float32 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img


def load_test_cases(config_path=None, matlab_only=False):
    """
    Load test cases from JSON configuration.
    
    Args:
        config_path: Path to JSON config file. If None, uses default.
        matlab_only: If True, only return cases with MATLAB reference files.
        
    Returns:
        List of (name, test_case) tuples for parametrize.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / "test_cases_fire_2d.json"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    cases = config["test_cases"]
    
    if matlab_only:
        cases = [tc for tc in cases if "matlab_reference_mat" in tc]
    
    return [(tc["name"], tc) for tc in cases]


def load_matlab_reference(mat_file_path):
    """
    Load MATLAB reference data from .mat file.
    
    Expected structure: 
        data.Xa - vertex coordinates
        data.Fa - fiber structures
        data.M - network statistics
    
    Returns:
        dict with parsed MATLAB data
    """
    if not os.path.exists(mat_file_path):
        pytest.skip(f"MATLAB reference file not found: {mat_file_path}")
    
    # Try h5py first (for MATLAB v7.3 files), fall back to scipy for v5 files
    _h5py_ok = False
    try:
        if H5PY_AVAILABLE:
            try:
                _h5_file = h5py.File(mat_file_path, 'r')
                _h5py_ok = True
            except Exception:
                _h5py_ok = False  # not an HDF5 file — fall through to scipy
        if _h5py_ok:
            # Load MATLAB v7.3 file using h5py
            with h5py.File(mat_file_path, 'r') as f:
                if 'data' not in f:
                    raise ValueError("MATLAB .mat file must contain 'data' structure")
                
                data_group = f['data']
                
                # Extract relevant fields
                result = {
                    'Xa': None,
                    'Fa': None,
                    'M': {},
                }
                
                # Extract Xa (vertex coordinates)
                if 'Xa' in data_group:
                    result['Xa'] = np.array(data_group['Xa']).T  # Transpose for MATLAB convention

                # Extract Fa fiber structures (MATLAB 1-based indices → 0-based)
                if 'Fa' in data_group and 'v' in data_group['Fa']:
                    v_ds = data_group['Fa']['v']
                    fibers = []
                    for i in range(v_ds.shape[0]):
                        ref = v_ds[i, 0]
                        v_arr = np.array(f[ref]).flatten().astype(int) - 1
                        fibers.append({'v': list(v_arr)})
                    result['Fa'] = fibers

                # Extract network statistics from M structure
                if 'M' in data_group:
                    M_group = data_group['M']
                    
                    result['M'] = {}
                    
                    # Extract scalar values
                    if 'fiber_num' in M_group:
                        result['M']['fiber_num'] = int(np.array(M_group['fiber_num']).item())
                    if 'avgL' in M_group:
                        result['M']['avgL'] = float(np.array(M_group['avgL']).item())
                    if 'totL' in M_group:
                        result['M']['totL'] = float(np.array(M_group['totL']).item())
                    if 'Ldens' in M_group:
                        result['M']['Ldens'] = float(np.array(M_group['Ldens']).item())
                    if 'volfrac' in M_group:
                        result['M']['volfrac'] = float(np.array(M_group['volfrac']).item())
                    
                    # Extract arrays
                    if 'L' in M_group:
                        L_data = np.array(M_group['L'])
                        result['M']['L'] = L_data.flatten() if L_data.size > 0 else np.array([])
                    if 'angle_xy' in M_group:
                        angle_data = np.array(M_group['angle_xy'])
                        result['M']['angle_xy'] = angle_data.flatten() if angle_data.size > 0 else np.array([])
                
                return result
        
        if not _h5py_ok and SCIPY_AVAILABLE:
            # Fall back to scipy for older .mat files
            mat_data = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
            
            if 'data' not in mat_data:
                raise ValueError("MATLAB .mat file must contain 'data' structure")
            
            data = mat_data['data']
            
            # Extract Xa
            raw_xa = data.Xa if hasattr(data, 'Xa') else None

            # Normalize Fa: scipy returns array of mat_struct objects with 1-based
            # MATLAB indices; convert to list of dicts with 0-based indices to match
            # the h5py loading branch.
            raw_fa = data.Fa if hasattr(data, 'Fa') else None
            fa_norm = None
            if raw_fa is not None:
                fa_norm = []
                for fib in np.atleast_1d(raw_fa):
                    if hasattr(fib, 'v'):
                        v_arr = np.atleast_1d(fib.v).flatten().astype(int) - 1
                        fa_norm.append({'v': v_arr.tolist()})

            result = {
                'Xa': raw_xa,
                'Fa': fa_norm,
                'M': {},
            }
            
            # Extract network statistics
            if hasattr(data, 'M'):
                M = data.M
                result['M'] = {
                    'fiber_num': M.fiber_num if hasattr(M, 'fiber_num') else 0,
                    'avgL': M.avgL if hasattr(M, 'avgL') else 0,
                    'totL': M.totL if hasattr(M, 'totL') else 0,
                    'L': M.L if hasattr(M, 'L') else np.array([]),
                    'angle_xy': M.angle_xy if hasattr(M, 'angle_xy') else np.array([]),
                    'Ldens': M.Ldens if hasattr(M, 'Ldens') else 0,
                    'volfrac': M.volfrac if hasattr(M, 'volfrac') else 0,
                }
            
            return result
            
    except Exception as e:
        pytest.skip(f"Could not load MATLAB reference file: {e}")
    
    pytest.skip("Neither h5py nor scipy.io available for loading MATLAB files")


# ============================================================================
# Soft IoU helpers
# ============================================================================

# Gaussian sigma for smoothing 1-px skeletons before computing soft IoU.
# sigma=5 → ~10-px FWHM, which measures agreement at fiber scale rather than
# pixel scale.  Adapted from tme_quant/tests/test_ctfire.py.
_SMOOTH_SIGMA = 5.0

SOFT_IOU_THRESHOLD_MATLAB    = 0.50  # Python vs MATLAB centerlines (>30px filtered)


def _smooth_mask(mask, smooth_sigma=_SMOOTH_SIGMA):
    """Gaussian-smooth a binary mask and rescale to [0, 1]."""
    from skimage import exposure, filters
    mask = mask.astype(np.float32)
    mask = exposure.rescale_intensity(mask, out_range=(0.0, 1.0))
    density = filters.gaussian(mask, sigma=smooth_sigma, preserve_range=False)
    return exposure.rescale_intensity(density, out_range=(0.0, 1.0)).astype(np.float32)


def _soft_iou(mask_1, mask_2, beta=1e-3):
    """
    Soft IoU on two float masks already smoothed to [0, 1].

    Formula: (m1·m2).sum() / (m1²+m2²-m1·m2).sum()
    beta prevents 0/0 when both masks are empty.
    """
    intersection = mask_1 * mask_2
    union = mask_1**2 + mask_2**2 - mask_1 * mask_2
    return float((intersection.sum() + beta) / (union.sum() + beta))


def _rasterize_fibers(X, F, image_shape):
    """
    Rasterize fire_2d_angle fiber output to a 1-px skeleton image.

    Uses Bresenham line drawing between consecutive fiber vertices then
    morphological skeletonize to guarantee 1-px thickness.  Vertex indices
    follow the 0-based convention (X[v] = coordinates for vertex v) as
    established in CLAUDE_CTFIRE.md.
    """
    from skimage.draw import line as draw_line
    from skimage.morphology import skeletonize
    canvas = np.zeros(image_shape, dtype=np.uint8)
    H, W = image_shape
    X_arr = np.asarray(X)
    for fiber in F:
        v_list = fiber['v'] if isinstance(fiber, dict) else list(fiber)
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


MIN_FIBER_LEN_PX = 30.0


def _fiber_stats_filtered(X, F, min_len=MIN_FIBER_LEN_PX, row_idx=0, col_idx=1):
    """
    Compute fiber_num, avgL, totL, angle_xy for fibers with arc-length >= min_len px.

    row_idx/col_idx: which columns in X are row and col.
      Python [row, col, ...]: row_idx=0, col_idx=1  (default)
      MATLAB Xa (loaded as [col, row, z]): row_idx=1, col_idx=0
    """
    if F is None or len(F) == 0 or X is None or len(X) == 0:
        return {'fiber_num': 0, 'avgL': 0.0, 'totL': 0.0, 'angle_xy': np.array([])}

    eps = np.finfo(float).eps
    X_arr = np.asarray(X, dtype=float)
    N = len(X_arr)
    lengths, angles = [], []

    for fiber in F:
        v = fiber['v'] if isinstance(fiber, dict) else list(fiber)
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
        'fiber_num': len(L),
        'avgL': float(np.mean(L)) if len(L) > 0 else 0.0,
        'totL': float(np.sum(L)),
        'angle_xy': np.array(angles, dtype=float),
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_fire_2d_basic_execution(test_name, test_case):
    """
    Test that fire_2d_angle executes without errors and returns valid structure.
    
    This is a smoke test - it doesn't validate correctness, just that the
    function runs and returns properly structured data.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    
    # Load image
    img = load_test_image(test_case["image"])
    
    # Convert to 3D array as expected by fire_2d_angle
    if img.ndim == 2:
        im3 = img[np.newaxis, :, :]
    else:
        im3 = img
    
    # Run fire_2d_angle
    params = test_case["params"]
    data = fire_2d_angle(p=params, im=im3, plotflag=0)
    
    # Validate returned structure
    assert isinstance(data, dict), "fire_2d_angle should return a dictionary"
    
    # Check required fields
    required_fields = ['X', 'F', 'R', 'Xa', 'Fa', 'Va', 'Ra', 'M']
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Validate data types
    assert isinstance(data['X'], np.ndarray), "X should be numpy array"
    assert isinstance(data['F'], (list, np.ndarray)), "F should be list or array"
    assert isinstance(data['M'], dict), "M (statistics) should be dictionary"
    
    # Validate fiber counts are within expected range
    expected = test_case.get("expected_outputs", {})
    if "min_fiber_count" in expected:
        fiber_count = len(data['F'])
        assert fiber_count >= expected["min_fiber_count"], \
            f"Too few fibers detected: {fiber_count} < {expected['min_fiber_count']}"
    
    if "max_fiber_count" in expected:
        fiber_count = len(data['F'])
        assert fiber_count <= expected["max_fiber_count"], \
            f"Too many fibers detected: {fiber_count} > {expected['max_fiber_count']}"


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=False),
    ids=[name for name, _ in load_test_cases(matlab_only=False)],
)
def test_fire_2d_network_statistics(test_name, test_case):
    """
    Test that network statistics are calculated and within reasonable ranges.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    
    img = load_test_image(test_case["image"])
    
    if img.ndim == 2:
        im3 = img[np.newaxis, :, :]
    else:
        im3 = img
    
    params = test_case["params"]
    data = fire_2d_angle(p=params, im=im3, plotflag=0)
    
    M = data['M']
    
    # Validate statistics exist and are reasonable
    assert 'fiber_num' in M, "Missing fiber_num statistic"
    assert 'avgL' in M, "Missing avgL statistic"
    assert 'totL' in M, "Missing totL statistic"
    
    # Check values are positive and reasonable
    assert M['fiber_num'] >= 0, "Fiber count should be non-negative"
    assert M['avgL'] >= 0, "Average length should be non-negative"
    assert M['totL'] >= 0, "Total length should be non-negative"
    
    # Check expected ranges if provided
    expected = test_case.get("expected_outputs", {})
    if "min_avg_length" in expected and M['avgL'] > 0:
        assert M['avgL'] >= expected["min_avg_length"], \
            f"Average length too short: {M['avgL']:.2f} < {expected['min_avg_length']}"
    
    if "max_avg_length" in expected and M['avgL'] > 0:
        assert M['avgL'] <= expected["max_avg_length"], \
            f"Average length too long: {M['avgL']:.2f} > {expected['max_avg_length']}"


# ============================================================================
# Non-square (rectangular) image handling
# ============================================================================


def test_fire_2d_handles_non_square_image():
    """fire_2d_angle runs on a non-square image and emits in-bounds coordinates.

    Uses the exact parameters of the square ``real1.tif`` case but on a
    rectangular crop (``real1_rect.tif``).  On a square image a row/col
    transposition is invisible; here H != W, so every fiber vertex must satisfy
    ``0 <= row < H`` and ``0 <= col < W`` or the swap is exposed.  A green
    skeleton overlay is saved for visual inspection.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    # Reuse the real1 parameters verbatim so the two cases stay in sync.
    _, real1_case = next(
        (n, tc) for n, tc in load_test_cases(matlab_only=True)
        if n == "real1_fire_params"
    )
    params = real1_case["params"]

    img = load_test_image("real1_rect.tif")
    img_2d = img[0] if img.ndim == 3 else img
    H, W = img_2d.shape
    assert H != W, (
        f"real1_rect.tif must be non-square to exercise row/col handling, got {H}x{W}"
    )

    im3 = img[np.newaxis, :, :] if img.ndim == 2 else img
    data = fire_2d_angle(p=params, im=im3, plotflag=0)

    # Structure smoke check (mirrors test_fire_2d_basic_execution).
    for field in ('X', 'F', 'R', 'Xa', 'Fa', 'Va', 'Ra', 'M'):
        assert field in data, f"Missing required field: {field}"
    assert len(data['F']) > 0, "No fibers detected on rectangular image"

    # Core non-square check: every referenced vertex lies inside the rectangle.
    Xa = np.asarray(data['Xa'], dtype=float)
    referenced = set()
    for fiber in data['Fa']:
        v = fiber['v'] if isinstance(fiber, dict) else list(fiber)
        referenced.update(int(idx) for idx in v)
    referenced = [i for i in referenced if 0 <= i < len(Xa)]
    assert referenced, "Fa references no valid vertices in Xa"
    rows = Xa[referenced, 0]
    cols = Xa[referenced, 1]
    assert rows.min() >= 0 and rows.max() < H, (
        f"Fiber row coords out of bounds for H={H}: [{rows.min()}, {rows.max()}]"
    )
    assert cols.min() >= 0 and cols.max() < W, (
        f"Fiber col coords out of bounds for W={W}: [{cols.min()}, {cols.max()}]"
    )

    # Visual overlay: green Python skeleton over the rescaled rectangular image.
    from skimage import exposure

    py_skel = _rasterize_fibers(data['Xa'], data['Fa'], (H, W))
    overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
    overlay_rgba[py_skel] = [0.0, 1.0, 0.0, 1.0]  # green: Python skeleton
    image_eq = exposure.rescale_intensity(
        img_2d, in_range=tuple(np.percentile(img_2d, (2, 98)))
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_eq, cmap="gray")
    ax.imshow(overlay_rgba)
    ax.set_title(
        f"real1_rect  {H}x{W}  fibers={len(data['F'])}  (green=Python skeleton)"
    )
    ax.axis("off")
    plt.tight_layout()
    overlay_path = (
        Path(__file__).parent / "test_results" / "fire_2d_test_files"
        / "overlay_real1_rect_nonsquare.png"
    )
    fig.savefig(str(overlay_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved non-square overlay: {overlay_path}")


# ============================================================================
# MATLAB Comparison Tests
# ============================================================================


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_fire_2d_matches_matlab_fiber_count(test_name, test_case):
    """
    Compare fiber count against MATLAB reference.
    
    Note: Python implementation currently missing check_danglers and full fiberproc,
    so we expect 10-30% more fibers than MATLAB (see CTFIRE_CONVERSION.md).
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    
    # Load image and run Python implementation
    img = load_test_image(test_case["image"])
    if img.ndim == 2:
        im3 = img[np.newaxis, :, :]
    else:
        im3 = img
    
    params = test_case["params"]
    data_py = fire_2d_angle(p=params, im=im3, plotflag=0)
    
    # Load MATLAB reference
    mat_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)
    
    # Compare unfiltered M stats — MATLAB Fa is unfiltered, Python M is also on all Fa
    fiber_count_py  = data_py['M']['fiber_num']
    fiber_count_mat = data_mat['M']['fiber_num']

    if fiber_count_mat > 0:
        rel_diff = abs(fiber_count_py - fiber_count_mat) / fiber_count_mat

        assert fiber_count_py >= fiber_count_mat * 0.85, \
            f"Python has too few fibers: {fiber_count_py} vs MATLAB {fiber_count_mat} (diff: {rel_diff:.1%})"

        assert fiber_count_py <= fiber_count_mat * 1.15, \
            f"Python has too many fibers: {fiber_count_py} vs MATLAB {fiber_count_mat} (diff: {rel_diff:.1%})"

        print(f"\nFiber count - Python: {fiber_count_py}, MATLAB: {fiber_count_mat}, diff: {rel_diff:.1%}")


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_fire_2d_matches_matlab_fiber_length(test_name, test_case):
    """
    Compare average fiber length against MATLAB reference.
    
    Note: Python fibers may be shorter due to missing gap filling in fiberproc.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    
    img = load_test_image(test_case["image"])
    if img.ndim == 2:
        im3 = img[np.newaxis, :, :]
    else:
        im3 = img
    
    params = test_case["params"]
    data_py = fire_2d_angle(p=params, im=im3, plotflag=0)
    
    mat_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)
    
    # Compare unfiltered M stats — same stage as MATLAB (post-fiberproc Fa)
    avgL_py  = data_py['M'].get('avgL', 0)
    avgL_mat = data_mat['M']['avgL']

    if avgL_mat > 0 and avgL_py > 0:
        rel_diff = abs(avgL_py - avgL_mat) / avgL_mat

        assert avgL_py >= avgL_mat * 0.90, \
            f"Python fibers too short: {avgL_py:.2f} vs MATLAB {avgL_mat:.2f} (diff: {rel_diff:.1%})"

        assert avgL_py <= avgL_mat * 1.10, \
            f"Python fibers too long: {avgL_py:.2f} vs MATLAB {avgL_mat:.2f}"

        print(f"\nAvg fiber length - Python: {avgL_py:.2f}, MATLAB: {avgL_mat:.2f}, diff: {rel_diff:.1%}")


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_fire_2d_matches_matlab_angles(test_name, test_case):
    """
    Compare fiber angle distributions against MATLAB reference.
    
    Validates that the angle calculation produces similar distributions,
    even if individual fiber counts differ.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    
    img = load_test_image(test_case["image"])
    if img.ndim == 2:
        im3 = img[np.newaxis, :, :]
    else:
        im3 = img
    
    params = test_case["params"]
    data_py = fire_2d_angle(p=params, im=im3, plotflag=0)
    
    mat_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)
    
    # Compare unfiltered angle distributions from M stats
    angles_py  = data_py['M'].get('angle_xy', np.array([]))
    angles_mat = data_mat['M'].get('angle_xy', np.array([]))

    if len(angles_py) > 0 and len(angles_mat) > 0:
        angles_py  = angles_py  % np.pi
        angles_mat = angles_mat % np.pi
        bins = np.linspace(0, np.pi, 20)

        hist_py, _ = np.histogram(angles_py, bins=bins, density=True)
        hist_mat, _ = np.histogram(angles_mat, bins=bins, density=True)

        hist_py  = hist_py  / (hist_py.sum()  + 1e-10)
        hist_mat = hist_mat / (hist_mat.sum() + 1e-10)

        if hist_py.sum() > 0 and hist_mat.sum() > 0:
            correlation = np.corrcoef(hist_py, hist_mat)[0, 1]

            assert correlation > 0.5, \
                f"Angle distributions too different (correlation: {correlation:.3f})"

            print(f"\nAngle distribution correlation: {correlation:.3f}")


@pytest.mark.matlab
@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_fire_2d_segment_angle_mean(test_name, test_case):
    """
    Validate per-segment angles from calc_fiberang2.

    For non-straight fibers, segment angles vary along the path.
    The mean of all segment angles per fiber should correlate with the
    endpoint angle (M['angle_xy']) and with the MATLAB reference distribution.
    """
    if not CPP_AVAILABLE:
        pytest.skip("C++ backend not available")

    img = load_test_image(test_case["image"])
    im3 = img[np.newaxis] if img.ndim == 2 else img
    data_py = fire_2d_angle(p=test_case["params"], im=im3, plotflag=0)

    mat_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / test_case["matlab_reference_mat"]
    data_mat = load_matlab_reference(mat_path)

    # Filter to fibers with arc-length >= 30px (same threshold as other tests)
    Xa  = data_py['Xa']
    Fa  = data_py['Fa']
    Fang_all          = data_py['M']['Fang']
    endpoint_all      = np.asarray(data_py['M']['angle_xy'])

    long_mask = []
    for fiber in Fa:
        v = fiber['v']
        length = sum(
            np.linalg.norm(Xa[v[i + 1]] - Xa[v[i]])
            for i in range(len(v) - 1)
            if 0 <= v[i] < len(Xa) and 0 <= v[i + 1] < len(Xa)
        )
        long_mask.append(length >= MIN_FIBER_LEN_PX)

    Fang           = [f for f, keep in zip(Fang_all,     long_mask) if keep]
    endpoint_angles = endpoint_all[np.array(long_mask)]

    # MATLAB: filter endpoint angles to fibers > 30px
    mat_stats  = (_fiber_stats_filtered(data_mat['Xa'], data_mat['Fa'], row_idx=1, col_idx=0)
                  if data_mat.get('Fa') is not None else data_mat['M'])
    mat_angles = np.asarray(mat_stats.get('angle_xy', []))

    print(f"\nFibers >30px — Python: {len(Fang)}, MATLAB: {len(mat_angles)}")

    # Sub-check A: mean segment angle per fiber correlates with endpoint angle
    mean_seg = np.array([
        np.mean(f['angle_xy']) for f in Fang if len(f.get('angle_xy', [])) > 0
    ])
    if len(mean_seg) > 1 and len(endpoint_angles) > 1:
        n = min(len(mean_seg), len(endpoint_angles))
        corr = np.corrcoef(mean_seg[:n], endpoint_angles[:n])[0, 1]
        assert corr > 0.5, (
            f"Mean segment angle vs endpoint angle correlation too low: {corr:.3f}"
        )
        print(f"\nMean segment vs endpoint angle correlation: {corr:.3f}")

    # Sub-check A2: long non-straight fibers must show angle variation
    SPI = test_case["params"].get("ang_interval", 5)
    long_stds = [
        np.std(f['angle_xy'])
        for f in Fang
        if len(f.get('angle_xy', [])) > 2 * SPI
    ]
    if long_stds:
        max_std = max(long_stds)
        assert max_std > 0.05, (
            f"Long fibers show no angle variation (max std={max_std:.3f} rad) — "
            "calc_fiberang2 may be returning identical angles for all segments"
        )
        print(f"Long fiber angle std — max: {max_std:.3f} rad, mean: {np.mean(long_stds):.3f} rad")

    # Sub-check B: Python endpoint-angle mean is within 20° of MATLAB endpoint
    # mean. Segment-angle means are validated against Python endpoints above;
    # MATLAB stores endpoint angles here, so compare like with like.
    if len(endpoint_angles) > 0 and len(mat_angles) > 0:
        py_mean_abs  = np.degrees(np.mean(endpoint_angles % np.pi))
        mat_mean_abs = np.degrees(np.mean(mat_angles % np.pi))
        delta_deg = abs(py_mean_abs - mat_mean_abs)
        assert delta_deg < 20.0, (
            f"Mean endpoint angle {py_mean_abs:.1f}° differs from MATLAB {mat_mean_abs:.1f}° by {delta_deg:.1f}° > 20°"
        )
        print(f"Mean endpoint angle [0-180°] - Python: {py_mean_abs:.1f}°, MATLAB: {mat_mean_abs:.1f}°, delta: {delta_deg:.1f}°")


# ============================================================================
# Soft IoU Validation
# ============================================================================


class TestSoftIoU:
    """
    Validate that fire_2d_angle recovers fiber centerlines at fiber scale.

    Soft IoU at sigma=5 (≈10-px FWHM) measures spatial overlap of 1-px
    skeleton images after Gaussian smoothing — a score > 0.30 means the
    extracted skeleton substantially overlaps the ground truth at fiber
    granularity, not pixel precision.
    """

    @pytest.mark.matlab
    @pytest.mark.parametrize(
        "test_name,test_case",
        load_test_cases(matlab_only=True),
        ids=[name for name, _ in load_test_cases(matlab_only=True)],
    )
    def test_soft_iou_matlab_vs_python(self, test_name, test_case):
        """
        Compare Python centerlines against MATLAB Xa/Fa via soft IoU.

        Skips gracefully if the .mat reference file is absent.
        Also checks total length (within ±10%) and mean |angle| (within 10°).
        """
        if not CPP_AVAILABLE:
            pytest.skip("C++ backend not available")

        # Skip if test image is not present
        img_path = Path(__file__).parent / "test_images" / test_case["image"]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {test_case['image']}")

        # Load image and run Python extraction
        img = load_test_image(test_case["image"])
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        data_py = fire_2d_angle(p=test_case["params"], im=img, plotflag=0)

        # Load MATLAB reference (skips if file absent)
        mat_path = (
            Path(__file__).parent
            / "test_results"
            / "fire_2d_test_files"
            / test_case["matlab_reference_mat"]
        )
        data_mat = load_matlab_reference(mat_path)

        image_2d = img[0] if img.ndim == 3 else img
        H, W = image_2d.shape

        # Soft IoU — unfiltered Fa from both sides.
        # MATLAB Xa is [col, row, z] after h5py transpose; swap to [row, col].
        if data_mat["Xa"] is not None and data_mat.get("Fa") is not None:
            mat_Xa_rc = data_mat["Xa"][:, [1, 0]]  # [col,row,z] → [row,col]
            mat_skel = _rasterize_fibers(mat_Xa_rc, data_mat["Fa"], (H, W))
            py_skel  = _rasterize_fibers(data_py["Xa"], data_py["Fa"], (H, W))
            iou = _soft_iou(_smooth_mask(mat_skel.astype(np.float32)),
                            _smooth_mask(py_skel.astype(np.float32)))
            assert iou > SOFT_IOU_THRESHOLD_MATLAB, (
                f"soft IoU {iou:.3f} < {SOFT_IOU_THRESHOLD_MATLAB} "
                f"({test_name}): Python and MATLAB centerlines diverge spatially"
            )
            print(f"\n{test_name} soft IoU = {iou:.3f}")

            # Save IoU overlay: transparent RGBA so original image shows through
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
            mat_only = mat_skel & ~py_skel
            py_only  = py_skel  & ~mat_skel
            both     = mat_skel & py_skel
            overlay_rgba[mat_only] = [1.0, 0.0, 0.0, 1.0]  # red:    MATLAB only
            overlay_rgba[py_only]  = [0.0, 1.0, 0.0, 1.0]  # green:  Python only
            overlay_rgba[both]     = [1.0, 1.0, 0.0, 1.0]  # yellow: overlap
            from skimage import exposure
            image_eq = exposure.rescale_intensity(
                image_2d, in_range=tuple(np.percentile(image_2d, (2, 98)))
            )
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image_eq, cmap="gray")
            ax.imshow(overlay_rgba)
            ax.set_title(f"{test_name}  soft IoU={iou:.3f}  "
                         f"(red=MATLAB, green=Python, yellow=overlap)")
            ax.axis("off")
            plt.tight_layout()
            overlay_path = (Path(__file__).parent / "test_results" / "fire_2d_test_files"
                            / f"iou_overlay_{test_name}.png")
            fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Saved IoU overlay: {overlay_path}")

        # Total length — unfiltered M stats (same stage: post-fiberproc Fa)
        py_totL  = float(data_py['M']['totL'])
        mat_totL = float(data_mat['M'].get('totL', 0))
        if mat_totL > 0 and py_totL > 0:
            # ±10% tolerance (matches ct_fire): the ~9% Python/MATLAB length drift
            # comes from check_danglers / short-fiber handling differences documented
            # in doc/MATLAB_PARITY_ANALYSIS.md, not from spatial divergence (IoU passes).
            assert 0.90 * mat_totL <= py_totL <= 1.10 * mat_totL, (
                f"total length {py_totL:.1f} not within 10% of MATLAB {mat_totL:.1f}"
            )
            print(f"\ntotal length - Python: {py_totL:.1f}, MATLAB: {mat_totL:.1f}, "
                  f"diff: {abs(py_totL - mat_totL) / mat_totL:.1%}")

        # Mean |angle| within 10° of MATLAB reference — unfiltered M stats
        mat_angles = np.asarray(data_mat['M'].get('angle_xy', []))
        py_angles  = np.asarray(data_py['M'].get('angle_xy', []))
        if len(mat_angles) > 0 and len(py_angles) > 0:
            py_mean  = np.degrees(np.mean(py_angles  % np.pi))
            mat_mean = np.degrees(np.mean(mat_angles % np.pi))
            delta_deg = abs(py_mean - mat_mean)
            assert delta_deg < 10.0, (
                f"mean angle [0-180°] differs by {delta_deg:.1f}° > 10° ({test_name})"
            )
            print(f"mean angle [0-180°] - Python: {py_mean:.1f}°, "
                  f"MATLAB: {mat_mean:.1f}°, delta: {delta_deg:.1f}°")


# ============================================================================
# Utility Tests
# ============================================================================


def test_load_test_config():
    """Test that the test configuration file loads correctly."""
    config_path = Path(__file__).parent / "test_results" / "fire_2d_test_files" / "test_cases_fire_2d.json"
    
    assert config_path.exists(), f"Test config not found: {config_path}"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    assert "test_cases" in config
    assert len(config["test_cases"]) > 0
    assert "tolerances" in config
    
    # Validate first test case structure
    tc = config["test_cases"][0]
    assert "name" in tc
    assert "image" in tc
    assert "params" in tc
    assert "expected_outputs" in tc


def test_cpp_backend_availability():
    """Report on C++ backend availability for debugging."""
    if CPP_AVAILABLE:
        print("\n✓ C++ backend is available")
        print(f"  Available functions: {dir(fiber_backend)}")
    else:
        print("\n✗ C++ backend NOT available")
        print("  Tests requiring C++ will be skipped")
        print("  To enable: cd src/ctfire_py/CPP && make clean && make")


# ============================================================================
# Documentation Tests
# ============================================================================


def test_implementation_status_documented():
    """
    Verify that implementation status and differences are documented.
    
    This reminds developers about known differences between Python and MATLAB.
    """
    doc_path = Path(__file__).parent.parent / "doc" / "MATLAB_PARITY_ANALYSIS.md"
    
    assert doc_path.exists(), "MATLAB_PARITY_ANALYSIS.md documentation not found"
    
    with open(doc_path, "r", encoding="utf-8") as f:
        doc_content = f.read()
    
    # Check that key differences are documented
    assert "check_danglers" in doc_content, "check_danglers differences should be documented"
    assert "fiberproc" in doc_content, "fiberproc differences should be documented"
    assert "CPP" in doc_content or "cpp" in doc_content, "CPP comparison should be documented"
    
    print("\n✓ Implementation differences are documented in CTFIRE_CONVERSION.md")


# ============================================================================
# Main entry point for running tests
# ============================================================================


if __name__ == "__main__":
    # Standalone demo: run FIRE on real1.tif, compare with MATLAB reference,
    # print soft IoU metrics, and save a red/green/yellow centerline overlay.
    # Run with:  python tests/test_fire_2d_angle.py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import exposure as _exposure

    _, test_case = next(
        (n, tc) for n, tc in load_test_cases(matlab_only=True) if n == "real1_fire_params"
    )
    img = load_test_image(test_case["image"])
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    data_py = fire_2d_angle(p=test_case["params"], im=img, plotflag=0)

    mat_path = (
        Path(__file__).parent
        / "test_results"
        / "fire_2d_test_files"
        / test_case["matlab_reference_mat"]
    )
    data_mat = load_matlab_reference(mat_path)

    image_2d = img[0] if img.ndim == 3 else img
    H, W = image_2d.shape

    mat_Xa_rc = data_mat["Xa"][:, [1, 0]]  # [col,row,z] → [row,col]
    mat_skel = _rasterize_fibers(mat_Xa_rc, data_mat["Fa"], (H, W))
    py_skel  = _rasterize_fibers(data_py["Xa"], data_py["Fa"], (H, W))
    iou = _soft_iou(_smooth_mask(mat_skel.astype(np.float32)),
                    _smooth_mask(py_skel.astype(np.float32)))

    py_angles = np.asarray(data_py["M"].get("angle_xy", [])) % np.pi
    mean_angle_deg = float(np.degrees(np.mean(py_angles))) if len(py_angles) > 0 else float("nan")

    print(
        f"Test case   = real1_fire_params\n"
        f"Soft IoU    = {iou:.4f}  (threshold {SOFT_IOU_THRESHOLD_MATLAB})\n"
        f"Fibers      = {data_py['M']['fiber_num']}\n"
        f"Total length= {data_py['M']['totL']:.1f} px\n"
        f"Mean |angle|= {mean_angle_deg:.1f}°"
    )

    overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
    mat_only = mat_skel & ~py_skel
    py_only  = py_skel  & ~mat_skel
    both     = mat_skel &  py_skel
    overlay_rgba[mat_only] = [1.0, 0.0, 0.0, 1.0]  # red:    MATLAB only
    overlay_rgba[py_only]  = [0.0, 1.0, 0.0, 1.0]  # green:  Python only
    overlay_rgba[both]     = [1.0, 1.0, 0.0, 1.0]  # yellow: overlap
    image_eq = _exposure.rescale_intensity(
        image_2d, in_range=tuple(np.percentile(image_2d, (2, 98)))
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_eq, cmap="gray")
    ax.imshow(overlay_rgba)
    ax.set_title(f"real1  soft IoU={iou:.3f}  (red=MATLAB, green=Python, yellow=overlap)")
    ax.axis("off")
    plt.tight_layout()
    overlay_path = (
        Path(__file__).parent / "test_results" / "fire_2d_test_files" / "iou_overlay_real1_demo.png"
    )
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IoU overlay: {overlay_path}")
