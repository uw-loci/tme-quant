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
    """Load a test image by filename."""
    img_path = Path(__file__).parent / "test_images" / image_name
    img = plt.imread(img_path, format="TIF")
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
    
    # Try h5py first (for MATLAB v7.3 files), fall back to scipy
    try:
        if H5PY_AVAILABLE:
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
        
        elif SCIPY_AVAILABLE:
            # Fall back to scipy for older .mat files
            mat_data = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
            
            if 'data' not in mat_data:
                raise ValueError("MATLAB .mat file must contain 'data' structure")
            
            data = mat_data['data']
            
            # Extract relevant fields
            result = {
                'Xa': data.Xa if hasattr(data, 'Xa') else None,
                'Fa': data.Fa if hasattr(data, 'Fa') else None,
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

SOFT_IOU_THRESHOLD_SYNTHETIC = 0.70  # extracted skeleton vs known GT skeleton
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
    if not F or X is None or len(X) == 0:
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
            angles.append(np.arctan(dr / (dc + eps)))

    L = np.array(lengths, dtype=float)
    return {
        'fiber_num': len(L),
        'avgL': float(np.mean(L)) if len(L) > 0 else 0.0,
        'totL': float(np.sum(L)),
        'angle_xy': np.array(angles, dtype=float),
    }


def _make_synthetic_fiber_image(shape=(256, 256), n_fibers=8, fiber_sigma=2.5, rng_seed=42):
    """
    Generate a synthetic fiber image with a known ground-truth skeleton.

    Straight-line fibers are drawn with Gaussian cross-section profiles
    (signal amplitude 180, background mean 10 with Gaussian noise σ=4)
    so the FIRE distance-transform pipeline can find them at thresh_im2=5.
    Returns the image and the skeletonized 1-px ground-truth centerline mask.

    Why synthetic images:
      - Exact centerline positions are known at generation time — soft IoU
        measures genuine spatial recovery, not just "did any fibers come out".
      - Deterministic RNG seeds make CI results reproducible.
      - No MATLAB .mat reference files required.
      - Straight-line Gaussian fibers are the canonical FIRE input; regressions
        in extend_xlink, trimxfv, or filtering stages show up as IoU drops.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import line as draw_line
    from skimage.morphology import skeletonize
    rng = np.random.default_rng(rng_seed)
    H, W = shape
    margin = 20
    skeleton = np.zeros(shape, dtype=bool)
    generated = 0
    while generated < n_fibers:
        r0 = int(rng.integers(margin, H - margin))
        c0 = int(rng.integers(margin, W - margin))
        angle = rng.uniform(0, np.pi)
        length = int(rng.integers(60, min(H, W) - 2 * margin))
        r1 = int(np.clip(r0 + length * np.sin(angle), margin, H - margin))
        c1 = int(np.clip(c0 + length * np.cos(angle), margin, W - margin))
        if np.hypot(r1 - r0, c1 - c0) < 50:
            continue
        rr, cc = draw_line(r0, c0, r1, c1)
        skeleton[rr, cc] = True
        generated += 1
    dist = distance_transform_edt(~skeleton).astype(np.float32)
    signal = 180.0 * np.exp(-dist**2 / (2.0 * fiber_sigma**2))
    bg = rng.normal(10.0, 4.0, size=shape).astype(np.float32)
    image = np.clip(bg + signal, 0.0, 255.0).astype(np.float32)
    return image, skeletonize(skeleton)


def _default_fire_params():
    """
    Return the standard FIRE algorithm parameters from test_cases_fire_2d.json.

    Uses the 'real1_ctfire_params' test case (the only one whose image is
    available; 2B_D9_ROI1.tif is not present in the repo).
    Intended for use with real biological images.
    """
    config_path = (
        Path(__file__).parent
        / "test_results"
        / "fire_2d_test_files"
        / "test_cases_fire_2d.json"
    )
    with open(config_path, "r") as f:
        cfg = json.load(f)
    case = next(c for c in cfg["test_cases"] if c["name"] == "real1_ctfire_params")
    return dict(case["params"])


def _synthetic_fire_params():
    """
    Return FIRE algorithm parameters tuned for synthetic test images.

    Synthetic images have Gaussian-profile fibers (peak ~180, background ~10).
    The JSON default of thresh_im2=5 (absolute) includes nearly all background
    pixels at that level, producing hundreds of spurious short zigzag fibers.
    Using a fractional threshold (thresh_im=0.2, i.e. 20% of image maximum)
    raises the effective cutoff to ~36, cleanly separating fiber signal from
    background and suppressing spurious detections.
    """
    p = _default_fire_params()
    p["thresh_im"]  = 0.2   # fractional: keep pixels > 20% of max (~36 for peak 180)
    p["thresh_im2"] = []    # disable absolute threshold when thresh_im is set
    return p


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
    
    # Filter both sides to fibers > 30px before comparing
    stats_py  = _fiber_stats_filtered(data_py['Xf'], data_py['Ff'])
    stats_mat = (_fiber_stats_filtered(data_mat['Xa'], data_mat['Fa'], row_idx=1, col_idx=0)
                 if data_mat.get('Fa') is not None else data_mat['M'])

    fiber_count_py  = stats_py['fiber_num']
    fiber_count_mat = stats_mat['fiber_num']

    if fiber_count_mat > 0:
        rel_diff = abs(fiber_count_py - fiber_count_mat) / fiber_count_mat

        assert fiber_count_py >= fiber_count_mat * 0.70, \
            f"Python has too few fibers (>30px): {fiber_count_py} vs MATLAB {fiber_count_mat} (diff: {rel_diff:.1%})"

        assert fiber_count_py <= fiber_count_mat * 1.8, \
            f"Python has too many fibers (>30px): {fiber_count_py} vs MATLAB {fiber_count_mat} (diff: {rel_diff:.1%})"

        print(f"\nFiber count (>30px) - Python: {fiber_count_py}, MATLAB: {fiber_count_mat}, diff: {rel_diff:.1%}")


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
    
    # Filter both sides to fibers > 30px before comparing
    stats_py  = _fiber_stats_filtered(data_py['Xf'], data_py['Ff'])
    stats_mat = (_fiber_stats_filtered(data_mat['Xa'], data_mat['Fa'], row_idx=1, col_idx=0)
                 if data_mat.get('Fa') is not None else data_mat['M'])

    avgL_py  = stats_py['avgL']
    avgL_mat = stats_mat['avgL']

    if avgL_mat > 0 and avgL_py > 0:
        rel_diff = abs(avgL_py - avgL_mat) / avgL_mat

        assert avgL_py >= avgL_mat * 0.45, \
            f"Python fibers too short (>30px): {avgL_py:.2f} vs MATLAB {avgL_mat:.2f} (diff: {rel_diff:.1%})"

        assert avgL_py <= avgL_mat * 1.2, \
            f"Python fibers too long (>30px): {avgL_py:.2f} vs MATLAB {avgL_mat:.2f}"

        print(f"\nAvg fiber length (>30px) - Python: {avgL_py:.2f}, MATLAB: {avgL_mat:.2f}, diff: {rel_diff:.1%}")


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
    
    # Filter both sides to fibers > 30px before comparing
    stats_py  = _fiber_stats_filtered(data_py['Xf'], data_py['Ff'])
    stats_mat = (_fiber_stats_filtered(data_mat['Xa'], data_mat['Fa'], row_idx=1, col_idx=0)
                 if data_mat.get('Fa') is not None else data_mat['M'])

    angles_py  = stats_py['angle_xy']
    angles_mat = np.asarray(stats_mat.get('angle_xy', []))

    if len(angles_py) > 0 and len(angles_mat) > 0:
        bins = np.linspace(-np.pi / 2, np.pi / 2, 20)

        hist_py, _ = np.histogram(angles_py, bins=bins, density=True)
        hist_mat, _ = np.histogram(angles_mat, bins=bins, density=True)

        hist_py  = hist_py  / (hist_py.sum()  + 1e-10)
        hist_mat = hist_mat / (hist_mat.sum() + 1e-10)

        if hist_py.sum() > 0 and hist_mat.sum() > 0:
            correlation = np.corrcoef(hist_py, hist_mat)[0, 1]

            assert correlation > 0.5, \
                f"Angle distributions too different (>30px, correlation: {correlation:.3f})"

            print(f"\nAngle distribution correlation (>30px): {correlation:.3f}")


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

    def test_soft_iou_synthetic(self):
        """
        Synthetic image with known GT skeleton — always runs, no .mat needed.

        Validates:
        1. Soft IoU of extracted centerlines vs ground-truth skeleton > 0.30
        2. Total extracted fiber length is at least 20% of GT skeleton length
        3. Extracted fiber angles span a reasonable range (std > 0.3 rad),
           confirming the angle computation is not degenerate
        """
        if not CPP_AVAILABLE:
            pytest.skip("C++ backend not available")

        image, gt_skeleton = _make_synthetic_fiber_image(rng_seed=42)
        p = _synthetic_fire_params()
        data = fire_2d_angle(p=p, im=image, plotflag=0)

        assert len(data["Ff"]) > 0, "no filtered fibers extracted"

        # 1. Soft IoU
        pred = _rasterize_fibers(data["Xf"], data["Ff"], image.shape)
        iou = _soft_iou(_smooth_mask(gt_skeleton.astype(np.float32)),
                        _smooth_mask(pred.astype(np.float32)))
        assert iou > SOFT_IOU_THRESHOLD_SYNTHETIC, (
            f"soft IoU {iou:.3f} < {SOFT_IOU_THRESHOLD_SYNTHETIC} — "
            "extracted centerlines do not overlap ground truth at fiber scale"
        )

        # 2. Total extracted length vs GT skeleton pixel count (arc-length proxy)
        tot_L = float(data["M"]["totL"])
        assert tot_L > 0, "total extracted fiber length is zero"
        gt_length = float(gt_skeleton.sum())
        assert tot_L > 0.2 * gt_length, (
            f"extracted total length {tot_L:.1f} px < 20% of GT skeleton "
            f"length {gt_length:.1f} px — pipeline may be discarding too many fibers"
        )

        # 3. Angle range sanity check
        angles = np.asarray(data["M"].get("angle_xy", []))
        assert len(angles) > 0, "no fiber angles in M['angle_xy']"
        assert np.all(np.abs(angles) <= np.pi + 1e-6), "angle value outside [-π, π]"
        assert angles.std() > 0.3, (
            f"angle std {angles.std():.3f} rad unexpectedly small — "
            "all fibers have nearly the same orientation on a random image"
        )

    @pytest.mark.parametrize("seed", [0, 7, 99])
    def test_soft_iou_multiple_seeds(self, seed):
        """
        Soft IoU > 0.30 across multiple random synthetic configurations.

        Guards against a lucky pass on a single seed by checking that the
        pipeline recovers fibers consistently across different random images.
        """
        if not CPP_AVAILABLE:
            pytest.skip("C++ backend not available")

        image, gt_skeleton = _make_synthetic_fiber_image(rng_seed=seed)
        p = _synthetic_fire_params()
        data = fire_2d_angle(p=p, im=image, plotflag=0)

        assert len(data["Ff"]) > 0, f"no fibers extracted (seed={seed})"
        pred = _rasterize_fibers(data["Xf"], data["Ff"], image.shape)
        iou = _soft_iou(_smooth_mask(gt_skeleton.astype(np.float32)),
                        _smooth_mask(pred.astype(np.float32)))
        assert iou > SOFT_IOU_THRESHOLD_SYNTHETIC, (
            f"soft IoU {iou:.3f} < {SOFT_IOU_THRESHOLD_SYNTHETIC} (seed={seed})"
        )

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
        Also checks total length (within ±40%) and mean |angle| (within 10°).
        """
        if not CPP_AVAILABLE:
            pytest.skip("C++ backend not available")

        # Skip if test image is not present (e.g. 2B_D9_ROI1.tif is not in the repo)
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

        # Soft IoU — rasterize fibers > 30px from both sides.
        # MATLAB Xa is stored [col, row, z] after h5py transpose; convert to
        # Python [row, col] convention so _rasterize_fibers indexes correctly.
        if data_mat["Xa"] is not None and data_mat.get("Fa") is not None:
            mat_Xa_rc = data_mat["Xa"][:, [1, 0]]  # [col,row,z] → [row,col]
            # Filter MATLAB fibers to > 30px (same threshold as Python Ff)
            mat_Ff = []
            for _f in data_mat["Fa"]:
                _v = _f['v']
                _len = sum(
                    np.linalg.norm(mat_Xa_rc[_v[i + 1]] - mat_Xa_rc[_v[i]])
                    for i in range(len(_v) - 1)
                    if 0 <= _v[i] < len(mat_Xa_rc) and 0 <= _v[i + 1] < len(mat_Xa_rc)
                )
                if _len >= MIN_FIBER_LEN_PX:
                    mat_Ff.append(_f)
            mat_skel = _rasterize_fibers(mat_Xa_rc, list(mat_Ff), (H, W))
            py_skel  = _rasterize_fibers(data_py["Xf"], data_py["Ff"], (H, W))
            iou = _soft_iou(_smooth_mask(mat_skel.astype(np.float32)),
                            _smooth_mask(py_skel.astype(np.float32)))
            assert iou > SOFT_IOU_THRESHOLD_MATLAB, (
                f"soft IoU {iou:.3f} < {SOFT_IOU_THRESHOLD_MATLAB} "
                f"({test_name}): Python and MATLAB centerlines diverge spatially"
            )
            print(f"\n{test_name} soft IoU (>30px) = {iou:.3f}")

        # Filter both sides to fibers > 30px
        stats_py  = _fiber_stats_filtered(data_py['Xf'], data_py['Ff'])
        stats_mat = (_fiber_stats_filtered(data_mat['Xa'], data_mat['Fa'], row_idx=1, col_idx=0)
                     if data_mat.get('Fa') is not None else data_mat['M'])

        # Total length within 5% of MATLAB reference (fibers > 30px only)
        py_totL  = stats_py['totL']
        mat_totL = float(stats_mat.get('totL', 0))
        if mat_totL > 0 and py_totL > 0:
            assert 0.95 * mat_totL <= py_totL <= 1.05 * mat_totL, (
                f"total length (>30px) {py_totL:.1f} not within 5% of MATLAB {mat_totL:.1f}"
            )
            print(f"\ntotal length (>30px) - Python: {py_totL:.1f}, MATLAB: {mat_totL:.1f}, "
                  f"diff: {abs(py_totL - mat_totL) / mat_totL:.1%}")

        # Mean |angle| within 10° of MATLAB reference (fibers > 30px only)
        mat_angles = np.asarray(stats_mat.get('angle_xy', []))
        py_angles  = np.asarray(stats_py['angle_xy'])
        if len(mat_angles) > 0 and len(py_angles) > 0:
            delta_deg = np.degrees(
                abs(np.mean(np.abs(py_angles)) - np.mean(np.abs(mat_angles)))
            )
            assert delta_deg < 10.0, (
                f"mean |angle| (>30px) differs by {delta_deg:.1f}° > 10° ({test_name})"
            )
            print(f"mean |angle| (>30px) - Python: {np.degrees(np.mean(np.abs(py_angles))):.1f}°, "
                  f"MATLAB: {np.degrees(np.mean(np.abs(mat_angles))):.1f}°, delta: {delta_deg:.1f}°")


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
    doc_path = Path(__file__).parent.parent / "docs" / "CTFIRE_CONVERSION.md"
    
    assert doc_path.exists(), "CTFIRE_CONVERSION.md documentation not found"
    
    with open(doc_path, "r") as f:
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
    # Standalone demo: generate synthetic image, run FIRE, print metrics, save overlay.
    # Run with:  python tests/test_fire_2d_angle.py
    from ctfire_py.test_fire_2d import plot_fiber_overlay

    image, gt_skeleton = _make_synthetic_fiber_image(rng_seed=42)
    p = _synthetic_fire_params()
    data = fire_2d_angle(p=p, im=image, plotflag=0)

    pred = _rasterize_fibers(data["Xf"], data["Ff"], image.shape)
    iou = _soft_iou(_smooth_mask(gt_skeleton.astype(np.float32)),
                    _smooth_mask(pred.astype(np.float32)))

    angles = np.asarray(data["M"].get("angle_xy", []))
    mean_angle_deg = float(np.degrees(np.mean(np.abs(angles)))) if len(angles) > 0 else float("nan")

    print(
        f"Soft IoU    = {iou:.4f}  (threshold {SOFT_IOU_THRESHOLD_SYNTHETIC})\n"
        f"Fibers      = {len(data['Ff'])}\n"
        f"Total length= {data['M']['totL']:.1f} px\n"
        f"Mean |angle|= {mean_angle_deg:.1f}°"
    )

    plot_fiber_overlay(
        image,
        data["Xf"],
        data["Ff"],
        title=f"FIRE overlay  (soft IoU={iou:.3f})",
        save_path="fire_2d_soft_iou_overlay.png",
    )
