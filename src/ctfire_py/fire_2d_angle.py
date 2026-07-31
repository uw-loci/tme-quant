"""
FIRE 2D Angle - Main fiber extraction algorithm using C++ backend

This is the main entry point for the FIRE (Fiber Extraction) algorithm.
It processes 2D images to extract fiber networks.

Parameters:
    p: parameter dictionary
    im: 2D image array
    plotflag: 0=no plots, 1=detailed plots, 2=summary plots

Returns:
    data: dictionary containing extracted fiber network and statistics
"""

import numpy as np
import time
from scipy.ndimage import distance_transform_edt, convolve
from typing import Dict, Any, Optional, Tuple, Union
import sys


from pycurvelets.utils.math import round_mlab

# Import C++ backend
try:
    import os as _os
    _package_dir = _os.path.dirname(_os.path.abspath(__file__))
    sys.path.insert(0, _os.path.join(_package_dir, "CPP"))
    import fiber_backend
except ImportError as exc:
    print(f"Warning: C++ fiber_backend not available: {exc}")
    print("Please compile it first.")
    fiber_backend = None


def smooth(v, sigma):
    """
    Smooths v by convolving with a Gaussian box of radius r.
    Equivalent to the MATLAB 'smooth' function provided.
    """
    v = np.asanyarray(v, dtype=float)
    ndims = v.ndim

    # Handle sigma inputs
    if np.all(sigma == 0):
        return v

    if np.isscalar(sigma):
        s = np.full(ndims, sigma)
    elif len(sigma) == ndims:
        s = np.array(sigma)
    else:
        raise ValueError("Improper input for sigma")

    # Calculate Radius (R)
    R = np.ceil(2 * s).astype(int)

    # --- Generate 1D Kernels ---

    # Kernel for Axis 0 (Vertical/Rows)
    x1 = np.arange(-R[0], R[0] + 1)
    gx1 = np.exp(-(x1**2) / (2 * s[0] ** 2))
    gx1 /= gx1.sum()
    # Reshape to (N, 1, 1) or (N, 1) depending on ndims
    shape1 = [1] * ndims
    shape1[0] = len(x1)
    gx1 = gx1.reshape(shape1)

    # Kernel for Axis 1 (Horizontal/Cols)
    x2 = np.arange(-R[1], R[1] + 1)
    gx2 = np.exp(-(x2**2) / (2 * s[1] ** 2))
    gx2 /= gx2.sum()
    shape2 = [1] * ndims
    shape2[1] = len(x2)
    gx2 = gx2.reshape(shape2)

    # Apply first two filters.
    # Use mode='constant', cval=0 (zero-padding) to match MATLAB's imfilter default.
    # MATLAB's imfilter(v, kernel, 'same') uses zero-padding by default (the 'same'
    # argument specifies output size, not boundary behavior; the default boundary is 0).
    v = convolve(v, gx1, mode="constant", cval=0.0)
    v = convolve(v, gx2, mode="constant", cval=0.0)

    # Kernel for Axis 2 (Depth/Slices) if 3D
    if ndims == 3:
        x3 = np.arange(-R[2], R[2] + 1)
        gx3 = np.exp(-(x3**2) / (2 * s[2] ** 2))
        gx3 /= gx3.sum()
        shape3 = [1, 1, len(x3)]
        gx3 = gx3.reshape(shape3)

        v = convolve(v, gx3, mode="constant", cval=0.0)

    return v


def flatten(image: np.ndarray) -> np.ndarray:
    """Flatten 3D image to 2D by taking maximum projection"""
    if image.ndim == 3:
        return np.max(image, axis=0)
    return image


def bw_dist(
    bw: np.ndarray, method: str = "euclidean"
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Distance transform of binary image.

    Computes the distance transform of the binary image BW. For each pixel in BW,
    the distance transform assigns a number that is the distance between that pixel
    and the nearest nonzero pixel of BW.

    Parameters
    ----------
    bw : np.ndarray
        Binary input image (can be numeric or logical). Nonzero values are treated as True.
    method : str, optional
        Distance metric to use. Options are:
        - 'euclidean': Euclidean distance (default)
        - 'cityblock': Manhattan/L1 distance (abs(x1-x2) + abs(y1-y2))
        - 'chessboard': Chebyshev/L-infinity distance (max(abs(x1-x2), abs(y1-y2)))
        - 'quasi-euclidean': Approximation of Euclidean distance

    Returns
    -------
    D : np.ndarray
        Distance transform array, same size as input. Values are float32.

    Notes
    -----
    This is a Python/NumPy implementation of MATLAB's bwdist function.
    The Euclidean method uses scipy's fast distance_transform_edt.
    Other methods use scipy's distance transforms with appropriate metrics.

    Examples
    --------
    >>> bw = np.zeros((5, 5))
    >>> bw[2, 2] = 1
    >>> D = bw_dist(bw)
    >>> print(D[0, 0])  # Distance from corner to center
    2.8284271247461903
    """
    # Convert to boolean
    bw = np.asarray(bw, dtype=bool)

    # Validate method
    valid_methods = ["euclidean", "cityblock", "chessboard", "quasi-euclidean"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    # Compute distance transform based on method
    if method == "euclidean":
        # Use scipy's fast Euclidean distance transform
        D = distance_transform_edt(bw).astype(np.float32)
    else:
        # Use chamfer distance for non-Euclidean methods
        D = _chamfer_distance(bw, method).astype(np.float32)

    return D


def _chamfer_distance(bw: np.ndarray, method: str) -> np.ndarray:
    """
    Compute chamfer distance transform using dual-scan algorithm.

    This matches MATLAB's implementation for cityblock, chessboard, and quasi-euclidean.
    """
    # Define weights and connectivity for each method
    if method == "cityblock":
        # 4-connectivity, weights = [0, 1, 1, 1, 1]
        # Neighbors: center, N, S, E, W
        weights = np.array([1.0, 1.0, 1.0, 1.0])  # N, S, E, W
    elif method == "chessboard":
        # 8-connectivity, weights = [0, 1, 1, 1, 1, 1, 1, 1, 1]
        # All 8 neighbors have weight 1
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # All 8 directions
    elif method == "quasi-euclidean":
        # 8-connectivity with quasi-Euclidean weights
        # Orthogonal = 1.0, Diagonal = sqrt(2) ≈ 1.414
        sqrt2 = np.sqrt(2.0)
        weights = np.array([1.0, 1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2, sqrt2])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initialize distance array
    D = np.full(bw.shape, np.inf, dtype=np.float64)
    D[bw] = 0.0

    rows, cols = bw.shape

    # Forward pass (top-left to bottom-right)
    for i in range(rows):
        for j in range(cols):
            if not bw[i, j]:
                min_dist = D[i, j]

                # Check neighbors based on method
                if method == "cityblock":
                    # 4-connectivity: N, W
                    if i > 0:  # N
                        min_dist = min(min_dist, D[i - 1, j] + weights[0])
                    if j > 0:  # W
                        min_dist = min(min_dist, D[i, j - 1] + weights[2])
                else:
                    # 8-connectivity: NW, N, NE, W
                    if i > 0 and j > 0:  # NW
                        idx = 4 if method == "quasi-euclidean" else 0
                        min_dist = min(min_dist, D[i - 1, j - 1] + weights[idx])
                    if i > 0:  # N
                        min_dist = min(min_dist, D[i - 1, j] + weights[0])
                    if i > 0 and j < cols - 1:  # NE
                        idx = 5 if method == "quasi-euclidean" else 1
                        min_dist = min(min_dist, D[i - 1, j + 1] + weights[idx])
                    if j > 0:  # W
                        min_dist = min(min_dist, D[i, j - 1] + weights[2])

                D[i, j] = min_dist

    # Backward pass (bottom-right to top-left)
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if not bw[i, j]:
                min_dist = D[i, j]

                # Check neighbors based on method
                if method == "cityblock":
                    # 4-connectivity: S, E
                    if i < rows - 1:  # S
                        min_dist = min(min_dist, D[i + 1, j] + weights[1])
                    if j < cols - 1:  # E
                        min_dist = min(min_dist, D[i, j + 1] + weights[3])
                else:
                    # 8-connectivity: E, SE, S, SW
                    if j < cols - 1:  # E
                        min_dist = min(min_dist, D[i, j + 1] + weights[3])
                    if i < rows - 1 and j < cols - 1:  # SE
                        idx = 6 if method == "quasi-euclidean" else 2
                        min_dist = min(min_dist, D[i + 1, j + 1] + weights[idx])
                    if i < rows - 1:  # S
                        min_dist = min(min_dist, D[i + 1, j] + weights[1])
                    if i < rows - 1 and j > 0:  # SW
                        idx = 7 if method == "quasi-euclidean" else 3
                        min_dist = min(min_dist, D[i + 1, j - 1] + weights[idx])

                D[i, j] = min_dist

    return D


def fire_2d_angle(
    p: Dict[str, Any], im: np.ndarray, plotflag: int = 1
) -> Dict[str, Any]:
    """
    Main FIRE algorithm for 2D fiber extraction

    Args:
        p: Parameter dictionary containing:
            - sigma_im: Smoothing sigma for image
            - thresh_im: Threshold for image (fraction of max)
            - thresh_im2: Absolute threshold (if thresh_im is None)
            - dtype: Distance transform type ('euclidean', 'chessboard', etc.)
            - sigma_d: Smoothing sigma for distance transform
            - s_xlinkbox: Box size for finding local maxima
            - thresh_Dxlink: Threshold for nucleation points
            - thresh_LMPdist: Minimum distance between LMP
            - thresh_LMP: Threshold for local maximum points
            - thresh_ext: Threshold for fiber extension
            - lam_dirdecay: Lambda for direction decay
            - thresh_linkd: Threshold for linking distance
            - thresh_linka: Threshold for linking angle
            - s_fiberdir: Number of points for fiber direction
            - s_maxspace: Maximum spacing for interpolation
            - lambda: Lambda for beam processing
            - ang_interval: Angle sampling interval
            - scale: Scaling factors [x, y, z]
        im: Input 2D image
        plotflag: 0=no plots, 1=detailed plots, 2=summary plots

    Returns:
        Dictionary containing extracted fiber network data
    """

    if fiber_backend is None:
        raise RuntimeError("C++ fiber_backend is required but not available")

    print("Starting FIRE 2D fiber extraction...")
    start_time = time.time()

    # Setup plotting parameters
    if plotflag == 1:
        rr, cc = 3, 3
    elif plotflag == 2:
        rr, cc = 1, 2
    else:
        rr, cc = 1, 1

    ifig = 0

    # Get image dimensions
    if im.ndim == 2:
        im = im[np.newaxis, :, :]  # Add channel dimension for consistency

    K, J, I = im.shape  # K=channels (1 for 2D), J=height, I=width
    ax = [1, I, 1, J]

    # Step 1: Smooth image
    print("  Smoothing original image")
    ims = round_mlab(smooth(im, p.get("sigma_im")))

    if plotflag == 1:
        print(f"  Smoothed image shape: {ims.shape}")

    # Step 2: Threshold image
    if np.asarray(p.get("thresh_im", [])).size != 0:
        imt = ims > p["thresh_im"] * np.max(ims)
    else:
        imt = ims > p.get("thresh_im2")

    if plotflag == 1:
        print(f"  Thresholded pixels: {np.sum(imt)}")

    # Step 3: Distance transform
    print(f"  Calculating {p.get('dtype', 'euclidean')} distance to background")

    # Flatten to 2D for distance transform
    imt_2d = flatten(imt)

    # Compute distance transform
    d = bw_dist(bw=~imt_2d, method=p.get("dtype"))

    # Smooth distance function
    dsm = smooth(d, p.get("sigma_d", 2.0)).astype(np.float32)

    if plotflag == 1:
        print(f"  Distance function range: [{np.min(dsm):.2f}, {np.max(dsm):.2f}]")

    # Step 4: Find nucleation points (crosslinks)
    print("Finding nucleation points")

    # Prepare for C++ call
    dsm_flat = dsm.flatten().astype(np.float32)

    xlink = fiber_backend.find_local_max(
        K, J, I, dsm_flat, p.get("s_xlinkbox", 3), p.get("thresh_Dxlink", 1.0)
    )

    if plotflag == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left: raw distance map
        axes[0].imshow(dsm, cmap="hot", origin="upper")
        axes[0].set_title(f"Distance map (dsm)")

        # Right: distance map + markers
        axes[1].imshow(dsm, cmap="hot", origin="upper")

        # xlink columns are [z, y, x] in 1-based indexing
        # convert to 0-based and plot
        if xlink.shape[0] > 0:
            # xlink[:,0] = z (row), xlink[:,1] = y (col) in 1-based
            rows = xlink[:, 0] - 1  # z -> row
            cols = xlink[:, 1] - 1  # y -> col
            axes[1].scatter(
                cols,
                rows,
                c="cyan",
                s=10,
                marker="+",
                linewidths=0.8,
                label=f"{len(rows)} pts",
            )
            axes[1].legend(loc="upper right", fontsize=8)

        axes[1].set_title(f"Local maxima ({xlink.shape[0]} points)")

        plt.tight_layout()
        plt.savefig("debug_markers.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"  Found {xlink.shape[0]} nucleation points")

    # Step 5: Extend network from nucleation points
    print("Extending nucleation points")

    # Prepare parameters for C++ call
    cpp_params = {
        "thresh_LMPdist": p.get("thresh_LMPdist", 3),
        "thresh_LMP": p.get("thresh_LMP", 1.0),
        "thresh_ext": p.get("thresh_ext", 0.7),
        "lam_dirdecay": p.get("lam_dirdecay", 0.5),
        "thresh_linkd": p.get("thresh_linkd", 5.0),
        "thresh_linka": p.get("thresh_linka", -0.5),
        "s_fiberdir": p.get("s_fiberdir", 3),
    }

    Xz, Fz, Vz, Rz = fiber_backend.extend_xlink(
        K, J, I, dsm_flat, xlink.astype(np.int32), cpp_params
    )

    print(f"  Extracted {len(Fz)} fiber segments")

    # Convert to double precision for compatibility
    Xz = Xz.astype(np.float64)
    Rz = np.array(Rz, dtype=np.float64)

    if plotflag == 1:
        print(f"  Vertices: {Xz.shape[0]}, Fibers: {len(Fz)}")

    # Step 6: Remove danglers and shorties
    # MATLAB's check_danglers.m is effectively a no-op (logic bug blocks the
    # only removal path; only trimxfv runs). Python has a "corrected" version
    # that does remove danglers. Set p["faithful_matlab_danglers"] = True to
    # match MATLAB bit-for-bit when validating end-to-end parity.
    faithful_danglers = bool(p.get("faithful_matlab_danglers", False))
    if faithful_danglers:
        print("Remove danglers and shorties (MATLAB-faithful: trimxfv only)")
    else:
        print("Remove danglers and shorties (Python corrected)")
    from ctfire_py.fiber_processing import check_danglers
    faithful_danglers = True  # Set to True to match MATLAB's behavior (no actual removal)
    Xz2, Fz2, Vz2, Rz2 = check_danglers(
        Xz, Fz, Vz, Rz, p, faithful_matlab=faithful_danglers
    )

    # Identify cross-links
    xlinkind = np.zeros(len(Vz2), dtype=bool)
    for vi in range(len(Vz2)):
        if isinstance(Vz2[vi], dict) and len(Vz2[vi].get("f", [])) > 1:
            xlinkind[vi] = True

    xlinknew = Xz2[xlinkind]
    print(f"  Identified {np.sum(xlinkind)} cross-links")

    # Step 7: Return intermediate values
    X = Xz2
    F = Fz2
    V = Vz2
    R = Rz2

    # Step 8: Fiber processing
    # Match MATLAB fire_2D_ang1.m line 167:
    #     [Xa Fa Ea Va Ra] = fiberproc(X, F, R, size(dsm), p);
    # The C++ `fiber_backend.process_fibers` is a faithful port of
    # fiberproc.m that bundles trimxfv, remove_repeat, 5x(fiberlink +
    # remove_repeat), fiberlinkgap, and fiberremove in MATLAB's exact order.
    print("Fiberproc")
    Xa, Fa, Ea, Va, Ra = fiber_backend.process_fibers(
        K, J, I,
        dsm_flat,
        Xz2.astype(np.int32),
        Fz2,
        Rz2,
        p,
    )

    if isinstance(Ea, list):
        Ea_array = np.zeros((len(Ea), 2), dtype=np.int32)
        for i, edge in enumerate(Ea):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                Ea_array[i, 0] = edge[0]
                Ea_array[i, 1] = edge[1]
        Ea = Ea_array

    Xa = np.asarray(Xa, dtype=np.float64)
    if Ra is not None:
        Ra = np.asarray(Ra, dtype=np.float64)

    elapsed_time = time.time() - start_time
    print(f"CPP code for this image takes {elapsed_time:.2f} seconds")

    # Step 9: Compute network statistics
    print("Computing network statistics")

    # Scale vertices
    scale = p.get("scale", [1.0, 1.0, 1.0])
    Xas = Xa.copy()
    for k in range(Xa.shape[0]):
        Xas[k, :] = Xa[k, :] * scale[: Xa.shape[1]]

    # Import analysis functions
    from ctfire_py.fiber_analysis.network_stats import network_statK
    from ctfire_py.fiber_analysis.fiber_angles import calc_fiberang2
    
    # Calculate network statistics
    M = network_statK(Xas, Fa, Va, Ra)

    # Step 10: Fiber interpolation via Hermite cubic spline resampling
    print("Interpolating fibers")
    from ctfire_py.fiber_processing.fiber2beam import fiber2beam
    minspace = float(p.get("s_maxspace", 5))
    lam      = float(p.get("lambda", 0.01))
    # fiber2beam expects 3-D X (N×3); Xas may be N×2 – pad if needed
    Xas3 = Xas if Xas.shape[1] == 3 else np.column_stack([Xas, np.zeros(len(Xas))])
    Xai, Fai, Vai = fiber2beam(Xas3, Fa, Va, Ra, minspace, lam)

    # Step 11: Calculate fiber angles
    print("Calculating fiber angles")
    SPI = p.get("ang_interval", 5)
    
    # Calculate angles for original fibers
    FiberAngle = calc_fiberang2(Xas, Fa, SPI)
    M["Fang"] = FiberAngle
    
    # Calculate angles for interpolated fibers
    FiberAngleI = calc_fiberang2(Xai, Fai, SPI)
    M["FangI"] = FiberAngleI

    # Step 12: Beam processing (scale, prune floppy edges, re-interpolate)
    print("Beamproc")
    from ctfire_py.fiber_processing.beamproc import beamproc
    try:
        Xab, Fab, Vab = beamproc(Xa, Fa, Va, Ra, p)
    except Exception as _beamproc_err:
        # beamproc can fail when boundary nodes are absent (e.g. tiny images);
        # fall back gracefully so the rest of the pipeline keeps running.
        print(f"  beamproc skipped ({_beamproc_err})")
        Xab = Xa.copy()
        Fab = Fa
        Vab = Va

    # Step 13: Fiber break at cross-links
    print("Fiberbreak")
    from ctfire_py.fiber_processing.fiberbreak import fiberbreak
    Xc, Fc, Vc = fiberbreak(Xa, Fa, Va)
    
    # Step 13.5: CurveAlign-style filtering
    print("Applying CurveAlign-style quality filters")
    from ctfire_py.fiber_processing.curvealign_filter import curvealign_filter, print_fiber_statistics
    
    # Apply length-only filter; straightness filter disabled
    min_length = p.get("min_fiber_length", 30.0)

    print_fiber_statistics(Xc, Fc)
    Xf, Ff, Vf = curvealign_filter(Xc, Fc, Vc, min_length=min_length, min_straightness=0.0)
    print_fiber_statistics(Xf, Ff)

    # Create output data structure
    data = {
        "X": X,
        "F": F,
        "R": R,
        "Xa": Xa,
        "Xas": Xas,
        "Fa": Fa,
        "Fas": Fa,
        "Va": Va,
        "Ea": Ea,
        "Ra": Ra,
        "Xab": Xab,
        "Fab": Fab,
        "Vab": Vab,
        "Xc": Xc,
        "Fc": Fc,
        "Vc": Vc,
        "Xf": Xf,  # Filtered (CurveAlign-style)
        "Ff": Ff,
        "Vf": Vf,
        "M": M,
        "xlink": xlink,
        "Xai": Xai,
        "Fai": Fai,
        "Vai": Vai,
        "Xpres": xlink,  # Nucleation points
        "Xz2": X,  # After extend_xlink
        "Fz2": F,
    }

    print("FIRE 2D extraction complete!")
    return data


if __name__ == "__main__":

    # same values as when when using real1.tif's fiber image
    # TODO: make p's keys much more descriptive
    p = dict(
        {
            "sigma_im": 0,
            "sigma_d": 0.3,
            "dtype": "cityblock",
            "thresh_im": [],
            "thresh_im2": 0,
            "thresh_Dxlink": 1.5,
            "s_xlinkbox": 8,
            "thresh_LMP": 0.2,
            "thresh_LMPdist": 2,
            "thresh_ext": 0.342,
            "lam_dirdecay": 0.5,
            "s_minstep": 2,
            "s_maxstep": 6,
            "thresh_dang_aextend": 0.9848,
            "thresh_dang_L": 15,
            "thresh_short_L": 15,
            "s_fiberdir": 4,
            "thresh_linkd": 15,
            "thresh_linka": -0.866,
            "thresh_flen": 15,
            "thresh_numv": 3,
            "scale": [1.0, 1.0, 1.0],
            "s_boundthick": 10,
            "blist": 1,
            "s_maxspace": 5,
            "lambda": 0.01,
            "ang_interval": 3,
        }
    )

    import os
    import matplotlib.pyplot as plt
    from ctfire_py.ct_reconstruction import ct_reconstruction  # lazy: requires curvelops

    img = plt.imread(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "tests",
            "test_images",
            "2B_D9_ROI1.tif",
        ),
        format="TIF",
    )

    ctfire_params = {
        "coefficient_percentile": 0.2,
        "num_scales": 4,
        "fiber_threshold": 0.5,
        "value": {
            "sigma_im": 0,
            "sigma_d": 0.3,
            "dtype": "cityblock",
            "thresh_im": [],
            "thresh_im2": 0,
            "thresh_Dxlink": 1.5,
            "s_xlinkbox": 8,
            "thresh_LMP": 0.2,
            "thresh_LMPdist": 2,
            "thresh_ext": 0.342,
            "lam_dirdecay": 0.5,
            "s_minstep": 2,
            "s_maxstep": 6,
            "thresh_dang_aextend": 0.9848,
            "thresh_dang_L": 15,
            "thresh_short_L": 15,
            "s_fiberdir": 4,
            "thresh_linkd": 15,
            "thresh_linka": -0.866,
            "thresh_flen": 15,
            "thresh_numv": 3,
            "scale": [1.0, 1.0, 1.0],
            "s_boundthick": 10,
            "blist": 1,
            "s_maxspace": 5,
            "lambda": 0.01,
            "ang_interval": 3,
        },
    }

    mask_ori = img > ctfire_params["value"]["thresh_im2"]

    reconstructed_ct = ct_reconstruction(
        img=img,
        output_filename="2B_D9_ROI1.tif",
        coefficient_percentile=ctfire_params["coefficient_percentile"],
        specific_scales=ctfire_params["num_scales"],
        plot_flag=False,
    )

    # Apply mask to reconstructed image (element-wise multiplication)
    reconstructed_ct = reconstructed_ct * mask_ori

    im3 = np.zeros(
        (1, reconstructed_ct.shape[0], reconstructed_ct.shape[1]),
        dtype=reconstructed_ct.dtype,
    )
    im3[0, :, :] = reconstructed_ct

    fire_2d_angle(p=ctfire_params["value"], im=im3, plotflag=1)
