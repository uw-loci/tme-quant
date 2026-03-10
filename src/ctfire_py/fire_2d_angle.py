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
from typing import Dict, Any, Optional
import sys

# Import C++ backend
try:
    sys.path.insert(0, "src/ctfire_py/CPP")
    import fiber_backend
except ImportError:
    print("Warning: C++ fiber_backend not available. Please compile it first.")
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

    # Apply first two filters
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
    ims = np.round(smooth(im, p.get("sigma_im")))

    if plotflag == 1:
        print(f"  Smoothed image shape: {ims.shape}")

    # Step 2: Threshold image
    if p.get("thresh_im") is not None:
        imt = ims > p["thresh_im"] * np.max(ims)
    else:
        imt = ims > p.get("thresh_im2", 0)

    if plotflag == 1:
        print(f"  Thresholded pixels: {np.sum(imt)}")

    # Step 3: Distance transform
    print(f"  Calculating {p.get('dtype', 'euclidean')} distance to background")

    # Flatten to 2D for distance transform
    imt_2d = flatten(imt)

    # Compute distance transform
    if p.get("dtype", "euclidean") == "euclidean":
        d = distance_transform_edt(imt_2d)
    else:
        # For other distance types, use euclidean as default
        d = distance_transform_edt(imt_2d)

    d = d.astype(np.float32)

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

    print(f"  Found {xlink.shape[0]} nucleation points")

    if plotflag == 1:
        print(f"  Nucleation points shape: {xlink.shape}")

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
    print("Remove danglers and shorties")
    # TODO: Implement check_danglers function
    # For now, pass through
    Xz2 = Xz.copy()
    Fz2 = Fz.copy() if isinstance(Fz, list) else Fz
    Vz2 = Vz.copy() if isinstance(Vz, list) else Vz
    Rz2 = Rz.copy()

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
    print("Fiberproc")
    # TODO: Implement full fiberproc when C++ backend is complete
    # For now, use the data we have
    Xa = Xz2
    Fa = Fz2
    Va = Vz2
    Ra = Rz2

    # Create edges array
    Ea = np.zeros((len(Fa), 2), dtype=np.int32)
    for i in range(len(Fa)):
        if isinstance(Fa[i], dict) and "v" in Fa[i]:
            v_list = Fa[i]["v"]
            if len(v_list) > 0:
                Ea[i, 0] = v_list[0]
                Ea[i, 1] = v_list[-1]

    elapsed_time = time.time() - start_time
    print(f"CPP code for this image takes {elapsed_time:.2f} seconds")

    # Step 9: Compute network statistics
    print("Computing network statistics")

    # Scale vertices
    scale = p.get("scale", [1.0, 1.0, 1.0])
    Xas = Xa.copy()
    for k in range(Xa.shape[0]):
        Xas[k, :] = Xa[k, :] * scale[: Xa.shape[1]]

    # TODO: Implement network_statK function
    M = {}  # Placeholder for network statistics

    # Step 10: Fiber interpolation
    print("Interpolating fibers")
    # TODO: Implement fiber2beam function
    Xai = Xas.copy()
    Fai = Fa
    Vai = Va

    # Step 11: Calculate fiber angles
    print("Calculating fiber angles")
    SPI = p.get("ang_interval", 5)
    # TODO: Implement calc_fiberang2 function
    FiberAngle = []  # Placeholder
    FiberAngleI = []  # Placeholder

    M["Fang"] = FiberAngle
    M["FangI"] = FiberAngleI

    # Step 12: Beam processing
    print("Beamproc")
    # TODO: Implement beamproc function
    Xab = Xa.copy()
    Fab = Fa
    Vab = Va

    # Step 13: Fiber break at cross-links
    # TODO: Implement fiberbreak function
    Xc = Xa.copy()
    Fc = Fa
    Vc = Va

    # Create output data structure
    data = {
        "X": X,
        "F": F,
        "R": R,
        "Xa": Xa,
        "Fa": Fa,
        "Va": Va,
        "Ea": Ea,
        "Ra": Ra,
        "Xab": Xab,
        "Fab": Fab,
        "Vab": Vab,
        "Xc": Xc,
        "Fc": Fc,
        "Vc": Vc,
        "M": M,
        "xlink": xlink,
        "Xai": Xai,
        "Fai": Fai,
        "Vai": Vai,
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
    height, width = img.shape
    im3 = np.zeros((1, height, width), dtype=img.dtype)

    im3[0, :, :] = img

    fire_2d_angle(p, im3, 0)
