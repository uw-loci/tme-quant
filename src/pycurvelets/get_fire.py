import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from ctfire_py.ct_fire import ct_fire as _run_ct_fire, _visualize_fiber_output
from pycurvelets.models import FeatureControlParameters
from pycurvelets.process_fibers import process_fibers

# Default CT-FIRE parameters, matching the values used in ct_fire.py __main__
DEFAULT_CTFIRE_PARAMS: Dict[str, Any] = {
    "coefficient_percentile": 0.2,
    "num_scales": 4,
    "fiber_threshold": 0.5,
    "widMAX": 20,
    "widcon": {
        "wid_mm": 1,     # minimum max-width threshold (pixels)
        "wid_mp": 10,    # minimum sample size for sigma-clipping path
        "wid_sigma": 1,  # ┬▒╧â confidence region for sigma-clipping
        "wid_max": 0,    # 0 = don't compute per-fiber maximum width
        "wid_opt": 1,    # 1 = use all points below threshold (no sigma clip)
    },
    "value": {
        "sigma_im": 0,
        "sigma_d": 0.3,
        "dtype": "cityblock",
        "thresh_im": [],
        "thresh_im2": 5,
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


def get_fire(
    img_name: str,
    fire_directory: str,
    fiber_mode: int,
    feature_cp: FeatureControlParameters,
    image_path: Optional[str] = None,
    ctfire_params: Optional[Dict[str, Any]] = None,
    img: Optional[np.ndarray] = None,
    show_plots: bool = False,
):
    """
    Run CT-FIRE on an image and convert the output to a structured DataFrame
    usable by CurveAlign/process_image.

    Mirrors the MATLAB ``getFIRE.m`` function: runs the CT-FIRE fiber extraction
    algorithm (via ``ct_fire.py``) and converts the resulting fiber network data
    into the DataFrame format expected by the rest of the Python pipeline.

    Parameters
    ----------
    img_name : str
        Name of the image file to process (e.g. ``"sample.tif"``).
    fire_directory : str
        Directory used as the CT-FIRE output path (``save_path``). Also used as
        ``image_path`` when ``image_path`` is not provided.
    fiber_mode : int
        Fiber processing mode:
        - ``1`` ΓÇö one row per fiber (``fibProcMeth == 1`` in MATLAB)
        - ``2`` ΓÇö one row per segment (``fibProcMeth == 0`` in MATLAB)
        - ``3`` ΓÇö two rows per fiber, one for each endpoint (``fibProcMeth == 2`` in MATLAB)
    feature_cp : FeatureControlParameters
        Control parameters for density and alignment feature extraction.
    image_path : str, optional
        Directory containing the image. Falls back to ``fire_directory`` when
        not provided.
    ctfire_params : dict, optional
        CT-FIRE algorithm parameters. Falls back to ``DEFAULT_CTFIRE_PARAMS``
        when not provided.
    img : np.ndarray, optional
        Pre-loaded image array. When provided, disk loading inside ``ct_fire``
        is skipped entirely. Callers such as ``process_image`` that already
        hold the image in memory should pass it here.
    show_plots : bool, optional
        When True, display a fiber overlay figure after extraction: extracted
        fibers coloured by angle are drawn over the original ``img``.
        Defaults to False.

    Returns
    -------
    fiber_structure : pd.DataFrame
        One row per fiber (or segment), with columns:
        ``angle``, ``center_row``, ``center_col``, ``total_length``,
        ``end_length``, ``curvature``, ``width``.
    density_df : pd.DataFrame
        Density features from :func:`process_fibers`.
    alignment_df : pd.DataFrame
        Alignment features from :func:`process_fibers`.
    """
    resolved_image_path = image_path or fire_directory
    resolved_ctfire_params = ctfire_params or DEFAULT_CTFIRE_PARAMS

    # Run the CT-FIRE algorithm.  When img is already in memory (the common
    # case when called from process_image) pass it directly to avoid a second
    # disk read.  image_path is still forwarded for output labelling purposes.
    # show_plots is kept False here ΓÇö the overlay is handled below in get_fire
    # so it can use the original img that was passed in by the caller.
    _fiber_out, ctfire_output = _run_ct_fire(
        image_path=resolved_image_path,
        image_name=img_name,
        save_path=fire_directory,
        control_params={
            "show_plots": False,
            "save_images": False,
            "output_format": "tif",
        },
        ctfire_params=resolved_ctfire_params,
        img=img,
    )

    data = ctfire_output["data"]
    LL1 = ctfire_output["cP"]["LL1"]

    # Overlay detected fibers on the original image passed in by the caller.
    # This is the correct place for the visualization: get_fire owns both the
    # original img and the fiber data, matching the described pipeline flow.
    if show_plots and img is not None:
        _visualize_fiber_output(img, data, img_name)

    # Convert CT-FIRE data dict ΓåÆ fiber_structure DataFrame
    img_shape = img.shape if img is not None else None
    fiber_structure = _build_fiber_dataframe(
        data, LL1, fiber_mode, feature_cp,
        ctfire_params=resolved_ctfire_params,
        img_shape=img_shape,
    )

    if fiber_structure.empty:
        return fiber_structure, pd.DataFrame(), pd.DataFrame()

    density_df, alignment_df = process_fibers(fiber_structure, feature_cp)
    return fiber_structure, density_df, alignment_df


def _build_fiber_dataframe(
    data: Dict[str, Any],
    LL1: float,
    fiber_mode: int,
    feature_cp: FeatureControlParameters,
    ctfire_params: Optional[Dict[str, Any]] = None,
    img_shape: Optional[Tuple[int, ...]] = None,
) -> pd.DataFrame:
    """
    Convert CT-FIRE output data dict to a fiber_structure DataFrame.

    Mirrors the fiber iteration logic in ``getFIRE.m``.

    Parameters
    ----------
    data : dict
        Output dict from :func:`fire_2d_angle`, containing keys:
        ``Fai``, ``Xai``, ``Fa``, ``Xa``, ``Ra``, ``M``.
    LL1 : float
        Minimum fiber length threshold (fibers shorter than this are excluded).
        Corresponds to ``cP.LL1`` in the MATLAB code.
    fiber_mode : int
        ``1`` = one row per fiber; ``2`` = one row per segment;
        ``3`` = two rows per fiber (start and end endpoints).
    feature_cp : FeatureControlParameters
        Used for ``fiber_midpoint_estimate`` (1 = endpoint mean, 2 = arc midpoint).
    ctfire_params : dict, optional
        Full CT-FIRE parameter dict. Used to read ``widMAX`` and ``widcon`` for
        the advanced width calculation. Falls back to simple mean when absent.
    img_shape : tuple, optional
        ``(height, width, ...)`` of the source image. When provided and
        ``fiber_mode == 2``, interpolated Xai coordinates are clamped to the
        valid pixel range before iterating (mirrors the MATLAB coordinate-clamp
        block that runs for ``fibProcMeth == 0``).

    Returns
    -------
    pd.DataFrame
        Columns: ``angle``, ``center_row``, ``center_col``, ``total_length``,
        ``end_length``, ``curvature``, ``width``.
    """
    Fai: List[Dict] = data.get("Fai", [])
    Xai: np.ndarray = data.get("Xai", np.empty((0, 2)))
    Fa: List[Dict] = data.get("Fa", [])
    Xa: np.ndarray = data.get("Xa", np.empty((0, 2)))
    Ra: np.ndarray = data.get("Ra", np.array([]))
    M: Dict = data.get("M", {})

    lengths: np.ndarray = M.get("L", np.array([]))
    angle_xy: np.ndarray = M.get("angle_xy", np.array([]))
    FangI: List[Dict] = M.get("FangI", [])

    # CT-FIRE always uses the arc midpoint (middle vertex of the Hermite-
    # interpolated Fai path), matching MATLAB getFIRE.m fibProcMeth==1:
    #   ctemp = ceil(length(Fai(i).v)/2);
    #   object(i).center = Xai(Fai(i).v(ctemp),:);
    # The fiber_midpoint_estimate parameter is only relevant for curvelet mode.
    use_midpoint_arc = True

    # --- Width calculation parameters (mirrors MATLAB widcon / widOPTflag) ---
    widcon = (ctfire_params or {}).get("widcon", {})
    wid_mm: float = float(widcon.get("wid_mm", 1))
    wid_mp: int = int(widcon.get("wid_mp", 10))
    wid_sigma: float = float(widcon.get("wid_sigma", 1))
    wid_opt: int = int(widcon.get("wid_opt", 1))
    wid_max_raw: float = float((ctfire_params or {}).get("widMAX", 20))
    wid_th: float = wid_max_raw if wid_max_raw >= wid_mm else wid_mm
    use_advanced_width: bool = ctfire_params is not None

    # --- Coordinate clamping for segment mode (mirrors MATLAB fibProcMeth == 0 block) ---
    # Work on a copy so the original data dict is not mutated.
    # For 512├ù512 (square) images the C++ column-major indexing of a row-major flat
    # array gives Xa[:,0] = row and Xa[:,1] = col.
    if fiber_mode == 2 and img_shape is not None and len(Xai) > 0:
        height, width = img_shape[0], img_shape[1]
        Xai = Xai.copy()
        Xai[:, 0] = np.clip(Xai[:, 0], 0, height - 1)  # row axis
        Xai[:, 1] = np.clip(Xai[:, 1], 0, width - 1)   # col axis

    rows: List[Dict] = []

    num_fib = len(Fai)
    for i in range(num_fib):
        if i >= len(lengths):
            continue
        fiber_length = float(lengths[i])
        if fiber_length <= LL1:
            continue

        fai_verts = Fai[i]["v"]
        if len(fai_verts) == 0:
            continue

        # --- End-to-end length (uses original Xa / Fa) ---
        fa_verts = Fa[i]["v"] if i < len(Fa) else []
        if len(fa_verts) >= 2 and len(Xa) > 0:
            # MATLAB: fsp = Fa(i).v(1); fep = Fa(i).v(end)
            #         sp  = Xa(fep,:);   ep  = Xa(fsp,:)
            # (MATLAB variable names are swapped; dse = norm(sp-ep) is symmetric)
            pt_a = Xa[fa_verts[-1], :2]   # matches MATLAB's "sp"
            pt_b = Xa[fa_verts[0], :2]    # matches MATLAB's "ep"
            end_length = float(np.linalg.norm(pt_a - pt_b))
        else:
            pt_a = np.full(2, np.nan)
            pt_b = np.full(2, np.nan)
            end_length = fiber_length

        # Straightness (curvature in CurveAlign terminology)
        curvature = end_length / fiber_length if fiber_length > 0 else 0.0

        # --- Width from radii ---
        width = _compute_width(fa_verts, Ra, wid_th, wid_opt, wid_mp, wid_sigma, use_advanced_width)

        if fiber_mode == 2:
            # Segment-level: one row per interpolated segment in this fiber
            if i >= len(FangI):
                continue
            seg_angles = FangI[i].get("angle_xy", [])
            for j, seg_ang in enumerate(seg_angles):
                if j >= len(fai_verts):
                    break
                v1 = fai_verts[j]
                if v1 >= len(Xai):
                    continue
                pt = Xai[v1, :2]
                rows.append(
                    {
                        "angle": _angle_rad_to_deg(float(seg_ang)),
                        "center_row": float(pt[0]),  # Xai[:,0] = row
                        "center_col": float(pt[1]),  # Xai[:,1] = col
                        "total_length": fiber_length,
                        "end_length": end_length,
                        "curvature": curvature,
                        "width": width,
                    }
                )

        elif fiber_mode == 3:
            # Endpoint mode: two rows per fiber ΓÇö one for each original endpoint.
            # Mirrors MATLAB fibProcMeth == 2.
            if i < len(angle_xy):
                theta_deg = _angle_rad_to_deg(float(angle_xy[i]))
            else:
                theta_deg = np.nan

            scalar_feats = {
                "angle": theta_deg,
                "total_length": fiber_length,
                "end_length": end_length,
                "curvature": curvature,
                "width": width,
            }

            # Row A ΓÇö pt_a (MATLAB's "sp", last original vertex)
            # Xa[:,0] = row, Xa[:,1] = col.
            rows.append(
                {
                    **scalar_feats,
                    "center_row": float(np.round(pt_a[0])) if not np.isnan(pt_a[0]) else np.nan,
                    "center_col": float(np.round(pt_a[1])) if not np.isnan(pt_a[1]) else np.nan,
                }
            )
            # Row B ΓÇö pt_b (MATLAB's "ep", first original vertex)
            rows.append(
                {
                    **scalar_feats,
                    "center_row": float(np.round(pt_b[0])) if not np.isnan(pt_b[0]) else np.nan,
                    "center_col": float(np.round(pt_b[1])) if not np.isnan(pt_b[1]) else np.nan,
                }
            )

        else:
            # Fiber-level (fiber_mode == 1): one row per fiber
            if i < len(angle_xy):
                theta_deg = _angle_rad_to_deg(float(angle_xy[i]))
            else:
                theta_deg = np.nan

            center_row, center_col = _compute_center(
                fai_verts, Xai, fa_verts, Xa, use_midpoint_arc
            )

            rows.append(
                {
                    "angle": theta_deg,
                    "center_row": center_row,
                    "center_col": center_col,
                    "total_length": fiber_length,
                    "end_length": end_length,
                    "curvature": curvature,
                    "width": width,
                }
            )

    return pd.DataFrame(rows)


def _compute_width(
    fa_verts: List[int],
    Ra: np.ndarray,
    wid_th: float,
    wid_opt: int,
    wid_mp: int,
    wid_sigma: float,
    use_advanced: bool,
) -> float:
    """
    Compute average fiber width from radii array.

    Mirrors the ``widOPTflag`` width calculation block in ``getFIRE.m``.

    When ``use_advanced`` is True (``widOPTflag == 1`` in MATLAB):
    - Points exceeding ``wid_th`` are excluded.
    - If ``wid_opt == 1``: mean of remaining points.
    - If ``wid_opt != 1`` and enough samples: sigma-clipped mean.

    When ``use_advanced`` is False (``widOPTflag == 0`` fallback):
    - Plain mean of all diameters.
    """
    if len(Ra) == 0 or len(fa_verts) == 0:
        return np.nan

    valid_verts = [v for v in fa_verts if v < len(Ra)]
    if not valid_verts:
        return np.nan

    widall = 2.0 * Ra[valid_verts]

    if not use_advanced:
        return float(np.mean(widall))

    # Advanced path (widOPTflag == 1)
    wtemp = widall[widall <= wid_th]
    if len(wtemp) == 0:
        return np.nan

    if wid_opt == 1:
        return float(np.mean(wtemp))

    # Sigma-clipping path
    if len(wtemp) > wid_mp:
        wid_mean = np.mean(wtemp)
        wid_std = np.std(wtemp, ddof=1)
        mask = np.abs(wtemp - wid_mean) <= wid_sigma * wid_std
        clipped = wtemp[mask]
        return float(np.mean(clipped)) if len(clipped) > 0 else float(np.mean(wtemp))

    return float(np.mean(wtemp))


def _angle_rad_to_deg(theta_rad: float) -> float:
    """
    Convert fiber angle from radians to degrees in [0, 180].

    Mirrors the MATLAB sign flip and range adjustment:
    ``theta = -1 * angle_xy; thetaDeg = theta * 180 / pi; if thetaDeg < 0: thetaDeg += 180``
    """
    theta_deg = -theta_rad * 180.0 / np.pi
    if theta_deg < 0:
        theta_deg += 180.0
    return theta_deg


def _compute_center(
    fai_verts: List[int],
    Xai: np.ndarray,
    fa_verts: List[int],
    Xa: np.ndarray,
    use_midpoint_arc: bool,
) -> Tuple[float, float]:
    """
    Compute fiber center coordinates.

    Parameters
    ----------
    use_midpoint_arc : bool
        If True, use the arc midpoint (midpoint vertex along the interpolated
        fiber, ``fiber_midpoint_estimate == 2`` in MATLAB). If False, use the
        mean of the two original endpoint coordinates
        (``fiber_midpoint_estimate == 1`` in MATLAB).

    Returns
    -------
    tuple
        ``(center_row, center_col)`` where ``[:,0]`` = row and ``[:,1]`` = col
        in the C++ backend output (verified for square images).
    """
    # C++ backend stores coordinates as [row, col] (for square images, the
    # column-major decomposition of a row-major flat index gives i=row, j=col).
    if use_midpoint_arc and len(fai_verts) > 0 and len(Xai) > 0:
        mid_idx = fai_verts[len(fai_verts) // 2]
        if mid_idx < len(Xai):
            pt = Xai[mid_idx, :2]
            return float(pt[0]), float(pt[1])  # row, col

    # Fallback: mean of the two original endpoints
    if len(fa_verts) >= 2 and len(Xa) > 0:
        sp = Xa[fa_verts[0], :2]
        ep = Xa[fa_verts[-1], :2]
        mean_pt = (sp + ep) / 2.0
        return float(mean_pt[0]), float(mean_pt[1])  # row, col

    # Last resort: first interpolated vertex
    if len(fai_verts) > 0 and len(Xai) > 0 and fai_verts[0] < len(Xai):
        pt = Xai[fai_verts[0], :2]
        return float(pt[0]), float(pt[1])  # row, col

    return np.nan, np.nan


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    _repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    _image_path = os.path.join(_repo_root, "tests", "test_images")
    _image_name = "fiber_image_1.tif"
    _fire_directory = os.path.join(_repo_root, "tests", "test_results")

    # Load and normalize the image to float32 [0, 255]
    _img = plt.imread(os.path.join(_image_path, _image_name)).astype("float32")
    _img_max = _img.max()
    if _img_max > 0:
        if _img_max <= 1.0:
            _img *= 255.0
        elif _img_max > 255.0:
            _img *= 255.0 / _img_max

    _feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=2,
        minimum_box_size=32,
        fiber_midpoint_estimate=1,
    )

    fiber_structure, density_df, alignment_df = get_fire(
        img_name=_image_name,
        fire_directory=_fire_directory,
        fiber_mode=1,             # 1 = one row per fiber
        feature_cp=_feature_cp,
        image_path=_image_path,
        ctfire_params=DEFAULT_CTFIRE_PARAMS,
        img=_img,
        show_plots=True,
    )

    print(f"\nExtracted {len(fiber_structure)} fibers")
    if not fiber_structure.empty:
        print(fiber_structure[["angle", "center_row", "center_col",
                                "total_length", "curvature", "width"]].describe())
