import csv as _csv_module
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Dict, Any, Tuple, Optional

from ctfire_py.ct_reconstruction import ct_reconstruction
from ctfire_py.fire_2d_angle import fire_2d_angle


def ct_fire(
    image_path: Optional[str],
    image_name: str,
    save_path: str,
    control_params: Dict[str, Any],
    ctfire_params: Dict[str, Any],
    img: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single image to extract fiber information using ctFIRE algorithm.

    Parameters
    ----------
    image_path : str or None
        Path to the directory containing the image. Only used when ``img`` is
        not provided.
    image_name : str
        Name of the image file (used for labelling outputs and disk load when
        ``img`` is not provided).
    save_path : str
        Path where output files will be saved.
    control_params : dict
        Control parameters for output images and files, e.g.:
        - ``show_plots``: bool, whether to display result images
        - ``save_images``: bool, whether to save result images
        - ``output_format``: str, format for saved images
    ctfire_params : dict
        ctFIRE algorithm parameters adjustable on the control panel, e.g.:
        - ``coefficient_percentile``: float, fraction of coefficients to keep
        - ``num_scales``: int, number of finest scales for reconstruction
        - ``fiber_threshold``: float, threshold for fiber detection
    img : np.ndarray, optional
        Pre-loaded image array. When provided, ``image_path`` / ``image_name``
        are not used for loading — the supplied array is used directly. This
        allows callers that already have the image in memory (e.g.
        ``process_image``) to avoid a redundant disk read.

    Returns
    -------
    fiber_output : dict
        Dictionary containing extracted fiber information from original processing
    ctfire_output : dict
        Dictionary containing extracted fiber information from curvelet transform processing

    Notes
    -----
    Results are saved in a subfolder: {image_path}/ctFIREout/
    """
    if img is None:
        img = cv2.imread(f"{image_path}/{image_name}")

    # Convert RGB to grayscale if needed.
    # Note: cv2.imread returns BGR; plt.imread returns RGB — for single-channel
    # microscopy images neither branch is taken and the array is used as-is.
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to float32 in [0, 255] so that thresh_im2 (default 5) behaves
    # consistently regardless of the source dtype (uint8, uint16, float 0-1, etc.).
    img = img.astype(np.float32)
    img_max = img.max()
    if img_max > 0:
        if img_max <= 1.0:
            # Float image already normalized to [0, 1] — scale up to [0, 255]
            img = img * 255.0
        elif img_max > 255.0:
            # 16-bit or other high-range image — rescale to [0, 255]
            img = img * (255.0 / img_max)

    # Mask is built from the ORIGINAL image (matches MATLAB ctFIRE: mask_ori = original_Image > thresh_im2).
    # The mask suppresses reconstruction artifacts in background regions while
    # preserving fiber signal where the original image is bright.
    mask_ori = img > ctfire_params["value"]["thresh_im2"]

    reconstructed_ct = ct_reconstruction(
        img=img,
        output_filename=image_name,
        coefficient_percentile=ctfire_params["coefficient_percentile"],
        specific_scales=ctfire_params["num_scales"],
        plot_flag=control_params["show_plots"],
    )

    reconstructed_ct = reconstructed_ct * mask_ori

    im3 = np.zeros(
        (1, reconstructed_ct.shape[0], reconstructed_ct.shape[1]),
        dtype=reconstructed_ct.dtype,
    )
    im3[0, :, :] = reconstructed_ct

    # Pass thresh_im2=0 so fire_2d_angle does not re-apply the threshold — the
    # background mask from the original image has already been applied above.
    import copy as _copy
    fire2d_params = _copy.copy(ctfire_params["value"])
    fire2d_params["thresh_im2"] = 0
    data = fire_2d_angle(p=fire2d_params, im=im3, plotflag=0)

    # LL1 is a CT-FIRE post-processing filter: selects fibers for output and
    # downstream analysis.  It is independent of thresh_flen (which is an
    # in-extraction dangling-fiber removal threshold inside fire_2d_angle).
    LL1 = ctfire_params.get("LL1", 30)
    ctfire_output = {
        "data": data,
        "cP": {
            "LL1": LL1,
            "widMAX": ctfire_params.get("widMAX", 20),
        },
        "saved_files": {},
    }

    if control_params.get("save_images", False) and save_path:
        ctfire_output["saved_files"] = _save_ct_fire_outputs(
            img=img,
            data=data,
            save_path=save_path,
            image_name=image_name,
            ctfire_params=ctfire_params,
        )

    return {}, ctfire_output


def _save_overlay_tiff(
    img: np.ndarray,
    data: Dict[str, Any],
    overlay_path: str,
) -> None:
    """Render fiber overlay onto *img* and save headlessly as a TIFF file.

    Uses ``data["Xf"]`` / ``data["Ff"]`` (CurveAlign-filtered fibers) coloured
    by absolute angle, identical to :func:`_visualize_fiber_output` but rendered
    off-screen via ``matplotlib.figure.Figure`` + ``FigureCanvasAgg`` so that no
    display is required.
    """
    import matplotlib.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    Xf: np.ndarray = data.get("Xf", np.empty((0, 2)))
    Ff: list = data.get("Ff", [])
    M: dict = data.get("M", {})
    angle_xy: np.ndarray = M.get("angle_xy", np.array([]))

    img_display = np.asarray(img, dtype=np.float32)
    vmin, vmax = img_display.min(), img_display.max()
    if vmax > vmin:
        img_display = (img_display - vmin) / (vmax - vmin)

    fig = matplotlib.figure.Figure(
        figsize=(img_display.shape[1] / 100, img_display.shape[0] / 100), dpi=100
    )
    ax = fig.add_subplot(111)
    ax.imshow(img_display, cmap="gray", origin="upper")
    ax.axis("off")

    _cmap = cm.get_cmap("hsv")
    _norm = mcolors.Normalize(vmin=0, vmax=180)

    def _to_deg(rad: float) -> float:
        deg = -rad * 180.0 / np.pi
        return deg + 180.0 if deg < 0 else deg

    drawn = 0
    for fi, fiber in enumerate(Ff):
        verts = fiber.get("v", [])
        valid = [v for v in verts if v < len(Xf)]
        if len(valid) < 2:
            continue
        coords = Xf[valid, :2]
        xs = coords[:, 1]
        ys = coords[:, 0]
        angle_deg = _to_deg(float(angle_xy[fi])) if fi < len(angle_xy) else 90.0
        color = _cmap(_norm(angle_deg))
        ax.plot(xs, ys, color=color, linewidth=0.8, solid_capstyle="round")
        drawn += 1

    sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fiber angle (°)", fontsize=8)
    cbar.set_ticks([0, 45, 90, 135, 180])
    ax.set_title(f"CT-FIRE fibers ({drawn}/{len(Ff)} shown)", fontsize=9, pad=4)

    fig.tight_layout()
    FigureCanvasAgg(fig)
    fig.savefig(overlay_path, dpi=100, bbox_inches="tight")


def _save_fiber_csv(
    data: Dict[str, Any],
    csv_path: str,
    min_fiber_len: float = 30.0,
) -> None:
    """Write per-fiber statistics to a CSV file for fibers with arc-length >= *min_fiber_len*.

    Uses ``data["Fa"]`` / ``data["Xa"]`` / ``data["Ra"]`` / ``data["M"]`` (post-fiberproc
    arrays).  Columns: ``fiber_id``, ``length_px``, ``angle_deg``, ``width_px``,
    ``straightness``.

    ``angle_deg`` is the endpoint angle in [0, 180) degrees.
    ``width_px`` is the mean distance-transform radius over the fiber's vertices.
    ``straightness`` is endpoint Euclidean distance / arc-length (∈ (0, 1]).
    """
    Xa = np.asarray(data.get("Xa", []))
    Fa = data.get("Fa", [])
    Ra_raw = data.get("Ra")
    M = data.get("M", {})
    lengths = np.asarray(M.get("L", []))
    angles = np.asarray(M.get("angle_xy", []))
    Ra = np.asarray(Ra_raw) if Ra_raw is not None else None

    rows = []
    for i, fiber in enumerate(Fa):
        if i >= len(lengths):
            break
        length = float(lengths[i])
        if length < min_fiber_len:
            continue

        verts = fiber["v"] if isinstance(fiber, dict) else list(fiber)

        # Angle in [0, 180) degrees
        if i < len(angles):
            angle_deg: Any = float(np.degrees(float(angles[i]) % np.pi))
        else:
            angle_deg = ""

        # Width: 2 × mean Ra (Ra is the radius, so diameter = 2*Ra)
        if Ra is not None and len(Ra) > 0:
            valid_v = [v for v in verts if 0 <= v < len(Ra)]
            width_px: Any = round(2.0 * float(np.mean([Ra[v] for v in valid_v])), 3) if valid_v else ""
        else:
            width_px = ""

        # Straightness: endpoint_dist / arc_length
        straightness: Any = ""
        if len(verts) >= 2 and len(Xa) > 0:
            v0, v_end = verts[0], verts[-1]
            if 0 <= v0 < len(Xa) and 0 <= v_end < len(Xa):
                endpoint_dist = float(np.linalg.norm(Xa[v_end, :2] - Xa[v0, :2]))
                straightness = round(endpoint_dist / length, 6) if length > 0 else ""

        rows.append({
            "fiber_id": i,
            "length_px": round(length, 3),
            "angle_deg": round(angle_deg, 3) if isinstance(angle_deg, float) else angle_deg,
            "width_px": width_px,
            "straightness": straightness,
        })

    with open(csv_path, "w", newline="") as fh:
        writer = _csv_module.DictWriter(
            fh,
            fieldnames=["fiber_id", "length_px", "angle_deg", "width_px", "straightness"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_params_json(ctfire_params: Dict[str, Any], path: str) -> None:
    """Write *ctfire_params* to *path* as a human-readable JSON file.

    Uses the cP-style layout so that ``LL1`` (the post-processing length filter)
    is explicit in the ``"ctfire"`` section, clearly separate from
    ``thresh_flen`` (the in-extraction dangling-fiber removal threshold kept in
    ``"fire2d"``).

    Output structure::

        {
          "ctfire": {"coefficient_percentile": ..., "num_scales": ...,
                     "LL1": ..., "widMAX": ...},
          "fire2d": {<all ctfire_params["value"] keys>}
        }

    Reload with :func:`load_ctfire_params`.
    """
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

    cP = {
        "ctfire": {
            "coefficient_percentile": ctfire_params.get("coefficient_percentile", 0.2),
            "num_scales": ctfire_params.get("num_scales", 4),
            "LL1": ctfire_params.get("LL1", 30),
            "widMAX": ctfire_params.get("widMAX", 20),
        },
        "fire2d": ctfire_params.get("value", {}),
    }
    with open(path, "w") as fh:
        json.dump(cP, fh, indent=2, default=_default)


def _save_ct_fire_outputs(
    img: np.ndarray,
    data: Dict[str, Any],
    save_path: str,
    image_name: str,
    ctfire_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save CT-FIRE outputs to *save_path*.

    Creates up to three files:

    * ``{stem}_overlay.tif``  — fibers drawn over the source image (TIFF).
    * ``{stem}_fibers.csv``   — per-fiber statistics for fibers ≥ ``LL1`` px.
    * ``{stem}_params.json``  — CT-FIRE parameters (only when *ctfire_params*
      is provided).

    Returns
    -------
    dict
        Absolute paths keyed by ``'overlay'``, ``'csv'``, and ``'params'``
        (``'params'`` only present when *ctfire_params* is not ``None``).
    """
    LL1 = ctfire_params.get("LL1", 30) if ctfire_params is not None else 30.0
    os.makedirs(save_path, exist_ok=True)
    stem = os.path.splitext(image_name)[0]
    saved: Dict[str, str] = {}

    overlay_path = os.path.join(save_path, f"{stem}_overlay.tif")
    _save_overlay_tiff(img, data, overlay_path)
    saved["overlay"] = overlay_path

    csv_path = os.path.join(save_path, f"{stem}_fibers.csv")
    _save_fiber_csv(data, csv_path, min_fiber_len=LL1)
    saved["csv"] = csv_path

    if ctfire_params is not None:
        params_path = os.path.join(save_path, f"{stem}_params.json")
        _save_params_json(ctfire_params, params_path)
        saved["params"] = params_path

    return saved


def load_ctfire_params(json_path: str) -> Dict[str, Any]:
    """Load CT-FIRE parameters from a JSON file saved by :func:`ct_fire`.

    Reconstructs the ``ctfire_params`` dict in the format expected by
    :func:`ct_fire`::

        {
            "coefficient_percentile": float,
            "num_scales": int,
            "LL1": float,    # post-processing fiber length filter (px)
            "widMAX": float,
            "value": {       # fire_2d_angle parameters ("fire2d" pass-through)
                "thresh_flen": ...,  # in-extraction dangling-fiber removal
                ...
            },
        }

    Parameters
    ----------
    json_path : str
        Path to a ``*_params.json`` file written by :func:`ct_fire`.

    Returns
    -------
    dict
        ``ctfire_params`` dict directly passable to :func:`ct_fire`.
    """
    with open(json_path, "r") as fh:
        cP = json.load(fh)
    ctfire_section = cP.get("ctfire", {})
    return {
        "coefficient_percentile": ctfire_section.get("coefficient_percentile", 0.2),
        "num_scales": ctfire_section.get("num_scales", 4),
        "LL1": ctfire_section.get("LL1", 30),
        "widMAX": ctfire_section.get("widMAX", 20),
        "value": cP.get("fire2d", {}),
    }


def _visualize_fiber_output(
    img: np.ndarray,
    data: Dict[str, Any],
    image_name: str = "",
) -> None:
    """
    Overlay extracted fibers on the source image.

    Uses the CurveAlign-filtered fibers (``data["Xf"]`` / ``data["Ff"]``) so
    only length- and straightness-qualified fibers are shown.  Each fiber is
    drawn as a line coloured by its absolute angle (0–180 °) using a cyclic
    HSV colormap; a colorbar is included for reference.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale image (float, 0–255 range) used as the background.
    data : dict
        Output dict from :func:`fire_2d_angle`.
    image_name : str, optional
        Used in the figure title.
    """
    Xf: np.ndarray = data.get("Xf", np.empty((0, 2)))
    Ff: list = data.get("Ff", [])
    M: dict = data.get("M", {})
    # Fa is used to index into M["angle_xy"] (same ordering as Ff after filtering)
    Fa: list = data.get("Fa", [])

    angle_xy: np.ndarray = M.get("angle_xy", np.array([]))

    # Build a map from original fiber index → absolute angle in degrees
    def _to_deg(rad: float) -> float:
        deg = -rad * 180.0 / np.pi
        return deg + 180.0 if deg < 0 else deg

    # Xf is the filtered coordinate array; Ff indexes into Xf.
    # M["angle_xy"] was computed from Fa (original fibers) so its length may
    # differ from Ff.  We use index-aligned access with a NaN fallback.
    n_fibers = len(Ff)

    # Normalize img for display: matplotlib's imshow clips float arrays to [0,1].
    # Rescale to [0,1] so the background is visible regardless of input range.
    img_display = np.asarray(img, dtype=np.float32)
    img_min, img_max_val = img_display.min(), img_display.max()
    if img_max_val > img_min:
        img_display = (img_display - img_min) / (img_max_val - img_min)

    fig, ax = plt.subplots(figsize=(img_display.shape[1] / 100, img_display.shape[0] / 100), dpi=100)
    ax.imshow(img_display, cmap="gray", origin="upper")
    ax.axis("off")

    cmap = cm.get_cmap("hsv")
    norm = mcolors.Normalize(vmin=0, vmax=180)

    drawn = 0
    for fi, fiber in enumerate(Ff):
        verts = fiber.get("v", [])
        if len(verts) < 2:
            continue
        # Guard against out-of-range vertex indices
        valid = [v for v in verts if v < len(Xf)]
        if len(valid) < 2:
            continue

        coords = Xf[valid, :2]  # (N, 2) — C++ backend stores [row, col]
        xs = coords[:, 1]       # col → horizontal axis
        ys = coords[:, 0]       # row → vertical axis

        angle_deg = _to_deg(float(angle_xy[fi])) if fi < len(angle_xy) else 90.0
        color = cmap(norm(angle_deg))

        ax.plot(xs, ys, color=color, linewidth=0.8, solid_capstyle="round")
        drawn += 1

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fiber angle (°)", fontsize=8)
    cbar.set_ticks([0, 45, 90, 135, 180])

    title = f"CT-FIRE fibers ({drawn}/{n_fibers} shown)"
    if image_name:
        title = f"{image_name}  —  {title}"
    ax.set_title(title, fontsize=9, pad=4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use the canonical test image rather than a co-located copy that could
    # be accidentally overwritten by a plot-save side-effect.
    image_path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "test_images")
    image_name = "2B_D9_ROI1.tif"
    save_path = os.path.join(os.path.dirname(__file__), "ctFIREout")

    control_params = {
        "show_plots": True,
        "save_images": True,
        "output_format": "tif",
    }

    ctfire_params = {
        "coefficient_percentile": 0.2,
        "num_scales": 4,
        "LL1": 30,
        "fiber_threshold": 0.5,
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

    fiber_out, ctfire_out = ct_fire(
        image_path=image_path,
        image_name=image_name,
        save_path=save_path,
        control_params=control_params,
        ctfire_params=ctfire_params,
    )

    if control_params["show_plots"] and ctfire_out:
        import matplotlib.pyplot as _plt
        img_display = _plt.imread(
            os.path.join(image_path, image_name)
        ).astype("float32")
        img_max = img_display.max()
        if img_max > 0 and img_max <= 1.0:
            img_display *= 255.0
        elif img_max > 255.0:
            img_display *= 255.0 / img_max
        _visualize_fiber_output(img_display, ctfire_out["data"], image_name)
