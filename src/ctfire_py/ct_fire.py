import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Dict, Any, Tuple, Optional

from ctfire_py import ct_reconstruction, fire_2d_angle


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

    # Create binary mask based on threshold (operates in 0–255 space after normalization)
    # TODO: if cP.RO ~= 2 --> then p2.thresh_im2 = 0?? right now we have it as 5
    mask_ori = img > ctfire_params["value"]["thresh_im2"]

    reconstructed_ct = ct_reconstruction(
        img=img,
        output_filename=image_name,
        coefficient_percentile=ctfire_params["coefficient_percentile"],
        specific_scales=ctfire_params["num_scales"],
        plot_flag=control_params["show_plots"],
    )

    # Apply mask to reconstructed image (element-wise multiplication)
    reconstructed_ct = reconstructed_ct * mask_ori

    im3 = np.zeros(
        (1, reconstructed_ct.shape[0], reconstructed_ct.shape[1]),
        dtype=reconstructed_ct.dtype,
    )
    im3[0, :, :] = reconstructed_ct

    data = fire_2d_angle(p=ctfire_params["value"], im=im3, plotflag=0)

    LL1 = ctfire_params["value"].get("thresh_flen", 15)
    ctfire_output = {
        "data": data,
        "cP": {
            "LL1": LL1,
            "widMAX": ctfire_params.get("widMAX", 20),
        },
    }

    return {}, ctfire_output


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
