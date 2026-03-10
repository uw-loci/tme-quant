import cv2
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ctfire_py.ct_reconstruction import ct_reconstruction
from ctfire_py.fire_2d_angle import fire_2d_angle


def ct_fire(
    image_path: str,
    image_name: str,
    save_path: str,
    control_params: Dict[str, Any],
    ctfire_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single image to extract fiber information using ctFIRE algorithm.

    Parameters
    ----------
    image_path : str
        Path to the directory containing the image to be processed
    image_name : str
        Name of the image file to be processed
    save_path : str
        Path where output files will be saved
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
        #TODO: add more details to ctfire_params (the p stuff)

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
    # TODO: Implement ctFIRE algorithm
    # 1. Load image from image_path/image_name
    # 2. Apply curvelet transform reconstruction (ct_rec)
    # 3. Extract fiber features from both original and reconstructed images
    # 4. Save results to save_path/ctFIREout/

    # Placeholder for loaded image (replace with actual image loading)
    img = cv2.imread(f"{image_path}/{image_name}")

    # TODO: if cP.RO ~= 2 --> then p2.thresh_im2 = 0?? right now we have it as 5
    # Not sure why there's a discrepancy between thresh_im2 of p2 and ctfire_params

    # Convert RGB to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        # OpenCV loads as BGR, convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create 3D array to store image
    height, width = img.shape
    im3 = np.zeros((1, height, width), dtype=img.dtype)
    im3[0, :, :] = img

    # Flip image vertically (associated with 'axis xy' in MATLAB)
    img_flipped = np.flipud(img)

    # Create binary mask based on threshold
    mask_ori = img > ctfire_params["value"]["thresh_im2"]

    reconstructed_ct = ct_reconstruction(
        img=img,
        output_filename="2B_D9_ROI1.tif",
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

    fire_2d_angle(p=ctfire_params["value"], im=im3, plotflag=0)

    fiber_output = {}
    ctfire_output = {}

    return fiber_output, ctfire_output


if __name__ == "__main__":
    image_path = os.path.dirname(__file__)
    image_name = "2B_D9_ROI1.tif"
    save_path = os.path.join(image_path, "ctFIREout")

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
