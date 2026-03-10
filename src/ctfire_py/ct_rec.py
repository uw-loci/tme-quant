import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import scoreatpercentile
from pycurvelets.utils.math import round_mlab

from curvelops import fdct2d_wrapper


def ct_rec(
    image,
    output_filename,
    coefficient_percentile,
    num_scales,
    plot_flag=False,
):
    """
    Reconstruct an image using curvelet transform for denoising and fiber edge enhancement.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array of the input image
    output_filename : str
        Name for the reconstructed image file (e.g., 'CTR_image.mat')
    coefficient_percentile : float
        Fraction of curvelet coefficients to keep (0.0 to 1.0).
        Higher values keep more detail, lower values denoise more
    num_scales : int
        Number of finest scales to use in reconstruction.
        Higher values include more detail/texture
    plot_flag : bool, optional
        If True, display and save the reconstructed image. Default is False

    Returns
    -------
    reconstructed_image : np.ndarray
        Reconstructed image as a 2D numpy array
    """
    # Setup output filename (convert .mat to .tif format)
    output_image_path = output_filename.replace("CTR_", "CTRimg_").replace(
        ".mat", ".tif"
    )

    # Ensure input is float64 for transform precision
    image_float = np.asanyarray(image, dtype=np.float64)
    height, width = image_float.shape

    # Apply forward curvelet transform
    M, N = image.shape
    is_real = 0  # Complex-valued transform
    ac = 0  # 0 = wavelets, 1 = curvelets
    num_scales = math.floor(math.log2(min(M, N)) - 3)
    num_angles_coarse = 16  # Default number of angles at coarsest scale
    coefficients = fdct2d_wrapper.fdct2d_forward_wrap(
        num_scales, num_angles_coarse, ac, image
    )

    # Compute threshold for coefficient filtering
    # Collect all coefficient magnitudes across all scales and orientations
    all_coefficients = []
    for scale in coefficients:
        for wedge in scale:
            all_coefficients.append(np.abs(wedge).flatten())

    all_coefficients = np.concatenate(all_coefficients)
    all_coefficients = np.sort(all_coefficients)[::-1]  # Sort descending

    # Determine threshold: keep top coefficient_percentile of coefficients
    num_to_keep = int(round_mlab(coefficient_percentile * len(all_coefficients)))
    idx = min(num_to_keep, len(all_coefficients) - 1)
    threshold = all_coefficients[idx]

    # Apply hard thresholding: zero out coefficients below threshold
    for scale_idx in range(len(coefficients)):
        for wedge_idx in range(len(coefficients[scale_idx])):
            coefficients[scale_idx][wedge_idx] = coefficients[scale_idx][wedge_idx] * (
                np.abs(coefficients[scale_idx][wedge_idx]) > threshold
            )

    # Select which scales to use for reconstruction (finest scales contain detail)
    # Use the finest num_scales scales, zero out coarser scales
    total_scales = len(coefficients)
    selected_scales = range(total_scales - num_scales, total_scales)

    # Create coefficient structure with only selected scales
    filtered_coefficients = []
    for scale_idx in range(total_scales):
        if scale_idx in selected_scales:
            filtered_coefficients.append(coefficients[scale_idx])
        else:
            # Zero out this scale
            zero_scale = [
                np.zeros_like(coefficients[scale_idx][w])
                for w in range(len(coefficients[scale_idx]))
            ]
            filtered_coefficients.append(zero_scale)

    # Apply inverse curvelet transform to reconstruct image
    reconstructed_complex = fdct2d_wrapper.fdct2d_inverse_wrap(
        M, N, num_scales, num_angles_coarse, ac, filtered_coefficients
    )
    reconstructed_image = np.real(reconstructed_complex)

    # Optionally display and save the reconstructed image
    if plot_flag:
        plt.figure(figsize=(width / 128, height / 128))
        plt.imshow(reconstructed_image, cmap="gray")
        plt.axis("image")
        plt.title(
            f"Curvelet reconstruction using scales {selected_scales[0]} - {selected_scales[-1]}"
        )
        plt.savefig(output_image_path, dpi=128)
        plt.show()

    print("Curvelet transform reconstruction complete")
    return reconstructed_image


if __name__ == "__main__":
    # Example usage
    test_image = plt.imread(
        os.path.join(os.path.dirname(__file__), "2B_D9_ROI1.tif"),
        format="TIF",
    )

    reconstructed = ct_rec(
        image=test_image,
        output_filename="2B_D9_ROI1.tif",
        coefficient_percentile=0.2,  # Keep top 20% of coefficients
        num_scales=4,  # Use 4 finest scales
        plot_flag=True,
    )
