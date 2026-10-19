import os
import numpy as np
import matplotlib.pyplot as plt
import math
from pycurvelets.utils.math import round_mlab


def ct_reconstruction(
    img, output_filename, coefficient_percentile, specific_scales, plot_flag=False
):
    """
    Reconstruct an image using curvelet transform for denoising and fiber edge enhancement.

    Applies forward curvelet transform, thresholds coefficients to remove noise, selects
    specific scales for reconstruction, and applies inverse transform to produce an
    enhanced image with reduced noise and emphasized fiber edges.

    Parameters
    ----------
    img : np.ndarray
        2D input image array to be processed
    output_filename : str
        Base filename for saving the reconstructed image (e.g., 'CTR_image.mat').
        Will be converted to .tif format with 'CTRimg_' prefix
    coefficient_percentile : float
        Fraction of curvelet coefficients to retain (0.0 to 1.0).
        Higher values preserve more detail, lower values increase denoising.
        Example: 0.2 keeps the top 20% of coefficients by magnitude
    specific_scales : int
        Number of finest scales to use in reconstruction.
        Finer scales contain high-frequency detail and texture.
        Example: 4 uses the 4 finest scales
    plot_flag : bool, optional
        If True, displays and saves the reconstructed image. Default is False

    Returns
    -------
    reconstructed_image : np.ndarray
        2D array of the reconstructed image with denoising and edge enhancement applied
    """
    try:
        from curvelops import fdct2d_wrapper
    except ImportError as e:
        raise ImportError(
            "curvelops is required for ct_reconstruction. "
            "Install it with: pip install 'tme-quant[curvelops]'"
        ) from e

    # Derive a safe output filename for the optional plot save.
    # Prefix with "CTRimg_" and force a .tif extension so the saved figure
    # never shares a name with the source image (which would overwrite it).
    base = os.path.splitext(os.path.basename(output_filename))[0]
    ct_img_name = f"CTRimg_{base}.tiff"

    # Ensure input is a float64 for transform precision
    is_img = np.asanyarray(img, dtype=np.float64)
    img_height, img_width = is_img.shape

    # Match fdct_wrapping(double(IS), 0): its default scale count uses ceil.
    # The native wrapper's ac flag selects the finest-scale representation;
    # ac=0 is a finest wavelet, matching MATLAB's default finest=2.
    M, N = is_img.shape
    ac = 0
    nbscales = math.ceil(math.log2(min(M, N)) - 3)
    nbangles_coarse = 16  # default
    c = fdct2d_wrapper.fdct2d_forward_wrap(nbscales, nbangles_coarse, ac, is_img)

    # 3. Thresholding Logic
    # Flatten all coefficients across all scales and wedges into one array
    all_coeffs = []
    for scale in c:
        for wedge in scale:
            all_coeffs.append(np.abs(wedge).flatten())

    all_coeffs = np.concatenate(all_coeffs)
    all_coeffs.sort()
    all_coeffs = all_coeffs[::-1]  # Descending sort

    # Get specific threshold based on percentile (pct)
    nb = int(round_mlab(coefficient_percentile * len(all_coeffs)))
    # MATLAB cfs(nb) is one-based. Keep the supported pct=0 behavior
    # (no retained coefficients) by clamping the index to the first value.
    idx = max(0, min(nb - 1, len(all_coeffs) - 1))
    cutoff = all_coeffs[idx]

    # 4. Filter coefficients by threshold
    for scale in range(len(c)):
        for wedge in range(len(c[scale])):
            # Apply hard thresholding
            c[scale][wedge] = c[scale][wedge] * (np.abs(c[scale][wedge]) > cutoff)

    # MATLAB: s = length(C)-SS : length(C)-1  (1-based inclusive, excludes finest scale)
    # Python 0-based equivalent: subtract 1 from each bound, use exclusive stop.
    #   start = (num_scales - SS) - 1 = num_scales - SS - 1
    #   stop  = (num_scales - 1)       = num_scales - 1  (exclusive → includes up to index num_scales-2)
    # This selects exactly SS scales and deliberately omits the very finest scale
    # (index num_scales-1), which typically contains high-frequency noise.
    num_scales = len(c)
    selected_scale_indices = range(
        max(0, num_scales - specific_scales - 1), num_scales - 1
    )

    # Create an empty coefficient structure (zeros)
    curvelet_coefficients = []
    for scale in range(num_scales):
        scale_wedges = []
        for wedge in range(len(c[scale])):
            if scale in selected_scale_indices:
                scale_wedges.append(c[scale][wedge])
            else:
                scale_wedges.append(np.zeros_like(c[scale][wedge]))
        curvelet_coefficients.append(scale_wedges)

    # 6. Inverse Curvelet Transform
    reconstructed_image = fdct2d_wrapper.fdct2d_inverse_wrap(
        M, N, nbscales, 16, 0, curvelet_coefficients
    )
    reconstructed_image = np.real(reconstructed_image)

    # 7. Plotting and Saving
    if plot_flag:
        # Save raw pixel data (no matplotlib borders) so the file can be
        # re-loaded as an image without any framing artifacts.
        plt.imsave(ct_img_name, reconstructed_image, cmap="gray")

        fig, ax = plt.subplots(
            figsize=(img_width / 128, img_height / 128), tight_layout=True
        )
        ax.imshow(reconstructed_image, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"CT partial reconstruction scales {list(selected_scale_indices)[0]}"
            f" - {list(selected_scale_indices)[-1]}"
        )
        plt.show()

    print("Curvelet transform based reconstruction is done")
    return reconstructed_image
