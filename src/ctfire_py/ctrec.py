import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import scoreatpercentile
from pycurvelets.utils.math import round_mlab

from curvelops import fdct2d_wrapper


def ct_rec_1(img, fctr, pct, SS, plot_flag=False):
    """
    Obtaining curvelet transform based reconstruction image for denoising
    and fiber edge enhancement.
    """
    # 1. Setup paths and filenames
    # Replaces CTR_ with CTRimg_ and .mat with .tif
    ct_img_name = fctr.replace("CTR_", "CTRimg_").replace(".mat", ".tif")

    # Ensure input is a float64 for transform precision
    is_img = np.asanyarray(img, dtype=np.float64)
    pix_h, pix_w = is_img.shape

    # Apply the FDCT to the image
    # Note: Python implementation uses different parameter ordering from MATLAB
    # is_real=0 in MATLAB corresponds to ac=1 in Python (complex-valued transform)
    M, N = img.shape
    is_real = 0  # 0 means complex
    ac = 0  # 1 is curvelets, 0 is wavelets
    nbscales = math.floor(math.log2(min(M, N)) - 3)
    nbangles_coarse = 16  # default
    c = fdct2d_wrapper.fdct2d_forward_wrap(nbscales, nbangles_coarse, ac, img)

    # 3. Thresholding Logic
    # Flatten all coefficients across all scales and wedges into one array
    all_cfs = []
    for scale in c:
        for wedge in scale:
            all_cfs.append(np.abs(wedge).flatten())

    all_cfs = np.concatenate(all_cfs)
    all_cfs.sort()
    all_cfs = all_cfs[::-1]  # Descending sort

    # Get specific threshold based on percentile (pct)
    nb = int(round_mlab(pct * len(all_cfs)))
    # Handle edge case where pct might be 0 or 1
    idx = min(nb, len(all_cfs) - 1)
    cutoff = all_cfs[idx]

    # 4. Filter coefficients by threshold
    for s in range(len(c)):
        for w in range(len(c[s])):
            # Apply hard thresholding
            c[s][w] = c[s][w] * (np.abs(c[s][w]) > cutoff)

    # 5. Select specific scales (SS)
    # Equivalent to MATLAB's s = length(C)-SS : length(C)-1
    num_scales = len(c)
    selected_scale_indices = range(num_scales - SS, num_scales - 1)

    # Create an empty coefficient structure (zeros)
    Ct = []
    for s in range(num_scales):
        scale_wedges = []
        for w in range(len(c[s])):
            if s in selected_scale_indices:
                scale_wedges.append(c[s][w])
            else:
                scale_wedges.append(np.zeros_like(c[s][w]))
        Ct.append(scale_wedges)

    # 6. Inverse Curvelet Transform
    Y = fdct2d_wrapper.fdct2d_inverse_wrap(M, N, nbscales, 16, 0, Ct)
    out_ct = np.real(Y)

    # 7. Plotting and Saving
    if plot_flag:
        plt.figure(figsize=(pix_w / 128, pix_h / 128))
        plt.imshow(out_ct, cmap="gray")
        plt.axis("image")
        plt.title(
            f"CT partial reconstruction scales {selected_scale_indices[0]} - {selected_scale_indices[-1]}"
        )

        # Save the image
        plt.savefig(ct_img_name, dpi=128)
        plt.show()

    print("Curvelet transform based reconstruction is done")
    return out_ct


if __name__ == "__main__":
    plot_flag = 1
    SS = 3
    pct = 0.2
    import os
    import matplotlib.pyplot as plt

    img = plt.imread(
        os.path.join(
            os.path.dirname(__file__),
            "2B_D9_ROI1.tif",
        ),
        format="TIF",
    )

    ct_rec_1(plot_flag=plot_flag, SS=SS, pct=pct, img=img, fctr="2B_D9_ROI1.tif")
