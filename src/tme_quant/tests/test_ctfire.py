"""
Tests for CTFireExtraction -- soft IoU against ground truth centerline.

Soft IoU formula (Gaussian-smoothed, following centerline.py):
    smooth each 1-px skeleton with a Gaussian (sigma=5) rescaled to [0,1],
    then  IoU = (m1*m2).sum() / (m1²+m2²-m1*m2).sum()

sigma=5 measures spatial agreement at the ~10-px scale, which is the right
tolerance for testing "did CT-FIRE find the fibers" rather than pixel precision.
"""
from __future__ import annotations
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from skimage import draw, filters, exposure, morphology
from skimage.transform import resize as sk_resize
from tme_quant.fiber_analysis.methods.ctfire import CTFireExtraction
from tme_quant.fiber_analysis.config import CTFireParams

# Gaussian sigma for smoothing 1-px skeletons before computing soft IoU.
# sigma=5 → ~10-px FWHM, appropriate for fiber-level (not pixel-level) matching.
_SMOOTH_SIGMA = 2.0


def _smooth_mask(mask, smooth_sigma=_SMOOTH_SIGMA):
    mask = mask.astype(np.float32)
    mask = exposure.rescale_intensity(mask, out_range=(0.0, 1.0))
    density = filters.gaussian(mask, sigma=smooth_sigma, preserve_range=False)
    return exposure.rescale_intensity(density, out_range=(0.0, 1.0)).astype(np.float32)


def _soft_iou(mask_1, mask_2, beta=1e-3):
    if mask_1.shape != mask_2.shape:
        mask_2 = sk_resize(
            mask_2, mask_1.shape, anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
    intersection = mask_1 * mask_2
    union = mask_1**2 + mask_2**2 - mask_1 * mask_2
    return float((intersection.sum() + beta) / (union.sum() + beta))


def _rasterize_centerlines(fibers, image_shape):
    canvas = np.zeros(image_shape, dtype=np.uint8)
    H, W = image_shape
    for fiber in fibers:
        pts = fiber.centerline
        if pts is None or len(pts) < 2:
            continue
        pts = np.round(pts[:, :2]).astype(int)
        for i in range(len(pts) - 1):
            r0 = int(np.clip(pts[i, 0], 0, H - 1))
            c0 = int(np.clip(pts[i, 1], 0, W - 1))
            r1 = int(np.clip(pts[i + 1, 0], 0, H - 1))
            c1 = int(np.clip(pts[i + 1, 1], 0, W - 1))
            rr, cc = draw.line(r0, c0, r1, c1)
            canvas[rr, cc] = 1
    return morphology.skeletonize(canvas > 0)


def _make_synthetic_fiber_image(
    shape=(256, 256),
    n_fibers=10,
    fiber_sigma=2.0,
    min_length=60,
    rng_seed=42,
):
    """
    Synthetic fiber image with Gaussian cross-section profiles.

    Fibers are drawn as straight lines; each pixel's intensity is
    exp(-d^2 / (2*fiber_sigma^2)) where d is the distance to the nearest
    centerline.  This matches the Gaussian ridge appearance of real SHG
    collagen images that CT-FIRE was designed for.

    Returns
    -------
    image : (H, W) float32, values in [0, 1]
    gt_skeleton : (H, W) bool  -- 1-px-thick ground-truth centerlines
    """
    rng = np.random.default_rng(rng_seed)
    H, W = shape
    margin = max(10, min_length // 4)
    skeleton_map = np.zeros(shape, dtype=bool)

    generated = 0
    attempts = 0
    while generated < n_fibers and attempts < n_fibers * 10:
        attempts += 1
        r0 = int(rng.integers(margin, H - margin))
        c0 = int(rng.integers(margin, W - margin))
        angle = rng.uniform(0, np.pi)  # avoid duplicate reversed segments
        length = int(rng.integers(min_length, max(min_length + 1, min(H, W) - 2 * margin)))
        r1 = int(np.clip(r0 + length * np.sin(angle), margin, H - margin))
        c1 = int(np.clip(c0 + length * np.cos(angle), margin, W - margin))
        actual_len = np.hypot(r1 - r0, c1 - c0)
        if actual_len < min_length:
            continue
        rr, cc = draw.line(r0, c0, r1, c1)
        skeleton_map[rr, cc] = True
        generated += 1

    # Gaussian cross-section via distance transform (fiber amplitude 0.85)
    dist = distance_transform_edt(~skeleton_map).astype(np.float32)
    fiber_signal = 0.85 * np.exp(-dist**2 / (2.0 * fiber_sigma**2))

    # SHG-like background: non-zero mean with Gaussian noise, as CT-FIRE's
    # percentile-based bright-pixel logic requires that fibers occupy the
    # top ~8% of pixel intensities (impossible on a near-zero background).
    background = rng.normal(0.15, 0.04, size=shape).astype(np.float32)
    image = np.clip(background + fiber_signal, 0.0, 1.0)

    gt_skeleton = morphology.skeletonize(skeleton_map)
    return image, gt_skeleton


# CT-FIRE parameters tuned for the synthetic images above.
# ctfire_threshold=0.25 keeps only strong Frangi ridges, suppressing the
# many short false-positive strands that appear at lower thresholds.
# min_fiber_width=2.0 further rejects thin noise strands (< 1 px radius).
_CTFIRE_PARAMS = CTFireParams(
    pixel_size=1.0,
    min_fiber_length=25.0,
    max_fiber_length=1000.0,
    min_fiber_width=2.0,
    max_fiber_width=15.0,
    ctfire_threshold=0.25,
    mask_closing_radius=1,
    spur_length_px=5,
    extract_centerlines=True,
)

SOFT_IOU_THRESHOLD = 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_centerline_overlay(
    image: np.ndarray,
    fibers,
    gt_skeleton: np.ndarray | None = None,
    title: str = "CT-FIRE centerline overlay",
    save_path: str | None = None,
):
    """
    Display extracted fiber centerlines overlaid on the source image.

    Each detected fiber is drawn in a distinct color from a qualitative
    colormap.  The optional ground-truth skeleton is shown in white.

    Parameters
    ----------
    image : (H, W) float32
        Grayscale source image (values in [0, 1]).
    fibers : list of FiberProperties
        CT-FIRE output fibers; only those with a ``centerline`` are drawn.
    gt_skeleton : (H, W) bool, optional
        Ground-truth 1-px skeleton.  Drawn in white if provided.
    title : str
        Figure title (also used as the window title).
    save_path : str, optional
        If given, save the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    if gt_skeleton is not None:
        # Green semi-transparent overlay for ground truth
        gt_rgba = np.zeros((*gt_skeleton.shape, 4), dtype=np.float32)
        gt_rgba[gt_skeleton, :] = [0.0, 1.0, 0.0, 0.8]
        ax.imshow(gt_rgba, interpolation="nearest")

    # Pick a qualitative colormap with enough distinct colors
    cmap = plt.get_cmap("tab20")
    fibers_with_cl = [f for f in fibers if f.centerline is not None and len(f.centerline) >= 2]
    H, W = image.shape

    for idx, fiber in enumerate(fibers_with_cl):
        color = cmap(idx % 20)
        pts = np.round(fiber.centerline[:, :2]).astype(int)
        # col (x) first, then row (y) for matplotlib
        xs = np.clip(pts[:, 1], 0, W - 1)
        ys = np.clip(pts[:, 0], 0, H - 1)
        ax.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.9)
        # Mark the start point
        ax.plot(xs[0], ys[0], "o", color=color, markersize=3, alpha=0.9)

    n_gt = int(gt_skeleton.sum()) if gt_skeleton is not None else 0
    ax.set_title(
        f"{title}\n"
        f"{len(fibers_with_cl)} detected fibers"
        + (f" | GT pixels: {n_gt}" if gt_skeleton is not None else ""),
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo  (python tests/test_ctfire.py)
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    """Generate a synthetic image, run CT-FIRE, and show the overlay."""
    image, gt_skeleton = _make_synthetic_fiber_image(
        shape=(256, 256), n_fibers=10, fiber_sigma=2.0, rng_seed=42
    )
    result = CTFireExtraction().extract_2d(image, _CTFIRE_PARAMS)
    pred_skeleton = _rasterize_centerlines(result.fibers, image.shape)
    ratio = _soft_iou(_smooth_mask(gt_skeleton), _smooth_mask(pred_skeleton))
    print(f"Detected {len(result.fibers)} fibers | soft IoU = {ratio:.4f}")
    plot_centerline_overlay(
        image,
        result.fibers,
        gt_skeleton=gt_skeleton,
        title=f"CT-FIRE overlay  (soft IoU = {ratio:.4f})",
    )


if __name__ == "__main__":
    _demo()


class TestCTFireSoftIoU:
    def test_soft_iou_synthetic(self):
        image, gt_skeleton = _make_synthetic_fiber_image(
            shape=(256, 256), n_fibers=10, fiber_sigma=2.0, rng_seed=42
        )
        result = CTFireExtraction().extract_2d(image, _CTFIRE_PARAMS)
        assert result.fibers, "CT-FIRE returned no fibers."
        pred_skeleton = _rasterize_centerlines(result.fibers, image.shape)
        ratio = _soft_iou(_smooth_mask(gt_skeleton), _smooth_mask(pred_skeleton))
        assert ratio > SOFT_IOU_THRESHOLD, (
            f"Soft IoU {ratio:.4f} < {SOFT_IOU_THRESHOLD}"
        )

    def test_soft_iou_uses_centerlines(self):
        image, _ = _make_synthetic_fiber_image(
            shape=(128, 128), n_fibers=5, fiber_sigma=2.0, rng_seed=7
        )
        result = CTFireExtraction().extract_2d(image, _CTFIRE_PARAMS)
        fibers_with_cl = [f for f in result.fibers if f.centerline is not None]
        assert fibers_with_cl, "No fibers have centerlines."
        assert _rasterize_centerlines(result.fibers, image.shape).any(), (
            "Skeleton is blank."
        )

    @pytest.mark.parametrize("n_fibers,seed", [(5, 0), (8, 13), (10, 99)])
    def test_soft_iou_multiple_configurations(self, n_fibers, seed):
        image, gt_skeleton = _make_synthetic_fiber_image(
            shape=(256, 256), n_fibers=n_fibers, fiber_sigma=2.0, rng_seed=seed
        )
        result = CTFireExtraction().extract_2d(image, _CTFIRE_PARAMS)
        assert result.fibers, f"No fibers (n_fibers={n_fibers}, seed={seed})."
        pred_skeleton = _rasterize_centerlines(result.fibers, image.shape)
        ratio = _soft_iou(_smooth_mask(gt_skeleton), _smooth_mask(pred_skeleton))
        assert ratio > SOFT_IOU_THRESHOLD, (
            f"Soft IoU {ratio:.4f} < {SOFT_IOU_THRESHOLD} "
            f"(n_fibers={n_fibers}, seed={seed})"
        )
