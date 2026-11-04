#!/usr/bin/env python3
"""
CurveAlign end-to-end pipeline: the reference for the Fiji plugin (issue #66).

Mirror run_curvealign() in the plugin. It is the only function that talks to the
library. Everything else here is what Fiji replaces: main() stands in for the
plugin's @Parameter fields, and load_grayscale() stands in for the open image.

    img = load_grayscale("real1.tif")
    out = run_curvealign(img, "real1", "CA_Out", boundary_mask=load_grayscale("mask_real1.tiff"))
    out["features"]          # per-fiber DataFrame, or None if no fibers were found
    out["files"]["overlay"]  # path to the overlay TIFF

Setup (curvelet mode needs the curvelops extra; see doc/INSTALL.md):
    Linux, WSL, macOS:  bash bin/install.sh, then prefix commands with "uv run"
    Windows:            activate your MSYS2 UCRT64 curvelops environment

Run from the repo root:
    python examples/curvealign_pipeline.py tests/test_images/real1.tif
    python examples/curvealign_pipeline.py tests/test_images/real1.tif \
        --boundary-mask tests/test_images/CA_Boundary/mask_real1.tiff

Outputs, written to --output-dir as <image name> + suffix:
    _overlay.tiff               fibers drawn on the image
    _procmap.tiff               alignment heatmap
    _FiberFeatures.xlsx         per-fiber table, same data as out["features"]
    _hist.csv, _values.csv      angle histogram and angle values
    _reconstructed.tiff         curvelet mode only
    _BoundaryMeasurements.xlsx  boundary mask only

Limitations of process_image to design around:
    - It returns only the feature table, so images must be read back from the files.
    - total_length, end_to_end_length, curvature and width are always NaN.
    - CSV boundaries (tif_boundary 1 or 2) are not implemented, so they are not offered here.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

# Must precede any pyplot import. process_image draws its figures with pyplot, and the
# default backend can need a display (WSL) or the main thread (macOS).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pycurvelets.models import (
    AdvancedAnalysisOptions,
    BoundaryParameters,
    FiberAnalysisParameters,
    ImageInputParameters,
    OutputControlParameters,
)


def load_grayscale(path):
    """Read an image as a 2-D array. In Fiji, the array comes from the open image instead."""
    img = plt.imread(path)
    if img.ndim == 3:  # RGB or RGBA; any alpha channel is ignored
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    return img


def collect_outputs(output_dir, img_name, fiber_mode=0, has_boundary=False):
    """Locate this run's output files by name, since process_image does not return them."""
    suffixes = {
        "overlay": "_overlay.tiff",
        "procmap": "_procmap.tiff",
        "features": "_FiberFeatures.xlsx",
        "histogram": "_hist.csv",
        "values": "_values.csv",
    }
    # Only ask for what this mode writes, so leftovers from an earlier run are not reported.
    if fiber_mode == 0:
        suffixes["reconstructed"] = "_reconstructed.tiff"
    if has_boundary:
        suffixes["boundary_measurements"] = "_BoundaryMeasurements.xlsx"

    paths = {label: Path(output_dir) / f"{img_name}{s}" for label, s in suffixes.items()}
    return {label: path for label, path in paths.items() if path.exists()}


def run_curvealign(
    img,
    img_name,
    output_dir,
    *,
    keep=0.1,
    fiber_mode=0,
    fire_dir=None,
    boundary_mask=None,
    distance_threshold=100.0,
):
    """
    Run the full CurveAlign analysis on one 2-D image.

    img, boundary_mask  2-D arrays; a mask switches on boundary analysis
    img_name            prefix for the output files; also selects the FIRE results
    keep                fraction of curvelet coefficients to keep
    fiber_mode          0 = curvelet; 1, 2, 3 = read existing CT-FIRE results
    fire_dir            folder holding ct_fire_out/ct_fire_out<img_name>.mat (FIRE only)
    distance_threshold  how far from the boundary to evaluate fibers, in pixels

    Returns {"features": DataFrame or None, "files": {label: Path}}.
    """
    if fiber_mode != 0 and fire_dir is None:
        raise ValueError("FIRE modes (fiber_mode 1, 2, 3) require fire_dir")

    # Imported here because this pulls in curvelops, the compiled curvelet backend.
    from pycurvelets.process_image import process_image

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    has_boundary = boundary_mask is not None

    # Step 1: the image. The name sets the output file prefix.
    image_params = ImageInputParameters(img=img, img_name=img_name)

    # Step 2: how fibers are found.
    fiber_params = FiberAnalysisParameters(
        fiber_mode=fiber_mode,
        keep=keep,
        fire_directory=None if fire_dir is None else str(fire_dir),
    )

    # Step 3: what gets written. The feature file also controls the returned table.
    output_params = OutputControlParameters(
        output_directory=str(output_dir),
        make_associations=True,
        make_map=True,
        make_overlay=True,
        make_feature_file=True,
    )

    # Step 4: the boundary. tif_boundary 3 = TIFF mask, 0 = no boundary.
    boundary_params = BoundaryParameters(
        distance_threshold=distance_threshold,
        tif_boundary=3 if has_boundary else 0,
        boundary_img=boundary_mask,
    )

    # Step 5: advanced options. A first plugin need not expose these. 16 matches
    # tests/example_process_image.py; the dataclass default is 19.
    advanced_options = AdvancedAnalysisOptions(
        exclude_fibers_in_mask=has_boundary,
        heatmap_STD_filter_size=16,
    )

    # Step 6: the single library call. With a mask it starts a multiprocessing.Pool, so
    # the calling script needs an `if __name__ == "__main__":` guard.
    result = process_image(
        image_params=image_params,
        fiber_params=fiber_params,
        output_params=output_params,
        boundary_params=boundary_params,
        advanced_options=advanced_options,
    )

    # Step 7: collect results. process_image returns None when it finds no fibers.
    return {
        "features": result["fib_feat_df"] if result else None,
        "files": collect_outputs(output_dir, img_name, fiber_mode, has_boundary),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CurveAlign on one image and write the results to a folder."
    )
    parser.add_argument("input", type=Path, help="2-D image file, such as a TIFF")
    parser.add_argument("--output-dir", type=Path, help="default: <image folder>/CA_Out")
    parser.add_argument("--keep", type=float, default=0.1, help="fraction of curvelets to keep")
    parser.add_argument("--boundary-mask", type=Path, help="TIFF mask; enables boundary analysis")
    parser.add_argument(
        "--distance-threshold", type=float, default=100.0, help="pixels from the boundary"
    )
    parser.add_argument(
        "--fiber-mode", type=int, default=0, choices=[0, 1, 2, 3], help="0 = curvelet, 1-3 = FIRE"
    )
    parser.add_argument("--fire-dir", type=Path, help="CT-FIRE results folder; required for FIRE")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input image not found: {args.input}")
    if args.boundary_mask and not args.boundary_mask.is_file():
        parser.error(f"boundary mask not found: {args.boundary_mask}")
    if args.fiber_mode != 0 and args.fire_dir is None:
        parser.error("--fire-dir is required when --fiber-mode is 1, 2 or 3")

    # process_image prints emoji, which crash a piped stdout using a legacy Windows encoding.
    # A plugin that runs Python as a subprocess should set UTF-8 too.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    output_dir = args.output_dir or args.input.parent / "CA_Out"
    out = run_curvealign(
        load_grayscale(args.input),
        args.input.stem,
        output_dir,
        keep=args.keep,
        fiber_mode=args.fiber_mode,
        fire_dir=args.fire_dir,
        boundary_mask=load_grayscale(args.boundary_mask) if args.boundary_mask else None,
        distance_threshold=args.distance_threshold,
    )

    features = out["features"]
    if features is None:
        print("No fibers detected.", file=sys.stderr)
        return 1

    angles = features["fiber_absolute_angle"]
    print(f"\nFibers detected: {len(features)}")
    print(f"Angle mean, std: {angles.mean():.2f}, {angles.std():.2f} degrees")
    if "alignment_mean" in features:
        print(f"Alignment mean:  {features['alignment_mean'].mean():.4f}")
    print(f"Output files in {output_dir}:")
    for label, path in out["files"].items():
        print(f"  {label}: {path.name}")
    return 0


if __name__ == "__main__":  # also required by multiprocessing; see Step 6
    sys.exit(main())
