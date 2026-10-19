"""Compare the raw-image CT-FIRE pipeline with a MATLAB runtime capture.

Example (parameters must match the capture's original CT-FIRE run):
    python scripts/compare_matlab_reconstruction.py runtime_reference.mat \
        tests/test_images/real1.tif --fraction 0.2 --scales 3 --threshold 5 \
        --output reconstruction_comparison.json

Requires h5py and the compiled fiber_backend. MATLAB and curvelops are not
needed at runtime. This command runs ct_fire, including its image conversion,
original-image mask, reconstruction, and extraction.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

from compare_matlab_runtime import path_multiset, read_capture, statistics


def compare(reference, image_path, fraction=0.2, scales=3, threshold=5):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    ct_module = importlib.import_module("ctfire_py.ct_fire")
    fire_module = importlib.import_module("ctfire_py.fire_2d_angle")
    capture = read_capture(reference)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    observed = {}
    find_local_max = fire_module.fiber_backend.find_local_max
    fire = ct_module.fire_2d_angle

    def record_points(*args):
        observed["dsm"] = args[3].copy()
        points = find_local_max(*args)
        observed["points"] = points.copy()
        return points

    def record_input(*args, **kwargs):
        observed["input"] = kwargs["im"].copy()
        return fire(*args, **kwargs)

    params = dict(capture["p"], thresh_im2=threshold)
    with contextlib.redirect_stdout(sys.stderr):
        with (
            patch.object(ct_module, "fire_2d_angle", side_effect=record_input),
            patch.object(
                fire_module.fiber_backend, "find_local_max", side_effect=record_points
            ),
        ):
            _, result = ct_module.ct_fire(
                None,
                image_path.name,
                "",
                {"show_plots": False, "save_images": False},
                {
                    "value": params,
                    "coefficient_percentile": fraction,
                    "num_scales": scales,
                },
                img=image,
            )
    data = result["data"]
    reference_points = (capture["xlink"][:, [1, 0, 2]] - 1).astype(int)
    difference = observed["input"] - capture["im"]
    return {
        "reference": str(reference.resolve()),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "raw_image": str(image_path.resolve()),
        "raw_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "parameters": {"fraction": fraction, "scales": scales, "threshold": threshold},
        "reconstruction": {
            "max_abs_error": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "foreground_different_pixels": int(
                np.count_nonzero((observed["input"] > 0) != (capture["im"] > 0))
            ),
        },
        "preprocessing": {
            "dsm_different_pixels": int(
                np.count_nonzero(observed["dsm"] != capture["dsm"].ravel())
            ),
            "dsm_max_abs_error": float(
                np.max(np.abs(observed["dsm"] - capture["dsm"].ravel()))
            ),
            "nucleation_count": len(observed["points"]),
            "shared_nucleation_points": len(
                set(map(tuple, observed["points"])) & set(map(tuple, reference_points))
            ),
        },
        "matlab_processed": statistics(capture["Xa"], capture["Fa"]),
        "raw_image_pipeline_processed": statistics(
            np.asarray(data["Xa"])[:, [1, 0, 2]] - [0, 0, 1],
            data["Fa"],
            path_multiset(capture["Xa"] - 1, capture["Fa"]),
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    rendered = json.dumps(
        compare(args.reference, args.image, args.fraction, args.scales, args.threshold),
        indent=2,
    )
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
