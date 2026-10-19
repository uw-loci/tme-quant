"""Compare extraction stages with a MATLAB v7.3 runtime capture.

Run from the repository with:
    python scripts/compare_matlab_runtime.py runtime_reference.mat --output result.json

Requires h5py and a compiled fiber_backend. Reconstruction is intentionally
bypassed: both implementations start from the captured FIRE input.
"""

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import numpy as np


def read_capture(path):
    with h5py.File(path) as handle:
        def array(name):
            return np.asarray(handle[name]).T

        def fibers(name):
            return [
                {"v": (np.asarray(handle[ref]).ravel().astype(int) - 1).tolist()}
                for ref in np.asarray(handle[name]["v"]).ravel()
            ]

        params = {}
        for key, value in handle["p"].items():
            data = np.asarray(value)
            if value.attrs.get("MATLAB_class") == b"char":
                params[key] = "".join(chr(int(x)) for x in data.ravel())
            elif value.attrs.get("MATLAB_empty", 0):
                params[key] = []
            elif data.size == 1:
                params[key] = data.item()
            else:
                params[key] = data.ravel().tolist()
        for key in ("s_xlinkbox", "s_fiberdir", "thresh_numv", "ang_interval"):
            params[key] = int(params[key])
        # The current extension wrapper requires an integer LMP separation.
        separation = params["thresh_LMPdist"]
        if separation != int(separation):
            raise ValueError("The current backend cannot represent fractional thresh_LMPdist")
        params["thresh_LMPdist"] = int(separation)
        return {
            "p": params, "im": array("im"), "dsm": array("dsm").squeeze(),
            "xlink": array("xlink"), "Xz": array("Xz"), "Fz": fibers("Fz"),
            "Rz": array("Rz").ravel(), "Xa": array("Xa"), "Fa": fibers("Fa"),
        }


def path_multiset(coordinates, fibers):
    paths = Counter()
    for fiber in fibers:
        path = tuple(map(tuple, np.asarray(coordinates)[fiber["v"], :2].astype(int)))
        paths[min(path, path[::-1])] += 1
    return paths


def statistics(coordinates, fibers, reference_paths=None):
    coordinates = np.asarray(coordinates)
    result = {
        "fibers": len(fibers),
        "total_length": float(sum(
            np.linalg.norm(np.diff(coordinates[fiber["v"]], axis=0), axis=1).sum()
            for fiber in fibers
        )),
    }
    if reference_paths is not None:
        result["exact_shared_paths"] = sum(
            (path_multiset(coordinates, fibers) & reference_paths).values()
        )
    return result


def compare(reference, backend_dir=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if backend_dir:
        sys.path.insert(0, str(backend_dir))
        importlib.import_module("fiber_backend")
    module = importlib.import_module("ctfire_py.fire_2d_angle")
    backend = module.fiber_backend
    capture = read_capture(reference)
    params = capture["p"]
    height, width = capture["dsm"].shape
    extension_paths = path_multiset(capture["Xz"] - 1, capture["Fz"])
    processed_paths = path_multiset(capture["Xa"] - 1, capture["Fa"])
    saved_dsm = capture["dsm"].ravel().astype(np.float32)
    points = (capture["xlink"][:, [1, 0, 2]] - 1).astype(np.int32)
    observed = {}
    find_local_max = backend.find_local_max

    def record_nucleation(*args):
        before = args[3].copy()
        result = find_local_max(*args)
        observed["dsm"] = before
        observed["mutated_pixels"] = int(np.count_nonzero(args[3] != before))
        observed["points"] = result.copy()
        return result

    # Keep the command's JSON output separate from ordinary pipeline logs.
    with contextlib.redirect_stdout(sys.stderr):
        with patch.object(backend, "find_local_max", side_effect=record_nucleation):
            data = module.fire_2d_angle(params, capture["im"], plotflag=0)
        x, fibers, _, radii = backend.extend_xlink(
            1, height, width, saved_dsm.copy(), points, params
        )
        extension = statistics(
            np.asarray(x)[:, [1, 0, 2]] - [0, 0, 1], fibers, extension_paths
        )
        # Retain MATLAB coordinate values and axis order to isolate fiberproc.
        xa, fa, _, _, _ = backend.process_fibers(
            1, height, width, saved_dsm.copy(), capture["Xz"].astype(np.int32),
            capture["Fz"], capture["Rz"].astype(np.float32), params,
        )
    return {
        "reference": str(Path(reference).resolve()), "backend": backend.__file__,
        "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        "parameters": params,
        "matlab_extension": statistics(capture["Xz"], capture["Fz"]),
        "matlab_processed": statistics(capture["Xa"], capture["Fa"]),
        "preprocessing": {
            "dsm_max_abs_error": float(np.max(np.abs(observed["dsm"] - saved_dsm))),
            "dsm_different_pixels": int(np.count_nonzero(observed["dsm"] != saved_dsm)),
            "dsm_mutated_by_nucleation_pixels": observed["mutated_pixels"],
            "nucleation_count": len(observed["points"]),
            "shared_nucleation_points": len(
                set(map(tuple, observed["points"])) & set(map(tuple, points))
            ),
        },
        "extension_on_matlab_inputs": extension,
        "fiberproc_on_matlab_inputs": statistics(np.asarray(xa) - 1, fa, processed_paths),
        "current_pipeline_processed": statistics(
            np.asarray(data["Xa"])[:, [1, 0, 2]] - [0, 0, 1], data["Fa"], processed_paths
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    result = compare(args.reference, args.backend_dir)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
