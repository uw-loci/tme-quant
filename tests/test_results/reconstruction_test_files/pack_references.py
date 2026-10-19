"""Convert MATLAB-generated .mat cases to portable, pickle-free NPZ files."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source-dir",
    type=Path,
    required=True,
    help="CurveLab fdct_wrapping_matlab directory",
)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = args.source_dir
metadata = {
    "matlab_version": None,
    "source_files": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            source / "fdct_wrapping.m",
            source / "ifdct_wrapping.m",
            source / "fdct_wrapping_window.m",
        ]
    },
    "cases": [],
}
for p in sorted(root.glob("matlab_case_*.mat")):
    m = loadmat(p)
    arrays = {
        k: m[k]
        for k in [
            "input",
            "cutoff",
            "reconstructed",
            "roundtrip",
            "scales",
            "angles",
            "selected_scales",
        ]
    }
    for field in ["C", "Ct"]:
        cells = m[field].ravel()
        arrays[field + "_wedges"] = np.array([s.size for s in cells])
        for j, s in enumerate(cells):
            for k, w in enumerate(s.ravel()):
                arrays[f"{field}_{j}_{k}"] = w
    target = p.with_suffix(".npz")
    np.savez_compressed(target, **arrays)
    metadata["matlab_version"] = m["matlab_version"].item()
    metadata["cases"].append(
        {
            "file": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "shape": list(m["input"].shape),
            "angles": int(m["angles"].item()),
            "description": "zero"
            if p.stem.endswith("09")
            else "impulse"
            if p.stem.endswith("10")
            else "modular ramp",
            "integration_threshold_test": not p.stem.endswith("10"),
        }
    )
(root / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
