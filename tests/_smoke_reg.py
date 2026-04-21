"""Ad-hoc smoke test: run the SHG/HE registration and print metrics.

Usage (from tme-quant/): ``uv run python tests/_smoke_reg.py [case_id]``
where case_id is one of test1/test2/test3; defaults to all three.

Override the algorithm via the ``SMOKE_METHOD`` env var:
``mi_ncc`` (default), ``mi``, ``ncc``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from skimage import io

from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

_ROOT = Path(__file__).resolve().parent
_FIXTURE = _ROOT / "test_for_shg_he_registration_BDcreation"

CASES: tuple[tuple[str, float, str], ...] = (
    ("test1", 1.5, "HE_registered_test1"),
    ("test2", 2.0, "HE_registered_test2"),
    ("test3", 3.0, "HE_registered_test3"),
)


def _run_case(case_id: str, ppm: float, gold_folder: str) -> None:
    gpath = _FIXTURE / "HE" / gold_folder / "patient_001.tif"
    golden = io.imread(str(gpath))
    if golden.dtype != np.uint8:
        golden = np.clip(golden, 0, 255).astype(np.uint8)

    import os
    method = os.environ.get("SMOKE_METHOD")  # None -> use dataclass default
    params_kwargs: dict[str, Any] = dict(
        HEfilepath=str(_FIXTURE / "HE"),
        HEfilename="patient_001.tif",
        pixelpermicron=ppm,
        SHGfilepath=str(_FIXTURE / "SHG"),
        areaThreshold=5000.0,
    )
    if method:
        params_kwargs["registration_method"] = method
    params = SHGHERegistrationParameters(**params_kwargs)
    t0 = time.perf_counter()
    reg_float, debug = shg_he_registration(
        params, save_output=False, return_debug=True
    )
    dt = time.perf_counter() - t0
    reg_uint8 = (np.clip(reg_float, 0, 1) * 255).astype(np.uint8)

    diff = np.abs(reg_uint8.astype(np.int32) - golden.astype(np.int32))
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    exact = float((reg_uint8 == golden).mean())
    within5 = float((diff <= 5).mean())
    within10 = float((diff <= 10).mean())
    within20 = float((diff <= 20).mean())

    print(f"[{case_id}] ppm={ppm}  shape={reg_uint8.shape}  backend={debug.get('registration_backend')}")
    print(f"    runtime      : {dt:6.2f} s")
    print(f"    MAE          : {mae:6.3f} / 255")
    print(f"    RMSE         : {rmse:6.3f} / 255")
    print(f"    exact match  : {exact*100:5.2f} %")
    print(f"    within  5    : {within5*100:5.2f} %")
    print(f"    within 10    : {within10*100:5.2f} %")
    print(f"    within 20    : {within20*100:5.2f} %")
    print(f"    seed dice    : {debug.get('seed_dice')}")
    print(f"    sim params   : angle={debug.get('sim_angle_rad')} scale={debug.get('sim_scale')} tx={debug.get('sim_tx')} ty={debug.get('sim_ty')}")
    fwd = debug.get("forward_2x3")
    if fwd is not None:
        print(f"    forward_2x3  : {fwd}")
    print()


def main() -> None:
    selected = sys.argv[1:] or [c[0] for c in CASES]
    for cid, ppm, folder in CASES:
        if cid in selected:
            _run_case(cid, ppm, folder)


if __name__ == "__main__":
    main()
