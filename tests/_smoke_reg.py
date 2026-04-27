"""Ad-hoc smoke test: run the SHG/HE registration and print metrics.

Usage (from tme-quant/): ``uv run python tests/_smoke_reg.py [case_id]``
where case_id is one of test1/test2/test3; defaults to all three.

Override the algorithm via env vars:
``SMOKE_METHOD`` -> ``mi_ncc`` (default), ``mi``, ``ncc``, ``oneplusone``.
``SMOKE_ECM`` -> ``hsv`` (default), ``rgb``, ``lab``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from skimage import io

from pycurvelets._registration_quality import compute_registration_quality_metrics
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
    ecm = os.environ.get("SMOKE_ECM")
    params_kwargs: dict[str, Any] = dict(
        HEfilepath=str(_FIXTURE / "HE"),
        HEfilename="patient_001.tif",
        pixelpermicron=ppm,
        SHGfilepath=str(_FIXTURE / "SHG"),
        areaThreshold=5000.0,
    )
    if method:
        params_kwargs["registration_method"] = method
    if ecm:
        params_kwargs["ecm_method"] = ecm
    params = SHGHERegistrationParameters(**params_kwargs)
    t0 = time.perf_counter()
    reg_float, debug = shg_he_registration(
        params, save_output=False, return_debug=True
    )
    dt = time.perf_counter() - t0
    reg_uint8 = (np.clip(reg_float, 0, 1) * 255).astype(np.uint8)

    metrics = compute_registration_quality_metrics(reg_uint8, golden)
    mae = float(metrics["mae_uint8"])
    rmse = float(metrics["rmse_uint8"])
    exact = float(metrics["exact_frac"])
    within5 = float(metrics["within5_frac"])
    within10 = float(metrics["within10_frac"])
    within20 = float(metrics["within20_frac"])
    psnr = float(metrics["psnr"])
    ssim = float(metrics["ssim"])

    print(
        f"[{case_id}] ppm={ppm} shape={reg_uint8.shape} "
        f"backend={debug.get('registration_backend')} "
        f"method={params_kwargs.get('registration_method', 'mi_ncc')} "
        f"ecm={params_kwargs.get('ecm_method', 'hsv')}"
    )
    print(f"    runtime      : {dt:6.2f} s")
    print(f"    MAE          : {mae:6.3f} / 255")
    print(f"    RMSE         : {rmse:6.3f} / 255")
    print(f"    PSNR         : {psnr:6.3f} dB")
    print(f"    SSIM         : {ssim:6.4f}")
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
