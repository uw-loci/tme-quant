"""
Tests for examples/curvealign_pipeline.py. The same file runs on Windows, WSL and macOS.

Fast tests only:     pytest tests/test_curvealign_pipeline.py
With real curvelets: TMEQ_RUN_CURVELETS=1 pytest tests/test_curvealign_pipeline.py -v -rs
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "curvealign_pipeline.py"
IMAGE = REPO / "tests" / "test_images" / "real1.tif"
MASK = REPO / "tests" / "test_images" / "CA_Boundary" / "mask_real1.tiff"

# examples/ and the test images may be absent outside a git clone, such as in an sdist.
if not SCRIPT.exists():
    pytest.skip("examples/ is not available", allow_module_level=True)

requires_curvelets = pytest.mark.skipif(
    os.environ.get("TMEQ_RUN_CURVELETS") != "1",
    reason="curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location("curvealign_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(*args, cwd):
    """Run the script as a headless, piped subprocess, as a plugin or CI job would."""
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "MPLBACKEND")}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
        timeout=1800,
    )


@pytest.mark.parametrize(
    "shape", [(16, 24), (16, 24, 3), (16, 24, 4)], ids=["gray", "rgb", "rgba"]
)
def test_load_grayscale_returns_2d(pipeline, tmp_path, shape):
    path = tmp_path / "image.tif"
    Image.fromarray(np.full(shape, 100, dtype=np.uint8)).save(path)

    img = pipeline.load_grayscale(path)

    assert img.shape == (16, 24)
    assert np.allclose(img, 100)


def test_collect_outputs_finds_only_what_this_mode_writes(pipeline, tmp_path):
    for name in ("a_overlay.tiff", "a_hist.csv", "a_BoundaryMeasurements.xlsx", "b_procmap.tiff"):
        (tmp_path / name).touch()

    assert set(pipeline.collect_outputs(tmp_path, "a")) == {"overlay", "histogram"}
    assert set(pipeline.collect_outputs(tmp_path, "a", has_boundary=True)) == {
        "overlay",
        "histogram",
        "boundary_measurements",
    }


def test_run_curvealign_requires_fire_dir_for_fire_modes(pipeline, tmp_path):
    with pytest.raises(ValueError, match="fire_dir"):
        pipeline.run_curvealign(np.zeros((8, 8)), "a", tmp_path, fiber_mode=1)


@pytest.mark.parametrize(
    "args, message",
    [
        (["missing.tif"], "input image not found"),
        ([IMAGE, "--boundary-mask", "missing.tiff"], "boundary mask not found"),
        ([IMAGE, "--fiber-mode", "1"], "--fire-dir is required"),
    ],
    ids=["missing-image", "missing-mask", "fire-without-dir"],
)
def test_cli_rejects_bad_arguments(tmp_path, args, message):
    result = run_cli(*args, cwd=tmp_path)

    assert result.returncode == 2
    assert message in result.stderr


@requires_curvelets
def test_cli_without_boundary(tmp_path):
    output_dir = tmp_path / "output with space"

    result = run_cli(IMAGE, "--output-dir", output_dir, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Fibers detected:" in result.stdout
    for suffix in (
        "_overlay.tiff",
        "_procmap.tiff",
        "_FiberFeatures.xlsx",
        "_hist.csv",
        "_values.csv",
        "_reconstructed.tiff",
    ):
        assert (output_dir / f"real1{suffix}").is_file(), suffix
    assert not (output_dir / "real1_BoundaryMeasurements.xlsx").exists()


@requires_curvelets
def test_cli_with_boundary_mask(tmp_path):
    """Boundary analysis starts a Pool, so this covers each OS's start method."""
    output_dir = tmp_path / "output with space"

    result = run_cli(IMAGE, "--boundary-mask", MASK, "--output-dir", output_dir, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert (output_dir / "real1_BoundaryMeasurements.xlsx").is_file()
    features = pd.read_excel(output_dir / "real1_FiberFeatures.xlsx")
    assert features["nearest_distance_to_boundary"].notna().any()
