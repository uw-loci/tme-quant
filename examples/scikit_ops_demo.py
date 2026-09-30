"""Discover and run the starter TME-Quant ops through Appose."""

from pathlib import Path

import numpy as np
import skop

from tme_quant_ops import (
    boundary_labels,
    enhance_tubeness,
    largest_boundary_points,
    segment_threshold,
)


root = Path(__file__).parents[1]
image = np.zeros((64, 64), dtype=np.float32)
image[16:48, 20:44] = 1

specs, failures = skop.discover("tme_quant_ops")
if failures:
    raise RuntimeError("\n".join(str(failure) for failure in failures))
print("Discovered:", *(spec.name for spec in specs), sep="\n  ")

with skop.Runner(root=root / "src", envs_dir=root / "envs") as runner:
    enhanced = runner.run(enhance_tubeness, image=image, sigma=1.0)
    labels = runner.run(
        segment_threshold,
        image=enhanced,
        min_area=10,
        remove_border_objects=False,
    )
    boundaries = runner.run(boundary_labels, labels=labels, thickness=2)
    points = runner.run(
        largest_boundary_points, labels=labels, simplify_tolerance=1.0
    )

print(
    f"objects={int(labels.max())}, "
    f"boundary_pixels={np.count_nonzero(boundaries)}, "
    f"boundary_points={len(points)}"
)
