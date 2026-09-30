# TME-Quant scikit-ops

The `tme_quant_ops` package exposes TME-Quant computations as ordinary Python
functions decorated with `skop.op`. They can be called directly or dispatched
to an isolated Appose environment.

## Install

```bash
uv sync --extra scikit-ops
```

## Discover the collection

```python
import skop

specs, failures = skop.discover("tme_quant_ops")
for spec in specs:
    print(spec.name, spec.env)
for failure in failures:
    print(failure)
```

The collection currently contains:

| Op | Environment | Inputs | Outputs |
| --- | --- | --- | --- |
| `enhance_tubeness` | `tme-quant` | image, sigma | enhanced image |
| `enhance_frangi` | `tme-quant` | image, scale range, beta, gamma | enhanced image |
| `segment_threshold` | `tme-quant` | image and threshold settings | label image |
| `boundary_labels` | `tme-quant` | label image, thickness | boundary label image |
| `largest_boundary_points` | `tme-quant` | label image, simplification | ordered points |
| `summarize_fiber_neighborhoods` | `tme-quant` | fiber vectors, neighbor count | scalar statistics |
| `analyze_tacs` | `tme-quant` | fiber vectors, ordered boundary | associations and statistics |
| `extract_curvelets` | `tme-quant-curvelets` | image and curvelet settings | fiber vectors and statistics |

## Call an op directly

Decorated ops remain ordinary functions. Direct calls use the packages in the
current Python environment and are best for development and unit tests:

```python
from skimage.io import imread
from tme_quant_ops import (
    enhance_tubeness,
    segment_threshold,
    largest_boundary_points,
)

image = imread("image.tif")
enhanced = enhance_tubeness(image, sigma=1.5)
labels = segment_threshold(
    enhanced,
    method="otsu",
    min_area=100,
    fill_holes=True,
    remove_border_objects=True,
)
boundary = largest_boundary_points(labels, simplify_tolerance=1.0)
```

## Run an op through Appose

`Runner` uses the op's `@op(env=...)` declaration and the matching
`envs/<env-id>/pixi.toml`. The first call builds the environment; later calls
reuse it and its worker process.

```python
from pathlib import Path

import skop
from skimage.io import imread
from tme_quant_ops import enhance_tubeness, segment_threshold

root = Path("/path/to/tme-quant")
image = imread("image.tif")

with skop.Runner(root=root / "src", envs_dir=root / "envs") as runner:
    enhanced = runner.run(enhance_tubeness, image=image, sigma=1.5)
    labels = runner.run(
        segment_threshold,
        image=enhanced,
        method="otsu",
        min_area=100,
    )
```

Keyword arguments are recommended. They are the same parameter names that a
generated napari or Fiji form will expose.

## Preprocessing ops

```python
from tme_quant_ops import enhance_frangi, enhance_tubeness

tubular = enhance_tubeness(image, sigma=1.0)
vessels = enhance_frangi(
    image,
    sigma_min=1.0,
    sigma_max=10.0,
    sigma_step=1.0,
    beta=0.5,
    gamma=15.0,
)
```

Both accept a 2-D grayscale image or an RGB(A) image and return a `float32`
image. Their `ImageData` roles tell a host to display the result as an image.

## Segmentation and boundaries

```python
from tme_quant_ops import (
    boundary_labels,
    largest_boundary_points,
    segment_threshold,
)

labels = segment_threshold(
    image,
    method="triangle",  # otsu, triangle, isodata, mean, or minimum
    min_area=100,
    fill_holes=True,
    remove_border_objects=True,
)

boundary_image = boundary_labels(labels, thickness=2)
boundary_points = largest_boundary_points(labels, simplify_tolerance=1.0)
```

`boundary_labels` is intended for display. `largest_boundary_points` returns
the ordered `(row, column)` contour needed for tangent-based TACS analysis. It
selects only the longest contour; images containing multiple independent tumor
boundaries will need a future multi-boundary representation.

## Fiber representation

Fiber ops use the scikit-ops `VectorsData` representation:

```text
shape: (number_of_fibers, 2, 2)
fiber[:, 0]: center in (row, column) order
fiber[:, 1]: direction/displacement in (row, column) order
```

To construct vectors from existing measurements:

```python
import numpy as np
from pycurvelets.fiber_ops import fibers_to_vectors

centers = np.array([[20, 30], [25, 35], [30, 40]], dtype=float)
angles = np.array([0, 45, 90], dtype=float)
fibers = fibers_to_vectors(centers, angles, length=10)
```

The angle convention is axial degrees in `[0, 180)`: 0 degrees points along
the column/x axis and 90 degrees along the row/y axis.

## Fiber neighborhood statistics

```python
from tme_quant_ops import summarize_fiber_neighborhoods

summary = summarize_fiber_neighborhoods(fibers, neighbors=4)
print(summary.fiber_count)
print(summary.mean_nearest_distance)
print(summary.mean_local_alignment)
print(summary.global_alignment)
```

Alignment values are axial resultant lengths between zero and one. One means
parallel fibers; values near zero indicate dispersed orientations.

## TACS analysis

```python
from tme_quant_ops import analyze_tacs

result = analyze_tacs(
    fibers,
    boundary_points,
    min_distance=0,
    max_distance=200,
    tacs3_threshold=60,
)

print(result.fiber_count)
print(result.mean_distance)
print(result.mean_relative_angle)
print(result.tacs3_fraction)
associations = result.associations
```

Relative angles range from 0 degrees (tangential) to 90 degrees
(perpendicular). `associations` is `VectorsData` connecting each retained
fiber center to its nearest boundary point, so GUI hosts can display the
matches directly.

## Curvelet extraction

```python
from tme_quant_ops import extract_curvelets

result = extract_curvelets(
    image,
    keep=0.05,
    scale=1,
    grouping_radius=10,
    vector_length=10,
)
print(result.fibers, result.fiber_count, result.global_alignment)
```

This op requires the separately licensed native CurveLab/FFTW backend. Its
wrapper and environment boundary are defined, but the
`tme-quant-curvelets` Pixi environment deliberately does not pretend to be
portable yet: `curvelops` needs a reproducible native build recipe before Fiji
or napari can build it unattended. Direct calls work in an environment where
the repository's normal CurveLab installation procedure has been completed.

## Complete starter pipeline

```python
enhanced = enhance_tubeness(image, sigma=1.0)
labels = segment_threshold(enhanced, min_area=100)
boundary = largest_boundary_points(labels)
curvelets = extract_curvelets(image)
neighborhood = summarize_fiber_neighborhoods(curvelets.fibers)
tacs = analyze_tacs(curvelets.fibers, boundary, max_distance=200)
```

Each stage remains independently testable. Rendering overlays, saving TIFF or
Excel files, selecting napari layers, and managing Fiji ROIs are host concerns
and intentionally remain outside these core ops.

## GUI discovery status

Explicit discovery and Appose execution work now. The stock `skop-napari` and
`skop-fiji` projects currently default to the built-in `skop.ops` package,
however, so this separate `tme_quant_ops` collection does not automatically
appear in their menus. External collection registration must be added upstream,
or these ops must be contributed to the built-in collection, before the stock
hosts show them without a project-specific adapter.
