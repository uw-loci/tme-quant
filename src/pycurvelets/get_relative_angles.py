import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from skimage.measure import regionprops, label
import csv
import os

from pycurvelets.utils.geometry import find_outline_slope, circ_r


def get_relative_angles(ROI, obj, angle_option=0, fig_flag=False):
    """
    Compute relative angle measurements between an object and a polygonal ROI (Region of Interest).

    This function extracts geometric properties of the ROI and compares them with the
    orientation and center of a given object. Depending on the selected `angle_option`,
    it calculates angles between:
      - the object's orientation and the ROI boundary at a specific point,
      - the object's orientation and the ROI's overall orientation,
      - the object's orientation and the line connecting the object and ROI centers.

    Optionally, a visualization can be generated showing the ROI, object, centers, and
    orientation vectors.

    Parameters
    ----------
    ROI : dict
        Dictionary describing the region of interest with keys:
          - "coords" : array_like of shape (N, 2)
              Polygon coordinates of the ROI boundary, given as (row, col) or (y, x).
          - "imageHeight" : int
              Height of the image containing the ROI.
          - "imageWidth" : int
              Width of the image containing the ROI.
          - "index2object" : int
              Index into `coords` specifying the boundary point closest to the object.
    obj : dict
        Dictionary describing the object with keys:
          - "center" : array_like of shape (2,)
              Object center coordinates (y, x).
          - "angle" : float
              Object orientation angle in degrees.
    angle_option : {0, 1, 2, 3}, default=0
        Specifies which relative angle(s) to compute:
          - 0 : Compute all available relative angles.
          - 1 : Only compute angle between object and ROI boundary edge.
          - 2 : Only compute angle between object and ROI center orientation.
          - 3 : Only compute angle between object and ROI center-to-center line.
    fig_flag : bool, default=False
        If True and `angle_option == 0`, display a matplotlib figure showing the ROI,
        object center, ROI center, orientations, and connecting lines.

    Returns
    -------
    relative_angles : dict
        Dictionary containing computed angles in degrees:
          - "angle2boundaryEdge" : float or None
              Angle between object orientation and ROI boundary edge orientation at the
              specified boundary point.
          - "angle2boundaryCenter" : float or None
              Angle between object orientation and overall ROI orientation.
          - "angle2centersLine" : float or None
              Angle between object orientation and the line connecting object and ROI centers.
    ROImeasurements : dict
        Dictionary containing measurements of the ROI:
          - "center" : ndarray of shape (2,)
              ROI centroid coordinates (x, y).
          - "orientation" : float
              ROI orientation angle in degrees.
          - "area" : int
              Pixel area of the ROI.
          - "boundary" : ndarray of shape (N, 2)
              ROI boundary coordinates (y, x).

    Raises
    ------
    ValueError
        If the provided ROI coordinates do not define exactly one connected region.

    Notes
    -----
    - ROI orientation is derived from `regionprops`, which may differ slightly
      from MATLAB's `regionprops` implementation.
    - Angles are normalized such that the maximum deviation is 90°.
    - If the boundary point lies on the image edge, `angle2boundaryEdge` defaults to 0.

    Examples
    --------
    >>> ROI = {
    ...     "coords": np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
    ...     "imageHeight": 50,
    ...     "imageWidth": 50,
    ...     "index2object": 1
    ... }
    >>> obj = {"center": np.array([15, 25]), "angle": 45}
    >>> rel_angles, roi_meas = get_relative_angles(ROI, obj, angle_option=0, fig_flag=False)
    >>> rel_angles["angle2boundaryCenter"]
    30.0
    """

    coords = np.array(ROI["coords"])  # [[y1, x1], [y2, x2], ...]
    image_height = ROI["imageHeight"]
    image_width = ROI["imageWidth"]
    index2object = ROI["index2object"]
    object_center = obj["center"][::-1]  # switch to [x, y]
    object_angle = obj["angle"]

    # Step 1: Create ROI mask and compute properties
    # coords_xy = np.flip(coords, axis=1)  # now shape: (N, 2), with (x, y)

    # Step 2: Use polygon2mask, which behaves more like MATLAB's roipoly
    mask = polygon2mask((image_height, image_width), coords)  # output is bool array

    # Step 3: Label and compute properties (regionprops values are slightly off though)
    labeled = label(mask.astype(np.uint8))
    props = regionprops(labeled)

    if len(props) != 1:
        raise ValueError("Coordinates must define a single region")

    prop = props[0]
    boundary_center = np.array(prop.centroid)[::-1]  # to [x, y]
    roi_angle = -90 + np.degrees(prop.orientation)
    if roi_angle < 0:
        roi_angle = 180 + roi_angle

    ROImeasurements = {
        "center": boundary_center,
        "orientation": roi_angle,
        "area": prop.area,
        "boundary": coords,
    }

    relative_angles = {
        "angle2boundaryEdge": None,
        "angle2boundaryCenter": None,
        "angle2centersLine": None,
    }

    if angle_option in [1, 0]:
        boundary_pt = coords[index2object]
        boundary_angle = find_outline_slope(coords, index2object)
        if (
            any(boundary_pt == 1)
            or boundary_pt[0] == image_height
            or boundary_pt[1] == image_width
        ):
            temp_ang = 0
        else:
            if boundary_angle is None:
                temp_ang = None
            else:
                temp_ang = circ_r(
                    [np.radians(2 * object_angle), np.radians(2 * boundary_angle)]
                )
                temp_ang = np.degrees(np.arcsin(temp_ang))
        relative_angles["angle2boundaryEdge"] = temp_ang

    if angle_option in [2, 0]:
        temp_ang = abs(object_angle - roi_angle)
        if temp_ang > 90:
            temp_ang = 180 - temp_ang
        relative_angles["angle2boundaryCenter"] = temp_ang

    if angle_option in [3, 0]:
        dx = object_center[1] - boundary_center[1]
        dy = object_center[0] - boundary_center[0]
        centers_line_angle = np.degrees(np.arctan(dx / dy))
        if centers_line_angle < 0:
            centers_line_angle = abs(centers_line_angle)
        else:
            centers_line_angle = 180 - centers_line_angle
        relative_angles["angle2centersLine"] = abs(centers_line_angle - object_angle)
        if relative_angles["angle2centersLine"] > 90:
            relative_angles["angle2centersLine"] = (
                180 - relative_angles["angle2centersLine"]
            )

    if fig_flag and angle_option == 0:
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap="gray")
        ax.plot(coords[:, 1], coords[:, 0], "c-", label="Boundary")
        ax.plot(object_center[0], object_center[1], "ro", label="Object Center")
        ax.plot(boundary_center[0], boundary_center[1], "go", label="Boundary Center")

        dx = 100 * np.cos(np.radians(object_angle))
        dy = -100 * np.sin(np.radians(object_angle))
        ax.arrow(object_center[0], object_center[1], dx, dy, color="g", head_width=5)

        dx_roi = 100 * np.cos(np.radians(roi_angle))
        dy_roi = -100 * np.sin(np.radians(roi_angle))
        ax.arrow(
            boundary_center[0],
            boundary_center[1],
            dx_roi,
            dy_roi,
            color="m",
            head_width=5,
        )

        ax.plot(
            [object_center[0], boundary_center[0]],
            [object_center[1], boundary_center[1]],
            "r--",
            label="Centers Line",
        )

        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_title("Angle Visualization")
        ax.legend()
        plt.show()

    return relative_angles, ROImeasurements


def load_coords(csv_path):
    """
    Loads tab-separated (Y, X) coordinates from a CSV file and returns (Y, X) NumPy array.
    """
    csv_path = os.path.normpath(os.path.expanduser(csv_path.strip()))

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    coords = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                y, x = map(float, row[:2])
                coords.append([y, x])
    return np.array(coords)


coords = load_coords(
    "/Users/dongwoolee/Documents/GitHub/tme-quant/tests/test_results/relative_angle_test_files/boundary_coords.csv"
)

ROI = {
    "coords": coords,
    "imageWidth": 512,
    "imageHeight": 512,
    "index2object": 403,
}

object_data = {"center": [145, 430], "angle": 14.0625}

angles, measurements = get_relative_angles(
    ROI, object_data, angle_option=0, fig_flag=False
)

pretty_angles = {k: float(v) for k, v in angles.items()}

print(pretty_angles)
