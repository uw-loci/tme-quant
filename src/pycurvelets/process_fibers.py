import numpy as np
import pandas as pd

from pycurvelets.models import FeatureControlParameters
from pycurvelets.utils.math import circ_r
from sklearn.neighbors import NearestNeighbors


def process_fibers(fiber_structure, feature_cp: FeatureControlParameters):
    """
    Parameters
    ----------
    fiber_structure: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
            - ``angle``: orientation angle in degrees
            - ``center_row``: row coordinate of the curvelet center
            - ``center_col``: column coordinate of the curvelet center

    feature_cp: FeatureControlParameters
        Dataclass for control parameters for extracted features, e.g.:
            - minimum_nearest_fibers : int
                Minimum nearest fibers for localized fiber density and alignment calculation
            - minimum_box_size : int
                Minimum box size for localized fiber
            - fiber_midpoint_estimation: int
                0 = based on end points coordinate, 1 = based on fiber length density and alignment calculation
    Returns:
    --------
    density_list : np.ndarray
        Density features array
    alignment_list : np.ndarray
        Alignment features array
    """
    # Addd weight column to fiber_structure
    fiber_structure = fiber_structure.copy()
    fiber_structure["weight"] = np.nan

    fiber_number = len(fiber_structure)
    x = fiber_structure["center_row"].to_numpy()
    y = fiber_structure["center_col"].to_numpy()
    centers = np.column_stack((x, y))
    centers_arr = np.asarray(centers, dtype=np.float64)
    angles = fiber_structure["angle"].to_numpy()

    minimum_nearest_fibers = feature_cp.minimum_nearest_fibers
    minimum_box_size = feature_cp.minimum_box_size

    # Keep 4 original nearest fiber features
    nearest_fibers = [2**i * minimum_nearest_fibers for i in range(4)]
    box_sizes = [2**i * minimum_box_size for i in range(3)]
    fiber_sizes = np.ceil(np.array(box_sizes) / 2)

    density_list = np.full(
        (fiber_number, len(fiber_sizes) + len(nearest_fibers)), np.nan
    )
    alignment_list = np.full(
        (fiber_number, len(fiber_sizes) + len(nearest_fibers)), np.nan
    )

    K = nearest_fibers[-1] + 1
    nbrs = NearestNeighbors(n_neighbors=K, metric="euclidean", algorithm="brute").fit(
        centers
    )
    nearest_neighbor_dist, nearest_neighbor_idx = nbrs.kneighbors(centers)

    for i in range(fiber_number):
        neighbor_angles = angles[nearest_neighbor_idx[i, :]]
        for j, num_neighbors in enumerate(nearest_fibers):
            if num_neighbors <= nearest_neighbor_dist.shape[1]:
                density_list[i][j] = nearest_neighbor_dist[
                    i, 1 : num_neighbors + 1
                ].mean()
                alignment_list[i][j] = circ_r(
                    neighbor_angles[1 : num_neighbors + 1] * 2 * np.pi / 180
                )
            else:
                # if fiber number is less than number of nearest neighbors, then don't calculate
                density_list[i][j] = np.nan
                alignment_list[i][j] = np.nan

        # density box filter
        for j in range(len(fiber_sizes)):
            # find any positions in square region around current fiber
            square_mask = (
                (x > x[i] - fiber_sizes[j])
                & (x < x[i] + fiber_sizes[j])
                & (y > y[i] - fiber_sizes[j])
                & (y < y[i] + fiber_sizes[j])
            )

            # get all fibers in that area
            vals = np.vstack(fiber_structure[square_mask].angle)
            col_idx = len(nearest_fibers) + j
            density_list[i, col_idx] = len(vals)
            alignment_list[i, col_idx] = circ_r(vals * 2 * np.pi / 180, axis=None)

        fiber_structure.loc[i, "weight"] = np.nan

    density_df = pd.DataFrame(
        {
            f"distance_to_nearest_{(2 ** 0) * minimum_nearest_fibers}_fibers": [
                density_list[i][0] for i in range(len(density_list))
            ],
            f"distance_to_nearest_{(2 ** 1) * minimum_nearest_fibers}_fibers": [
                density_list[i][1] for i in range(len(density_list))
            ],
            f"distance_to_nearest_{(2 ** 2) * minimum_nearest_fibers}_fibers": [
                density_list[i][2] for i in range(len(density_list))
            ],
            f"distance_to_nearest_{(2 ** 3) * minimum_nearest_fibers}_fibers": [
                density_list[i][3] for i in range(len(density_list))
            ],
            "distance_to_nearest_fiber_mean": np.mean(
                density_list[:, : len(nearest_fibers)], axis=1
            ),
            "distance_to_nearest_fiber_std": np.std(
                density_list[:, : len(nearest_fibers)], axis=1, ddof=1
            ),
            f"fibers_within_box_density{(2 ** 0) * minimum_box_size}": [
                density_list[i][4] for i in range(len(density_list))
            ],
            f"fibers_within_box_density{(2 ** 1) * minimum_box_size}": [
                density_list[i][5] for i in range(len(density_list))
            ],
            f"fibers_within_box_density{(2 ** 2) * minimum_box_size}": [
                density_list[i][6] for i in range(len(density_list))
            ],
        }
    )
    alignment_df = pd.DataFrame(
        {
            f"alignment_of_nearest_{(2 ** 0) * minimum_nearest_fibers}_fibers": [
                alignment_list[i][0] for i in range(len(alignment_list))
            ],
            f"alignment_of_nearest_{(2 ** 1) * minimum_nearest_fibers}_fibers": [
                alignment_list[i][1] for i in range(len(alignment_list))
            ],
            f"alignment_of_nearest_{(2 ** 2) * minimum_nearest_fibers}_fibers": [
                alignment_list[i][2] for i in range(len(alignment_list))
            ],
            f"alignment_of_nearest_{(2 ** 3) * minimum_nearest_fibers}_fibers": [
                alignment_list[i][3] for i in range(len(alignment_list))
            ],
            "alignment_mean": np.mean(alignment_list[:, : len(nearest_fibers)], axis=1),
            "alignment_std": np.std(
                alignment_list[:, : len(nearest_fibers)], axis=1, ddof=1
            ),
            f"fiber_alignment_in_box_{(2 ** 0) * minimum_box_size}": [
                alignment_list[i][4] for i in range(len(alignment_list))
            ],
            f"fiber_alignment_in_box_{(2 ** 1) * minimum_box_size}": [
                alignment_list[i][5] for i in range(len(alignment_list))
            ],
            f"fiber_alignment_in_box_{(2 ** 2) * minimum_box_size}": [
                alignment_list[i][6] for i in range(len(alignment_list))
            ],
        }
    )

    return density_df, alignment_df
