import pandas as pd
import numpy as np
from pycurvelets.utils.math import circ_r


class FiberProcessor:
    """
    Unified processor for CT or FIRE fiber data
    """

    def __init__(self, feature_cp):
        """
        Parameters
        ----------
        feature_cp: dict
            Dictionary for control parameters for extracted features, e.g.:
                - minimum_nearest_fibers : int
                    Minimum nearest fibers for localized fiber density and alignment calculation
                - minimum_box_size : int
                    Minimum box size for localized fiber
                - fiber_midpoint_estimation: int
                    0 = based on end points coordinate, 1 = based on fiber length density and alignment calculation
        """
        self.feature_cp = feature_cp

    def process(self, object_list):
        """
        Given list of fiber dicts: [{'center':(x,y), 'angle':deg, ...}, ...]
        Compute density, alignment, etc.
        Returns pd.DataFrame
        """

        minimum_nearest_fibers = self.feature_cp["minimum_nearest_fibers"]
        minimum_box_size = self.feature_cp["minimum_box_size"]

        # Keep 4 original nearest fiber features
        nearest_fibers = [2**i * minimum_nearest_fibers for i in range(4)]
        box_sizes = [2**i * mbs for i in range(3)]
        fiber_sizes = np.ceil(np.array(box_sizes) / 2)

        len_fiber_sizes = len(fiber_sizes)
        len_nearest_fibers = len(nearest_fibers)

        density_list = []
        alignment_list = []

        centers = np.vstack(obj.center for obj in object_list)
        angles = np.vstack(obj.angle for obj in object_list)
        x = centers[:, 0]
        y = centers[:, 1]

        K = nearest_fibers[-1] + 1
        nbrs = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(centers)
        nearest_neighbor_dist, nearest_neighbor_idx = nbrs.kneighbors(centers)

        for i in range(len(object_list)):
            angles_i = angles[nearest_neighbor_idx[i, :]]
            for j in range(len_nearest_fibers):
                if nearest_fibers[j] <= nearest_neighbor_dist.shape[1] - 1:
                    # skip the first column (the self-distance)
                    try:
                        density_list[i][j] = nearest_neighbor_dist[
                            i, 1 : nearest_neighbor[j] + 1
                        ].mean()
                        alignment_list[i][j] = circ_r(
                            angles_i[1 : nearest_neighbor[j] + 1] * 2 * np.pi / 180
                        )
                    except:
                        print(f"size of density list: {len(density_list)}")
                else:
                    # if fiber number is less than number of nearest neighbors, then don't calculate
                    density_list[i][j] = np.nan
                    alignment_list[i][j] = np.nan

            # density box filter
            for j in range(len_fiber_sizes):
                # find any positions in square region around current fiber
                square_mask = (
                    (x > x[i] - fiber_sizes[j])
                    & (x < x[i] + fiber_sizes[j])
                    & (y > y[i] - fiber_sizes[j])
                    & (y < y[i] + fiber_sizes[j])
                )

                # get all fibers in that area
                vals = np.vstack(object)

        # --- assemble final DataFrame ---
        df = pd.DataFrame(
            {
                "x": centers[:, 0],
                "y": centers[:, 1],
                "angle": angles,
                "density_mean": density_mean,
                "density_std": density_std,
                "alignment_mean": alignment_mean,
                "alignment_std": alignment_std,
            }
        )

        for k in range(len(n_list)):
            df[f"density_n{k+1}"] = density_array[:, k]
            df[f"alignment_n{k+1}"] = alignment_array[:, k]
        for k in range(len(box_sizes)):
            df[f"density_box{k+1}"] = box_density[:, k]
            df[f"alignment_box{k+1}"] = box_alignment[:, k]

        return df
