from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from preprocess_functions import boundary_tuning


class RegisteredCellBoundaryTuningTests(unittest.TestCase):
    def test_cell_idx_accepts_registered_cell_columns(self) -> None:
        self.assertEqual(boundary_tuning.cell_idx_from_col("cell_17"), 17)
        self.assertEqual(boundary_tuning.cell_idx_from_col("registered_cell_17"), 17)

    def test_egocentric_boundary_map_accepts_registered_cell_column(self) -> None:
        arena_mask = np.ones((12, 12), dtype=bool)
        df = pd.DataFrame(
            {
                "ear_mid_x": [5.0, 6.0, 7.0],
                "ear_mid_y": [5.0, 5.0, 5.0],
                "nose.x": [6.0, 7.0, 8.0],
                "nose.y": [5.0, 5.0, 5.0],
                "registered_cell_0": [0.2, 0.5, 0.8],
            }
        )

        result = boundary_tuning.compute_egocentric_boundary_map(
            df,
            arena_mask,
            cell_col="registered_cell_0",
            angle_bins=8,
            distance_bins=4,
            boundary_stride=2,
        )

        self.assertEqual(result["cell_col"], "registered_cell_0")
        self.assertEqual(result["cell_idx"], 0)
        self.assertEqual(result["mean_activity"].shape, (4, 8))

    def test_boundary_shuffle_results_accept_registered_cell_column(self) -> None:
        arena_mask = np.ones((12, 12), dtype=bool)
        df = pd.DataFrame(
            {
                "ear_mid_x": np.linspace(3.0, 8.0, 12),
                "ear_mid_y": np.full(12, 5.0),
                "nose.x": np.linspace(4.0, 9.0, 12),
                "nose.y": np.full(12, 5.0),
                "registered_cell_0": np.linspace(0.1, 1.2, 12),
            }
        )

        results = boundary_tuning.compute_boundary_tuning_results(
            df,
            arena_mask,
            cell_cols=["registered_cell_0"],
            n_bins=3,
            n_shuffles=3,
            random_state=0,
        )

        self.assertEqual(results.loc[0, "cell_col"], "registered_cell_0")
        self.assertEqual(results.loc[0, "cell_idx"], 0)
        self.assertEqual(len(results.loc[0, "shuffle_rhos"]), 3)


if __name__ == "__main__":
    unittest.main()
