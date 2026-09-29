from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from preprocess_functions import plot
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    if exc.name != "matplotlib":
        raise
    plt = None
    plot = None


class PlotVideoAxisTests(unittest.TestCase):
    @unittest.skipIf(plot is None, "matplotlib is required for plot axis tests")
    def test_cell_summary_trajectory_uses_video_y_axis(self) -> None:
        df = pd.DataFrame(
            {
                "ear_mid_x": np.linspace(1, 8, 8),
                "ear_mid_y": np.linspace(2, 9, 8),
                "head_dir_rad": np.linspace(-np.pi, np.pi, 8),
                "registered_cell_0": np.linspace(0, 1, 8),
            }
        )

        fig, axes = plot.plot_cell_summary(
            df,
            cell_col="registered_cell_0",
            arena_mask=None,
            show=False,
        )

        try:
            self.assertTrue(axes[0, 0].yaxis_inverted())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
