from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.build_aligned_sessions import add_registered_cell_aliases


class RegisteredCellAliasTests(unittest.TestCase):
    def test_adds_registered_aliases_without_fragmentation_warning(self) -> None:
        n_cells = 160
        n_rows = 12
        aligned = pd.DataFrame(
            {
                f"cell_{idx}": np.arange(n_rows, dtype=float) + idx
                for idx in range(n_cells)
            }
        )
        record = SimpleNamespace(
            recording_id="recording_1",
            mouse_id="mouse_1",
            sheet_name=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            registration_path = (
                output_root
                / "coregistration"
                / "mouse_1"
                / "mouse_1_cell_registration.csv"
            )
            registration_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "recording_id": ["recording_1"] * n_cells,
                    "registered_cell_id": [
                        f"registered_cell_{idx}" for idx in range(n_cells)
                    ],
                    "cell_col": [f"cell_{idx}" for idx in range(n_cells)],
                }
            ).to_csv(registration_path, index=False)

            with warnings.catch_warnings():
                warnings.simplefilter("error", pd.errors.PerformanceWarning)
                out, summary = add_registered_cell_aliases(
                    aligned,
                    record,
                    output_root,
                )

        self.assertEqual(summary["registered_cell_status"], "registered_aliases_added")
        self.assertEqual(summary["n_registered_cell_aliases"], n_cells)
        self.assertEqual(
            out["registered_cell_42"].tolist(),
            aligned["cell_42"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
