from __future__ import annotations

import re
import unittest
from pathlib import Path

import pandas as pd

from preprocess_functions.manifest import SessionRecord, add_manifest_ids, normalize_manifest_table
from scripts.run_matlab_neural_extraction import choose_neural_input_h5


class MotionCorrectionManifestTests(unittest.TestCase):
    def test_motion_corrected_h5_alias_loads_into_record(self) -> None:
        table = pd.DataFrame(
            {
                "date": ["2026-08-06"],
                "mouse_id": ["C5731A"],
                "motion_corrected_h5": [r"D:\preprocess_out\rec\neural\rec_miniscope_mc.h5"],
            }
        )

        normalized = add_manifest_ids(normalize_manifest_table(table))
        record = SessionRecord.from_row(normalized.iloc[0])

        self.assertEqual(
            normalize_windowsish(record.neu_mc_h5),
            r"d:\preprocess_out\rec\neural\rec_miniscope_mc.h5",
        )

    def test_extraction_prefers_manifest_motion_corrected_h5(self) -> None:
        record = SessionRecord(
            row_index=0,
            sheet_name=None,
            mouse_id="C5731A",
            session_id="C5731A_20260806",
            recording_id="C5731A_20260806",
            date=None,
            neu_mc_h5=Path(r"D:\preprocess_out\rec\neural\rec_miniscope_mc.h5"),
        )
        raw_h5 = Path(r"D:\preprocess_out\rec\neural\rec_miniscope.h5")

        chosen, input_kind = choose_neural_input_h5(
            record,
            Path(r"D:\preprocess_out"),
            raw_h5=raw_h5,
            ignore_motion_corrected=False,
            use_motion_corrected=False,
        )

        self.assertEqual(normalize_windowsish(chosen), normalize_windowsish(record.neu_mc_h5))
        self.assertEqual(input_kind, "motion_corrected_manifest")

    def test_extraction_can_ignore_motion_corrected_h5(self) -> None:
        record = SessionRecord(
            row_index=0,
            sheet_name=None,
            mouse_id="C5731A",
            session_id="C5731A_20260806",
            recording_id="C5731A_20260806",
            date=None,
            neu_mc_h5=Path(r"D:\preprocess_out\rec\neural\rec_miniscope_mc.h5"),
        )
        raw_h5 = Path(r"D:\preprocess_out\rec\neural\rec_miniscope.h5")

        chosen, input_kind = choose_neural_input_h5(
            record,
            Path(r"D:\preprocess_out"),
            raw_h5=raw_h5,
            ignore_motion_corrected=True,
            use_motion_corrected=False,
        )

        self.assertEqual(normalize_windowsish(chosen), normalize_windowsish(raw_h5))
        self.assertEqual(input_kind, "raw")


def normalize_windowsish(path) -> str:
    return re.sub(r"\\+", r"\\", str(path).replace("/", "\\")).lower()


if __name__ == "__main__":
    unittest.main()
