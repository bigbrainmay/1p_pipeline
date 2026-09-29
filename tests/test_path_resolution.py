from __future__ import annotations

import unittest
import re

try:
    from preprocess_functions.manifest import resolve_lab_path
except ImportError:
    resolve_lab_path = None


@unittest.skipIf(resolve_lab_path is None, "pandas is required to import manifest helpers")
class PathResolutionTests(unittest.TestCase):
    def test_lab_drive_preserves_absolute_output_drive_paths(self) -> None:
        path = resolve_lab_path(
            r"D:\preprocess_out\C5731A\neural\cells.csv",
            lab_drive=r"Z:\\",
        )

        self.assertEqual(
            normalize_windowsish(path),
            r"d:\preprocess_out\c5731a\neural\cells.csv",
        )

    def test_lab_drive_remaps_lab_relative_drive_paths(self) -> None:
        path = resolve_lab_path(
            r"C:\Data\May\RSC-PPC\file.csv",
            lab_drive=r"Z:\\",
        )

        self.assertEqual(
            normalize_windowsish(path),
            r"z:\data\may\rsc-ppc\file.csv",
        )

    def test_lab_drive_remaps_plain_lab_relative_paths(self) -> None:
        path = resolve_lab_path(
            r"Data\May\RSC-PPC\file.csv",
            lab_drive=r"Z:\\",
        )

        self.assertEqual(
            normalize_windowsish(path),
            r"z:\data\may\rsc-ppc\file.csv",
        )


def normalize_windowsish(path) -> str:
    return re.sub(r"\\+", r"\\", str(path).replace("/", "\\")).lower()


if __name__ == "__main__":
    unittest.main()
