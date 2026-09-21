from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records
from preprocess_functions.pipeline import default_neu_h5_path
from preprocess_functions.video import repair_extract_h5_layout


def main() -> None:
    args = parse_args()
    target = Path(args.input)
    rows = []

    if target.suffix.lower() in {".h5", ".hdf5"}:
        rows.append(repair_one(target, fail_fast=args.fail_fast))
    else:
        output_root = Path(args.output_root)
        records = load_session_records(
            target,
            sheet_name=args.sheet,
            lab_drive=args.lab_drive,
        )
        for record in records:
            if not selected(record, args.only):
                continue

            h5_path = record.neu_h5 if args.use_manifest_h5 and record.neu_h5 else default_neu_h5_path(output_root, record)
            row = {
                "recording_id": record.recording_id,
                "session_id": record.session_id,
                "neu_h5": str(h5_path),
            }
            if h5_path is None or not Path(h5_path).exists():
                rows.append({**row, "status": "missing_h5"})
                print(f"{record.recording_id}: missing H5 {h5_path}")
                continue

            summary = repair_one(h5_path, fail_fast=args.fail_fast)
            rows.append({**row, **summary})

    index_path = Path(args.index_out) if args.index_out else Path(args.output_root) / "h5_extract_layout_repair_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")


def repair_one(h5_path: str | Path, *, fail_fast: bool) -> dict:
    try:
        summary = repair_extract_h5_layout(h5_path)
        print(
            f"{h5_path}: {summary['status']} "
            f"root_datasets={summary['root_datasets_after']}"
        )
        return summary
    except Exception as exc:
        print(f"{h5_path}: ERROR {exc}")
        if fail_fast:
            raise
        return {"h5_path": str(h5_path), "status": "error", "error": repr(exc)}


def selected(record, only: str | None) -> bool:
    if only is None:
        return True
    return only in {
        record.recording_id,
        record.session_id,
        record.trial_type,
        record.mouse_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair converted miniscope H5 files so EXTRACT sees only /data at "
            "the H5 root. Existing /valid_frame_mask is moved to "
            "/pipeline/valid_frame_mask."
        )
    )
    parser.add_argument("input", help="H5 path or manifest CSV/XLSX")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--index-out")
    parser.add_argument("--only", help="Recording ID, session ID, trial type, or mouse ID to process")
    parser.add_argument(
        "--use-manifest-h5",
        action="store_true",
        help="Use an existing neu_h5 column instead of the local output-root H5 path",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
