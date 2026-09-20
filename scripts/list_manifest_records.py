from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records


def main() -> None:
    args = parse_args()
    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )

    rows = []
    for record in records:
        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "mouse_id": record.mouse_id,
            "date": record.date.strftime("%Y-%m-%d") if record.date is not None else None,
            "trial_type": record.trial_type,
        }
        if args.show_paths:
            row["neu_vid"] = str(record.neu_vid) if record.neu_vid else None
            row["beh_vid"] = str(record.beh_vid) if record.beh_vid else None
        if args.check_files:
            row["neu_vid_exists"] = bool(record.neu_vid and record.neu_vid.exists())
            row["beh_vid_exists"] = bool(record.beh_vid and record.beh_vid.exists())
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")
    else:
        print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List recording IDs generated from a manifest spreadsheet."
    )
    parser.add_argument("manifest", help="CSV or XLSX manifest spreadsheet")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--show-paths", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--output-csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
