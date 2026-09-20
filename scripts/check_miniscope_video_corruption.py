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
from preprocess_functions.video import (
    check_leading_video_corruption,
    default_corruption_report_path,
    write_json_report,
)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )

    rows = []
    for record in records:
        if not selected(record, args.only):
            continue

        report_path = default_corruption_report_path(
            default_neu_h5_path(output_root, record)
        )
        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "neu_vid": str(record.neu_vid) if record.neu_vid else None,
            "corruption_report_path": str(report_path),
        }

        if record.neu_vid is None:
            rows.append({**row, "status": "missing_neu_vid"})
            continue

        try:
            report = check_leading_video_corruption(
                record.neu_vid,
                probe_frames=args.corruption_probe_frames,
                clean_run_frames=args.clean_run_frames,
                stripe_score_threshold=args.stripe_score_threshold,
                banding_score_threshold=args.banding_score_threshold,
            )
            write_json_report(report_path, report)
            rows.append(
                {
                    **row,
                    "status": report["status"],
                    "leading_bad_frames": report["leading_bad_frames"],
                    "first_clean_frame": report["first_clean_frame"],
                    "n_suspicious_later_frames": report["n_suspicious_later_frames"],
                    "metadata_warnings": ";".join(report["metadata_warnings"]),
                }
            )
            print(
                f"{record.recording_id}: {report['status']} "
                f"(gap leading frames: {report['leading_bad_frames']})"
            )
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "video_corruption_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check miniscope AVI starts for corrupt/striated leading frames."
    )
    parser.add_argument("manifest", help="CSV or XLSX manifest spreadsheet")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument("--corruption-probe-frames", type=int, default=600)
    parser.add_argument("--clean-run-frames", type=int, default=30)
    parser.add_argument("--stripe-score-threshold", type=float, default=0.75)
    parser.add_argument("--banding-score-threshold", type=float, default=1.8)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def selected(record, only: str | None) -> bool:
    if only is None:
        return True
    return only in {
        record.recording_id,
        record.session_id,
        record.trial_type,
        record.mouse_id,
    }


if __name__ == "__main__":
    main()
