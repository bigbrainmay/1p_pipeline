from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import (
    load_manifest,
    load_session_records,
    write_manifest_with_outputs,
)
from preprocess_functions.pipeline import default_neu_h5_path
from preprocess_functions.video import convert_avi_to_h5


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest, sheet_name=args.sheet)
    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )

    rows = []
    for record in records:
        if not selected(record, args.only):
            continue

        neu_h5 = (
            record.neu_h5
            if args.use_manifest_h5 and record.neu_h5
            else default_neu_h5_path(output_root, record)
        )
        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "neu_vid": str(record.neu_vid) if record.neu_vid else None,
            "neu_h5": str(neu_h5),
        }

        if record.neu_vid is None:
            rows.append({**row, "status": "missing_neu_vid"})
            continue

        if args.dry_run:
            print(f"[dry-run] {record.recording_id}: {record.neu_vid} -> {neu_h5}")
            rows.append({**row, "status": "dry_run"})
            continue

        try:
            summary = convert_avi_to_h5(
                record.neu_vid,
                neu_h5,
                overwrite=args.overwrite,
                compression=args.compression,
                gap_leading_corrupt=not args.no_corruption_check,
                manual_gap_leading_frames=args.manual_gap_leading_frames,
                gap_fill_strategy=args.gap_fill_strategy,
                gap_fill_value=args.gap_fill_value,
                corruption_probe_frames=args.corruption_probe_frames,
                clean_run_frames=args.clean_run_frames,
                stripe_score_threshold=args.stripe_score_threshold,
                banding_score_threshold=args.banding_score_threshold,
            )
            rows.append({**row, **summary})
            print(f"{record.recording_id}: {summary['status']} -> {neu_h5}")
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "miniscope_h5_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")

    manifest_out = Path(args.manifest_out) if args.manifest_out else output_root / "manifest_with_h5.csv"
    output_rows = [
        {"recording_id": row["recording_id"], "neu_h5": row["neu_h5"]}
        for row in rows
        if row.get("status") in {"converted", "exists", "dry_run"}
    ]
    write_manifest_with_outputs(manifest, manifest_out, output_rows)
    print(f"Wrote {manifest_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert miniscope AVI videos listed in a manifest to H5 stacks."
    )
    parser.add_argument("manifest", help="CSV or XLSX manifest spreadsheet")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument(
        "--output-root",
        default="preprocess_out",
        help="Local root for generated files; H5 files are written here by default",
    )
    parser.add_argument("--manifest-out")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument(
        "--use-manifest-h5",
        action="store_true",
        help="Use an existing neu_h5 column instead of writing a local H5 under output-root",
    )
    parser.add_argument("--compression", choices=["gzip", "lzf"], help="Optional H5 compression")
    parser.add_argument(
        "--no-corruption-check",
        action="store_true",
        help="Disable leading corruption detection and write all readable frames as-is",
    )
    parser.add_argument(
        "--manual-gap-leading-frames",
        type=int,
        default=0,
        help="Force this many leading H5 frames to be gap placeholders",
    )
    parser.add_argument(
        "--gap-fill-strategy",
        choices=["nearest", "constant"],
        default="nearest",
        help="How to fill gap placeholder frames; nearest copies the closest clean frame",
    )
    parser.add_argument(
        "--gap-fill-value",
        type=float,
        default=0.0,
        help="Pixel value used only when --gap-fill-strategy constant",
    )
    parser.add_argument("--corruption-probe-frames", type=int, default=600)
    parser.add_argument("--clean-run-frames", type=int, default=30)
    parser.add_argument("--stripe-score-threshold", type=float, default=0.75)
    parser.add_argument("--banding-score-threshold", type=float, default=1.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
