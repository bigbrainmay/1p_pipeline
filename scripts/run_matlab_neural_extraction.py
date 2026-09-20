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
from preprocess_functions.neural import run_matlab_extraction
from preprocess_functions.pipeline import default_neu_h5_path, matlab_output_dir


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
    extra_vars = parse_extra_vars(args.extra)

    rows = []
    for record in records:
        if not selected(record, args.only):
            continue

        neu_h5 = (
            record.neu_h5
            if args.use_manifest_h5 and record.neu_h5
            else default_neu_h5_path(output_root, record)
        )
        out_dir = record.matlab_output_dir or matlab_output_dir(output_root, record)
        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "neu_h5": str(neu_h5),
            "matlab_output_dir": str(out_dir),
        }

        if not Path(neu_h5).exists() and not args.dry_run:
            rows.append({**row, "status": "missing_neu_h5"})
            print(f"{record.recording_id}: missing H5 {neu_h5}")
            continue

        try:
            result = run_matlab_extraction(
                recording_id=record.recording_id,
                session_id=record.session_id,
                neu_h5_path=neu_h5,
                output_dir=out_dir,
                matlab_script=args.matlab_script,
                matlab_bin=args.matlab_bin,
                extra_vars=extra_vars,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print("[dry-run]", " ".join(result))
                rows.append({**row, "status": "dry_run"})
            else:
                rows.append({**row, "status": "completed", "returncode": result.returncode})
                print(f"{record.recording_id}: MATLAB extraction completed")
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "matlab_extraction_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")

    manifest_out = Path(args.manifest_out) if args.manifest_out else output_root / "manifest_with_matlab.csv"
    output_rows = [
        {
            "recording_id": row["recording_id"],
            "neu_h5": row["neu_h5"],
            "matlab_output_dir": row["matlab_output_dir"],
        }
        for row in rows
        if row.get("status") in {"completed", "dry_run"}
    ]
    write_manifest_with_outputs(manifest, manifest_out, output_rows)
    print(f"Wrote {manifest_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a MATLAB neural extraction script for each converted H5."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_h5.csv")
    parser.add_argument("--matlab-script", required=True, help="MATLAB .m script to run")
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument(
        "--output-root",
        default="preprocess_out",
        help="Local root containing generated H5 files and MATLAB outputs",
    )
    parser.add_argument("--manifest-out")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument(
        "--use-manifest-h5",
        action="store_true",
        help="Use an existing neu_h5 column instead of the local output-root H5 path",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra MATLAB workspace variable; can be repeated",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_extra_vars(items: list[str]) -> dict[str, str]:
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE for --extra, got {item!r}")
        name, value = item.split("=", 1)
        values[name.strip()] = value
    return values


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
