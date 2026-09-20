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
from preprocess_functions.neural import find_extract_outputs, import_extract_curated_cells
from preprocess_functions.pipeline import (
    default_cell_csv_path,
    default_curated_neurons_path,
    matlab_output_dir,
)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
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

        out_dir = record.matlab_output_dir or matlab_output_dir(output_root, record)
        cell_csv = record.cell_csv or default_cell_csv_path(
            output_root,
            record,
            trace_kind=args.trace_kind,
        )
        curated_csv = default_curated_neurons_path(output_root, record)

        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "cell_csv": str(cell_csv),
            "curated_neurons_csv": str(curated_csv),
        }

        try:
            traces_mat, labels_mat = explicit_or_discovered_outputs(record, out_dir)
            summary = import_extract_curated_cells(
                traces_mat_path=traces_mat,
                labels_mat_path=labels_mat,
                output_csv=cell_csv,
                curated_csv=curated_csv,
                recording_id=record.recording_id,
                session_id=record.session_id,
                trace_kind=args.trace_kind,
                fps=args.fps,
                tau_seconds=args.tau_seconds,
            )
            rows.append(
                {
                    **row,
                    **summary,
                    "extract_output_mat": str(traces_mat),
                    "extract_labels_mat": str(labels_mat),
                    "status": "imported",
                }
            )
            print(
                f"{record.recording_id}: imported "
                f"{summary['n_traces_written']} curated traces"
            )
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "curated_neuron_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")

    manifest_out = Path(args.manifest_out) if args.manifest_out else output_root / "manifest_with_cells.csv"
    output_rows = [
        {
            "recording_id": row["recording_id"],
            "cell_csv": row["cell_csv"],
            "extract_output_mat": row.get("extract_output_mat"),
            "extract_labels_mat": row.get("extract_labels_mat"),
        }
        for row in rows
        if row.get("status") == "imported"
    ]
    write_manifest_with_outputs(manifest, manifest_out, output_rows)
    print(f"Wrote {manifest_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import MATLAB/EXTRACT outputs as curated cell trace CSVs."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_matlab.csv")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--manifest-out")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument("--trace-kind", choices=["raw", "deconvolved"], default="raw")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tau-seconds", type=float, default=1.0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def explicit_or_discovered_outputs(record, output_dir: Path) -> tuple[Path, Path]:
    if record.extract_output_mat and record.extract_labels_mat:
        return record.extract_output_mat, record.extract_labels_mat
    return find_extract_outputs(
        output_dir,
        recording_id=record.recording_id,
        session_id=record.session_id,
    )


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
