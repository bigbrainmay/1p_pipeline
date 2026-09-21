from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records
from preprocess_functions.neural import find_extract_outputs
from preprocess_functions.pipeline import (
    default_cell_registration_path,
    matlab_output_dir,
)
from preprocess_functions.registration import (
    CaimanNotInstalledError,
    caiman_available,
    load_registration_input,
    register_accepted_cells_across_sessions,
)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )
    selected_records = [record for record in records if selected(record, args.only)]
    records_by_mouse = group_records_by_mouse(selected_records)

    rows = []
    if not caiman_available() and not args.require_caiman:
        for mouse_id, mouse_records in records_by_mouse.items():
            rows.append(
                {
                    "mouse_id": mouse_id,
                    "status": "skipped_missing_caiman",
                    "n_sessions": len(mouse_records),
                    "cell_registration_csv": str(
                        default_cell_registration_path(output_root, mouse_id)
                    ),
                    "error": (
                        "CaImAn is not installed in this Python environment. "
                        "Database formation can continue from manifest_with_cells.csv."
                    ),
                }
            )
            print(f"{mouse_id}: skipped because CaImAn is not installed")
        write_index(output_root, rows)
        return

    for mouse_id, mouse_records in records_by_mouse.items():
        out_path = default_cell_registration_path(output_root, mouse_id)
        row = {
            "mouse_id": mouse_id,
            "n_sessions": len(mouse_records),
            "cell_registration_csv": str(out_path),
        }

        if len(mouse_records) < 2 and not args.allow_single_session:
            rows.append({**row, "status": "skipped_less_than_two_sessions"})
            print(f"{mouse_id}: skipped because there is only one selected session")
            continue

        try:
            registration_inputs = [
                load_record_registration_input(
                    record,
                    output_root=output_root,
                    accepted_label=args.accepted_label,
                )
                for record in sorted(mouse_records, key=record_sort_key)
            ]
            registration_df, summary = register_accepted_cells_across_sessions(
                registration_inputs,
                output_csv=out_path,
                align_flag=not args.no_template_align,
                max_thr=args.max_thr,
                use_opt_flow=args.use_opt_flow,
                thresh_cost=args.thresh_cost,
                max_dist=args.max_dist,
                enclosed_thr=args.enclosed_thr,
            )
            rows.append(
                {
                    **row,
                    **summary,
                    "status": "registered",
                    "n_rows": len(registration_df),
                }
            )
            print(
                f"{mouse_id}: registered {summary['n_registered_cells']} "
                f"cross-session cells -> {out_path}"
            )
        except CaimanNotInstalledError as exc:
            rows.append({**row, "status": "skipped_missing_caiman", "error": str(exc)})
            print(f"{mouse_id}: skipped because CaImAn is not installed")
            if args.require_caiman or args.fail_fast:
                raise
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{mouse_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    write_index(output_root, rows)


def load_record_registration_input(record, *, output_root: Path, accepted_label: int):
    traces_mat, labels_mat = explicit_or_discovered_outputs(
        record,
        record.matlab_output_dir or matlab_output_dir(output_root, record),
    )
    return load_registration_input(
        traces_mat_path=traces_mat,
        labels_mat_path=labels_mat,
        recording_id=record.recording_id,
        session_id=record.session_id,
        mouse_id=record.mouse_id or record.sheet_name,
        accepted_label=accepted_label,
    )


def explicit_or_discovered_outputs(record, output_dir: Path) -> tuple[Path, Path]:
    if record.extract_output_mat and record.extract_labels_mat:
        return record.extract_output_mat, record.extract_labels_mat
    return find_extract_outputs(
        output_dir,
        recording_id=record.recording_id,
        session_id=record.session_id,
    )


def group_records_by_mouse(records) -> dict[str, list]:
    grouped = defaultdict(list)
    for record in records:
        mouse_id = record.mouse_id or record.sheet_name or "unknown_mouse"
        grouped[str(mouse_id)].append(record)
    return dict(grouped)


def record_sort_key(record) -> tuple:
    date_key = record.date.isoformat() if record.date is not None else ""
    return date_key, record.session_id, record.recording_id


def selected(record, only: str | None) -> bool:
    if only is None:
        return True
    return only in {
        record.recording_id,
        record.session_id,
        record.trial_type,
        record.mouse_id,
    }


def write_index(output_root: Path, rows: list[dict]) -> None:
    index_path = output_root / "coregistration_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optionally register accepted EXTRACT/ActSort cells across sessions with CaImAn."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_cells.csv")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--only", help="Mouse ID, recording ID, session ID, or trial type to process")
    parser.add_argument("--accepted-label", type=int, default=1)
    parser.add_argument("--no-template-align", action="store_true")
    parser.add_argument("--no-opt-flow", dest="use_opt_flow", action="store_false")
    parser.add_argument("--max-thr", type=float, default=0)
    parser.add_argument("--thresh-cost", type=float, default=0.7)
    parser.add_argument("--max-dist", type=float, default=10)
    parser.add_argument("--enclosed-thr", type=float)
    parser.add_argument("--allow-single-session", action="store_true")
    parser.add_argument(
        "--require-caiman",
        action="store_true",
        help="Fail if CaImAn is not installed instead of writing a skipped index row.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.set_defaults(use_opt_flow=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
