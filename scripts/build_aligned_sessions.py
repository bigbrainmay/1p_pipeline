from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.bpod import bpod_port_intervals, load_bpod_byte_events
from preprocess_functions.manifest import (
    experiment_metadata_from_record,
    load_session_records,
)
from preprocess_functions.pipeline import (
    default_aligned_session_path,
    default_bpod_events_path,
    default_bpod_intervals_path,
    default_cell_registration_path,
    default_registered_cell_map_path,
    default_video_port_events_path,
)
from preprocess_functions.port_signal import load_video_port_events


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    import preprocess_functions.align as align
    import preprocess_functions.pose as pose

    cell_csv_by_recording = load_cell_index(args.cell_index, output_root)
    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )

    rows = []
    for record in records:
        if not selected(record, args.only):
            continue

        cell_csv = record.cell_csv or cell_csv_by_recording.get(record.recording_id)
        out_path = default_aligned_session_path(output_root, record)
        bpod_events_path = (
            default_bpod_events_path(output_root, record) if record.bpod_ts else None
        )
        bpod_intervals_path = (
            default_bpod_intervals_path(output_root, record) if record.bpod_ts else None
        )
        video_port_events_path = record.video_port_events_csv
        if video_port_events_path is None:
            default_video_port_path = default_video_port_events_path(output_root, record)
            video_port_events_path = (
                default_video_port_path if default_video_port_path.exists() else None
            )

        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "mouse_id": record.mouse_id,
            "trial_type": record.trial_type,
            **experiment_metadata_from_record(record),
            "beh_csv": str(record.beh_csv) if record.beh_csv else None,
            "sleap_csv": str(record.sleap_csv) if record.sleap_csv else None,
            "bpod_ts": str(record.bpod_ts) if record.bpod_ts else None,
            "bpod_events_csv": str(bpod_events_path) if bpod_events_path else None,
            "bpod_intervals_csv": str(bpod_intervals_path) if bpod_intervals_path else None,
            "video_port_events_csv": str(video_port_events_path) if video_port_events_path else None,
            "neu_csv": str(record.neu_csv) if record.neu_csv else None,
            "cell_csv": str(cell_csv) if cell_csv else None,
            "aligned_csv": str(out_path),
        }

        if record.beh_csv is None or record.sleap_csv is None:
            rows.append({**row, "status": "missing_behavior_inputs"})
            continue

        try:
            aligned = align.align_session(
                beh_path=record.beh_csv,
                sleap_path=record.sleap_csv,
                neu_path=record.neu_csv,
                cell_path=cell_csv,
                bpod_path=record.bpod_ts,
                video_port_events_path=video_port_events_path,
                fps=args.fps,
                ts_col=args.ts_col,
            )
            aligned, registration_summary = add_registered_cell_aliases(
                aligned,
                record,
                output_root,
                enabled=args.add_registered_cell_aliases,
            )

            if args.add_pose:
                aligned = pose.compute_position_from_df(
                    aligned,
                    ts_col="global_ts",
                    fps=args.fps,
                )

            aligned["recording_id"] = record.recording_id
            aligned["session_id"] = record.session_id
            aligned["mouse_id"] = record.mouse_id
            aligned["trial_type"] = record.trial_type
            for col, value in experiment_metadata_from_record(record).items():
                aligned[col] = value
            aligned["cue_ts"] = record.cue_ts
            aligned["beh_vid_path"] = str(record.beh_vid) if record.beh_vid else None
            aligned["neu_vid_path"] = str(record.neu_vid) if record.neu_vid else None
            aligned["cell_csv_path"] = str(cell_csv) if cell_csv else None

            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned.to_csv(out_path, index=False)
            bpod_summary = write_bpod_outputs(
                record.bpod_ts,
                bpod_events_path,
                bpod_intervals_path,
            )
            video_port_summary = summarize_video_port_events(video_port_events_path)
            rows.append(
                {
                    **row,
                    **bpod_summary,
                    **video_port_summary,
                    **registration_summary,
                    "status": "aligned",
                    "n_rows": len(aligned),
                }
            )
            print(f"{record.recording_id}: wrote {out_path}")
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "aligned_session_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aligned behavior/SLEAP/neural session CSVs from the manifest."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_cells.csv")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--cell-index", help="curated_neuron_index.csv; defaults to output-root copy")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ts-col", default="Timestamp")
    parser.add_argument("--no-pose", dest="add_pose", action="store_false")
    parser.add_argument(
        "--no-registered-cell-aliases",
        dest="add_registered_cell_aliases",
        action="store_false",
        help=(
            "Do not add registered_cell_* alias columns from "
            "preprocess_out/coregistration/<mouse>/<mouse>_cell_registration.csv"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.set_defaults(add_pose=True, add_registered_cell_aliases=True)
    return parser.parse_args()


def load_cell_index(path: str | None, output_root: Path) -> dict[str, Path]:
    index_path = Path(path) if path else output_root / "curated_neuron_index.csv"
    if not index_path.exists():
        return {}

    df = pd.read_csv(index_path)
    if "status" in df.columns:
        df = df[df["status"] == "imported"]
    if "recording_id" not in df.columns or "cell_csv" not in df.columns:
        return {}

    return {
        str(row["recording_id"]): Path(row["cell_csv"])
        for _, row in df.dropna(subset=["recording_id", "cell_csv"]).iterrows()
    }


def write_bpod_outputs(
    bpod_ts_path: Path | None,
    bpod_events_path: Path | None,
    bpod_intervals_path: Path | None,
) -> dict[str, int | None]:
    if bpod_ts_path is None or bpod_events_path is None or bpod_intervals_path is None:
        return {"n_bpod_events": None, "n_bpod_intervals": None}

    events = load_bpod_byte_events(bpod_ts_path)
    intervals = bpod_port_intervals(events)

    bpod_events_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(bpod_events_path, index=False)
    intervals.to_csv(bpod_intervals_path, index=False)

    return {
        "n_bpod_events": int(len(events)),
        "n_bpod_intervals": int(len(intervals)),
    }


def summarize_video_port_events(
    video_port_events_path: Path | None,
) -> dict[str, int | None]:
    if video_port_events_path is None:
        return {"n_video_port_events": None}
    events = load_video_port_events(video_port_events_path)
    return {"n_video_port_events": int(len(events))}


def add_registered_cell_aliases(
    aligned: pd.DataFrame,
    record,
    output_root: Path,
    *,
    enabled: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | str | None]]:
    summary: dict[str, int | str | None] = {
        "cell_registration_csv": None,
        "registered_cell_map_csv": None,
        "n_registered_cell_links": None,
        "n_registered_cell_aliases": None,
        "registered_cell_status": "disabled" if not enabled else "not_checked",
    }
    if not enabled:
        return aligned, summary

    mouse_id = record.mouse_id or record.sheet_name
    if not mouse_id:
        summary["registered_cell_status"] = "missing_mouse_id"
        return aligned, summary

    registration_path = default_cell_registration_path(output_root, str(mouse_id))
    summary["cell_registration_csv"] = str(registration_path)
    if not registration_path.exists():
        summary["registered_cell_status"] = "missing_cell_registration"
        summary["n_registered_cell_links"] = 0
        summary["n_registered_cell_aliases"] = 0
        return aligned, summary

    registration = pd.read_csv(registration_path)
    required = {"recording_id", "registered_cell_id", "cell_col"}
    missing = required - set(registration.columns)
    if missing:
        missing_cols = ",".join(sorted(missing))
        summary["registered_cell_status"] = f"registration_missing_columns:{missing_cols}"
        summary["n_registered_cell_links"] = 0
        summary["n_registered_cell_aliases"] = 0
        return aligned, summary

    rows = registration[
        registration["recording_id"].astype(str) == str(record.recording_id)
    ].copy()
    if rows.empty:
        summary["registered_cell_status"] = "no_registered_cells_for_recording"
        summary["n_registered_cell_links"] = 0
        summary["n_registered_cell_aliases"] = 0
        return aligned, summary

    out = aligned.copy()
    map_rows = []
    alias_count = 0
    for _, row in rows.iterrows():
        local_col = str(row["cell_col"])
        registered_col = str(row["registered_cell_id"])
        has_local_trace = local_col in out.columns
        if has_local_trace:
            out[registered_col] = out[local_col]
            alias_count += 1

        map_row = row.to_dict()
        map_row["local_cell_col"] = local_col
        map_row["registered_cell_col"] = registered_col
        map_row["has_local_trace"] = bool(has_local_trace)
        map_rows.append(map_row)

    map_path = default_registered_cell_map_path(output_root, record)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(map_rows).to_csv(map_path, index=False)

    summary["registered_cell_map_csv"] = str(map_path)
    summary["n_registered_cell_links"] = int(len(rows))
    summary["n_registered_cell_aliases"] = int(alias_count)
    if alias_count:
        summary["registered_cell_status"] = "registered_aliases_added"
    else:
        summary["registered_cell_status"] = "registered_cells_missing_local_traces"
    return out, summary


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
