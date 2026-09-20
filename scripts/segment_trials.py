from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records
from preprocess_functions.pipeline import (
    cue_timestamp_overrides_path,
    default_aligned_session_path,
    default_cue_events_path,
    default_trials_path,
)
from preprocess_functions.segment import (
    add_cue_event_frame_columns,
    load_cue_timestamp_overrides,
    parse_cue_events,
    resolve_cue_timestamp,
    segment_trials_gap_window,
    trials_to_dataframe,
)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cue_overrides_path = (
        Path(args.cue_overrides)
        if args.cue_overrides
        else cue_timestamp_overrides_path(output_root)
    )
    cue_overrides = load_cue_timestamp_overrides(cue_overrides_path)

    records = load_session_records(
        args.manifest,
        sheet_name=args.sheet,
        lab_drive=args.lab_drive,
    )

    rows = []
    for record in records:
        if not selected(record, args.only):
            continue

        aligned_csv = default_aligned_session_path(output_root, record)
        trials_csv = default_trials_path(output_root, record)
        cue_events_csv = default_cue_events_path(output_root, record)
        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "aligned_csv": str(aligned_csv),
            "trials_csv": str(trials_csv),
            "cue_events_csv": str(cue_events_csv),
        }

        if not aligned_csv.exists():
            rows.append({**row, "status": "missing_aligned_csv"})
            continue

        try:
            df = pd.read_csv(aligned_csv)
            missing = required_columns(args) - set(df.columns)
            if missing:
                rows.append(
                    {
                        **row,
                        "status": "missing_columns",
                        "missing_columns": ";".join(sorted(missing)),
                    }
                )
                continue

            cue_ts, cue_source, cue_note = resolve_cue_timestamp(
                recording_id=record.recording_id,
                default_cue_ts=default_cue_ts(df, record.cue_ts),
                cue_overrides=cue_overrides,
            )
            cue_events = parse_cue_events(
                cue_ts,
                cue_duration_s=args.cue_duration_s,
            )
            cue_events = add_cue_event_frame_columns(cue_events, df)

            trials = segment_trials_gap_window(
                df,
                startbox_col=(args.startbox_left_col, args.startbox_right_col),
                arena_only_col=args.arena_only_col,
                fps=args.fps,
                max_gap_s=args.max_gap_s,
                dwell_frames=args.dwell_frames,
                min_trial_s=args.min_trial_s,
                require_opposite_side=args.require_opposite_side,
            )
            trials_df = trials_to_dataframe(
                trials,
                df,
                recording_id=record.recording_id,
                session_id=record.session_id,
                mouse_id=record.mouse_id,
                trial_type=record.trial_type,
                cue_ts=cue_ts,
                cue_ts_source=cue_source,
                cue_ts_note=cue_note,
                cue_events=cue_events,
                cue_duration_s=args.cue_duration_s,
                fps=args.fps,
            )
            trials_csv.parent.mkdir(parents=True, exist_ok=True)
            cue_events_csv.parent.mkdir(parents=True, exist_ok=True)
            cue_events.to_csv(cue_events_csv, index=False)
            trials_df.to_csv(trials_csv, index=False)

            rows.append(
                {
                    **row,
                    "status": "segmented",
                    "n_trials": len(trials_df),
                    "n_cue_events": len(cue_events),
                    "cue_ts": cue_ts,
                    "cue_ts_source": cue_source,
                    "cue_overrides_csv": str(cue_overrides_path),
                }
            )
            print(f"{record.recording_id}: wrote {len(trials_df)} trials")
        except Exception as exc:
            rows.append({**row, "status": "error", "error": repr(exc)})
            print(f"{record.recording_id}: ERROR {exc}")
            if args.fail_fast:
                raise

    index_path = output_root / "trial_segment_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"Wrote {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment aligned sessions into trial metadata CSVs."
    )
    parser.add_argument("manifest", help="CSV/XLSX manifest, preferably manifest_with_cells.csv")
    parser.add_argument("--sheet", help="Excel sheet to process; defaults to all sheets")
    parser.add_argument("--lab-drive", help="Local mount for lab paths such as \\Data\\May")
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument("--cue-overrides", help="Manual cue timestamp override CSV")
    parser.add_argument("--only", help="Recording ID, session ID, or trial type to process")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--startbox-left-col", default="in_startbox_L")
    parser.add_argument("--startbox-right-col", default="in_startbox_R")
    parser.add_argument("--arena-only-col", default="arena_only")
    parser.add_argument("--max-gap-s", type=float, default=3.0)
    parser.add_argument("--dwell-frames", type=int, default=1)
    parser.add_argument("--min-trial-s", type=float, default=0.5)
    parser.add_argument(
        "--cue-duration-s",
        type=float,
        help=(
            "Fixed cue duration in seconds. If omitted, cue end is the next cue "
            "timestamp, and the final cue has no end."
        ),
    )
    parser.add_argument("--require-opposite-side", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def required_columns(args: argparse.Namespace) -> set[str]:
    return {
        args.startbox_left_col,
        args.startbox_right_col,
        args.arena_only_col,
    }


def default_cue_ts(df: pd.DataFrame, record_cue_ts: str | None) -> str | None:
    if "cue_ts" in df.columns:
        values = df["cue_ts"].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin({"", "nan", "none", "null"})]
        if not values.empty:
            return values.iloc[0]
    return record_cue_ts


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
