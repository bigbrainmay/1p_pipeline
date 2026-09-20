from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.manifest import load_session_records
from preprocess_functions.pipeline import default_aligned_session_path


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

        row = {
            "recording_id": record.recording_id,
            "session_id": record.session_id,
            "beh_csv": str(record.beh_csv) if record.beh_csv else None,
            "sleap_csv": str(record.sleap_csv) if record.sleap_csv else None,
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
                fps=args.fps,
                ts_col=args.ts_col,
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
            aligned["cue_ts"] = record.cue_ts
            aligned["beh_vid_path"] = str(record.beh_vid) if record.beh_vid else None
            aligned["neu_vid_path"] = str(record.neu_vid) if record.neu_vid else None
            aligned["cell_csv_path"] = str(cell_csv) if cell_csv else None

            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned.to_csv(out_path, index=False)
            rows.append({**row, "status": "aligned", "n_rows": len(aligned)})
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
    parser.add_argument("--fail-fast", action="store_true")
    parser.set_defaults(add_pose=True)
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
