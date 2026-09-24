import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

from .bpod import add_bpod_events_to_aligned, load_bpod_byte_events
from .manifest import add_manifest_ids, normalize_manifest_table
from .port_signal import add_video_port_events_to_aligned


LAB_RELATIVE_ROOTS = {"data", "users"}

#fixes lab path string
def resolve_lab_path(p, lab_drive=None):
    if pd.isna(p) or p is None:
        return None

    p = str(p).strip().strip('"')

    if p == "":
        return None

    lab_drive_value = lab_drive or os.environ.get("LAB_DRIVE_PATH")
    lab_drive_path = Path(lab_drive_value) if lab_drive_value else None

    drive_match = re.match(r"^[A-Za-z]:[\\/]*(.*)$", p)
    if drive_match:
        if lab_drive_path:
            return lab_drive_path.joinpath(*_split_lab_path_parts(drive_match.group(1)))
        return Path(p)

    # true UNC path (\\server\share)
    if p.startswith("\\\\") and not p.startswith("\\\\Data\\"):
        return Path(p)

    # relative lab path (\Data\...)
    parts = _split_lab_path_parts(p.lstrip("\\/"))
    if (
        lab_drive_path
        and parts
        and (p.startswith(("\\", "/")) or parts[0].lower() in LAB_RELATIVE_ROOTS)
    ):
        return lab_drive_path.joinpath(*parts)
    return Path(*parts) if parts else None


def _split_lab_path_parts(p):
    return [part for part in re.split(r"[\\/]+", str(p)) if part]

#loads in excel sheets as dictionary, adds session_id & mouse_id columns, checks for beh and sleap csvs, returns dictionary
def load_sessions(sheet_path, date_col="date"):
    sheet_path = Path(sheet_path)

    all_sheets = pd.read_excel(sheet_path, sheet_name=None, engine="openpyxl")

    for mouse_name, df in all_sheets.items():
        df = normalize_manifest_table(df, sheet_name=mouse_name)
        if df.empty:
            continue

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        df = add_manifest_ids(df)

        required = ["beh_csv", "sleap_csv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

        if "neu_csv" not in df.columns:
                df["neu_csv"] = None

        if "cell_csv" not in df.columns:
                df["cell_csv"] = None

        if "bpod_ts" not in df.columns:
                df["bpod_ts"] = None

        if "beh_vid" not in df.columns:
                df["beh_vid"] = None
        all_sheets[mouse_name] = df
    return all_sheets

#creates global timeline based off min & max timestamps from beh and neu streams, returns global index
def build_timeline(
    streams: list[pd.DataFrame],
    fps: float, 
    ts_col: str = "Timestamp",) -> pd.DataFrame:

    dt = pd.to_timedelta(1.0 / fps, unit="s")

    mins = []
    maxs = []

    for df in streams:
        if df is None or df.empty:
            continue

        if ts_col not in df.columns:
            raise ValueError(
                f"Timestamp column {ts_col!r} missing from stream. "
                f"Found columns: {df.columns.tolist()}"
            )

        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        if ts.notna().any():
            mins.append(ts.min())
            maxs.append(ts.max())

    if not mins:
        raise ValueError("No valid timestamps found in any stream")

    global_ts = pd.date_range(
        start=min(mins),
        end=max(maxs),
        freq=pd.to_timedelta(1.0 / fps, unit="s"),
    )

    return pd.DataFrame({
        "global_idx": np.arange(len(global_ts), dtype=int),
        "global_ts": global_ts,
    })

#maps streams together, merges on global index, returns dataframe
def map_stream(stream_df: pd.DataFrame, 
    union_df: pd.DataFrame, 
    fps: float, 
    ts_col: str, 
    prefix: str, 
    add_frame_idx: bool = False,) -> pd.DataFrame:
    
    tol = pd.Timedelta(seconds=0.5 / fps)

    s = stream_df.copy()
    s[ts_col] = pd.to_datetime(s[ts_col], utc=True, errors="coerce")
    s = s.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    s["recorded_idx"] = np.arange(len(s), dtype=int)
    if add_frame_idx:
        s["beh_frame_idx"] = s["recorded_idx"]

    mapped = pd.merge_asof(
        union_df.sort_values("global_ts"),
        s.sort_values(ts_col),
        left_on="global_ts",
        right_on=ts_col,
        direction="nearest",
        tolerance=tol,
    )

    mapped[f"{prefix}_dropped"] = mapped["recorded_idx"].isna()
    mapped = mapped.rename(columns={ts_col: f"{prefix}_ts"})

    return mapped

#builds a timeline, merges sleap timestamps with aligned dataframe, returns dataframe with sleap position columns
def align_session(
    beh_path: Path,
    sleap_path: Path,
    fps: float,
    ts_col: str = "Timestamp",
    neu_path: Path | None = None,
    cell_path: Path | None = None,
    bpod_path: Path | None = None,
    video_port_events_path: Path | list[Path] | None = None,
    sleap_cols: list[str] | None = None,
) -> pd.DataFrame:

    beh_df = pd.read_csv(beh_path)
    sleap_df = pd.read_csv(sleap_path)
    neu_df = pd.read_csv(neu_path) if neu_path is not None else None
    cell_df = pd.read_csv(cell_path) if cell_path is not None else None
    bpod_df = None
    if bpod_path is not None:
        bpod_df = load_bpod_byte_events(bpod_path).rename(columns={"bpod_ts": ts_col})

    if cell_df is not None:
        if neu_df is None:
            raise ValueError(
                "cell_path was provided, but neu_df is None. "
                "Need neu_df timestamps to align cell traces."
            )

        if len(cell_df) != len(neu_df):
            raise ValueError(
                f"Cell trace row count does not match neural timestamp row count: "
                f"cell_df={len(cell_df)}, neu_df={len(neu_df)}"
            )

        if ts_col not in neu_df.columns:
            raise ValueError(
                f"neu_df is missing timestamp column {ts_col!r}. "
                f"Found columns: {neu_df.columns.tolist()}"
            )
        
        if "index" in cell_df.columns:
            cell_df = cell_df.rename(columns={"index": "cell_frame_idx"})
        else:
            cell_df.insert(0, "cell_frame_idx", np.arange(len(cell_df), dtype=int))

        cell_df[ts_col] = neu_df[ts_col].values

        # Sanity checks.
        if cell_df[ts_col].isna().any():
            raise ValueError("Some copied cell timestamps are missing/NaN")

        cell_cols = [c for c in cell_df.columns if c.startswith("cell_")]
        if not cell_cols:
            raise ValueError(
                "No cell trace columns found. Expected columns like cell_000, cell_001, ..."
            )

        print(f"Loaded cell traces: {len(cell_df)} frames x {len(cell_cols)} cells")

    timeline = build_timeline(
        [neu_df, beh_df, cell_df, bpod_df],
        fps=fps, 
        ts_col=ts_col,
        )


    beh_stream = map_stream(
        beh_df, 
        timeline, 
        fps=fps, 
        ts_col=ts_col,
        prefix="beh", 
        add_frame_idx=True
    )

    aligned = beh_stream.copy()

    if neu_df is not None:
        neu_stream = map_stream(
            neu_df, 
            timeline, 
            fps=fps, 
            ts_col=ts_col,
            prefix="neu", 
            add_frame_idx=False
        )

        aligned = aligned.merge(
            neu_stream.drop(columns=["global_ts"]),
            on="global_idx",
            how="left",
             suffixes=("", "_neu"),
        )
    else:
        aligned["neu_ts"] = pd.NaT
        aligned["neu_dropped"] = True

    if cell_df is not None:
        cell_stream = map_stream(
            cell_df,
            timeline,
            fps=fps,
            ts_col=ts_col,
            prefix="cell",
            add_frame_idx=False,
        )
        aligned = aligned.merge(
            cell_stream.drop(columns=["global_ts"]),
            on="global_idx",
            how="left",
            suffixes=("", "_cell"),
        )
    else:
        aligned["cell_ts"] = pd.NaT
        aligned["cell_dropped"] = True

    # --- SLEAP merge ---
    sleap = sleap_df.copy().rename(columns={"frame_idx": "beh_frame_idx"})
    sleap["beh_frame_idx"] = pd.to_numeric(sleap["beh_frame_idx"], errors="coerce")
    aligned["beh_frame_idx"] = pd.to_numeric(aligned["beh_frame_idx"], errors="coerce")

    if sleap_cols is None:
        sleap_cols = ["nose.x", "nose.y", "ear_L.x", "ear_L.y", "ear_R.x", "ear_R.y"]

    keep = ["beh_frame_idx"] + [c for c in sleap_cols if c in sleap.columns]
    aligned = aligned.merge(sleap[keep], on="beh_frame_idx", how="left")

    if bpod_path is not None:
        aligned = add_bpod_events_to_aligned(aligned, bpod_path, fps=fps)
        aligned["bpod_ts_path"] = str(bpod_path)
    else:
        aligned["bpod_event_dropped"] = True
        aligned["bpod_any_port_active"] = False
        aligned["bpod_ts_path"] = None

    if video_port_events_path is not None:
        aligned = add_video_port_events_to_aligned(aligned, video_port_events_path)
        aligned["video_port_events_path"] = str(video_port_events_path)
    else:
        aligned["video_port_event_dropped"] = True
        aligned["video_any_port_active"] = False
        aligned["video_port_events_path"] = None

    return aligned


#resolving all paths & timestamps for each dataframe
def align_all(sheet_path, fps, ts_col="Timestamp", date_col="date"):
    sessions = load_sessions(sheet_path, date_col=date_col)
    aligned_by_session = {}

    for mouse_names, df in sessions.items():
        for _, row in df.iterrows():
            session_id = row["session_id"]
            recording_id = row.get("recording_id", session_id)

            beh_path = resolve_lab_path(row["beh_csv"])
            sleap_path = resolve_lab_path(row["sleap_csv"])
            neu_path = resolve_lab_path(row["neu_csv"]) if pd.notna(row["neu_csv"]) else None
            cell_path = resolve_lab_path(row["cell_csv"]) if pd.notna(row["cell_csv"]) else None
            bpod_path = resolve_lab_path(row["bpod_ts"]) if pd.notna(row["bpod_ts"]) else None
            video_port_events_path = (
                resolve_lab_path(row["video_port_events_csv"])
                if "video_port_events_csv" in row and pd.notna(row["video_port_events_csv"])
                else None
            )
            beh_vid = resolve_lab_path(row["beh_vid"]) if pd.notna(row["beh_vid"]) else None

            print(f"\nRecording: {recording_id}")
            print("session_id:", session_id)
            print("beh_path  :", beh_path)
            print("sleap_path:", sleap_path)
            print("neu_path  :", neu_path)
            print("cell_path :", cell_path)
            print("bpod_path :", bpod_path)
            print("video_port_events_path:", video_port_events_path)

        
            aligned = align_session(
                beh_path=beh_path,
                sleap_path=sleap_path,
                neu_path=neu_path,
                cell_path=cell_path,
                bpod_path=bpod_path,
                video_port_events_path=video_port_events_path,
                fps=fps,
                ts_col=ts_col,
            )

            aligned["session_id"] = session_id
            aligned["recording_id"] = recording_id
            aligned["beh_vid_path"] = str(beh_vid) if beh_vid is not None else None
            if "mouse_id" in row:
                aligned["mouse_id"] = row["mouse_id"]
            if "trial_type" in row:
                aligned["trial_type"] = row["trial_type"]
            if "cue_ts" in row:
                aligned["cue_ts"] = row["cue_ts"]

            aligned_by_session[recording_id] = aligned

    return aligned_by_session
