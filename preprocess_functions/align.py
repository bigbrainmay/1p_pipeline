import os
from pathlib import Path
import pandas as pd
import numpy as np
import openpyxl
import json
import time
import cv2
import re
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output

#fixes lab path string
def resolve_lab_path(p, lab_drive=None):
    if lab_drive is None:
        lab_drive = Path(os.environ.get("LAB_DRIVE_PATH", "Z:"))

    if pd.isna(p) or p is None:
        return None

    p = str(p).strip().strip('"')

    if p == "":
        return None

    # already absolute (Z:\...)
    if len(p) >= 2 and p[1] == ":":
        return Path(p)

    # true UNC path (\\server\share)
    if p.startswith("\\\\") and not p.startswith("\\\\Data\\"):
        return Path(p)

    # relative lab path (\Data\...)
    p = p.lstrip("\\/")
    return lab_drive / p

#loads in excel sheets as dictionary, adds session_id & mouse_id columns, checks for beh and sleap csvs, returns dictionary
def load_sessions(sheet_path, date_col="date"):
    sheet_path = Path(sheet_path)

    all_sheets = pd.read_excel(sheet_path, sheet_name=None, engine="openpyxl")

    for mouse_name, df in all_sheets.items():
        df = df.copy().dropna(how="all")
        if df.empty:
            continue

        # clean column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        df["mouse_id"] = mouse_name

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if "session_id" not in df.columns:
            if date_col in df.columns:
                df["session_id"] = (
                    df["mouse_id"].astype(str) + "_" +
                    df[date_col].dt.strftime("%Y%m%d")
                )
            else:
                df["session_id"] = df["mouse_id"].astype(str) + "_" + df.index.astype(str)

        required = ["beh_csv", "sleap_csv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

        if "neu_csv" not in df.columns:
                df["neu_csv"] = None

        if "cell_csv" not in df.columns:
                df["cell_csv"] = None

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
    sleap_cols: list[str] | None = None,
) -> pd.DataFrame:

    beh_df = pd.read_csv(beh_path)
    sleap_df = pd.read_csv(sleap_path)
    neu_df = pd.read_csv(neu_path) if neu_path is not None else None
    cell_df = pd.read_csv(cell_path) if cell_path is not None else None

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
        [neu_df, beh_df, cell_df], 
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

    return aligned


#resolving all paths & timestamps for each dataframe
def align_all(sheet_path, fps, ts_col="Timestamp", date_col="date"):
    sessions = load_sessions(sheet_path, date_col=date_col)
    aligned_by_session = {}

    for mouse_names, df in sessions.items():
        for _, row in df.iterrows():
            session_id = row["session_id"]

            beh_path = resolve_lab_path(row["beh_csv"])
            sleap_path = resolve_lab_path(row["sleap_csv"])
            neu_path = resolve_lab_path(row["neu_csv"]) if pd.notna(row["neu_csv"]) else None
            cell_path = resolve_lab_path(row["cell_csv"]) if pd.notna(row["cell_csv"]) else None
            beh_vid = resolve_lab_path(row["beh_vid"]) if pd.notna(row["beh_vid"]) else None

            print(f"\nSession: {session_id}")
            print("beh_path  :", beh_path)
            print("sleap_path:", sleap_path)
            print("neu_path  :", neu_path)
            print("cell_path :", cell_path)

        
            aligned = align_session(
                beh_path=beh_path,
                sleap_path=sleap_path,
                neu_path=neu_path,
                cell_path=cell_path,
                fps=fps,
                ts_col=ts_col,
            )

            aligned["session_id"] = session_id
            aligned["beh_vid_path"] = str(beh_vid) if beh_vid is not None else None
            if "mouse_id" in row:
                aligned["mouse_id"] = row["mouse_id"]

            aligned_by_session[session_id] = aligned

    return aligned_by_session