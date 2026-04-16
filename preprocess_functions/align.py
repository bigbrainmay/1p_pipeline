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

        if "beh_vid" not in df.columns:
                df["beh_vid"] = None
        all_sheets[mouse_name] = df
    return all_sheets

#creates global timeline based off min & max timestamps from beh and neu streams, returns global index
def build_timeline(neu_df: pd.DataFrame,
    beh_df: pd.DataFrame, 
    fps: float, 
    ts_col: str = "Timestamp") -> pd.DataFrame:

    dt = pd.to_timedelta(1.0 / fps, unit="s")

    neu_ts = pd.to_datetime(neu_df[ts_col], utc=True, errors="coerce")
    beh_ts = pd.to_datetime(beh_df[ts_col], utc=True, errors="coerce")

    if neu_ts.isna().all() or beh_ts.isna().all():
        raise ValueError("All timestamps are NaT in neu or beh after parsing")

    t0 = min(neu_ts.min(), beh_ts.min())
    t1 = max(neu_ts.max(), beh_ts.max())

    global_ts = pd.date_range(start=t0, end=t1, freq=dt)

    return pd.DataFrame(
        {"global_idx": np.arange(len(global_ts), dtype=int), "global_ts": global_ts}
    )

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
    sleap_cols: list[str] | None = None,
) -> pd.DataFrame:

    beh_df = pd.read_csv(beh_path)
    sleap_df = pd.read_csv(sleap_path)

    if neu_path is not None:
        neu_df = pd.read_csv(neu_path)
        timeline = build_timeline(neu_df, beh_df, fps=fps, ts_col=ts_col)
    else:
        # build timeline from behavior only
        beh_ts = pd.to_datetime(beh_df[ts_col], utc=True, errors="coerce")
        dt = pd.to_timedelta(1.0 / fps, unit="s")
        global_ts = pd.date_range(start=beh_ts.min(), end=beh_ts.max(), freq=dt)
        timeline = pd.DataFrame({
            "global_idx": np.arange(len(global_ts), dtype=int),
            "global_ts": global_ts
        })

    beh_stream = map_stream(
        beh_df, timeline, fps=fps, ts_col=ts_col,
        prefix="beh", add_frame_idx=True
    )

    if neu_path is not None:
        neu_stream = map_stream(
            neu_df, timeline, fps=fps, ts_col=ts_col,
            prefix="neu", add_frame_idx=False
        )
        aligned = beh_stream.merge(
            neu_stream.drop(columns=["global_ts"]),
            on="global_idx",
            how="left"
        )
    else:
        aligned = beh_stream.copy()
        aligned["neu_ts"] = pd.NaT
        aligned["neu_dropped"] = True

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
            beh_vid = resolve_lab_path(row["beh_vid"]) if pd.notna(row["beh_vid"]) else None

            print(f"\nSession: {session_id}")
            print("beh_path  :", beh_path)
            print("sleap_path:", sleap_path)
            print("neu_path  :", neu_path)

        
            aligned = align_session(
                beh_path=beh_path,
                sleap_path=sleap_path,
                neu_path=neu_path,
                fps=fps,
                ts_col=ts_col,
            )

            aligned["session_id"] = session_id
            aligned["beh_vid_path"] = str(beh_vid) if beh_vid is not None else None
            if "mouse_id" in row:
                aligned["mouse_id"] = row["mouse_id"]

            aligned_by_session[session_id] = aligned

    return aligned_by_session