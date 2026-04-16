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

#computes mouse ear midpoint, head direction & angular velocity  
def compute_position_from_df(
    df: pd.DataFrame,
    ts_col: str = "global_ts",
    use_dt_from_timestamps: bool = True,
    fps: float = 30.0,
    assume_sorted: bool = False,
) -> pd.DataFrame:

    df = df.copy()

    required = ["ear_L.x", "ear_L.y", "ear_R.x", "ear_R.y", "nose.x", "nose.y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # --- timestamps ---
    if use_dt_from_timestamps:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        if not assume_sorted:
            df = df.sort_values(ts_col).reset_index(drop=True)

        dt = df[ts_col].diff().dt.total_seconds().to_numpy()
    else:
        if not assume_sorted:
            df = df.reset_index(drop=True)
        dt = np.full(len(df), 1.0 / float(fps), dtype=float)
        dt[0] = np.nan

    # --- ear midpoint ---
    df["ear_mid_x"] = (df["ear_L.x"] + df["ear_R.x"]) / 2.0
    df["ear_mid_y"] = (df["ear_L.y"] + df["ear_R.y"]) / 2.0

    # --- head direction ---
    df["hd_dx"] = df["nose.x"] - df["ear_mid_x"]
    df["hd_dy"] = df["nose.y"] - df["ear_mid_y"]
    df["head_dir_rad"] = np.arctan2(df["hd_dy"], df["hd_dx"])

    # --- angular velocity ---
    ang = df["head_dir_rad"].to_numpy(dtype=float)
    ang_unwrapped = np.full_like(ang, np.nan)

    valid = np.isfinite(ang)
    if valid.any():
        ang_unwrapped[valid] = np.unwrap(ang[valid])

    dtheta = np.diff(ang_unwrapped)
    ang_vel = np.full(len(df), np.nan)

    dt_step = dt[1:]
    good_steps = np.isfinite(dtheta) & np.isfinite(dt_step) & (dt_step > 0)

    ang_vel[1:][good_steps] = dtheta[good_steps] / dt_step[good_steps]

    df["ang_vel_rad_s"] = ang_vel
    df["ang_vel_deg_s"] = np.degrees(ang_vel)
    df["ang_vel_speed"] = np.abs(ang_vel)

    return df

#loops across all sessions & dataframes in dictionary, returns dictionary
def compute_position_for_sessions(
    session_dict: dict[str, pd.DataFrame],
    **kwargs
) -> dict[str, pd.DataFrame]:

    out = {}

    for session_id, df in session_dict.items():
        print(f"Processing session: {session_id}")
        out[session_id] = compute_position_from_df(df, **kwargs)

    return out