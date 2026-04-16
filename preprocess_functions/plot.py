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

def overlay_mask(ax, mask, alpha=0.25, draw_outline=True, outline_lw=2):
    """
    mask: uint8 or bool, shape (H,W). Nonzero/True means inside.
    """
    m = mask.astype(bool)
    ax.imshow(m, alpha=alpha)  # simple grayscale overlay (no manual colors)

    if draw_outline:
        # draw contour from mask
        m8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.squeeze()
            if c.ndim == 2 and len(c) >= 3:
                ax.plot(c[:, 0], c[:, 1], linewidth=outline_lw)

def filter_points_by_mask(df, mask, x_col, y_col):
    H, W = mask.shape[:2]
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
 
    valid = np.isfinite(xs) & np.isfinite(ys)
    xi = xs[valid].astype(int)
    yi = ys[valid].astype(int)

    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    keep_idx = np.where(valid)[0][in_bounds]

    inside = np.zeros(len(df), dtype=bool)
    inside[keep_idx] = mask[yi[in_bounds], xi[in_bounds]] > 0
    return df.loc[inside].copy()


def plot_trajectory_over_arena(
    df_or_csv,
    masks=None,
    x_col: str = "ear_mid_x",
    y_col: str = "ear_mid_y",
    ts_col: str = "global_idx",
    use_mask_filter: bool = False,
    mask_name: str = "in_arena",
    downsample: int = 1,
    mask_alpha: float = 0.05,
    cmap: str = "viridis",
    date_col: str ="session_id",
):
    # Load dataframe
    df = pd.read_csv(df_or_csv) if isinstance(df_or_csv, str) else df_or_csv.copy()
    video_path = df["beh_vid_path"].iloc[0]
    # Remove NaNs
    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()

    # Downsample for performance
    if downsample > 1:
        df = df.iloc[::downsample].copy()

    # Load first video frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read video frame.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Optional mask filtering
    if use_mask_filter and masks is not None:
        mask = masks[mask_name] if isinstance(masks, dict) else masks
        df = filter_points_by_mask(df, mask, x_col=x_col, y_col=y_col)

    # Time for coloring
    if ts_col in df.columns:
        t_norm = df[ts_col].to_numpy()
    else:
        t_norm = np.arange(len(df))

    cmap = sns.color_palette(cmap, as_cmap=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame_rgb)

    # Overlay mask(s)
    if masks is not None:
        if isinstance(masks, dict):
            # overlay all masks
            for name, m in masks.items():
                overlay_mask(ax, m, alpha=mask_alpha, draw_outline=True)
        else:
            overlay_mask(ax, masks, alpha=mask_alpha, draw_outline=True)

    sc = ax.scatter(df[x_col], df[y_col], c=t_norm, cmap=cmap, s=3)
    fig.colorbar(sc, ax=ax, label="Time progression")

    ax.set_title(f"{df[date_col].iloc[0]} - Mouse Trajectory Over Arena (+ ROIs)")
    ax.invert_yaxis()
    plt.show()



def plot_head_direction_over_arena(
    df_or_csv,
    masks=None,
    x_col: str = "ear_mid_x",
    y_col: str = "ear_mid_y",
    angle_col: str = "head_dir_rad",
    ts_col: str = "global_idx",
    use_mask_filter: bool = False,
    mask_name: str = "in_arena",
    downsample: int = 2,       
    arrow_len: float = 30.0,       # pixels
    mask_alpha: float = 0.05,
    cmap: str = "viridis",
    date_col: str = "session_id",
    title: str | None=None,
):
    df = pd.read_csv(df_or_csv) if isinstance(df_or_csv, str) else df_or_csv.copy()
    video_path = df["beh_vid_path"].iloc[0]

    valid = (
        np.isfinite(df[x_col].to_numpy()) &
        np.isfinite(df[y_col].to_numpy()) &
        np.isfinite(df[angle_col].to_numpy())
    )
    df = df.loc[valid].copy()

    if downsample > 1:
        df = df.iloc[::downsample].copy()

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Optional mask filtering
    if use_mask_filter and masks is not None:
        mask = masks[mask_name] if isinstance(masks, dict) else masks
        df = filter_points_by_mask(df, mask, x_col=x_col, y_col=y_col)

    t = df[ts_col].to_numpy() if ts_col in df.columns else np.arange(len(df))
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    ang = df[angle_col].to_numpy()

    u = np.cos(ang) * arrow_len
    v = np.sin(ang) * arrow_len  # +y down in image coords

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame_rgb)

    if masks is not None:
        if isinstance(masks, dict):
            for name, m in masks.items():
                overlay_mask(ax, m, alpha=mask_alpha, draw_outline=True)
        else:
            overlay_mask(ax, masks, alpha=mask_alpha, draw_outline=True)
            
    cmap = sns.color_palette(cmap, as_cmap=True)

    q = ax.quiver(
        x, y, u, v, t,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap=cmap,
        alpha=1,
        width=0.003
    )
    fig.colorbar(q, ax=ax, label="Time progression")
    ax.set_title(title or f"{df[date_col].iloc[0]} - Head Direction Over Arena (+ ROIs)")
    ax.invert_yaxis()
    plt.show()


def trial_viewer(
    aligned_sessions_with_rois,
    trials_by_session,
    masks_by_session,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    video_col="beh_vid_path",
):
    session_ids = sorted(
        session for session in aligned_sessions_with_rois.keys()
        if session in trials_by_session and len(trials_by_session[session]) > 0
    )

    if not session_ids:
        raise ValueError("No sessions found with both aligned data and at least one trial.")

    session_dropdown = widgets.Dropdown(
        options=session_ids,
        value=session_ids[0],
        description="Session:"
    )

    trial_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(trials_by_session[session_dropdown.value]) - 1,
        step=1,
        description="Trial",
        continuous_update=False
    )

    output = widgets.Output()

    def update_slider_range(change):
        session = change["new"]
        trial_slider.max = len(trials_by_session[session]) - 1
        trial_slider.value = 0

    def show_trial(trial_idx, session):
        with output:
            clear_output(wait=True)

            trials = trials_by_session[session]
            s, e = map(int, trials[trial_idx])

            df_trial = aligned_sessions_with_rois[session].iloc[s:e+1].copy()

            plot_head_direction_over_arena(
                df_trial,
                masks=masks_by_session[session],
                x_col=x_col,
                y_col=y_col,
                title=f"{session} - Trial {trial_idx} Head Direction Over Arena"
            )

    session_dropdown.observe(update_slider_range, names="value")

    controls = widgets.VBox([session_dropdown, trial_slider])

    widgets.interactive_output(
        show_trial,
        {
            "trial_idx": trial_slider,
            "session": session_dropdown,
        }
    )

    display(controls, output)

    return session_dropdown, trial_slider, output