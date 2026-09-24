import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_functions import boundary_tuning 

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    display = None
    clear_output = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for video-frame plotting functions.")
    return cv2


def _require_widgets():
    if widgets is None or display is None:
        raise ImportError("ipywidgets is required for interactive viewer functions.")
    return widgets

def overlay_mask(ax, mask, alpha=0.25, draw_outline=True, outline_lw=2):
    """
    mask: uint8 or bool, shape (H,W). Nonzero/True means inside.
    """
    cv2_local = _require_cv2()
    m = mask.astype(bool)
    ax.imshow(m, alpha=alpha)  # simple grayscale overlay (no manual colors)

    if draw_outline:
        # draw contour from mask
        m8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2_local.findContours(m8, cv2_local.RETR_EXTERNAL, cv2_local.CHAIN_APPROX_SIMPLE)
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


def _load_plot_dataframe(df_or_csv):
    if isinstance(df_or_csv, (str, Path)):
        return pd.read_csv(df_or_csv)
    return df_or_csv.copy()


def _numeric_column(df, col):
    if col not in df.columns:
        raise ValueError(f"Missing column {col!r}. Found: {df.columns.tolist()}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


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
    cv2_local = _require_cv2()
    # Load dataframe
    df = pd.read_csv(df_or_csv) if isinstance(df_or_csv, str) else df_or_csv.copy()
    video_path = df["beh_vid_path"].iloc[0]
    # Remove NaNs
    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()

    # Downsample for performance
    if downsample > 1:
        df = df.iloc[::downsample].copy()

    # Load first video frame
    cap = cv2_local.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read video frame.")
    frame_rgb = cv2_local.cvtColor(frame, cv2_local.COLOR_BGR2RGB)

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
    cv2_local = _require_cv2()
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

    cap = cv2_local.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    frame_rgb = cv2_local.cvtColor(frame, cv2_local.COLOR_BGR2RGB)

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


def plot_trajectory(
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
    date_col: str = "session_id",
):
    """Plot the animal trajectory over the first behavior-video frame."""
    return plot_trajectory_over_arena(
        df_or_csv,
        masks=masks,
        x_col=x_col,
        y_col=y_col,
        ts_col=ts_col,
        use_mask_filter=use_mask_filter,
        mask_name=mask_name,
        downsample=downsample,
        mask_alpha=mask_alpha,
        cmap=cmap,
        date_col=date_col,
    )


def plot_hd_trajectory(
    df_or_csv,
    masks=None,
    x_col: str = "ear_mid_x",
    y_col: str = "ear_mid_y",
    angle_col: str = "head_dir_rad",
    ts_col: str = "global_idx",
    use_mask_filter: bool = False,
    mask_name: str = "in_arena",
    downsample: int = 2,
    arrow_len: float = 30.0,
    mask_alpha: float = 0.05,
    cmap: str = "viridis",
    date_col: str = "session_id",
    title: str | None = None,
):
    """Plot head-direction arrows over the animal trajectory."""
    return plot_head_direction_over_arena(
        df_or_csv,
        masks=masks,
        x_col=x_col,
        y_col=y_col,
        angle_col=angle_col,
        ts_col=ts_col,
        use_mask_filter=use_mask_filter,
        mask_name=mask_name,
        downsample=downsample,
        arrow_len=arrow_len,
        mask_alpha=mask_alpha,
        cmap=cmap,
        date_col=date_col,
        title=title,
    )


def trial_viewer(
    aligned_sessions_with_rois,
    trials_by_session,
    masks_by_session,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    video_col="beh_vid_path",
):
    widgets_local = _require_widgets()
    session_ids = sorted(
        session
        for session in aligned_sessions_with_rois
        if (
            session in trials_by_session
            and len(trials_by_session[session]) > 0
        )
    )

    if not session_ids:
        raise ValueError(
            "No sessions found with both aligned data and at least one trial."
        )

    session_dropdown = widgets_local.Dropdown(
        options=session_ids,
        value=session_ids[0],
        description="Session:",
    )

    trial_slider = widgets_local.IntSlider(
        value=0,
        min=0,
        max=len(trials_by_session[session_dropdown.value]) - 1,
        step=1,
        description="Trial:",
        continuous_update=False,
    )

    def update_slider_range(change):
        session = change["new"]
        trial_slider.max = len(trials_by_session[session]) - 1
        trial_slider.value = 0

    def show_trial(trial_idx, session):
        trials = trials_by_session[session]
        trial = trials[trial_idx]

        # New dictionary format
        if isinstance(trial, dict):
            s = int(trial["start_frame"])
            e = int(trial["end_frame"])

            start_side = trial.get("start_side", "?")
            end_side = trial.get("end_side", "?")

            title = (
                f"{session} - Trial {trial_idx} "
                f"({start_side} → {end_side}) "
                f"Head Direction Over Arena"
            )

        # Old tuple/list format: (start_frame, end_frame)
        else:
            s, e = map(int, trial)

            title = (
                f"{session} - Trial {trial_idx} "
                f"Head Direction Over Arena"
            )

        df_trial = (
            aligned_sessions_with_rois[session]
            .iloc[s:e + 1]
            .copy()
        )

        plot_hd_trajectory(
            df_trial,
            masks=masks_by_session[session],
            x_col=x_col,
            y_col=y_col,
            title=title,
        )

    session_dropdown.observe(
        update_slider_range,
        names="value",
    )

    output = widgets_local.interactive_output(
        show_trial,
        {
            "trial_idx": trial_slider,
            "session": session_dropdown,
        },
    )

    controls = widgets_local.VBox([
        session_dropdown,
        trial_slider,
    ])

    display(controls, output)

    return session_dropdown, trial_slider, output


def _plot_start_end(ax, x, y):
    ax.scatter(x[0], y[0], marker="^", s=80, label="start")
    ax.scatter(x[-1], y[-1], marker="s", s=80, label="end")


def _activity_map(x, y, activity, bins):
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)

    occupancy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    activity_sum, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=activity,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_activity = activity_sum / occupancy

    mean_activity[occupancy == 0] = np.nan
    return mean_activity, x_edges, y_edges


def _arena_mask_array(arena_mask, mask_name="arena"):
    if isinstance(arena_mask, dict):
        if mask_name not in arena_mask:
            raise KeyError(
                f"Mask dictionary is missing {mask_name!r}. "
                f"Found masks: {list(arena_mask.keys())}"
            )
        arena_mask = arena_mask[mask_name]
    mask = np.asarray(arena_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"arena_mask must be 2D, got shape {mask.shape}")
    return mask


def _points_inside_mask(mask, x, y):
    h, w = mask.shape
    inside = np.zeros(len(x), dtype=bool)

    valid = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return inside

    xi = np.rint(x[valid_idx]).astype(int)
    yi = np.rint(y[valid_idx]).astype(int)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    checked_idx = valid_idx[in_bounds]
    inside[checked_idx] = mask[yi[in_bounds], xi[in_bounds]]
    return inside


def _trajectory_coverage_axes():
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 3)

    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[0, 2] = fig.add_subplot(grid[0, 2])
    axes[1, 0] = fig.add_subplot(grid[1, 0])
    axes[1, 1] = fig.add_subplot(grid[1, 1], projection="polar")
    axes[1, 2] = fig.add_subplot(grid[1, 2], projection="polar")

    return fig, axes


def _head_direction_tuning(head_dir, activity, n_bins):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    wrapped = (head_dir + np.pi) % (2 * np.pi) - np.pi
    bin_idx = np.digitize(wrapped, edges) - 1
    bin_idx = np.where(bin_idx == n_bins, n_bins - 1, bin_idx)

    tuning = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (bin_idx == i) & np.isfinite(activity)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            tuning[i] = np.nanmean(activity[mask])

    return centers, tuning, counts


def _plot_head_direction_tuning(ax, head_dir, activity, n_bins, cell_col):
    centers, tuning, _ = _head_direction_tuning(head_dir, activity, n_bins)

    theta = np.r_[centers, centers[0]]
    r = np.r_[tuning, tuning[0]]

    ax.plot(theta, r, marker="o", linewidth=1.5)
    ax.fill(theta, r, alpha=0.15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.grid(False)
    ax.set_title(f"{cell_col} head direction tuning")
    return centers, tuning


def plot_2d_ratemap(
    df_or_csv,
    cell_col="cell_0",
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    bins=20,
    arena_mask=None,
    arena_mask_name="arena",
    occupancy_min=1,
    cmap="viridis",
    ax=None,
    title=None,
    show=True,
):
    """Plot an occupancy-normalized 2D activity map for one cell."""
    df = _load_plot_dataframe(df_or_csv)
    required = [x_col, y_col, cell_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    x = _numeric_column(df, x_col)
    y = _numeric_column(df, y_col)
    activity = _numeric_column(df, cell_col)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(activity)

    if arena_mask is not None:
        mask = _arena_mask_array(arena_mask, mask_name=arena_mask_name)
        valid &= _points_inside_mask(mask, x, y)

    if not valid.any():
        raise ValueError("No valid points available for the 2D ratemap")

    x_plot = x[valid]
    y_plot = y[valid]
    activity_plot = activity[valid]

    x_edges = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), bins + 1)
    y_edges = np.linspace(np.nanmin(y_plot), np.nanmax(y_plot), bins + 1)
    occupancy, _, _ = np.histogram2d(x_plot, y_plot, bins=[x_edges, y_edges])
    activity_sum, _, _ = np.histogram2d(
        x_plot,
        y_plot,
        bins=[x_edges, y_edges],
        weights=activity_plot,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_activity = activity_sum / occupancy
    mean_activity[occupancy < occupancy_min] = np.nan

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        mean_activity.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        cmap=cmap,
    )
    ax.set_title(title or f"{cell_col} 2D ratemap")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.colorbar(im, ax=ax, label=f"mean {cell_col} activity")

    if show:
        plt.show()
    return ax


def plot_hd(
    df_or_csv,
    cell_col="cell_0",
    head_dir_col="head_dir_rad",
    n_bins=36,
    ax=None,
    title=None,
    show=True,
):
    """Plot head-direction tuning for one cell."""
    df = _load_plot_dataframe(df_or_csv)
    head_dir = _numeric_column(df, head_dir_col)
    activity = _numeric_column(df, cell_col)
    valid = np.isfinite(head_dir) & np.isfinite(activity)

    if not valid.any():
        raise ValueError("No valid head-direction/activity samples available")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    _plot_head_direction_tuning(
        ax,
        head_dir[valid],
        activity[valid],
        n_bins,
        cell_col,
    )
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
    return ax


def plot_hd_tuning(*args, **kwargs):
    """Alias for plot_hd."""
    return plot_hd(*args, **kwargs)


def event_active_columns(df, prefixes=("bpod_", "video_"), suffix="_active"):
    """Return Bpod/video port active columns that can be plotted as event traces."""
    return [
        col
        for col in df.columns
        if col.endswith(suffix)
        and any(col.startswith(prefix) for prefix in prefixes)
        and col not in {"bpod_any_port_active", "video_any_port_active"}
    ]


def slice_trial_dataframe(aligned_df, trial_row):
    """Return the aligned-dataframe rows covered by one trial metadata row."""
    start = max(0, int(trial_row["start_frame"]))
    end = min(len(aligned_df) - 1, int(trial_row["end_frame"]))
    if end < start:
        return aligned_df.iloc[0:0].copy()
    return aligned_df.iloc[start:end + 1].copy()


def compute_trial_behavior_metrics(
    aligned_df,
    trials_df,
    *,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    fps=30.0,
    event_cols=None,
):
    """
    Summarize each trial's trajectory and Bpod/video-port event activity.

    Output keeps all original trial metadata columns and adds path, speed,
    straightness, and event fraction/count columns.
    """
    event_cols = event_cols or event_active_columns(aligned_df)
    rows = []

    for _, trial in trials_df.iterrows():
        df_trial = slice_trial_dataframe(aligned_df, trial)
        row = trial.to_dict()
        row["n_frames"] = int(len(df_trial))
        row["duration_s_metric"] = float(len(df_trial) / fps) if fps else np.nan

        if {x_col, y_col}.issubset(df_trial.columns):
            x = pd.to_numeric(df_trial[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df_trial[y_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 2:
                xv = x[valid]
                yv = y[valid]
                steps = np.sqrt(np.diff(xv) ** 2 + np.diff(yv) ** 2)
                path_length = float(np.nansum(steps))
                displacement = float(np.sqrt((xv[-1] - xv[0]) ** 2 + (yv[-1] - yv[0]) ** 2))
                row["path_length_px"] = path_length
                row["displacement_px"] = displacement
                row["straightness"] = displacement / path_length if path_length > 0 else np.nan
                row["mean_speed_px_s"] = path_length / row["duration_s_metric"] if row["duration_s_metric"] > 0 else np.nan
                row["start_x"] = float(xv[0])
                row["start_y"] = float(yv[0])
                row["end_x"] = float(xv[-1])
                row["end_y"] = float(yv[-1])

        for col in event_cols:
            if col not in df_trial.columns:
                continue
            active = pd.to_numeric(df_trial[col], errors="coerce").fillna(0).astype(bool)
            row[f"frac_{col}"] = float(active.mean()) if len(active) else np.nan
            row[f"n_{col}_onsets"] = int((active & ~active.shift(1, fill_value=False)).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def plot_trial_behavior_events(
    aligned_df,
    trial_row,
    *,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    event_cols=None,
    cell_col=None,
    neural_cols=None,
    cue_color="tab:red",
    show=True,
):
    """
    Plot one trial's trajectory, port-event activity, and optional neural trace.
    """
    df_trial = slice_trial_dataframe(aligned_df, trial_row)
    if df_trial.empty:
        raise ValueError("Trial slice is empty")

    event_cols = event_cols or event_active_columns(df_trial)
    if neural_cols is None:
        neural_cols = [cell_col] if cell_col is not None else []
    neural_cols = [col for col in neural_cols if col in df_trial.columns]

    n_rows = 2 + int(bool(event_cols)) + int(bool(neural_cols))
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows), squeeze=False)
    axes = axes[:, 0]
    ax_i = 0

    ax = axes[ax_i]
    ax_i += 1
    if {x_col, y_col}.issubset(df_trial.columns):
        x = pd.to_numeric(df_trial[x_col], errors="coerce")
        y = pd.to_numeric(df_trial[y_col], errors="coerce")
        t = np.arange(len(df_trial))
        ax.plot(x, y, color="0.75", linewidth=1)
        sc = ax.scatter(x, y, c=t, cmap="viridis", s=18)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            first = int(np.flatnonzero(valid)[0])
            last = int(np.flatnonzero(valid)[-1])
            ax.scatter([x.iloc[first]], [y.iloc[first]], c="lime", edgecolor="black", s=70, label="start")
            ax.scatter([x.iloc[last]], [y.iloc[last]], c="red", edgecolor="black", s=70, label="end")
        plt.colorbar(sc, ax=ax, label="trial frame")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
    ax.set_title(f"{trial_row.get('recording_id', '')} trial {trial_row.get('trial_idx', '')}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    ax = axes[ax_i]
    ax_i += 1
    frame = np.arange(len(df_trial))
    if "head_dir_rad" in df_trial.columns:
        ax.plot(frame, np.degrees(pd.to_numeric(df_trial["head_dir_rad"], errors="coerce")), label="head_dir_deg")
        ax.set_ylabel("head dir deg")
    elif {x_col, y_col}.issubset(df_trial.columns):
        x = pd.to_numeric(df_trial[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df_trial[y_col], errors="coerce").to_numpy(dtype=float)
        speed = np.r_[np.nan, np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]
        ax.plot(frame, speed, label="frame speed px")
        ax.set_ylabel("speed px/frame")
    ax.set_xlabel("trial frame")
    ax.legend(loc="best")

    if event_cols:
        ax = axes[ax_i]
        ax_i += 1
        for offset, col in enumerate(event_cols):
            active = pd.to_numeric(df_trial[col], errors="coerce").fillna(0).astype(float)
            ax.fill_between(frame, offset, offset + active, step="pre", alpha=0.35)
            ax.text(frame[0] if len(frame) else 0, offset + 0.5, col, va="center")
        ax.set_ylim(-0.25, len(event_cols) + 0.25)
        ax.set_yticks([])
        ax.set_xlabel("trial frame")
        ax.set_title("Bpod/video port activity")

    if neural_cols:
        ax = axes[ax_i]
        for col in neural_cols:
            ax.plot(frame, pd.to_numeric(df_trial[col], errors="coerce"), linewidth=1, label=col)
        ax.set_xlabel("trial frame")
        ax.set_ylabel("activity")
        ax.set_title("Neural trace during trial")
        ax.legend(loc="best")

    for ax in axes:
        _mark_trial_cue_window(ax, trial_row, len(df_trial), color=cue_color)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_trial_metric_summary(
    metrics_df,
    *,
    metric_cols=("duration_s_metric", "path_length_px", "mean_speed_px_s", "straightness"),
    group_col=None,
    show=True,
):
    """Plot behavior metric distributions, optionally grouped by outcome/label."""
    metric_cols = [col for col in metric_cols if col in metrics_df.columns]
    if not metric_cols:
        raise ValueError("No requested metric columns are present")

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4), squeeze=False)
    axes = axes[0]
    for ax, metric in zip(axes, metric_cols):
        if group_col and group_col in metrics_df.columns:
            groups = [
                (str(name), pd.to_numeric(group[metric], errors="coerce").dropna())
                for name, group in metrics_df.groupby(group_col, dropna=False)
            ]
            ax.boxplot([values for _, values in groups], labels=[name for name, _ in groups])
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.hist(pd.to_numeric(metrics_df[metric], errors="coerce").dropna(), bins=20)
        ax.set_title(metric)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_trial_trajectories_by_group(
    aligned_df,
    trials_df,
    *,
    group_col,
    group_values=None,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    max_trials_per_group=25,
    show=True,
):
    """Overlay trial trajectories split by a grouping column such as success/label."""
    if group_col not in trials_df.columns:
        raise KeyError(f"{group_col!r} is not in trials_df")
    group_values = group_values or sorted(trials_df[group_col].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, len(group_values), figsize=(6 * len(group_values), 5), squeeze=False)
    axes = axes[0]

    for ax, value in zip(axes, group_values):
        subset = trials_df[trials_df[group_col].astype(str) == str(value)].head(max_trials_per_group)
        for _, trial in subset.iterrows():
            df_trial = slice_trial_dataframe(aligned_df, trial)
            if {x_col, y_col}.issubset(df_trial.columns):
                ax.plot(
                    pd.to_numeric(df_trial[x_col], errors="coerce"),
                    pd.to_numeric(df_trial[y_col], errors="coerce"),
                    linewidth=1,
                    alpha=0.45,
                )
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{group_col}={value} (n={len(subset)})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_event_aligned_neural(
    aligned_df,
    events,
    *,
    cell_cols,
    event_frame_col=None,
    event_ts_col=None,
    window_s=2.0,
    fps=30.0,
    ts_col="global_ts",
    show=True,
):
    """
    Plot neural traces aligned to trajectory or behavior event timestamps/frames.

    Pass either event_frame_col or event_ts_col. `events` can be a trial/event
    dataframe; each row contributes one event.
    """
    if isinstance(cell_cols, str):
        cell_cols = [cell_cols]
    cell_cols = [col for col in cell_cols if col in aligned_df.columns]
    if not cell_cols:
        raise ValueError("No requested cell columns are present")

    half_window = int(round(float(window_s) * float(fps)))
    offsets = np.arange(-half_window, half_window + 1)
    event_indices = _event_indices_from_table(
        aligned_df,
        events,
        event_frame_col=event_frame_col,
        event_ts_col=event_ts_col,
        ts_col=ts_col,
    )

    if not event_indices:
        raise ValueError("No event indices could be resolved")

    fig, axes = plt.subplots(len(cell_cols), 1, figsize=(9, 3 * len(cell_cols)), squeeze=False)
    axes = axes[:, 0]
    x = offsets / float(fps)
    for ax, cell_col in zip(axes, cell_cols):
        traces = []
        values = pd.to_numeric(aligned_df[cell_col], errors="coerce").to_numpy(dtype=float)
        for idx in event_indices:
            take = idx + offsets
            ok = (take >= 0) & (take < len(values))
            trace = np.full(len(offsets), np.nan)
            trace[ok] = values[take[ok]]
            traces.append(trace)
            ax.plot(x, trace, color="0.75", linewidth=0.8, alpha=0.5)
        mean_trace = np.nanmean(np.vstack(traces), axis=0)
        ax.plot(x, mean_trace, color="black", linewidth=2, label="mean")
        ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(f"{cell_col} aligned to event")
        ax.set_xlabel("seconds from event")
        ax.set_ylabel("activity")
        ax.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _mark_trial_cue_window(ax, trial_row, n_frames, color="tab:red"):
    if "cue_start_frame" not in trial_row:
        return
    try:
        start = int(trial_row["cue_start_frame"]) - int(trial_row["start_frame"])
    except Exception:
        return
    if start < -n_frames or start > n_frames:
        return
    ax.axvline(start, color=color, linestyle="--", linewidth=1, alpha=0.8)
    if pd.notna(trial_row.get("cue_end_frame", np.nan)):
        end = int(trial_row["cue_end_frame"]) - int(trial_row["start_frame"])
        ax.axvspan(start, end, color=color, alpha=0.08)


def _event_indices_from_table(
    aligned_df,
    events,
    *,
    event_frame_col=None,
    event_ts_col=None,
    ts_col="global_ts",
):
    event_indices = []
    if event_frame_col is not None and event_frame_col in events.columns:
        for value in pd.to_numeric(events[event_frame_col], errors="coerce").dropna():
            event_indices.append(int(value))
        return event_indices

    if event_ts_col is None or event_ts_col not in events.columns:
        return event_indices

    aligned_ts = pd.to_datetime(aligned_df[ts_col], utc=True, errors="coerce")
    for value in events[event_ts_col].dropna():
        target = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(target):
            continue
        delta = (aligned_ts - target).abs()
        if delta.notna().any():
            valid = delta.notna().to_numpy()
            valid_positions = np.flatnonzero(valid)
            closest = int(np.argmin(delta[valid].to_numpy()))
            event_indices.append(int(valid_positions[closest]))
    return event_indices


def _compute_ebc_map_for_plot(
    df_or_csv,
    arena_mask,
    cell_col="cell_0",
    angle_bins=36,
    distance_bins=20,
    distance_max_px=None,
    boundary_stride=1,
    frame_stride=1,
    chunk_size=128,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    nose_x_col="nose.x",
    nose_y_col="nose.y",
    cell_dropped_col="cell_dropped",
    mask_name="arena",
):
    df = _load_plot_dataframe(df_or_csv)
    mask = _arena_mask_array(arena_mask, mask_name=mask_name)
    return boundary_tuning.compute_egocentric_boundary_map(
        df,
        mask,
        cell_col=cell_col,
        angle_bins=angle_bins,
        distance_bins=distance_bins,
        distance_max_px=distance_max_px,
        boundary_stride=boundary_stride,
        frame_stride=frame_stride,
        chunk_size=chunk_size,
        x_col=x_col,
        y_col=y_col,
        nose_x_col=nose_x_col,
        nose_y_col=nose_y_col,
        cell_dropped_col=cell_dropped_col,
    )


def plot_ebc(
    df_or_csv,
    arena_mask,
    cell_col="cell_0",
    angle_bins=36,
    distance_bins=20,
    distance_max_px=None,
    boundary_stride=1,
    frame_stride=1,
    chunk_size=128,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    nose_x_col="nose.x",
    nose_y_col="nose.y",
    cell_dropped_col="cell_dropped",
    arena_mask_name="arena",
    occupancy_min=1,
    radius_max=None,
    smooth_sigma=None,
    cmap="viridis",
    ax=None,
    show=True,
):
    """Plot an egocentric boundary-cell map as a polar plot."""
    egocentric_map = _compute_ebc_map_for_plot(
        df_or_csv,
        arena_mask,
        cell_col=cell_col,
        angle_bins=angle_bins,
        distance_bins=distance_bins,
        distance_max_px=distance_max_px,
        boundary_stride=boundary_stride,
        frame_stride=frame_stride,
        chunk_size=chunk_size,
        x_col=x_col,
        y_col=y_col,
        nose_x_col=nose_x_col,
        nose_y_col=nose_y_col,
        cell_dropped_col=cell_dropped_col,
        mask_name=arena_mask_name,
    )
    return boundary_tuning.plot_egocentric_boundary_polar(
        egocentric_map,
        ax=ax,
        cmap=cmap,
        occupancy_min=occupancy_min,
        radius_max=radius_max,
        smooth_sigma=smooth_sigma,
        show=show,
    )


def plot_ebc_heatmap(
    df_or_csv,
    arena_mask,
    cell_col="cell_0",
    angle_bins=36,
    distance_bins=20,
    distance_max_px=None,
    boundary_stride=1,
    frame_stride=1,
    chunk_size=128,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    nose_x_col="nose.x",
    nose_y_col="nose.y",
    cell_dropped_col="cell_dropped",
    arena_mask_name="arena",
    occupancy_min=1,
    cmap="viridis",
    ax=None,
    show=True,
):
    """Plot an egocentric boundary-cell map as angle x distance heatmap."""
    egocentric_map = _compute_ebc_map_for_plot(
        df_or_csv,
        arena_mask,
        cell_col=cell_col,
        angle_bins=angle_bins,
        distance_bins=distance_bins,
        distance_max_px=distance_max_px,
        boundary_stride=boundary_stride,
        frame_stride=frame_stride,
        chunk_size=chunk_size,
        x_col=x_col,
        y_col=y_col,
        nose_x_col=nose_x_col,
        nose_y_col=nose_y_col,
        cell_dropped_col=cell_dropped_col,
        mask_name=arena_mask_name,
    )
    return boundary_tuning.plot_egocentric_boundary_map(
        egocentric_map,
        ax=ax,
        cmap=cmap,
        occupancy_min=occupancy_min,
        show=show,
    )


def plot_ebc_occupancy(
    df_or_csv,
    arena_mask,
    cell_col="cell_0",
    angle_bins=36,
    distance_bins=20,
    distance_max_px=None,
    boundary_stride=1,
    frame_stride=1,
    chunk_size=128,
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    nose_x_col="nose.x",
    nose_y_col="nose.y",
    cell_dropped_col="cell_dropped",
    arena_mask_name="arena",
    cmap="magma",
    ax=None,
    show=True,
):
    """Plot EBC sample occupancy for checking map coverage."""
    egocentric_map = _compute_ebc_map_for_plot(
        df_or_csv,
        arena_mask,
        cell_col=cell_col,
        angle_bins=angle_bins,
        distance_bins=distance_bins,
        distance_max_px=distance_max_px,
        boundary_stride=boundary_stride,
        frame_stride=frame_stride,
        chunk_size=chunk_size,
        x_col=x_col,
        y_col=y_col,
        nose_x_col=nose_x_col,
        nose_y_col=nose_y_col,
        cell_dropped_col=cell_dropped_col,
        mask_name=arena_mask_name,
    )
    return boundary_tuning.plot_egocentric_boundary_occupancy(
        egocentric_map,
        ax=ax,
        cmap=cmap,
        show=show,
    )


def plot_cell_summary(*args, **kwargs):
    """Plot trajectory, occupancy, trace, 2D ratemap, EBC, and HD panels."""
    return plot_trajectory_coverage_with_cell(*args, **kwargs)


def plot_trajectory_coverage_with_cell(
    df,
    cell_col="cell_0",
    x_col="ear_mid_x",
    y_col="ear_mid_y",
    nose_x_col="nose.x",
    nose_y_col="nose.y",
    head_dir_col="head_dir_rad",
    time_col="global_ts",
    activity_threshold=0,
    point_size=8,
    bins=20,
    arena_mask=None,
    arena_mask_name="arena",
    egocentric_angle_bins=36,
    egocentric_distance_bins=20,
    egocentric_boundary_stride=2,
    egocentric_frame_stride=1,
    egocentric_smooth_sigma=None,
    egocentric_radius_max=None,
    egocentric_occupancy_min=1,
    egocentric_cmap="viridis",
    head_direction_bins=36,
    show=True,
):
    """
    Plot trajectory, occupancy, one cell trace, and occupancy-normalized
    spatial activity for a single cell.

    If arena_mask is provided, also plots an egocentric boundary polar map.
    """
    required = [x_col, y_col, head_dir_col, cell_col]
    if arena_mask is not None:
        required += [nose_x_col, nose_y_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    keep_cols = required.copy()
    if time_col in df.columns:
        keep_cols.append(time_col)

    plot_df = df[keep_cols].copy()
    numeric_cols = [x_col, y_col, head_dir_col, cell_col]
    for col in numeric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=numeric_cols)

    if plot_df.empty:
        raise ValueError(
            "No valid rows after dropping NaNs for "
            f"{x_col}, {y_col}, {head_dir_col}, and {cell_col}"
        )

    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)
    activity = plot_df[cell_col].to_numpy(dtype=float)
    head_dir = plot_df[head_dir_col].to_numpy(dtype=float)
    head_dir_deg = np.degrees(head_dir) % 360

    if arena_mask is not None:
        if isinstance(arena_mask, dict):
            if arena_mask_name not in arena_mask:
                raise KeyError(
                    f"Mask dictionary is missing {arena_mask_name!r}. "
                    f"Found masks: {list(arena_mask.keys())}"
                )
            arena_mask_resolved = arena_mask[arena_mask_name]
        else:
            arena_mask_resolved = arena_mask

        arena_mask_resolved = np.asarray(arena_mask_resolved).astype(bool)
        if arena_mask_resolved.ndim != 2:
            raise ValueError(
                f"arena_mask must be 2D, got shape {arena_mask_resolved.shape}"
            )

        h, w = arena_mask_resolved.shape
        arena_points = np.zeros(len(plot_df), dtype=bool)
        valid_xy = np.isfinite(x) & np.isfinite(y)
        valid_idx = np.where(valid_xy)[0]
        xi = np.rint(x[valid_idx]).astype(int)
        yi = np.rint(y[valid_idx]).astype(int)
        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        checked_idx = valid_idx[in_bounds]
        arena_points[checked_idx] = arena_mask_resolved[yi[in_bounds], xi[in_bounds]]
    else:
        arena_mask_resolved = None
        arena_points = np.ones(len(plot_df), dtype=bool)

    active = (activity > activity_threshold) & arena_points

    x_plot = x[arena_points]
    y_plot = y[arena_points]
    activity_plot = activity[arena_points]

    if len(x_plot) == 0:
        raise ValueError("No valid points remain inside the arena mask")

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 3)
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[0, 2] = fig.add_subplot(grid[0, 2])
    axes[1, 0] = fig.add_subplot(grid[1, 0])
    axes[1, 1] = fig.add_subplot(grid[1, 1], projection="polar")
    axes[1, 2] = fig.add_subplot(grid[1, 2], projection="polar")

    ax = axes[0, 0]
    ax.plot(x_plot, y_plot, linewidth=1, alpha=0.6, color="gray")
    if active.any():
        sc = ax.scatter(
            x[active],
            y[active],
            c=head_dir_deg[active],
            s=point_size,
            cmap="hsv",
            vmin=0,
            vmax=360,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Head direction (degrees)")
        cbar.set_ticks([0, 90, 180, 270, 360])
    else:
        ax.text(
            0.5,
            0.5,
            "No active frames above threshold",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    if len(x_plot) > 0:
        ax.scatter(x_plot[0], y_plot[0], marker="^", s=80, )
        ax.scatter(x_plot[-1], y_plot[-1], marker="s", s=80, )
    ax.set_aspect("equal")
    ax.set_title(f"{cell_col}: active frames colored by head direction")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()

    ax = axes[0, 1]
    occupancy = ax.hist2d(x_plot, y_plot, bins=bins)
    ax.set_aspect("equal")
    ax.set_title("Spatial occupancy")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.colorbar(occupancy[3], ax=ax, label="visits")

    ax = axes[0, 2]
    if time_col in plot_df.columns:
        ax.plot(plot_df[time_col], activity, linewidth=1)
        ax.set_xlabel(time_col)
    else:
        ax.plot(activity, linewidth=1)
        ax.set_xlabel("frame")
    ax.axhline(activity_threshold, linestyle="--")
    ax.set_title(f"{cell_col} activity over time")
    ax.set_ylabel("activity")
    

    ax = axes[1, 0]
    x_edges = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), bins + 1)
    y_edges = np.linspace(np.nanmin(y_plot), np.nanmax(y_plot), bins + 1)
    spatial_occupancy, _, _ = np.histogram2d(x_plot, y_plot, bins=[x_edges, y_edges])
    activity_sum, _, _ = np.histogram2d(
        x_plot,
        y_plot,
        bins=[x_edges, y_edges],
        weights=activity_plot,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_activity = activity_sum / spatial_occupancy
    mean_activity[spatial_occupancy == 0] = np.nan
    im = ax.imshow(
        mean_activity.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
    )
    ax.set_title(f"{cell_col} activity / occupancy")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.colorbar(im, ax=ax, label=f"mean {cell_col} activity")

    ego_ax = axes[1, 1]
    if arena_mask is not None:
        from preprocess_functions import boundary_tuning as bt

        egocentric_map = bt.compute_egocentric_boundary_map(
            df,
            arena_mask_resolved,
            cell_col=cell_col,
            angle_bins=egocentric_angle_bins,
            distance_bins=egocentric_distance_bins,
            boundary_stride=egocentric_boundary_stride,
            frame_stride=egocentric_frame_stride,
            x_col=x_col,
            y_col=y_col,
            nose_x_col=nose_x_col,
            nose_y_col=nose_y_col,
        )
        bt.plot_egocentric_boundary_polar(
            egocentric_map,
            ax=ego_ax,
            cmap=egocentric_cmap,
            occupancy_min=egocentric_occupancy_min,
            radius_max=egocentric_radius_max,
            smooth_sigma=egocentric_smooth_sigma,
            show=False,
        )
    else:
        ego_ax.set_axis_off()
        ego_ax.text(
            0.5,
            0.5,
            "Pass arena_mask for\negocentric boundary map",
            transform=ego_ax.transAxes,
            ha="center",
            va="center",
        )

    hd_ax = axes[1, 2]
    hd_edges = np.linspace(-np.pi, np.pi, head_direction_bins + 1)
    hd_centers = 0.5 * (hd_edges[:-1] + hd_edges[1:])
    wrapped_hd = (head_dir + np.pi) % (2 * np.pi) - np.pi
    hd_bin_idx = np.digitize(wrapped_hd, hd_edges) - 1
    hd_bin_idx = np.where(hd_bin_idx == head_direction_bins, head_direction_bins - 1, hd_bin_idx)

    hd_tuning = np.full(head_direction_bins, np.nan, dtype=float)
    for i in range(head_direction_bins):
        mask = (hd_bin_idx == i) & np.isfinite(activity)
        if mask.any():
            hd_tuning[i] = np.nanmean(activity[mask])

    hd_theta = np.r_[hd_centers, hd_centers[0]]
    hd_radius = np.r_[hd_tuning, hd_tuning[0]]
    hd_ax.plot(hd_theta, hd_radius, marker="o", linewidth=1.5)
    hd_ax.fill(hd_theta, hd_radius, alpha=0.15)
    hd_ax.set_theta_zero_location("N")
    hd_ax.set_theta_direction(1)
    hd_ax.grid()
    hd_ax.set_title(f"{cell_col} head direction tuning")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


plot_2d_rate_map = plot_2d_ratemap
plot_2D_ratemap = plot_2d_ratemap
plot_EBC = plot_ebc
plot_EBC_heatmap = plot_ebc_heatmap
plot_EBC_occupancy = plot_ebc_occupancy
plot_HD = plot_hd
plot_HD_tuning = plot_hd_tuning
EBCplot = plot_ebc
HDplot = plot_hd
