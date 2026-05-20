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
from preprocess_functions import boundary_tuning 

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
