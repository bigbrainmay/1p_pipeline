from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .manifest import safe_id


def read_video_frame(video_path: str | Path, frame_idx: int = 0) -> np.ndarray:
    """Read one video frame as RGB for plotting/inspection."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def video_metadata(video_path: str | Path) -> dict[str, int | float]:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")
    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def polygon_to_mask(roi_points: Iterable[Iterable[float]], shape: tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    points = np.asarray(list(roi_points), dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("ROI must contain at least three x,y points")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 1)
    return mask.astype(bool)


def extract_roi_intensity_signal(
    video_path: str | Path,
    roi_points: Iterable[Iterable[float]],
    *,
    frame_step: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
    grayscale: bool = True,
    progress_every: int | None = 1000,
) -> pd.DataFrame:
    """
    Extract a frame-by-frame intensity trace from a polygon ROI.

    The returned frame indices always refer to the original behavior video.
    Use ``frame_step=1`` for event timing that will be written back to the
    dataframe; larger steps are only meant for quick scouting.
    """
    cv2 = _cv2()
    frame_step = int(frame_step)
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    metadata = video_metadata(video_path)
    fps = float(metadata["fps"]) if metadata["fps"] else np.nan

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    rows: list[dict[str, float | int]] = []
    frame_idx = int(start_frame)
    processed = 0
    mask = None

    while True:
        if max_frames is not None and processed >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            if grayscale:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                image = frame

            if mask is None:
                mask = polygon_to_mask(roi_points, image.shape[:2])
                if not mask.any():
                    raise ValueError("ROI mask has no pixels")

            values = image[mask]
            if values.ndim > 1:
                values = values.mean(axis=1)

            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "elapsed_s": float(frame_idx / fps) if fps and np.isfinite(fps) else np.nan,
                    "roi_mean": float(np.mean(values)),
                    "roi_median": float(np.median(values)),
                    "roi_p95": float(np.percentile(values, 95)),
                    "roi_std": float(np.std(values)),
                    "video_fps": fps,
                }
            )
            processed += 1
            if progress_every and processed % int(progress_every) == 0:
                print(f"processed {processed} sampled frames; current video frame {frame_idx}")

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def smooth_port_signal(
    signal_df: pd.DataFrame,
    *,
    value_col: str = "roi_mean",
    smooth_window: int = 5,
    baseline_quantile: float = 0.1,
) -> pd.DataFrame:
    out = signal_df.copy()
    if out.empty:
        out[f"{value_col}_smooth"] = []
        out[f"{value_col}_delta"] = []
        out[f"{value_col}_z"] = []
        return out

    window = max(1, int(smooth_window))
    values = pd.to_numeric(out[value_col], errors="coerce")
    smooth = values.rolling(window=window, min_periods=1, center=True).median()
    baseline = float(smooth.quantile(float(baseline_quantile)))
    spread = float(smooth.std(ddof=0))
    if not np.isfinite(spread) or spread == 0:
        spread = 1.0

    out[f"{value_col}_smooth"] = smooth
    out[f"{value_col}_delta"] = smooth - baseline
    out[f"{value_col}_z"] = (smooth - baseline) / spread
    return out


def auto_signal_threshold(
    signal_df: pd.DataFrame,
    *,
    signal_col: str = "roi_mean_smooth",
    low_quantile: float = 0.1,
    high_quantile: float = 0.95,
    high_fraction: float = 0.5,
) -> float:
    values = pd.to_numeric(signal_df[signal_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric values in {signal_col!r}")
    low = float(values.quantile(float(low_quantile)))
    high = float(values.quantile(float(high_quantile)))
    return low + float(high_fraction) * (high - low)


def detect_signal_intervals(
    signal_df: pd.DataFrame,
    *,
    threshold: float | None = None,
    port_name: str = "port_1",
    signal_col: str = "roi_mean_smooth",
    frame_col: str = "frame_idx",
    min_on_frames: int = 2,
) -> pd.DataFrame:
    if signal_df.empty:
        return _empty_video_port_events()

    if threshold is None:
        threshold = auto_signal_threshold(signal_df, signal_col=signal_col)

    values = pd.to_numeric(signal_df[signal_col], errors="coerce").to_numpy()
    frames = pd.to_numeric(signal_df[frame_col], errors="coerce").to_numpy()
    active = np.isfinite(values) & (values >= float(threshold))

    rows: list[dict[str, Any]] = []
    idx = 0
    event_idx = 0
    while idx < len(active):
        if not active[idx]:
            idx += 1
            continue

        start_i = idx
        while idx < len(active) and active[idx]:
            idx += 1
        stop_i = idx

        run_len = stop_i - start_i
        if run_len < int(min_on_frames):
            continue

        run_values = values[start_i:stop_i]
        peak_local = int(np.nanargmax(run_values))
        peak_i = start_i + peak_local
        first_off_i = stop_i if stop_i < len(frames) else stop_i - 1

        start_frame = int(frames[start_i])
        last_active_frame = int(frames[stop_i - 1])
        stop_frame = int(frames[first_off_i])
        fps = _constant_or_nan(signal_df.get("video_fps"))
        duration_frames = max(0, stop_frame - start_frame)

        rows.append(
            {
                "port_signal_event_idx": event_idx,
                "port_name": str(port_name),
                "start_frame": start_frame,
                "last_active_frame": last_active_frame,
                "stop_frame": stop_frame,
                "peak_frame": int(frames[peak_i]),
                "start_signal": float(values[start_i]),
                "stop_signal": float(values[first_off_i]),
                "peak_signal": float(values[peak_i]),
                "threshold": float(threshold),
                "signal_col": signal_col,
                "sample_count": int(run_len),
                "duration_frames": int(duration_frames),
                "duration_s": float(duration_frames / fps) if fps and np.isfinite(fps) else np.nan,
                "event_source": "video_roi",
            }
        )
        event_idx += 1

    return pd.DataFrame(rows) if rows else _empty_video_port_events()


def add_aligned_times_to_port_events(
    events: pd.DataFrame,
    aligned: pd.DataFrame,
    *,
    frame_col: str = "beh_frame_idx",
    ts_col: str = "global_ts",
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if frame_col not in aligned.columns:
        raise KeyError(f"Aligned dataframe is missing {frame_col!r}")

    frame_map = aligned[[frame_col, "global_idx", ts_col]].dropna(subset=[frame_col]).copy()
    frame_map[frame_col] = pd.to_numeric(frame_map[frame_col], errors="coerce")
    frame_map = frame_map.dropna(subset=[frame_col]).sort_values(frame_col)

    out = events.copy()
    for prefix, source_col in [
        ("start", "start_frame"),
        ("stop", "stop_frame"),
        ("peak", "peak_frame"),
    ]:
        mapped = [
            _nearest_aligned_frame(frame_map, frame, frame_col=frame_col, ts_col=ts_col)
            for frame in pd.to_numeric(out[source_col], errors="coerce")
        ]
        out[f"{prefix}_global_idx"] = [row["global_idx"] for row in mapped]
        out[f"{prefix}_global_ts"] = [row[ts_col] for row in mapped]

    out["port_on_ts"] = out["start_global_ts"]
    out["port_off_ts"] = out["stop_global_ts"]
    return out


def add_video_port_events_to_aligned(
    aligned: pd.DataFrame,
    events_path: str | Path | Iterable[str | Path] | None,
    *,
    frame_col: str = "beh_frame_idx",
) -> pd.DataFrame:
    out = aligned.copy()
    events = load_video_port_events(events_path)
    out["video_port_event_dropped"] = True
    out["video_any_port_active"] = False

    if events.empty:
        return out

    out[frame_col] = pd.to_numeric(out[frame_col], errors="coerce")

    active_cols: list[str] = []
    for port_name in sorted(events["port_name"].dropna().astype(str).unique()):
        col = f"video_{safe_id(port_name)}_active"
        active_cols.append(col)
        out[col] = False

    out["video_port_signal_source"] = "video_roi"
    out["video_port_event_idx"] = pd.NA
    out["video_port_name"] = pd.NA
    out["video_port_state"] = pd.NA

    for _, event in events.iterrows():
        port_name = str(event.get("port_name", "port"))
        col = f"video_{safe_id(port_name)}_active"
        start = _maybe_int(event.get("start_frame"))
        active_stop = _maybe_int(event.get("last_active_frame", event.get("stop_frame")))
        stop = _maybe_int(event.get("stop_frame", active_stop))
        if start is None or active_stop is None or stop is None:
            continue

        active_mask = (out[frame_col] >= start) & (out[frame_col] <= active_stop)
        if col in out.columns:
            out.loc[active_mask, col] = True
        out.loc[active_mask, "video_any_port_active"] = True
        out.loc[active_mask, "video_port_event_dropped"] = False

        start_mask = out[frame_col] == start
        stop_mask = out[frame_col] == stop
        event_idx = event.get("port_signal_event_idx", pd.NA)
        out.loc[start_mask, ["video_port_event_idx", "video_port_name", "video_port_state"]] = [
            event_idx,
            port_name,
            "on",
        ]
        out.loc[stop_mask, ["video_port_event_idx", "video_port_name", "video_port_state"]] = [
            event_idx,
            port_name,
            "off",
        ]

    return out


def load_video_port_events(
    events_path: str | Path | Iterable[str | Path] | None,
) -> pd.DataFrame:
    if events_path is None:
        return _empty_video_port_events()

    if isinstance(events_path, (str, Path)):
        paths = [Path(events_path)]
    else:
        paths = [Path(path) for path in events_path]

    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "port_name" not in df.columns:
            df["port_name"] = _infer_port_name_from_path(path)
        if "event_source" not in df.columns:
            df["event_source"] = "video_roi"
        frames.append(df)

    if not frames:
        return _empty_video_port_events()

    events = pd.concat(frames, ignore_index=True)
    events["port_name"] = events["port_name"].fillna("port").astype(str)
    if "port_signal_event_idx" not in events.columns:
        events["port_signal_event_idx"] = np.arange(len(events), dtype=int)
    return events


def write_port_signal_outputs(
    behavior_dir: str | Path,
    recording_id: str,
    port_name: str,
    signal_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    update_combined: bool = True,
) -> dict[str, Path]:
    behavior_dir = Path(behavior_dir)
    behavior_dir.mkdir(parents=True, exist_ok=True)

    recording_safe = safe_id(recording_id)
    port_safe = safe_id(port_name)
    signal_path = behavior_dir / f"{recording_safe}_{port_safe}_port_signal.csv"
    events_path = behavior_dir / f"{recording_safe}_{port_safe}_port_events_from_video.csv"
    combined_path = behavior_dir / f"{recording_safe}_video_port_events.csv"

    signal_df.to_csv(signal_path, index=False)
    events_to_write = events_df.copy()
    events_to_write["recording_id"] = recording_id
    events_to_write["port_name"] = str(port_name)
    events_to_write.to_csv(events_path, index=False)

    if update_combined:
        upsert_combined_port_events(combined_path, events_to_write, port_name=port_name)

    return {
        "port_signal_csv": signal_path,
        "port_events_csv": events_path,
        "video_port_events_csv": combined_path,
    }


def upsert_combined_port_events(
    combined_path: str | Path,
    events_df: pd.DataFrame,
    *,
    port_name: str,
) -> pd.DataFrame:
    combined_path = Path(combined_path)
    existing = pd.DataFrame()
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        if "port_name" in existing.columns:
            existing = existing[existing["port_name"].astype(str) != str(port_name)]

    combined = pd.concat([existing, events_df], ignore_index=True)
    if not combined.empty and "start_frame" in combined.columns:
        combined = combined.sort_values(["start_frame", "port_name"]).reset_index(drop=True)
        combined["video_port_event_idx"] = np.arange(len(combined), dtype=int)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    return combined


def crop_frame_around_roi(
    frame_rgb: np.ndarray,
    roi_points: Iterable[Iterable[float]],
    *,
    padding: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(list(roi_points), dtype=float)
    if points.ndim != 2 or points.shape[0] < 3:
        raise ValueError("ROI must contain at least three points")
    height, width = frame_rgb.shape[:2]
    x0 = max(0, int(np.floor(points[:, 0].min())) - int(padding))
    x1 = min(width, int(np.ceil(points[:, 0].max())) + int(padding))
    y0 = max(0, int(np.floor(points[:, 1].min())) - int(padding))
    y1 = min(height, int(np.ceil(points[:, 1].max())) + int(padding))
    crop = frame_rgb[y0:y1, x0:x1]
    shifted_points = points - np.array([x0, y0], dtype=float)
    return crop, shifted_points


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video ROI signal extraction. "
            "Install opencv-python in the analysis environment."
        ) from exc
    return cv2


def _constant_or_nan(values: Any) -> float:
    if values is None:
        return np.nan
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    return float(series.iloc[0])


def _nearest_aligned_frame(
    frame_map: pd.DataFrame,
    frame: Any,
    *,
    frame_col: str,
    ts_col: str,
) -> dict[str, Any]:
    if pd.isna(frame) or frame_map.empty:
        return {"global_idx": pd.NA, ts_col: pd.NaT}
    values = frame_map[frame_col].to_numpy(dtype=float)
    idx = int(np.abs(values - float(frame)).argmin())
    row = frame_map.iloc[idx]
    return {"global_idx": row.get("global_idx", pd.NA), ts_col: row.get(ts_col, pd.NaT)}


def _maybe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(float(value))


def _infer_port_name_from_path(path: Path) -> str:
    stem = path.stem
    match = re.search(
        r"(port[_-]?\d+(?:[_-](?:left|right|up|down))?)",
        stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        return "port"
    return match.group(1).replace("-", "_").lower()


def _empty_video_port_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "port_signal_event_idx",
            "port_name",
            "start_frame",
            "last_active_frame",
            "stop_frame",
            "peak_frame",
            "start_signal",
            "stop_signal",
            "peak_signal",
            "threshold",
            "signal_col",
            "sample_count",
            "duration_frames",
            "duration_s",
            "event_source",
        ]
    )
