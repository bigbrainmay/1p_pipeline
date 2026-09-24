from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_LABEL_OPTIONS = [
    "left_small_loop",
    "left_inverse_small_loop",
    "left_big_loop",
    "right_small_loop",
    "right_inverse_small_loop",
    "right_big_loop",
    "non_characteristic",
]

DEFAULT_XY_COLUMN_PAIRS = [
    ("ear_mid_x", "ear_mid_y"),
    ("nose.x", "nose.y"),
    ("centroid_x", "centroid_y"),
    ("center_x", "center_y"),
    ("x", "y"),
]

DEFAULT_TIME_SERIES_CANDIDATES = [
    "ear_mid_x",
    "ear_mid_y",
    "nose.x",
    "nose.y",
    "ear_L.x",
    "ear_L.y",
    "ear_R.x",
    "ear_R.y",
    "head_dir_rad",
    "ang_vel_speed",
    "ang_vel_rad_s",
    "in_arena",
    "in_startbox_L",
    "in_startbox_R",
    "arena_only",
    "bpod_any_port_active",
    "bpod_port_1_active",
    "bpod_port_2_active",
    "bpod_port_3_active",
    "bpod_port_4_active",
    "video_any_port_active",
    "video_port_1_left_active",
    "video_port_2_right_active",
    "video_port_3_up_active",
    "video_port_4_down_active",
    "video_port_1_active",
    "video_port_2_active",
    "video_port_3_active",
    "video_port_4_active",
]


@dataclass
class TrialFeatureResult:
    X: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]


def load_trial_catalog(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = Path(output_root)
    aligned_index = _read_csv(output_root / "aligned_session_index.csv")
    trial_index = _read_csv(output_root / "trial_segment_index.csv")

    aligned_lookup = (
        aligned_index.set_index("recording_id")["aligned_csv"].to_dict()
        if "recording_id" in aligned_index
        else {}
    )
    trial_rows = []

    for _, row in trial_index.iterrows():
        if str(row.get("status", "")).lower() not in {"segmented", "dry_run"}:
            continue
        trials_csv = _clean_path(row.get("trials_csv"))
        if trials_csv is None or not trials_csv.exists():
            continue

        trials = pd.read_csv(trials_csv)
        recording_id = str(row.get("recording_id"))
        session_id = str(row.get("session_id"))
        aligned_csv = _clean_path(row.get("aligned_csv")) or _clean_path(
            aligned_lookup.get(recording_id)
        )

        trials["recording_id"] = recording_id
        trials["session_id"] = session_id
        trials["trials_csv"] = str(trials_csv)
        trials["aligned_csv"] = str(aligned_csv) if aligned_csv is not None else None
        trial_rows.append(trials)

    if not trial_rows:
        raise ValueError("No segmented trials found. Run scripts/segment_trials.py first.")

    catalog = pd.concat(trial_rows, ignore_index=True)
    catalog["trial_idx"] = pd.to_numeric(catalog["trial_idx"], errors="coerce").astype("Int64")
    catalog["start_frame"] = pd.to_numeric(catalog["start_frame"], errors="coerce").astype("Int64")
    catalog["end_frame"] = pd.to_numeric(catalog["end_frame"], errors="coerce").astype("Int64")
    catalog["trial_uid"] = (
        catalog["recording_id"].astype(str) + "::" + catalog["trial_idx"].astype(str)
    )
    catalog = catalog.dropna(
        subset=["trial_idx", "start_frame", "end_frame", "aligned_csv"]
    ).reset_index(drop=True)
    return catalog, aligned_index, trial_index


def make_aligned_loader() -> Callable[[pd.Series], pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}

    def load_aligned_for_row(row: pd.Series) -> pd.DataFrame:
        recording_id = str(row["recording_id"])
        if recording_id not in cache:
            aligned_path = Path(str(row["aligned_csv"]))
            cache[recording_id] = pd.read_csv(aligned_path)
        return cache[recording_id]

    return load_aligned_for_row


def select_xy_columns(
    df: pd.DataFrame,
    xy_column_pairs: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    for x_col, y_col in xy_column_pairs or DEFAULT_XY_COLUMN_PAIRS:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError(
        "Could not find x/y columns. "
        f"Tried: {xy_column_pairs or DEFAULT_XY_COLUMN_PAIRS}. "
        f"Found: {df.columns.tolist()}"
    )


def slice_trial(
    row: pd.Series,
    load_aligned_for_row: Callable[[pd.Series], pd.DataFrame],
) -> pd.DataFrame:
    df = load_aligned_for_row(row)
    start = max(0, int(row["start_frame"]))
    end = min(len(df) - 1, int(row["end_frame"]))
    if end < start:
        return df.iloc[0:0].copy()
    return df.iloc[start : end + 1].copy()


class TrialFeaturizer:
    def __init__(
        self,
        *,
        fixed_t: int = 200,
        fps: float = 30.0,
        time_series_candidates: list[str] | None = None,
        xy_column_pairs: list[tuple[str, str]] | None = None,
    ):
        self.fixed_t = int(fixed_t)
        self.fps = float(fps)
        self.time_series_candidates = time_series_candidates or DEFAULT_TIME_SERIES_CANDIDATES
        self.xy_column_pairs = xy_column_pairs or DEFAULT_XY_COLUMN_PAIRS
        self.time_series_cols_: list[str] | None = None
        self.scalar_names_: list[str] | None = None

    def prepare_schema(
        self,
        catalog: pd.DataFrame,
        *,
        load_aligned_for_row: Callable[[pd.Series], pd.DataFrame],
    ) -> None:
        available = set()
        for _, row in catalog.drop_duplicates("recording_id").iterrows():
            df = load_aligned_for_row(row)
            available.update(df.columns)

        cols = [col for col in self.time_series_candidates if col in available]
        active_port_cols = sorted(
            col
            for col in available
            if col.endswith("_active")
            and (col.startswith("bpod_port_") or col.startswith("video_port_"))
        )
        extra_active = [col for col in active_port_cols if col not in cols]
        self.time_series_cols_ = [*cols, *extra_active]

        scalar_names = [
            "duration_s",
            "duration_frames",
            "path_length_px",
            "displacement_px",
            "straightness",
            "x_range_px",
            "y_range_px",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "mean_speed_px_s",
            "frac_in_arena",
            "frac_in_startbox_L",
            "frac_in_startbox_R",
            "frac_arena_only",
            "frac_bpod_any_port_active",
            "frac_video_any_port_active",
            "start_side_L",
            "start_side_R",
            "end_side_L",
            "end_side_R",
        ]
        scalar_names.extend(f"frac_{col}" for col in active_port_cols)
        self.scalar_names_ = list(dict.fromkeys(scalar_names))

    def apply_bundle_schema(self, bundle: dict[str, Any]) -> None:
        time_series_cols = bundle.get("time_series_cols")
        scalar_names = bundle.get("scalar_names")
        if time_series_cols is None or scalar_names is None:
            time_series_cols, scalar_names = infer_schema_from_feature_names(
                bundle.get("feature_names", []),
                fixed_t=int(bundle.get("fixed_t", self.fixed_t)),
            )
        self.time_series_cols_ = list(time_series_cols)
        self.scalar_names_ = list(scalar_names)

    def schema(self) -> dict[str, Any]:
        return {
            "fixed_t": self.fixed_t,
            "fps": self.fps,
            "time_series_cols": list(self.time_series_cols_ or []),
            "scalar_names": list(self.scalar_names_ or []),
            "feature_names": self.feature_names(),
        }

    def feature_names(self) -> list[str]:
        if self.time_series_cols_ is None or self.scalar_names_ is None:
            return []
        names = []
        for col in self.time_series_cols_:
            names.extend([f"{col}_t{i:03d}" for i in range(self.fixed_t)])
        names.extend(self.scalar_names_)
        return names

    def numeric_series(self, values: Any) -> np.ndarray:
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        if len(arr) == 0:
            return np.full(self.fixed_t, np.nan)
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return np.full(self.fixed_t, np.nan)
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, self.fixed_t)
        return np.interp(x_new, x_old[valid], arr[valid])

    def scalar_features(self, df_trial: pd.DataFrame, row: pd.Series) -> dict[str, float]:
        scalars: dict[str, float] = {}
        duration = row.get("duration_s", np.nan)
        scalars["duration_s"] = (
            float(duration) if pd.notna(duration) else len(df_trial) / self.fps
        )
        scalars["duration_frames"] = float(len(df_trial))

        try:
            x_col, y_col = select_xy_columns(df_trial, self.xy_column_pairs)
            x = pd.to_numeric(df_trial[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df_trial[y_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 2:
                xv = x[valid]
                yv = y[valid]
                steps = np.sqrt(np.diff(xv) ** 2 + np.diff(yv) ** 2)
                path_length = float(np.nansum(steps))
                displacement = float(np.sqrt((xv[-1] - xv[0]) ** 2 + (yv[-1] - yv[0]) ** 2))
                scalars.update(
                    {
                        "path_length_px": path_length,
                        "displacement_px": displacement,
                        "straightness": displacement / path_length if path_length > 0 else np.nan,
                        "x_range_px": float(np.nanmax(xv) - np.nanmin(xv)),
                        "y_range_px": float(np.nanmax(yv) - np.nanmin(yv)),
                        "start_x": float(xv[0]),
                        "start_y": float(yv[0]),
                        "end_x": float(xv[-1]),
                        "end_y": float(yv[-1]),
                        "mean_speed_px_s": (
                            float(path_length / scalars["duration_s"])
                            if scalars["duration_s"] > 0
                            else np.nan
                        ),
                    }
                )
        except Exception:
            pass

        for col in [
            "in_arena",
            "in_startbox_L",
            "in_startbox_R",
            "arena_only",
            "bpod_any_port_active",
            "video_any_port_active",
        ]:
            if col in df_trial.columns:
                scalars[f"frac_{col}"] = float(
                    pd.to_numeric(df_trial[col], errors="coerce").mean()
                )

        for col in [
            col
            for col in df_trial.columns
            if col.endswith("_active")
            and (col.startswith("bpod_port_") or col.startswith("video_port_"))
        ]:
            scalars[f"frac_{col}"] = float(
                pd.to_numeric(df_trial[col], errors="coerce").mean()
            )

        for side_col in ["start_side", "end_side"]:
            value = row.get(side_col)
            scalars[f"{side_col}_L"] = float(value == "L")
            scalars[f"{side_col}_R"] = float(value == "R")

        return scalars

    def featurize_one(
        self,
        row: pd.Series,
        *,
        load_aligned_for_row: Callable[[pd.Series], pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        if self.time_series_cols_ is None or self.scalar_names_ is None:
            raise RuntimeError("Prepare or apply a schema before featurize_one")

        df_trial = slice_trial(row, load_aligned_for_row)
        series_features = []
        for col in self.time_series_cols_:
            values = (
                df_trial[col]
                if col in df_trial.columns
                else pd.Series([np.nan] * len(df_trial))
            )
            series_features.append(self.numeric_series(values))

        scalars = self.scalar_features(df_trial, row)
        scalar_values = np.array(
            [scalars.get(name, np.nan) for name in self.scalar_names_],
            dtype=float,
        )

        if series_features:
            vector = np.concatenate([np.concatenate(series_features), scalar_values])
        else:
            vector = scalar_values
        return vector, self.feature_names()

    def build_matrix(
        self,
        catalog: pd.DataFrame,
        *,
        load_aligned_for_row: Callable[[pd.Series], pd.DataFrame],
        skipped_path: str | Path | None = None,
    ) -> TrialFeatureResult:
        if self.time_series_cols_ is None or self.scalar_names_ is None:
            self.prepare_schema(catalog, load_aligned_for_row=load_aligned_for_row)

        vectors = []
        metas = []
        feature_names = None
        skipped = []

        for _, row in catalog.iterrows():
            try:
                vector, names = self.featurize_one(
                    row,
                    load_aligned_for_row=load_aligned_for_row,
                )
                if feature_names is None:
                    feature_names = names
                vectors.append(vector)
                metas.append(
                    {
                        "recording_id": row["recording_id"],
                        "session_id": row.get("session_id"),
                        "trial_idx": int(row["trial_idx"]),
                        "start_frame": int(row["start_frame"]),
                        "end_frame": int(row["end_frame"]),
                        "trial_uid": row["trial_uid"],
                    }
                )
            except Exception as exc:
                skipped.append({"trial_uid": row.get("trial_uid"), "error": repr(exc)})

        if skipped and skipped_path is not None:
            skipped_path = Path(skipped_path)
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(skipped).to_csv(skipped_path, index=False)

        if not vectors:
            raise ValueError("No trial features could be built")
        return TrialFeatureResult(np.vstack(vectors), pd.DataFrame(metas), feature_names or [])


def infer_schema_from_feature_names(
    feature_names: list[str],
    *,
    fixed_t: int,
) -> tuple[list[str], list[str]]:
    time_series_cols = []
    scalar_names = []
    seen_series = set()
    pattern = re.compile(r"(.+)_t\d{3}$")
    for name in feature_names:
        match = pattern.match(str(name))
        if match:
            col = match.group(1)
            if col not in seen_series:
                time_series_cols.append(col)
                seen_series.add(col)
        else:
            scalar_names.append(str(name))
    if fixed_t <= 0:
        raise ValueError("fixed_t must be positive")
    return time_series_cols, scalar_names


def labels_for_features(meta: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["trial_idx"] = pd.to_numeric(labels["trial_idx"], errors="coerce").astype("Int64")
    return meta.merge(
        labels[["recording_id", "trial_idx", "label"]],
        on=["recording_id", "trial_idx"],
        how="left",
    )


def predict_confidence(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.nanmax(proba, axis=1)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X))
        if decision.ndim == 1:
            return 1 / (1 + np.exp(-np.abs(decision)))
        best = np.nanmax(decision, axis=1)
        second = np.partition(decision, -2, axis=1)[:, -2] if decision.shape[1] > 1 else 0
        return 1 / (1 + np.exp(-(best - second)))
    return np.full(X.shape[0], np.nan)


def load_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    columns = ["recording_id", "session_id", "trial_idx", "label", "notes", "start_frame", "end_frame"]
    if path.exists():
        labels = pd.read_csv(path)
        for col in columns:
            if col not in labels.columns:
                labels[col] = np.nan
        labels["trial_idx"] = pd.to_numeric(labels["trial_idx"], errors="coerce").astype("Int64")
        return labels
    return pd.DataFrame(columns=columns)


def save_labels(labels: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.sort_values(["recording_id", "trial_idx"]).to_csv(path, index=False)


def upsert_label(
    labels: pd.DataFrame,
    row: pd.Series,
    label: str,
    *,
    path: str | Path,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    labels = labels.copy()
    extra = extra or {}
    for col in extra:
        if col not in labels.columns:
            labels[col] = np.nan

    mask = (
        (labels["recording_id"].astype(str) == str(row["recording_id"]))
        & (pd.to_numeric(labels["trial_idx"], errors="coerce") == int(row["trial_idx"]))
    )
    new_row = {
        "recording_id": row["recording_id"],
        "session_id": row.get("session_id"),
        "trial_idx": int(row["trial_idx"]),
        "label": label,
        "notes": notes,
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        **extra,
    }
    if mask.any():
        for key, value in new_row.items():
            if key not in labels.columns:
                labels[key] = np.nan
            labels.loc[mask, key] = value
    else:
        labels = pd.concat([labels, pd.DataFrame([new_row])], ignore_index=True)
    labels["trial_idx"] = pd.to_numeric(labels["trial_idx"], errors="coerce").astype("Int64")
    save_labels(labels, path)
    return labels


def plot_trial(
    row: pd.Series,
    *,
    load_aligned_for_row: Callable[[pd.Series], pd.DataFrame],
    title_prefix: str = "",
    prediction_text: str | None = None,
    show_video_background: bool = False,
    invert_y_axis: bool = True,
):
    import matplotlib.pyplot as plt

    df_trial = slice_trial(row, load_aligned_for_row)
    if df_trial.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("Empty trial slice")
        return fig

    x_col, y_col = select_xy_columns(df_trial)
    x = pd.to_numeric(df_trial[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_trial[y_col], errors="coerce").to_numpy(dtype=float)
    t = np.arange(len(df_trial))

    fig, ax = plt.subplots(figsize=(7, 6))

    if show_video_background and "beh_vid_path" in df_trial.columns:
        frame = _read_first_video_frame(
            df_trial["beh_vid_path"].dropna().iloc[0]
            if df_trial["beh_vid_path"].notna().any()
            else None
        )
        if frame is not None:
            ax.imshow(frame)

    ax.plot(x, y, color="0.75", linewidth=1, zorder=1)
    sc = ax.scatter(x, y, c=t, cmap="viridis", s=18, zorder=2)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        first = np.flatnonzero(valid)[0]
        last = np.flatnonzero(valid)[-1]
        ax.scatter([x[first]], [y[first]], c="lime", edgecolor="black", s=80, label="start", zorder=3)
        ax.scatter([x[last]], [y[last]], c="red", edgecolor="black", s=80, label="end", zorder=3)

    if invert_y_axis:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best")
    plt.colorbar(sc, ax=ax, label="trial frame")

    pieces = [title_prefix.strip()] if title_prefix else []
    pieces.append(
        f"{row['recording_id']} trial {int(row['trial_idx'])} "
        f"frames {int(row['start_frame'])}-{int(row['end_frame'])}"
    )
    if pd.notna(row.get("cue_key", np.nan)):
        pieces.append(f"cue={row.get('cue_key')}")
    if prediction_text:
        pieces.append(prediction_text)
    ax.set_title("\n".join(pieces))
    fig.tight_layout()
    return fig


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the upstream pipeline stage first.")
    return pd.read_csv(path)


def _clean_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _read_first_video_frame(video_path: str | None):
    if not video_path or pd.isna(video_path):
        return None
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
