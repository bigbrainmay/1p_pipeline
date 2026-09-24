from pathlib import Path
import re

import pandas as pd
import numpy as np


CUE_OVERRIDE_COLUMNS = ["recording_id", "cue_ts", "cue_ts_note"]
CUE_EVENT_COLUMNS = [
    "cue_event_idx",
    "cue_key",
    "cue_start_utc",
    "cue_end_utc",
    "cue_duration_s",
    "cue_start_frame",
    "cue_end_frame",
    "cue_start_global_idx",
    "cue_end_global_idx",
]

CUE_KEY_ALIASES = {
    "LEFT": "left",
    "ARROWLEFT": "left",
    "LEFTARROW": "left",
    "←": "left",
    "RIGHT": "right",
    "ARROWRIGHT": "right",
    "RIGHTARROW": "right",
    "→": "right",
    "DOWN": "down",
    "ARROWDOWN": "down",
    "DOWNARROW": "down",
    "↓": "down",
    "UP": "up",
    "ARROWUP": "up",
    "UPARROW": "up",
    "↑": "up",
}
KEY_RE = re.compile(
    r"(?<![A-Za-z])("
    r"ArrowLeft|ArrowRight|ArrowDown|ArrowUp|"
    r"LeftArrow|RightArrow|DownArrow|UpArrow|"
    r"Left|Right|Down|Up|"
    r"←|→|↓|↑"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)
UTC_TS_RE = re.compile(
    r"("
    r"\d{4}-\d{2}-\d{2}"
    r"[ T]"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:\s*(?:Z|UTC)|[+-]\d{2}:?\d{2})?"
    r")"
)


def load_cue_timestamp_overrides(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CUE_OVERRIDE_COLUMNS)

    overrides = pd.read_csv(path, dtype=str).fillna("")
    for col in CUE_OVERRIDE_COLUMNS:
        if col not in overrides.columns:
            overrides[col] = ""
    return overrides


def resolve_cue_timestamp(
    *,
    recording_id,
    default_cue_ts=None,
    cue_overrides: pd.DataFrame | None = None,
) -> tuple[str | None, str, str | None]:
    """
    Return cue timestamp plus provenance.

    cue_overrides should have columns recording_id, cue_ts, cue_ts_note. A
    non-empty cue_ts in that file wins over the manifest/aligned dataframe value.
    """
    default_value = _clean_optional_text(default_cue_ts)

    if cue_overrides is not None and not cue_overrides.empty:
        required = {"recording_id", "cue_ts"}
        missing = required - set(cue_overrides.columns)
        if missing:
            raise ValueError(f"cue_overrides is missing columns: {sorted(missing)}")

        matches = cue_overrides[
            cue_overrides["recording_id"].astype(str) == str(recording_id)
        ]
        matches = matches[matches["cue_ts"].astype(str).str.strip() != ""]
        if not matches.empty:
            row = matches.iloc[-1]
            note = row.get("cue_ts_note", "")
            return str(row["cue_ts"]).strip(), "manual_override", _clean_optional_text(note)

    return default_value, "manifest", None


def parse_cue_events(cue_ts, cue_duration_s: float | None = None) -> pd.DataFrame:
    """
    Parse cue metadata containing arrow-key cue labels and UTC timestamps.

    Multiple cue events can be separated by newlines, semicolons, or pipes.
    If cue_duration_s is None, each event's end time is the next cue start.
    The final event has no end unless cue_duration_s is provided.
    """
    text = _clean_optional_text(cue_ts)
    if text is None:
        return pd.DataFrame(columns=CUE_EVENT_COLUMNS)

    chunks = [chunk.strip() for chunk in re.split(r"[\n;|]+", text) if chunk.strip()]
    if not chunks:
        chunks = [text]

    rows = []
    for chunk in chunks:
        ts_match = UTC_TS_RE.search(chunk)
        if not ts_match:
            continue

        timestamp = pd.to_datetime(ts_match.group(1), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue

        key_match = KEY_RE.search(chunk)
        cue_key = normalize_cue_key(key_match.group(1)) if key_match else None
        rows.append(
            {
                "cue_key": cue_key,
                "cue_start_utc": timestamp,
                "raw_cue_event": chunk,
            }
        )

    if not rows:
        return pd.DataFrame(columns=[*CUE_EVENT_COLUMNS, "raw_cue_event"])

    events = (
        pd.DataFrame(rows)
        .sort_values("cue_start_utc")
        .reset_index(drop=True)
    )
    events.insert(0, "cue_event_idx", events.index.astype(int))

    if cue_duration_s is None:
        events["cue_end_utc"] = events["cue_start_utc"].shift(-1)
    else:
        events["cue_end_utc"] = (
            events["cue_start_utc"] + pd.to_timedelta(float(cue_duration_s), unit="s")
        )

    events["cue_duration_s"] = (
        events["cue_end_utc"] - events["cue_start_utc"]
    ).dt.total_seconds()

    for col in (
        "cue_start_frame",
        "cue_end_frame",
        "cue_start_global_idx",
        "cue_end_global_idx",
    ):
        events[col] = np.nan

    return events


def normalize_cue_key(value) -> str | None:
    """Normalize actual arrow-key cue labels to left/right/down/up."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    compact = re.sub(r"[\s_-]+", "", raw).upper()
    return CUE_KEY_ALIASES.get(compact) or CUE_KEY_ALIASES.get(raw)


def add_cue_event_frame_columns(
    cue_events: pd.DataFrame,
    df: pd.DataFrame,
    *,
    ts_col: str = "global_ts",
) -> pd.DataFrame:
    events = cue_events.copy()
    if events.empty or ts_col not in df.columns:
        return events

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    global_idx_values = df["global_idx"].to_numpy() if "global_idx" in df.columns else None

    for event_idx, row in events.iterrows():
        start_pos = _nearest_frame_position(ts, row.get("cue_start_utc"))
        end_pos = _nearest_frame_position(ts, row.get("cue_end_utc"))

        if start_pos is not None:
            events.at[event_idx, "cue_start_frame"] = int(start_pos)
            if global_idx_values is not None:
                events.at[event_idx, "cue_start_global_idx"] = global_idx_values[start_pos]

        if end_pos is not None:
            events.at[event_idx, "cue_end_frame"] = int(end_pos)
            if global_idx_values is not None:
                events.at[event_idx, "cue_end_global_idx"] = global_idx_values[end_pos]

    return events


def trials_to_dataframe(
    trials,
    df: pd.DataFrame,
    *,
    recording_id=None,
    session_id=None,
    mouse_id=None,
    trial_type=None,
    cue_ts=None,
    cue_ts_source="manifest",
    cue_ts_note=None,
    cue_events: pd.DataFrame | None = None,
    cue_duration_s: float | None = None,
    ts_col: str = "global_ts",
    fps=30.0,
) -> pd.DataFrame:
    rows = []
    if cue_events is None:
        cue_events = parse_cue_events(cue_ts, cue_duration_s=cue_duration_s)
    cue_events = add_cue_event_frame_columns(cue_events, df, ts_col=ts_col)

    for trial_idx, trial in enumerate(trials):
        if isinstance(trial, dict):
            start_frame = int(trial["start_frame"])
            end_frame = int(trial["end_frame"])
            start_side = trial.get("start_side")
            end_side = trial.get("end_side")
        else:
            start_frame, end_frame = map(int, trial)
            start_side = None
            end_side = None

        row = {
            "recording_id": recording_id,
            "session_id": session_id,
            "mouse_id": mouse_id,
            "trial_idx": int(trial_idx),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_side": start_side,
            "end_side": end_side,
            "trial_type": trial_type,
            "cue_ts": cue_ts,
            "cue_ts_source": cue_ts_source,
            "cue_ts_note": cue_ts_note,
            "duration_frames": int(end_frame - start_frame + 1),
            "duration_s": float((end_frame - start_frame + 1) / fps) if fps else np.nan,
        }

        for col in ("global_ts", "global_idx", "beh_frame_idx", "neu_frame_idx"):
            if col in df.columns:
                row[f"start_{col}"] = _safe_row_value(df, start_frame, col)
                row[f"end_{col}"] = _safe_row_value(df, end_frame, col)

        cue_row = _cue_event_for_trial(
            cue_events,
            df,
            start_frame=start_frame,
            end_frame=end_frame,
            ts_col=ts_col,
        )
        row.update(_cue_row_to_trial_columns(cue_row))

        rows.append(row)

    return pd.DataFrame(rows)


def _cue_event_for_trial(
    cue_events: pd.DataFrame,
    df: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    ts_col: str = "global_ts",
) -> pd.Series | None:
    if cue_events is None or cue_events.empty:
        return None

    frame_matches = pd.Series(False, index=cue_events.index)
    if "cue_start_frame" in cue_events.columns:
        cue_start_frame = pd.to_numeric(cue_events["cue_start_frame"], errors="coerce")
        frame_matches |= cue_start_frame.between(start_frame, end_frame, inclusive="both")
    if "cue_end_frame" in cue_events.columns:
        cue_end_frame = pd.to_numeric(cue_events["cue_end_frame"], errors="coerce")
        cue_start_frame = pd.to_numeric(cue_events["cue_start_frame"], errors="coerce")
        frame_matches |= (
            cue_start_frame.notna()
            & cue_end_frame.notna()
            & (cue_start_frame <= end_frame)
            & (cue_end_frame >= start_frame)
        )
    if frame_matches.any():
        return cue_events.loc[frame_matches].iloc[0]

    if ts_col in df.columns:
        trial_start = pd.to_datetime(_safe_row_value(df, start_frame, ts_col), utc=True, errors="coerce")
        trial_end = pd.to_datetime(_safe_row_value(df, end_frame, ts_col), utc=True, errors="coerce")
        if pd.notna(trial_start) and pd.notna(trial_end):
            cue_start = pd.to_datetime(cue_events["cue_start_utc"], utc=True, errors="coerce")
            cue_end = pd.to_datetime(cue_events["cue_end_utc"], utc=True, errors="coerce")
            overlap = (
                (cue_start <= trial_end)
                & (
                    cue_end.ge(trial_start)
                    | (cue_end.isna() & cue_start.ge(trial_start))
                )
            )
            if overlap.any():
                return cue_events.loc[overlap].iloc[0]

    if len(cue_events) == 1:
        return cue_events.iloc[0]
    return None


def _cue_row_to_trial_columns(cue_row: pd.Series | None) -> dict:
    values = {col: np.nan for col in CUE_EVENT_COLUMNS}
    if cue_row is None:
        return values

    for col in CUE_EVENT_COLUMNS:
        if col in cue_row:
            values[col] = cue_row[col]
    return values


def _clean_optional_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return None
    return text


def _safe_row_value(df: pd.DataFrame, row_idx: int, col: str):
    if row_idx < 0 or row_idx >= len(df):
        return np.nan
    value = df.iloc[row_idx][col]
    if pd.isna(value):
        return np.nan
    return value


def _nearest_frame_position(ts: pd.Series, value):
    if value is None or pd.isna(value):
        return None
    target = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(target):
        return None

    deltas = (ts - target).abs()
    valid = deltas.notna()
    if not valid.any():
        return None
    valid_positions = np.flatnonzero(valid.to_numpy())
    closest_valid_position = int(np.argmin(deltas[valid].to_numpy()))
    return int(valid_positions[closest_valid_position])


#fills false frame gaps in ROI mask if surrounded by valid xy coordinates
def fill_short_gaps(mask: np.ndarray, gap_frames: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            i += 1
            continue
        j = i
        while j < n and not m[j]:
            j += 1
        left_true = (i - 1 >= 0 and m[i - 1])
        right_true = (j < n and m[j])
        if left_true and right_true and (j - i) <= gap_frames:
            m[i:j] = True
        i = j
    return m

#eliminates noise by removing short runs
def remove_short_runs(mask: np.ndarray, min_run_frames: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if (j - i) < min_run_frames:
            m[i:j] = False
        i = j
    return m

#sets parameters and calls functions for filling short gaps and runs
def debounce(mask: np.ndarray, gap_frames=10, min_run_frames=3) -> np.ndarray:
    m = fill_short_gaps(mask, gap_frames=gap_frames)
    m = remove_short_runs(m, min_run_frames=min_run_frames)
    return m

#
def segment_trials_gap_window(
    df,
    startbox_col=("in_startbox_L", "in_startbox_R"),
    arena_only_col="arena_only",
    fps=30.0,
    max_gap_s=3.0,
    dwell_frames=1,      # start with 1; increase to 3–6 after it works
    min_trial_s=0.5,
    require_opposite_side=False,
):

    """
    Segment behavioral trials using startbox → arena → startbox transitions.

    Logic:
        - Start: stable presence in startbox
        - Transition: enters arena within a time window
        - End: returns to startbox after being in arena

    Constraints:
        - max_gap_s: allowed delay between states
        - dwell_frames: minimum consecutive frames for stability
        - min_trial_s: minimum trial duration

    Returns:
        list[tuple[int, int]]:
            List of (start_frame, end_frame) for each trial
    """
    left_col, right_col = startbox_col
    
    sb_left = df[left_col].fillna(False).astype(bool).to_numpy()
    sb_right = df[right_col].fillna(False).astype(bool).to_numpy()
    ar = df[arena_only_col].fillna(False).astype(bool).to_numpy()

    sb = sb_left | sb_right

    max_gap = int(round(max_gap_s * fps))
    min_trial = int(round(min_trial_s * fps))

    n = len(df)
    trials = []

    def stable(mask, i):
        """True when the state lasts for at least dwell_frames."""
        if i + dwell_frames > n:
            return False
        return mask[i:i + dwell_frames].all()

    def startbox_side(i):
        """Return the stable start-box side at frame i."""
        left_stable = stable(sb_left, i)
        right_stable = stable(sb_right, i)

        if left_stable and not right_stable:
            return "L"
        if right_stable and not left_stable:
            return "R"

        # Ambiguous or not stably in either box
        return None

    i = 0

    while i < n:

        # Find stable occupancy in either start box
        start_side = None

        while i < n:
            start_side = startbox_side(i)
            if start_side is not None:
                break
            i += 1

        if i >= n:
            break

        # Leave whichever start box the mouse began in
        start_mask = sb_left if start_side == "L" else sb_right

        j = i
        while j < n and start_mask[j]:
            j += 1

        if j >= n:
            break

        # Find arena entry shortly after leaving the start box
        window_end = min(n, j + max_gap)
        arena_start = None

        for k in range(j, window_end):
            if stable(ar, k):
                arena_start = k
                break

        if arena_start is None:
            i = j
            continue

        # Find stable entry into either start box
        end_frame = None
        end_side = None

        m = arena_start + 1

        while m < n:
            candidate_side = startbox_side(m)

            if candidate_side is not None:
                # Optionally require the mouse to finish on the other side
                correct_side = (
                    not require_opposite_side
                    or candidate_side != start_side
                )

                # Confirm arena occupancy occurred recently
                recent_start = max(arena_start, m - max_gap)
                recently_in_arena = ar[recent_start:m].any()

                if correct_side and recently_in_arena:
                    end_frame = m
                    end_side = candidate_side
                    break

            m += 1

        if end_frame is None:
            break

        if end_frame - arena_start >= min_trial:
            trials.append({
                "start_frame": int(arena_start),
                "end_frame": int(end_frame),
                "start_side": start_side,
                "end_side": end_side,
            })

        i = end_frame + 1

    return trials
