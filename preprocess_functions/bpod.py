from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_bpod_byte_events(
    path: str | Path,
    *,
    timestamp_col: str | int | None = None,
    code_col: str | int | None = None,
    off_code_offset: int = 10,
) -> pd.DataFrame:
    """
    Load a Bpod byte/event CSV into a normalized event table.

    Expected rows can be headerless, for example:
    ``1,1,1,False,True,False,2026-07-22T15:14:41.7699200-07:00,1``.

    Byte codes below ``off_code_offset`` are treated as port on events. Codes at
    or above that offset are treated as port off events for ``code - offset``.
    """
    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return _empty_events()

    ts_name = _resolve_column(raw, timestamp_col, _detect_timestamp_col(raw))
    code_name = _resolve_column(raw, code_col, raw.columns[-1])

    ts = pd.to_datetime(raw[ts_name], utc=True, errors="coerce")
    codes = pd.to_numeric(raw[code_name], errors="coerce")

    events = raw.copy()
    events.columns = [f"bpod_raw_{idx}" for idx in range(events.shape[1])]
    events["bpod_ts"] = ts
    events["bpod_code"] = codes.astype("Int64")
    events = events.dropna(subset=["bpod_ts", "bpod_code"]).reset_index(drop=True)
    if events.empty:
        return _empty_events()

    code_values = events["bpod_code"].astype(int)
    is_off = code_values >= off_code_offset
    ports = np.where(is_off, code_values - off_code_offset, code_values)

    events["bpod_event_idx"] = np.arange(len(events), dtype=int)
    events["bpod_port"] = ports.astype(int)
    events["bpod_state"] = np.where(is_off, "off", "on")
    events["bpod_is_on"] = ~is_off
    events["bpod_is_off"] = is_off
    return events.sort_values("bpod_ts").reset_index(drop=True)


def add_bpod_events_to_aligned(
    aligned: pd.DataFrame,
    bpod_path: str | Path,
    *,
    fps: float,
    timestamp_col: str | int | None = None,
    code_col: str | int | None = None,
    off_code_offset: int = 10,
) -> pd.DataFrame:
    events = load_bpod_byte_events(
        bpod_path,
        timestamp_col=timestamp_col,
        code_col=code_col,
        off_code_offset=off_code_offset,
    )
    out = aligned.copy()

    if events.empty:
        out["bpod_event_dropped"] = True
        out["bpod_ts"] = pd.NaT
        out["bpod_code"] = pd.NA
        out["bpod_port"] = pd.NA
        out["bpod_state"] = pd.NA
        out["bpod_any_port_active"] = False
        return out

    out["global_ts"] = pd.to_datetime(out["global_ts"], utc=True, errors="coerce")
    event_cols = [
        "bpod_ts",
        "bpod_event_idx",
        "bpod_code",
        "bpod_port",
        "bpod_state",
        "bpod_is_on",
        "bpod_is_off",
    ]
    mapped = pd.merge_asof(
        out[["global_idx", "global_ts"]].sort_values("global_ts"),
        events[event_cols].sort_values("bpod_ts"),
        left_on="global_ts",
        right_on="bpod_ts",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=0.5 / fps),
    )
    mapped["bpod_event_dropped"] = mapped["bpod_event_idx"].isna()

    merge_cols = ["global_idx", *event_cols, "bpod_event_dropped"]
    out = out.merge(mapped[merge_cols], on="global_idx", how="left")

    active_cols = []
    for port in sorted(events["bpod_port"].dropna().astype(int).unique()):
        col = f"bpod_port_{port}_active"
        active_cols.append(col)
        out[col] = False

    for interval in bpod_port_intervals(events).itertuples(index=False):
        col = f"bpod_port_{int(interval.bpod_port)}_active"
        if col not in out.columns:
            continue
        in_interval = (
            (out["global_ts"] >= interval.bpod_on_ts)
            & (out["global_ts"] < interval.bpod_off_ts)
        )
        out.loc[in_interval, col] = True

    out["bpod_any_port_active"] = out[active_cols].any(axis=1) if active_cols else False
    return out


def bpod_port_intervals(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_by_port: dict[int, list[pd.Series]] = {}

    for _, event in events.sort_values("bpod_ts").iterrows():
        port = int(event["bpod_port"])
        if bool(event["bpod_is_on"]):
            open_by_port.setdefault(port, []).append(event)
            continue

        queue = open_by_port.setdefault(port, [])
        if not queue:
            rows.append(
                {
                    "bpod_port": port,
                    "bpod_on_ts": pd.NaT,
                    "bpod_off_ts": event["bpod_ts"],
                    "bpod_on_event_idx": pd.NA,
                    "bpod_off_event_idx": event["bpod_event_idx"],
                    "bpod_duration_s": np.nan,
                    "bpod_interval_status": "off_without_on",
                }
            )
            continue

        on_event = queue.pop(0)
        duration_s = (event["bpod_ts"] - on_event["bpod_ts"]).total_seconds()
        rows.append(
            {
                "bpod_port": port,
                "bpod_on_ts": on_event["bpod_ts"],
                "bpod_off_ts": event["bpod_ts"],
                "bpod_on_event_idx": on_event["bpod_event_idx"],
                "bpod_off_event_idx": event["bpod_event_idx"],
                "bpod_duration_s": duration_s,
                "bpod_interval_status": "paired",
            }
        )

    for port, queue in open_by_port.items():
        for on_event in queue:
            rows.append(
                {
                    "bpod_port": port,
                    "bpod_on_ts": on_event["bpod_ts"],
                    "bpod_off_ts": pd.NaT,
                    "bpod_on_event_idx": on_event["bpod_event_idx"],
                    "bpod_off_event_idx": pd.NA,
                    "bpod_duration_s": np.nan,
                    "bpod_interval_status": "on_without_off",
                }
            )

    return pd.DataFrame(rows)


def _detect_timestamp_col(df: pd.DataFrame) -> Any:
    scores = {}
    for col in df.columns:
        text = df[col].astype(str)
        iso_like = int(text.str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).sum())
        parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
        scores[col] = (iso_like, int(parsed.notna().sum()))
    best_col = max(scores, key=scores.get)
    if scores[best_col] == (0, 0):
        raise ValueError("Could not detect a timestamp column in Bpod byte CSV")
    return best_col


def _resolve_column(df: pd.DataFrame, requested: str | int | None, detected: Any) -> Any:
    if requested is None:
        return detected
    if requested in df.columns:
        return requested
    if isinstance(requested, str) and requested.isdigit():
        idx = int(requested)
        if idx in df.columns:
            return idx
    raise KeyError(f"Column {requested!r} not found in Bpod CSV")


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bpod_ts",
            "bpod_code",
            "bpod_event_idx",
            "bpod_port",
            "bpod_state",
            "bpod_is_on",
            "bpod_is_off",
        ]
    )
