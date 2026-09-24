from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


NULL_STRINGS = {"", "nan", "none", "null", "na", "n/a"}
GENERIC_SHEET_NAMES = {"sheet", "sheet1", "paths"}
LAB_RELATIVE_ROOTS = {"data", "users"}

PATH_COLUMNS = (
    "beh_csv",
    "neu_csv",
    "sleap_csv",
    "bpod_ts",
    "video_port_events_csv",
    "beh_vid",
    "neu_vid",
    "neu_h5",
    "cell_csv",
    "extract_output_mat",
    "extract_labels_mat",
    "matlab_output_dir",
)

EXPERIMENT_METADATA_COLUMNS = (
    "manipulation",
    "cue_condition",
    "cue_rotation_deg",
    "visual_cues_present",
)

COLUMN_ALIASES = {
    "beh_csv_path": "beh_csv",
    "behavior_csv": "beh_csv",
    "behavior_csv_path": "beh_csv",
    "neu_csv_path": "neu_csv",
    "neural_csv": "neu_csv",
    "neural_csv_path": "neu_csv",
    "miniscope_csv": "neu_csv",
    "miniscope_csv_path": "neu_csv",
    "sleap_csv_path": "sleap_csv",
    "bpod_csv": "bpod_ts",
    "bpod_csv_path": "bpod_ts",
    "bpod_ts_path": "bpod_ts",
    "bpod_bytes": "bpod_ts",
    "bpod_bytes_path": "bpod_ts",
    "bpod_events": "bpod_ts",
    "bpod_events_path": "bpod_ts",
    "port_events_csv": "video_port_events_csv",
    "port_events_path": "video_port_events_csv",
    "video_port_events": "video_port_events_csv",
    "video_port_events_path": "video_port_events_csv",
    "beh_vid_path": "beh_vid",
    "behavior_video": "beh_vid",
    "behavior_video_path": "beh_vid",
    "neu_vid_path": "neu_vid",
    "neural_video": "neu_vid",
    "neural_video_path": "neu_vid",
    "miniscope_vid": "neu_vid",
    "miniscope_video": "neu_vid",
    "miniscope_video_path": "neu_vid",
    "cell_csv_path": "cell_csv",
    "neuron_csv": "cell_csv",
    "neuron_csv_path": "cell_csv",
    "neu_h5_path": "neu_h5",
    "miniscope_h5": "neu_h5",
    "miniscope_h5_path": "neu_h5",
    "cue_rotation": "cue_rotation_deg",
    "rotation_deg": "cue_rotation_deg",
    "visual_cue_condition": "cue_condition",
    "visual_cues": "visual_cues_present",
    "cues_present": "visual_cues_present",
}


def clean_column_name(name: Any) -> str:
    text = str(name).replace("\ufeff", "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def safe_id(value: Any, fallback: str = "unknown") -> str:
    if _is_missing(value):
        return fallback
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return text or fallback


def resolve_lab_path(value: Any, lab_drive: str | Path | None = None) -> Path | None:
    """
    Resolve handmade spreadsheet paths.

    The lab sheets often store Windows-looking paths such as ``\\Data\\May\\...``.
    When ``lab_drive`` or ``LAB_DRIVE_PATH`` is set, those become paths under that
    root, with backslashes converted to the local platform separator.
    """
    if _is_missing(value):
        return None

    text = str(value).strip().strip('"').strip("'")
    if text.lower() in NULL_STRINGS:
        return None

    text = os.path.expandvars(os.path.expanduser(text))
    lab_drive_value = lab_drive or os.environ.get("LAB_DRIVE_PATH")
    lab_drive_path = Path(lab_drive_value) if lab_drive_value else None

    drive_match = re.match(r"^[A-Za-z]:[\\/]*(.*)$", text)
    if drive_match:
        if lab_drive_path:
            return lab_drive_path.joinpath(*_split_path_parts(drive_match.group(1)))
        return Path(text)

    if text.startswith("\\\\") and not re.match(r"^\\\\Data[\\/]", text, re.I):
        return Path(text)

    path = Path(text)
    if path.is_absolute() and not text.startswith("\\"):
        return path

    parts = _split_path_parts(text.lstrip("\\/"))
    if (
        lab_drive_path
        and parts
        and (text.startswith(("\\", "/")) or parts[0].lower() in LAB_RELATIVE_ROOTS)
    ):
        return lab_drive_path.joinpath(*parts)
    return Path(*parts) if parts else None


@dataclass(frozen=True)
class SessionRecord:
    row_index: int
    sheet_name: str | None
    mouse_id: str | None
    session_id: str
    recording_id: str
    date: pd.Timestamp | None
    beh_csv: Path | None = None
    neu_csv: Path | None = None
    sleap_csv: Path | None = None
    beh_vid: Path | None = None
    neu_vid: Path | None = None
    neu_h5: Path | None = None
    cell_csv: Path | None = None
    extract_output_mat: Path | None = None
    extract_labels_mat: Path | None = None
    matlab_output_dir: Path | None = None
    trial_type: str | None = None
    cue_ts: str | None = None
    bpod_ts: Path | None = None
    video_port_events_csv: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def bpod_csv(self) -> Path | None:
        return self.bpod_ts

    @classmethod
    def from_row(
        cls,
        row: pd.Series,
        *,
        lab_drive: str | Path | None = None,
    ) -> "SessionRecord":
        extras = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "row_index",
                "sheet_name",
                "mouse_id",
                "session_id",
                "recording_id",
                "date",
                "trial_type",
                "cue_ts",
                *PATH_COLUMNS,
            }
        }

        date = row.get("date")
        if _is_missing(date):
            date = None
        else:
            date = pd.to_datetime(date, errors="coerce")
            date = None if pd.isna(date) else date

        paths = {
            col: resolve_lab_path(row.get(col), lab_drive=lab_drive)
            for col in PATH_COLUMNS
        }

        return cls(
            row_index=int(row.get("row_index", 0)),
            sheet_name=_clean_optional_text(row.get("sheet_name")),
            mouse_id=_clean_optional_text(row.get("mouse_id")),
            session_id=str(row["session_id"]),
            recording_id=str(row["recording_id"]),
            date=date,
            trial_type=_clean_optional_text(row.get("trial_type")),
            cue_ts=_clean_optional_text(row.get("cue_ts")),
            extras=extras,
            **paths,
        )


def load_manifest(
    manifest_path: str | Path,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    path = Path(manifest_path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(
            path,
            sheet_name=None if sheet_name is None else sheet_name,
            engine="openpyxl",
        )
        if isinstance(sheets, pd.DataFrame):
            sheets = {sheet_name or path.stem: sheets}
    else:
        sheets = {sheet_name: pd.read_csv(path)}

    frames = []
    for current_sheet, df in sheets.items():
        normalized = normalize_manifest_table(df, sheet_name=current_sheet)
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    manifest = pd.concat(frames, ignore_index=True)
    return add_manifest_ids(manifest)


def load_session_records(
    manifest_path: str | Path,
    *,
    sheet_name: str | None = None,
    lab_drive: str | Path | None = None,
) -> list[SessionRecord]:
    manifest = load_manifest(manifest_path, sheet_name=sheet_name)
    return [
        SessionRecord.from_row(row, lab_drive=lab_drive)
        for _, row in manifest.iterrows()
    ]


def experiment_metadata_from_record(record: SessionRecord) -> dict[str, Any]:
    """Return optional experimental-analysis metadata carried by a manifest row."""
    metadata = {}
    for col in EXPERIMENT_METADATA_COLUMNS:
        value = record.extras.get(col)
        if _is_missing(value):
            value = None
        metadata[col] = value
    return metadata


def normalize_manifest_table(
    df: pd.DataFrame,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    out = df.copy().dropna(how="all")
    if out.empty:
        return out

    out.columns = [clean_column_name(col) for col in out.columns]
    out = _coalesce_alias_columns(out)
    out = out.dropna(how="all").reset_index(drop=True)
    if "row_index" not in out.columns:
        out.insert(0, "row_index", out.index)
    if "sheet_name" not in out.columns:
        out["sheet_name"] = sheet_name
    elif sheet_name is not None:
        out["sheet_name"] = out["sheet_name"].fillna(sheet_name)

    if "mouse_id" not in out.columns:
        mouse_id = None
        if sheet_name and sheet_name.lower() not in GENERIC_SHEET_NAMES:
            mouse_id = sheet_name
        out["mouse_id"] = mouse_id

    for path_col in PATH_COLUMNS:
        if path_col not in out.columns:
            out[path_col] = None

    return out


def add_manifest_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        out["date"] = pd.NaT

    if "session_id" not in out.columns:
        out["session_id"] = [
            _make_base_session_id(row) for _, row in out.iterrows()
        ]
    else:
        out["session_id"] = [
            safe_id(value, fallback=_make_base_session_id(row))
            for _, (value, row) in enumerate(zip(out["session_id"], out.to_dict("records")))
        ]

    if "recording_id" not in out.columns:
        out["recording_id"] = _make_unique_recording_ids(out)
    else:
        out["recording_id"] = [
            safe_id(value, fallback=session_id)
            for value, session_id in zip(out["recording_id"], out["session_id"])
        ]

    return out


def write_manifest_with_outputs(
    manifest: pd.DataFrame,
    output_path: str | Path,
    output_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    updates = pd.DataFrame(output_rows)
    merged = manifest.copy()
    if not updates.empty:
        merged = merged.merge(
            updates,
            on="recording_id",
            how="left",
            suffixes=("", "_new"),
        )
        for col in updates.columns:
            new_col = f"{col}_new"
            if col == "recording_id" or new_col not in merged.columns:
                continue
            if col not in merged.columns:
                merged[col] = merged[new_col]
            else:
                merged[col] = merged[new_col].combine_first(merged[col])
            merged = merged.drop(columns=[new_col])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def _coalesce_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for alias, canonical in COLUMN_ALIASES.items():
        if alias not in out.columns:
            continue
        if canonical not in out.columns:
            out = out.rename(columns={alias: canonical})
        else:
            out[canonical] = out[canonical].combine_first(out[alias])
            out = out.drop(columns=[alias])
    return out


def _make_base_session_id(row: pd.Series | dict[str, Any]) -> str:
    mouse_id = safe_id(row.get("mouse_id"), fallback="")
    date = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.notna(date):
        date_part = date.strftime("%Y%m%d")
    else:
        date_part = f"row{int(row.get('row_index', 0)):03d}"
    return f"{mouse_id}_{date_part}" if mouse_id else date_part


def _make_unique_recording_ids(df: pd.DataFrame) -> list[str]:
    counts = df["session_id"].value_counts()
    used: dict[str, int] = {}
    ids = []

    for _, row in df.iterrows():
        base = str(row["session_id"])
        if counts[base] == 1:
            candidate = base
        else:
            suffix = safe_id(row.get("trial_type"), fallback="")
            if not suffix:
                video_path = row.get("neu_vid") or row.get("beh_vid")
                suffix = safe_id(_path_stem_text(video_path), fallback="")
            if not suffix:
                suffix = f"row{int(row.get('row_index', 0)):03d}"
            candidate = f"{base}_{suffix}"

        duplicate_number = used.get(candidate, 0)
        used[candidate] = duplicate_number + 1
        ids.append(candidate if duplicate_number == 0 else f"{candidate}_{duplicate_number + 1:02d}")

    return ids


def _split_path_parts(text: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", text) if part]


def _path_stem_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    parts = _split_path_parts(str(value))
    return Path(parts[-1]).stem if parts else ""


def _clean_optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return None if text.lower() in NULL_STRINGS else text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return True
    return False
