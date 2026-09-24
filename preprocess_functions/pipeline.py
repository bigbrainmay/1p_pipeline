from __future__ import annotations

from pathlib import Path

from .manifest import SessionRecord, safe_id


def recording_output_dir(output_root: str | Path, record: SessionRecord) -> Path:
    return Path(output_root) / safe_id(record.recording_id)


def neural_output_dir(output_root: str | Path, record: SessionRecord) -> Path:
    return recording_output_dir(output_root, record) / "neural"


def matlab_output_dir(output_root: str | Path, record: SessionRecord) -> Path:
    return recording_output_dir(output_root, record) / "matlab"


def behavior_output_dir(output_root: str | Path, record: SessionRecord) -> Path:
    return recording_output_dir(output_root, record) / "behavior"


def coregistration_output_dir(output_root: str | Path, mouse_id: str) -> Path:
    return Path(output_root) / "coregistration" / safe_id(mouse_id)


def default_neu_h5_path(output_root: str | Path, record: SessionRecord) -> Path:
    return neural_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_miniscope.h5"


def default_cell_csv_path(
    output_root: str | Path,
    record: SessionRecord,
    *,
    trace_kind: str = "raw",
) -> Path:
    return neural_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_{trace_kind}_cell_traces.csv"


def default_curated_neurons_path(output_root: str | Path, record: SessionRecord) -> Path:
    return neural_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_curated_neurons.csv"


def default_aligned_session_path(output_root: str | Path, record: SessionRecord) -> Path:
    return (
        Path(output_root)
        / "aligned_sessions"
        / f"{safe_id(record.recording_id)}_session.csv"
    )


def default_trials_path(output_root: str | Path, record: SessionRecord) -> Path:
    return behavior_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_trials.csv"


def default_cue_events_path(output_root: str | Path, record: SessionRecord) -> Path:
    return behavior_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_cue_events.csv"


def default_bpod_events_path(output_root: str | Path, record: SessionRecord) -> Path:
    return behavior_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_bpod_events.csv"


def default_bpod_intervals_path(output_root: str | Path, record: SessionRecord) -> Path:
    return behavior_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_bpod_intervals.csv"


def default_video_port_events_path(output_root: str | Path, record: SessionRecord) -> Path:
    return behavior_output_dir(output_root, record) / f"{safe_id(record.recording_id)}_video_port_events.csv"


def default_cell_registration_path(output_root: str | Path, mouse_id: str) -> Path:
    return (
        coregistration_output_dir(output_root, mouse_id)
        / f"{safe_id(mouse_id)}_cell_registration.csv"
    )


def cue_timestamp_overrides_path(output_root: str | Path) -> Path:
    return Path(output_root) / "cue_ts_overrides.csv"
