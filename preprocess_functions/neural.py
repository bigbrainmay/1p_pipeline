from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .boundary_tuning import cell_col_from_idx


def run_matlab_extraction(
    *,
    recording_id: str,
    neu_h5_path: str | Path,
    output_dir: str | Path,
    matlab_script: str | Path,
    matlab_bin: str = "matlab",
    session_id: str | None = None,
    extra_vars: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess | list[str]:
    """
    Run a MATLAB extraction script with standard workspace variables.

    The MATLAB script can read ``recording_id``, ``session_id``, ``neu_h5_path``,
    and ``output_dir`` from its workspace.
    """
    neu_h5_path = Path(neu_h5_path)
    output_dir = Path(output_dir)
    matlab_script = Path(matlab_script)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    assignments = {
        "recording_id": recording_id,
        "session_id": session_id or recording_id,
        "neu_h5_path": str(neu_h5_path),
        "output_dir": str(output_dir),
    }
    assignments.update(extra_vars or {})

    matlab_parts = [
        f"{name} = {_matlab_string(value)};"
        for name, value in assignments.items()
    ]
    matlab_parts.append(f"run({_matlab_string(str(matlab_script))});")
    command = " ".join(matlab_parts)
    args = [matlab_bin, "-batch", command]

    if dry_run:
        return args

    return subprocess.run(args, check=True)


def find_extract_outputs(
    output_dir: str | Path,
    *,
    recording_id: str | None = None,
    session_id: str | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    prefixes = [value for value in (recording_id, session_id) if value]

    for prefix in prefixes:
        traces = output_dir / f"{prefix}_precomputed_output.mat"
        labels = output_dir / f"{prefix}_precomputed_output_LABELS.mat"
        if traces.exists() and labels.exists():
            return traces, labels

    trace_matches = [
        path
        for path in output_dir.glob("*precomputed_output.mat")
        if "LABELS" not in path.name.upper()
    ]
    label_matches = list(output_dir.glob("*precomputed_output_LABELS.mat"))

    if len(trace_matches) == 1 and len(label_matches) == 1:
        return trace_matches[0], label_matches[0]

    raise FileNotFoundError(
        "Could not uniquely find EXTRACT outputs in "
        f"{output_dir}. Found traces={trace_matches}, labels={label_matches}"
    )


def import_extract_curated_cells(
    *,
    traces_mat_path: str | Path,
    labels_mat_path: str | Path,
    output_csv: str | Path,
    curated_csv: str | Path | None = None,
    recording_id: str | None = None,
    session_id: str | None = None,
    cell_label: int = 1,
    trace_kind: str = "raw",
    fps: float = 30.0,
    tau_seconds: float = 1.0,
) -> dict[str, Any]:
    traces = load_extract_traces(traces_mat_path)
    labels = load_extract_labels(labels_mat_path)

    if traces.shape[0] == labels.shape[0]:
        traces = traces.T

    if traces.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Mismatch: traces shape {traces.shape}, labels shape {labels.shape}"
        )

    cell_indices = np.where(labels == cell_label)[0]
    raw_df = pd.DataFrame(
        traces[:, cell_indices],
        columns=[cell_col_from_idx(idx) for idx in cell_indices],
    )

    failures = pd.DataFrame()
    if trace_kind == "raw":
        cell_df = raw_df
    elif trace_kind == "deconvolved":
        cell_df, failures = deconvolve_cell_traces(
            raw_df,
            fps=fps,
            tau_seconds=tau_seconds,
        )
    else:
        raise ValueError("trace_kind must be 'raw' or 'deconvolved'")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(output_csv, index=False)

    curated = pd.DataFrame(
        {
            "recording_id": recording_id,
            "session_id": session_id,
            "component_idx": cell_indices,
            "cell_col": [cell_col_from_idx(idx) for idx in cell_indices],
            "label": labels[cell_indices],
            "trace_kind": trace_kind,
            "included_in_cell_csv": [
                cell_col_from_idx(idx) in cell_df.columns for idx in cell_indices
            ],
        }
    )

    if curated_csv is not None:
        curated_csv = Path(curated_csv)
        curated_csv.parent.mkdir(parents=True, exist_ok=True)
        curated.to_csv(curated_csv, index=False)

    if not failures.empty:
        failures_path = output_csv.with_name(f"{output_csv.stem}_failures.csv")
        failures.to_csv(failures_path, index=False)

    return {
        "cell_csv": str(output_csv),
        "curated_csv": str(curated_csv) if curated_csv is not None else None,
        "n_components": int(labels.shape[0]),
        "n_cells": int(len(cell_indices)),
        "n_traces_written": int(len(cell_df.columns)),
        "trace_kind": trace_kind,
        "failures": int(len(failures)),
    }


def load_extract_traces(traces_mat_path: str | Path) -> np.ndarray:
    import h5py

    traces_mat_path = Path(traces_mat_path)
    with h5py.File(traces_mat_path, "r") as h5f:
        if "precomputedOutput" in h5f and "traces" in h5f["precomputedOutput"]:
            return np.asarray(h5f["precomputedOutput"]["traces"])

        matches: list[np.ndarray] = []

        def collect_trace_dataset(name: str, obj: Any) -> None:
            if name.endswith("traces") and hasattr(obj, "shape"):
                matches.append(np.asarray(obj))

        h5f.visititems(collect_trace_dataset)

    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Could not find a unique traces dataset in {traces_mat_path}")


def load_extract_labels(labels_mat_path: str | Path) -> np.ndarray:
    from scipy.io import loadmat

    labels_mat_path = Path(labels_mat_path)
    mat = loadmat(labels_mat_path, struct_as_record=False, squeeze_me=False)

    if "labels_overall" in mat:
        return np.asarray(mat["labels_overall"]).squeeze()

    if "labels" not in mat:
        raise KeyError(f"No 'labels' variable found in {labels_mat_path}")

    labels_obj = mat["labels"]
    labels_overall = _extract_matlab_field(labels_obj, "labels_overall")
    return np.asarray(labels_overall).squeeze()


def deconvolve_cell_traces(
    cell_trace_df: pd.DataFrame,
    *,
    fps: float = 30.0,
    tau_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from oasis.functions import deconvolve

    g = np.exp(-1 / (fps * tau_seconds))
    deconvolved: dict[str, np.ndarray] = {}
    failures = []

    for col in cell_trace_df.columns:
        try:
            trace = cell_trace_df[col].to_numpy(dtype=float).squeeze()
            if trace.ndim != 1:
                raise ValueError(f"{col} is not 1D: {trace.shape}")

            trace = (
                pd.Series(trace)
                .interpolate(limit_direction="both")
                .to_numpy(dtype=float)
            )
            if not np.isfinite(trace).all():
                raise ValueError("non-finite values remain after interpolation")
            if np.std(trace) == 0:
                raise ValueError("flat trace")

            y = trace - np.percentile(trace, 10)
            y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
            dy = np.diff(y)
            sn = np.median(np.abs(dy - np.median(dy))) / 0.6745 / np.sqrt(2)
            if not np.isfinite(sn) or sn <= 0:
                raise ValueError(f"bad noise estimate sn={sn}")

            _, spikes, _, _, _ = deconvolve(
                y,
                g=(g,),
                sn=sn,
                penalty=1,
                optimize_g=0,
            )
            deconvolved[col] = spikes
        except Exception as exc:  # keep batch import moving and report failures
            failures.append({"cell_col": col, "error": repr(exc)})

    return pd.DataFrame(deconvolved, index=cell_trace_df.index), pd.DataFrame(failures)


def _extract_matlab_field(value: Any, field_name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.names and field_name in value.dtype.names:
            return _extract_matlab_field(value[field_name], field_name)
        if value.size == 1:
            return _extract_matlab_field(value.item(), field_name)

    if hasattr(value, field_name):
        return getattr(value, field_name)

    if isinstance(value, dict) and field_name in value:
        return value[field_name]

    raise KeyError(f"Could not extract MATLAB field {field_name!r}")


def _matlab_string(value: Any) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"
