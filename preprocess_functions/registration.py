from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .boundary_tuning import cell_col_from_idx
from .neural import _extract_matlab_field, load_extract_labels


class CaimanNotInstalledError(ImportError):
    """Raised when CaImAn is needed for registration but is unavailable."""


@dataclass(frozen=True)
class RegistrationInput:
    recording_id: str
    session_id: str
    mouse_id: str
    traces_mat_path: Path
    labels_mat_path: Path
    A: Any
    dims: tuple[int, int]
    template: np.ndarray | None
    component_indices: np.ndarray
    cell_cols: list[str]


def caiman_available() -> bool:
    try:
        from caiman.base.rois import register_multisession  # noqa: F401
    except ImportError:
        return False
    return True


def scipy_available() -> bool:
    try:
        import scipy.sparse  # noqa: F401
    except ImportError:
        return False
    return True


def load_registration_input(
    *,
    traces_mat_path: str | Path,
    labels_mat_path: str | Path,
    recording_id: str,
    session_id: str,
    mouse_id: str | None,
    accepted_label: int = 1,
) -> RegistrationInput:
    labels = load_extract_labels(labels_mat_path)
    component_indices = np.where(labels == accepted_label)[0]
    if len(component_indices) == 0:
        raise ValueError(f"{recording_id} has no accepted cells with label {accepted_label}")

    spatial_weights = load_extract_spatial_weights(traces_mat_path)
    template = load_extract_summary_image(traces_mat_path)
    A_all, dims = spatial_weights_to_caiman_A(
        spatial_weights,
        labels=labels,
        template_shape=None if template is None else template.shape,
    )

    if A_all.shape[1] != labels.shape[0]:
        raise ValueError(
            f"{recording_id} has {A_all.shape[1]} spatial components but "
            f"{labels.shape[0]} ActSort labels"
        )

    template = match_template_dims(template, dims)
    return RegistrationInput(
        recording_id=recording_id,
        session_id=session_id,
        mouse_id=mouse_id or "unknown_mouse",
        traces_mat_path=Path(traces_mat_path),
        labels_mat_path=Path(labels_mat_path),
        A=A_all[:, component_indices].tocsc(),
        dims=dims,
        template=template,
        component_indices=component_indices.astype(int),
        cell_cols=[cell_col_from_idx(idx) for idx in component_indices],
    )


def register_accepted_cells_across_sessions(
    sessions: list[RegistrationInput],
    *,
    output_csv: str | Path | None = None,
    align_flag: bool = True,
    max_thr: float = 0,
    use_opt_flow: bool = True,
    thresh_cost: float = 0.7,
    max_dist: float = 10,
    enclosed_thr: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(sessions) < 2:
        raise ValueError("Cell registration needs at least two sessions")

    dims = sessions[0].dims
    mismatched = [session.recording_id for session in sessions if session.dims != dims]
    if mismatched:
        raise ValueError(
            f"All sessions for one registration batch must have the same FOV dims. "
            f"Expected {dims}; mismatched recordings: {mismatched}"
        )

    if align_flag and any(session.template is None for session in sessions):
        raise ValueError(
            "Template alignment was requested but at least one session is missing "
            "output.info.summary_image. Rerun with --no-template-align to register "
            "without template alignment."
        )

    register_multisession = _load_caiman_register_multisession()
    templates = [session.template for session in sessions] if align_flag else [None] * len(sessions)
    spatial_union, assignments, matchings = register_multisession(
        A=[session.A for session in sessions],
        dims=dims,
        templates=templates,
        align_flag=align_flag,
        max_thr=max_thr,
        use_opt_flow=use_opt_flow,
        thresh_cost=thresh_cost,
        max_dist=max_dist,
        enclosed_thr=enclosed_thr,
    )

    registration_df = assignments_to_long_dataframe(assignments, sessions)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        registration_df.to_csv(output_csv, index=False)

    summary = {
        "n_sessions": len(sessions),
        "n_registered_cells": int(assignments.shape[0]),
        "n_links": int(len(registration_df)),
        "dims": f"{dims[0]}x{dims[1]}",
        "spatial_union_shape": str(spatial_union.shape),
        "n_matchings": len(matchings),
    }
    return registration_df, summary


def assignments_to_long_dataframe(
    assignments: np.ndarray,
    sessions: list[RegistrationInput],
) -> pd.DataFrame:
    rows = []
    for registered_idx, assignment_row in enumerate(np.asarray(assignments)):
        registered_cell_id = f"registered_cell_{registered_idx}"
        for session_order, local_value in enumerate(assignment_row):
            if not np.isfinite(local_value):
                continue

            local_accepted_idx = int(local_value)
            session = sessions[session_order]
            if local_accepted_idx >= len(session.component_indices):
                raise IndexError(
                    f"Registration assignment {local_accepted_idx} is outside "
                    f"{session.recording_id}'s accepted-cell list"
                )

            rows.append(
                {
                    "mouse_id": session.mouse_id,
                    "registered_cell_id": registered_cell_id,
                    "registered_cell_idx": registered_idx,
                    "session_order": session_order,
                    "recording_id": session.recording_id,
                    "session_id": session.session_id,
                    "local_accepted_idx": local_accepted_idx,
                    "component_idx": int(session.component_indices[local_accepted_idx]),
                    "cell_col": session.cell_cols[local_accepted_idx],
                    "traces_mat_path": str(session.traces_mat_path),
                    "labels_mat_path": str(session.labels_mat_path),
                }
            )
    return pd.DataFrame(rows)


def load_extract_spatial_weights(traces_mat_path: str | Path) -> Any:
    traces_mat_path = Path(traces_mat_path)
    h5_value = _load_h5_field(
        traces_mat_path,
        field_paths=[
            "precomputedOutput/spatial_weights",
            "extractOutput/spatial_weights",
            "output/spatial_weights",
        ],
        suffix="spatial_weights",
        required=False,
    )
    if h5_value is not None:
        return h5_value

    from scipy.io import loadmat

    mat = loadmat(traces_mat_path, struct_as_record=False, squeeze_me=True)
    for key in ("precomputedOutput", "extractOutput", "output"):
        if key not in mat:
            continue
        try:
            return np.asarray(_extract_matlab_field(mat[key], "spatial_weights"))
        except KeyError:
            continue

    raise KeyError(f"Could not find spatial_weights in {traces_mat_path}")


def load_extract_summary_image(traces_mat_path: str | Path) -> np.ndarray | None:
    traces_mat_path = Path(traces_mat_path)
    h5_value = _load_h5_field(
        traces_mat_path,
        field_paths=[
            "precomputedOutput/info/summary_image",
            "extractOutput/info/summary_image",
            "output/info/summary_image",
        ],
        suffix="summary_image",
        required=False,
    )
    if h5_value is not None:
        return np.asarray(h5_value).squeeze()

    try:
        from scipy.io import loadmat

        mat = loadmat(traces_mat_path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        return None

    for key in ("precomputedOutput", "extractOutput", "output"):
        if key not in mat:
            continue
        try:
            info = _extract_matlab_field(mat[key], "info")
            return np.asarray(_extract_matlab_field(info, "summary_image")).squeeze()
        except KeyError:
            continue
    return None


def spatial_weights_to_caiman_A(
    spatial_weights: Any,
    *,
    labels: np.ndarray | None = None,
    template_shape: tuple[int, ...] | None = None,
) -> tuple[Any, tuple[int, int]]:
    sparse = _load_scipy_sparse()
    label_count = None if labels is None else int(np.asarray(labels).size)
    template_dims = _template_dims(template_shape)

    if sparse.issparse(spatial_weights):
        A = spatial_weights.tocsc()
        if template_dims is None:
            template_dims = _infer_square_dims(A.shape[0])
        A = _orient_component_matrix(A, label_count, template_dims)
        return A, template_dims

    arr = np.asarray(spatial_weights).squeeze()
    if arr.ndim == 3:
        masks = _orient_spatial_stack(arr, label_count, template_dims)
        dims = (int(masks.shape[0]), int(masks.shape[1]))
        A = masks.reshape((dims[0] * dims[1], masks.shape[2]), order="F")
        return sparse.csc_matrix(A), dims

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D spatial weights, got shape {arr.shape}")

    A = sparse.csc_matrix(arr)
    if template_dims is None:
        pixel_axis = _pixel_axis_from_labels(A.shape, label_count)
        template_dims = _infer_square_dims(A.shape[pixel_axis])
    A = _orient_component_matrix(A, label_count, template_dims)
    return A, template_dims


def match_template_dims(
    template: np.ndarray | None,
    dims: tuple[int, int],
) -> np.ndarray | None:
    if template is None:
        return None
    arr = np.asarray(template).squeeze()
    if arr.shape == dims:
        return arr
    if arr.shape == (dims[1], dims[0]):
        return arr.T
    raise ValueError(f"Template shape {arr.shape} does not match spatial dims {dims}")


def _load_caiman_register_multisession():
    try:
        from caiman.base.rois import register_multisession
    except ImportError as exc:
        raise CaimanNotInstalledError(
            "CaImAn is not installed in this Python environment. "
            "Run this registration stage from a CaImAn environment, or skip it "
            "and continue database formation from manifest_with_cells.csv."
        ) from exc
    return register_multisession


def _load_h5_field(
    mat_path: Path,
    *,
    field_paths: list[str],
    suffix: str,
    required: bool,
) -> np.ndarray | sparse.csc_matrix | None:
    import h5py

    try:
        with h5py.File(mat_path, "r") as h5f:
            for field_path in field_paths:
                if field_path in h5f:
                    return _h5_node_to_array_or_sparse(h5f[field_path])

            matches = []

            def collect_field(name: str, obj: Any) -> None:
                if name.endswith(suffix):
                    matches.append(obj)

            h5f.visititems(collect_field)
            if len(matches) == 1:
                return _h5_node_to_array_or_sparse(matches[0])
    except OSError:
        return None

    if required:
        raise KeyError(f"Could not find {suffix} in {mat_path}")
    return None


def _h5_node_to_array_or_sparse(node: Any) -> Any:
    if hasattr(node, "shape") and not hasattr(node, "keys"):
        return np.asarray(node).squeeze()

    keys = set(node.keys())
    if {"data", "ir", "jc"}.issubset(keys):
        sparse = _load_scipy_sparse()
        data = np.asarray(node["data"]).squeeze()
        ir = np.asarray(node["ir"]).squeeze().astype(np.int64)
        jc = np.asarray(node["jc"]).squeeze().astype(np.int64)
        n_rows = _matlab_sparse_n_rows(node)
        n_cols = len(jc) - 1
        return sparse.csc_matrix((data, ir, jc), shape=(n_rows, n_cols))

    raise TypeError(f"Unsupported H5 node for MATLAB field: {node.name}")


def _matlab_sparse_n_rows(node: Any) -> int:
    value = node.attrs.get("MATLAB_sparse")
    if value is None:
        raise ValueError(f"Sparse MATLAB node {node.name} is missing MATLAB_sparse")
    arr = np.asarray(value).squeeze()
    return int(arr.item() if arr.shape == () else arr[0])


def _orient_spatial_stack(
    arr: np.ndarray,
    label_count: int | None,
    template_dims: tuple[int, int] | None,
) -> np.ndarray:
    component_axis = _component_axis_from_labels(arr.shape, label_count)
    masks = np.moveaxis(arr, component_axis, -1)

    if template_dims is None:
        return masks
    if masks.shape[:2] == template_dims:
        return masks
    if masks.shape[:2] == (template_dims[1], template_dims[0]):
        return np.transpose(masks, (1, 0, 2))

    raise ValueError(
        f"Spatial stack shape {arr.shape} does not match template dims {template_dims}"
    )


def _component_axis_from_labels(
    shape: tuple[int, ...],
    label_count: int | None,
) -> int:
    if label_count is None:
        return len(shape) - 1

    matches = [axis for axis, size in enumerate(shape) if size == label_count]
    if not matches:
        raise ValueError(
            f"Could not find a component axis matching {label_count} labels "
            f"in spatial weight shape {shape}"
        )
    return matches[-1]


def _orient_component_matrix(
    A: Any,
    label_count: int | None,
    dims: tuple[int, int],
) -> Any:
    pixels = dims[0] * dims[1]
    if A.shape[0] == pixels and (label_count is None or A.shape[1] == label_count):
        return A.tocsc()
    if A.shape[1] == pixels and (label_count is None or A.shape[0] == label_count):
        return A.T.tocsc()
    raise ValueError(
        f"Could not orient spatial component matrix shape {A.shape} for "
        f"dims {dims} and label_count {label_count}"
    )


def _pixel_axis_from_labels(
    shape: tuple[int, int],
    label_count: int | None,
) -> int:
    if label_count is not None:
        if shape[0] == label_count:
            return 1
        if shape[1] == label_count:
            return 0
    return 0


def _template_dims(template_shape: tuple[int, ...] | None) -> tuple[int, int] | None:
    if template_shape is None:
        return None
    shape = tuple(int(size) for size in template_shape if int(size) > 1)
    if len(shape) < 2:
        return None
    return shape[:2]


def _infer_square_dims(n_pixels: int) -> tuple[int, int]:
    side = int(round(np.sqrt(n_pixels)))
    if side * side != n_pixels:
        raise ValueError(
            f"Cannot infer FOV dims from {n_pixels} pixels. Provide/load a summary_image."
        )
    return side, side


def _load_scipy_sparse():
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "scipy is required for cross-session cell registration. "
            "Run this stage from the main analysis environment or a CaImAn "
            "environment that includes scipy."
        ) from exc
    return sparse
