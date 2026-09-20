from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def convert_avi_to_h5(
    video_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str = "data",
    dtype: str = "float32",
    grayscale: bool = True,
    overwrite: bool = False,
    compression: str | None = None,
    gap_leading_corrupt: bool = True,
    manual_gap_leading_frames: int = 0,
    gap_fill_strategy: str = "nearest",
    gap_fill_value: float = 0.0,
    pad_to_frame_count_hint: bool = True,
    corruption_probe_frames: int = 600,
    clean_run_frames: int = 30,
    stripe_score_threshold: float = 0.75,
    banding_score_threshold: float = 1.8,
    corruption_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Convert a miniscope AVI into an HDF5 stack.

    The output dataset shape is ``frames x height x width`` when ``grayscale`` is
    true. This matches the shape expected by many MATLAB calcium-imaging tools.
    """
    import cv2
    import h5py

    if gap_fill_strategy not in {"nearest", "constant"}:
        raise ValueError("gap_fill_strategy must be 'nearest' or 'constant'")

    video_path = Path(video_path)
    output_path = Path(output_path)
    if corruption_report_path is None:
        corruption_report_path = default_corruption_report_path(output_path)
    else:
        corruption_report_path = Path(corruption_report_path)

    if output_path.exists() and not overwrite:
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "exists",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    corruption_report = None
    detected_leading_bad_frames = 0
    if gap_leading_corrupt:
        corruption_report = check_leading_video_corruption(
            video_path,
            probe_frames=corruption_probe_frames,
            clean_run_frames=clean_run_frames,
            stripe_score_threshold=stripe_score_threshold,
            banding_score_threshold=banding_score_threshold,
        )
        detected_leading_bad_frames = int(corruption_report["leading_bad_frames"])
        if not corruption_report["clean_run_found"]:
            write_json_report(corruption_report_path, corruption_report)
            raise ValueError(
                "No stable clean frame run found near the start of "
                f"{video_path}. Increase --corruption-probe-frames or inspect manually."
            )
        write_json_report(corruption_report_path, corruption_report)

    total_leading_gapped = max(
        int(manual_gap_leading_frames),
        detected_leading_bad_frames,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    metadata_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        _seek_to_frame(cap, total_leading_gapped, cv2)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Could not read first valid frame {total_leading_gapped} from {video_path}"
            )

        first_frame = _prepare_frame(first_frame, cv2=cv2, grayscale=grayscale, dtype=dtype)
        height, width = first_frame.shape

        initial_frames = max(frame_count_hint, total_leading_gapped + 1, 1)
        shape = (initial_frames, height, width)
        maxshape = (None, height, width)
        chunks = (1, height, width)

        source_frame_idx = total_leading_gapped
        valid_frame_mask = np.zeros(initial_frames, dtype=bool)
        last_valid_frame = first_frame
        with h5py.File(output_path, "w") as h5f:
            dset = h5f.create_dataset(
                dataset_name,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
                fillvalue=gap_fill_value,
            )

            if total_leading_gapped and gap_fill_strategy == "nearest":
                _write_gap_frames(dset, 0, total_leading_gapped, first_frame)

            dset[source_frame_idx] = first_frame
            valid_frame_mask[source_frame_idx] = True
            source_frame_idx += 1

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = _prepare_frame(frame, cv2=cv2, grayscale=grayscale, dtype=dtype)
                if frame.shape != (height, width):
                    raise ValueError(
                        f"Frame shape changed at source frame {source_frame_idx}: "
                        f"expected {(height, width)}, got {frame.shape}"
                    )

                if source_frame_idx >= dset.shape[0]:
                    new_size = max(dset.shape[0] * 2, source_frame_idx + 1)
                    dset.resize((new_size, height, width))
                    valid_frame_mask = _resize_bool_mask(valid_frame_mask, new_size)

                dset[source_frame_idx] = frame
                valid_frame_mask[source_frame_idx] = True
                last_valid_frame = frame
                source_frame_idx += 1

            output_frame_count = dset.shape[0] if pad_to_frame_count_hint and frame_count_hint > 0 else source_frame_idx
            if output_frame_count != dset.shape[0]:
                dset.resize((output_frame_count, height, width))
                valid_frame_mask = valid_frame_mask[:output_frame_count]

            if (
                gap_fill_strategy == "nearest"
                and source_frame_idx < output_frame_count
                and last_valid_frame is not None
            ):
                _write_gap_frames(dset, source_frame_idx, output_frame_count, last_valid_frame)

            h5f.create_dataset(
                "valid_frame_mask",
                data=valid_frame_mask.astype(np.uint8, copy=False),
                compression=compression,
            )
            h5f.attrs["source_video"] = str(video_path)
            h5f.attrs["fps"] = fps
            h5f.attrs["frame_count"] = output_frame_count
            h5f.attrs["valid_frame_count"] = int(valid_frame_mask.sum())
            h5f.attrs["source_frame_count_hint"] = frame_count_hint
            h5f.attrs["source_metadata_height"] = metadata_height
            h5f.attrs["source_metadata_width"] = metadata_width
            h5f.attrs["height"] = height
            h5f.attrs["width"] = width
            h5f.attrs["dataset_name"] = dataset_name
            h5f.attrs["gapped_leading_frames"] = total_leading_gapped
            h5f.attrs["detected_leading_corrupt_frames"] = detected_leading_bad_frames
            h5f.attrs["manual_gap_leading_frames"] = int(manual_gap_leading_frames)
            h5f.attrs["gap_fill_strategy"] = gap_fill_strategy
            h5f.attrs["gap_fill_value"] = float(gap_fill_value)
            h5f.attrs["corruption_check_enabled"] = bool(gap_leading_corrupt)
            h5f.attrs["corruption_report_path"] = str(corruption_report_path)
    finally:
        cap.release()

    output_frame_count = int(output_frame_count)
    valid_frame_count = int(valid_frame_mask.sum())
    return {
        "video_path": str(video_path),
        "output_path": str(output_path),
        "status": "converted",
        "frame_count": output_frame_count,
        "valid_frame_count": valid_frame_count,
        "source_frame_count_hint": frame_count_hint,
        "fps": fps,
        "height": height,
        "width": width,
        "gapped_leading_frames": total_leading_gapped,
        "detected_leading_corrupt_frames": detected_leading_bad_frames,
        "manual_gap_leading_frames": int(manual_gap_leading_frames),
        "gap_fill_strategy": gap_fill_strategy,
        "gap_fill_value": float(gap_fill_value),
        "corruption_report_path": str(corruption_report_path),
        "corruption_status": (
            corruption_report["status"] if corruption_report is not None else "not_checked"
        ),
    }


def check_leading_video_corruption(
    video_path: str | Path,
    *,
    probe_frames: int = 600,
    clean_run_frames: int = 30,
    stripe_score_threshold: float = 0.75,
    banding_score_threshold: float = 1.8,
    min_dynamic_std: float = 2.0,
    max_saturation_fraction: float = 0.98,
) -> dict[str, Any]:
    """
    Probe the start of a video and identify leading frames to gap.

    This is intentionally conservative: only the leading span before the first
    stable clean run is converted into gap placeholders. Later suspicious frames
    are reported, but not altered by the AVI-to-H5 converter.
    """
    import cv2

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    metadata = {
        "frame_count_hint": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    metadata_warnings = []
    if metadata["frame_count_hint"] <= 0:
        metadata_warnings.append("non_positive_frame_count_hint")
    if metadata["fps"] <= 0 or not np.isfinite(metadata["fps"]):
        metadata_warnings.append("non_positive_fps")
    if metadata["width"] <= 0 or metadata["height"] <= 0:
        metadata_warnings.append("non_positive_dimensions")

    metrics = []
    expected_shape = None
    try:
        probe_limit = int(probe_frames)
        if metadata["frame_count_hint"] > 0:
            probe_limit = min(probe_limit, metadata["frame_count_hint"])

        for frame_idx in range(probe_limit):
            ok, frame = _read_frame_at(cap, frame_idx, cv2)
            if not ok:
                metrics.append(
                    {
                        "frame_idx": frame_idx,
                        "decoded": False,
                        "bad": True,
                        "reasons": ["decode_failed"],
                    }
                )
                continue

            gray = _to_gray(frame, cv2).astype(np.float32, copy=False)
            if expected_shape is None:
                expected_shape = gray.shape

            frame_metrics = _frame_quality_metrics(
                gray,
                frame_idx=frame_idx,
                expected_shape=expected_shape,
                min_dynamic_std=min_dynamic_std,
                max_saturation_fraction=max_saturation_fraction,
            )
            metrics.append(frame_metrics)
    finally:
        cap.release()

    stripe_threshold = _robust_artifact_threshold(
        [m.get("stripe_score") for m in metrics],
        floor=stripe_score_threshold,
    )
    banding_threshold = _robust_artifact_threshold(
        [m.get("banding_score") for m in metrics],
        floor=banding_score_threshold,
    )

    for metric in metrics:
        reasons = list(metric.get("reasons", []))
        if metric.get("decoded"):
            if metric.get("stripe_score", 0.0) >= stripe_threshold:
                reasons.append("line_striation")
            if metric.get("banding_score", 0.0) >= banding_threshold:
                reasons.append("banding_striation")
        metric["reasons"] = sorted(set(reasons))
        metric["bad"] = bool(metric["reasons"])

    bad = [bool(metric["bad"]) for metric in metrics]
    clean_run_start = _find_first_clean_run(bad, int(clean_run_frames))
    clean_run_found = clean_run_start is not None
    leading_bad_frames = int(clean_run_start) if clean_run_found else 0

    suspicious_later_frames = [
        int(metric["frame_idx"])
        for metric in metrics[leading_bad_frames:]
        if metric.get("bad")
    ]

    if not clean_run_found:
        status = "no_clean_run_found"
    elif leading_bad_frames > 0:
        status = "leading_corruption_found"
    elif metadata_warnings:
        status = "metadata_warnings_only"
    else:
        status = "clean"

    return {
        "video_path": str(video_path),
        "status": status,
        "metadata": metadata,
        "metadata_warnings": metadata_warnings,
        "probe_frames": int(probe_frames),
        "probed_frames": len(metrics),
        "clean_run_frames": int(clean_run_frames),
        "clean_run_found": clean_run_found,
        "first_clean_frame": int(clean_run_start) if clean_run_found else None,
        "leading_bad_frames": leading_bad_frames,
        "stripe_score_threshold": float(stripe_threshold),
        "banding_score_threshold": float(banding_threshold),
        "suspicious_later_frames": suspicious_later_frames,
        "n_suspicious_later_frames": len(suspicious_later_frames),
        "frame_metrics": metrics,
    }


def default_corruption_report_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_corruption_report.json")


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)


def _read_frame_at(cap, frame_idx: int, cv2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    return cap.read()


def _seek_to_frame(cap, frame_idx: int, cv2) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))


def _resize_bool_mask(mask: np.ndarray, new_size: int) -> np.ndarray:
    out = np.zeros(new_size, dtype=bool)
    out[:len(mask)] = mask
    return out


def _write_gap_frames(dset, start: int, stop: int, frame: np.ndarray) -> None:
    for frame_idx in range(int(start), int(stop)):
        dset[frame_idx] = frame


def _prepare_frame(frame, *, cv2, grayscale: bool, dtype: str) -> np.ndarray:
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3:
        raise ValueError("Color H5 export is not implemented for this pipeline")
    return frame.astype(dtype, copy=False)


def _to_gray(frame, cv2) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _frame_quality_metrics(
    gray: np.ndarray,
    *,
    frame_idx: int,
    expected_shape: tuple[int, int],
    min_dynamic_std: float,
    max_saturation_fraction: float,
) -> dict[str, Any]:
    gray = np.asarray(gray, dtype=np.float32)
    reasons = []
    if gray.shape != expected_shape:
        reasons.append("shape_mismatch")

    finite = np.isfinite(gray)
    if not finite.all():
        reasons.append("non_finite_pixels")

    mean = float(np.nanmean(gray))
    std = float(np.nanstd(gray))
    if std < min_dynamic_std:
        reasons.append("low_dynamic_range")

    saturation_fraction = float(np.mean((gray <= 1) | (gray >= 254)))
    if saturation_fraction >= max_saturation_fraction:
        reasons.append("high_saturation")

    row_profile = np.nanmean(gray, axis=1)
    col_profile = np.nanmean(gray, axis=0)
    eps = 1e-6
    row_jump = _profile_jump_score(row_profile) / (std + eps)
    col_jump = _profile_jump_score(col_profile) / (std + eps)
    row_banding = float(np.nanstd(row_profile) / (std + eps))
    col_banding = float(np.nanstd(col_profile) / (std + eps))

    return {
        "frame_idx": int(frame_idx),
        "decoded": True,
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
        "mean": mean,
        "std": std,
        "saturation_fraction": saturation_fraction,
        "stripe_score": float(max(row_jump, col_jump)),
        "row_stripe_score": float(row_jump),
        "col_stripe_score": float(col_jump),
        "banding_score": float(max(row_banding, col_banding)),
        "row_banding_score": row_banding,
        "col_banding_score": col_banding,
        "bad": bool(reasons),
        "reasons": reasons,
    }


def _profile_jump_score(profile: np.ndarray) -> float:
    if len(profile) < 2:
        return 0.0
    jumps = np.abs(np.diff(profile))
    return float(np.nanpercentile(jumps, 95))


def _robust_artifact_threshold(values, *, floor: float) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 10:
        return float(floor)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    robust_sigma = 1.4826 * mad
    return float(max(floor, median + 8.0 * robust_sigma))


def _find_first_clean_run(bad: list[bool], clean_run_frames: int) -> int | None:
    if clean_run_frames <= 1:
        clean_run_frames = 1
    if len(bad) < clean_run_frames:
        return None

    for start in range(0, len(bad) - clean_run_frames + 1):
        if not any(bad[start:start + clean_run_frames]):
            return start
    return None
