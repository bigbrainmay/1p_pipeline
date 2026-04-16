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

#creates window for defining arena ROI
def collect_roi_opencv(video_path, roi_json_path, title="ROI (L-add, R-undo, s-save, q-quit)"):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame from video: {video_path}")

    points = []

    def redraw(img, pts):
        out = img.copy()
        for (x, y) in pts:
            cv2.circle(out, (x, y), 4, (255, 0, 0), -1)
        if len(pts) >= 2:
            p = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [p], False, (0, 255, 0), 2)
        cv2.putText(out, f"Points: {len(pts)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return out

    display = redraw(frame, points)

    def on_mouse(event, x, y, flags, param):
        nonlocal points, display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
            display = redraw(frame, points)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                display = redraw(frame, points)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, display)
    cv2.waitKey(1)

    cv2.setMouseCallback(title, on_mouse)

    while True:
        cv2.imshow(title, display)

        # Detect if user closed the window manually
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(title)
            cv2.waitKey(1)
            raise RuntimeError("Window closed without saving ROI.")

        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            if len(points) < 3:
                print("Need at least 3 points to save a polygon.")
                continue
            roi_json_path = Path(roi_json_path)
            roi_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(roi_json_path, "w") as f:
                json.dump(points, f)
            print("Saved ROI:", str(roi_json_path))

            cv2.destroyWindow(title)
            cv2.waitKey(1)
            break

        if key == ord('q') or key == 27:
            cv2.destroyWindow(title)
            cv2.waitKey(1)
            raise RuntimeError("Quit without saving ROI.")


    return frame, points

#generates filepath for storing ROIS
def roi_json_path(session_id, roi_name, root=None, folder_name="arena_rois"):
    """
    Generate and ensure a directory for ROI JSON files.

    Args:
        session_id (str): Session identifier
        roi_name (str): Name of ROI (e.g., "arena", "startbox_L")
        root (str | Path | None): Base directory (default = current working dir)
        folder_name (str): Subfolder for ROI files

    Returns:
        Path: Full path to ROI JSON file
    """
    root = Path(root or Path.cwd())
    roi_dir = root / folder_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    return roi_dir / f"{session_id}__{roi_name}.json"

#loads existing ROI, or prompts generation of new one
def load_or_collect_roi(video_path, roi_name, session_id, title=None, root=None, folder_name="arena_rois"):
    video_path = Path(video_path)
    json_path = roi_json_path(session_id, roi_name, root=root, folder_name=folder_name)

    if json_path.exists():
        with open(json_path, "r") as f:
            points = json.load(f)
        print(f"Loaded existing ROI: {json_path}")
        return points, json_path

    if title is None:
        title = f"{roi_name} ROI (L-add, R-undo, s-save, q-quit)"

    _, points = collect_roi_opencv(video_path, json_path, title=title)
    return points, json_path

#loads multiple ROIs for each session from dictionaryu of sessions
def collect_rois_for_sessions(
    sessions_df,
    video_col="beh_vid_path",
    roi_names=("arena", "startbox_L", "startbox_R"),
    root=None,
    folder_name="arena_rois",
):
    rois = {}

    for sessions, df in sessions_df.items():
        session_id = str(sessions)
        video_path = df[video_col].iloc[0]

        print(f"\n=== {session_id} ===")
        print("video:", video_path)

        rois[session_id] = {}

        for roi_name in roi_names:
            pts, json_path = load_or_collect_roi(
                video_path=video_path,
                roi_name=roi_name,
                session_id=session_id,
                title=f"{session_id} - {roi_name} ROI",
                root=root,
                folder_name=folder_name,
            )

            rois[session_id][roi_name] = {
                "points": pts,
                "json_path": json_path,
            }

    return rois

#loads or collects ROI multiple polygons for a single video
def load_or_collect_rois(video_path, roi_dir, roi_names):
    """
    roi_names e.g. ["arena", "startbox_L", "startbox_R", "goal"]
    Saves/loads each polygon as <roi_dir>/<name>.json
    Returns: rois dict {name: np.ndarray shape (N,2)}, plus frame0 size
    """
    video_path = Path(video_path)
    roi_dir = Path(roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # grab frame0 for shape
    cap = cv2.VideoCapture(str(video_path))
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    H, W = frame0.shape[:2]

    rois = {}
    for name in roi_names:
        roi_json = roi_dir / f"{name}.json"
        if roi_json.exists():
            with open(roi_json, "r") as f:
                pts = json.load(f)
            print(f"Loaded ROI {name}: {roi_json}")
        else:
            _, pts = collect_roi_opencv(video_path, roi_json, title=f"{name} ROI (L-add, R-undo, s-save, q-quit)")
        rois[name] = np.array(pts, dtype=np.float32)  # (N,2)

    return rois, (H, W)

#compute distance from point to polygon
def signed_dist_to_poly(poly_xy, x, y):
    contour = poly_xy.astype(np.float32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(x), float(y)), measureDist=True))

#computes ear_mid distance from polygon
def add_roi_features(df, rois, ref_x="ear_mid_x", ref_y="ear_mid_y"):
    out = df.copy()
    xs = out[ref_x].to_numpy()
    ys = out[ref_y].to_numpy()

    valid = np.isfinite(xs) & np.isfinite(ys)

    for name, poly in rois.items():
        inside = np.zeros(len(out), dtype=bool)
        sdist = np.full(len(out), np.nan, dtype=float)

        for i in np.where(valid)[0]:
            x, y = xs[i], ys[i]
            d = signed_dist_to_poly(poly, x, y)
            sdist[i] = d
            inside[i] = d >= 0  # inside if signed dist >= 0

        out[f"in_{name}"] = inside
        out[f"{name}_signed_dist_px"] = sdist
        out[f"{name}_dist_px"] = np.abs(sdist)

    return out

#converts polygon into binary image mask
def polygon_to_mask(poly_xy, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = poly_xy.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask

#builds masks for all ROIs
def build_roi_masks(rois, H, W):
    return {name: polygon_to_mask(poly, H, W) for name, poly in rois.items()}

#loads in polygons for each ROI from json 
def load_roi_polygons_for_session(rois_entry: dict) -> dict[str, np.ndarray]:
    """
    rois_entry is like: rois_by_session[session_id]
      { "arena": {"json_path": Path(...)}, "startbox_L": {...}, ... }

    Returns: { "arena": np.ndarray(N,2), ... }
    """
    polys = {}
    for name, meta in rois_entry.items():
        jp = Path(meta["json_path"]) if isinstance(meta, dict) else Path(meta)
        with open(jp, "r") as f:
            pts = json.load(f)
        polys[name] = np.asarray(pts, dtype=float)  # shape (N,2)
    return polys

#pulls first frame of video & dimensions
def get_video_hw(video_path: str | Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    H, W = frame.shape[:2]
    return H, W

#average x & y centerpoint of ROI
def polygon_centroid(poly_xy: np.ndarray) -> tuple[float, float]:
    # simple centroid; good for convex ROIs / hand-drawn arena polygons
    c = np.asarray(poly_xy, dtype=float)
    return float(c[:, 0].mean()), float(c[:, 1].mean())

#maximum radius from centerpoint
def polygon_max_radius(poly_xy: np.ndarray, cx: float, cy: float) -> float:
    c = np.asarray(poly_xy, dtype=float)
    r = np.sqrt((c[:, 0] - cx)**2 + (c[:, 1] - cy)**2)
    return float(np.nanmax(r))

#adds ROI features to each dataframe
def add_rois_to_all_sessions(
    aligned_sessions: dict[str, pd.DataFrame],
    rois_by_session: dict[str, dict],
    video_col: str = "beh_vid",
    ref_x: str = "ear_mid_x",
    ref_y: str = "ear_mid_y",
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:

    """
    Add ROI-derived features and masks to all session DataFrames.

    For each session:
        - Loads ROI polygons
        - Adds ROI inclusion + distance features
        - Computes arena center distance (raw + normalized)
        - Builds binary masks

    Returns:
        aligned_with_rois (dict[str, DataFrame]):
            Updated session DataFrames

        masks_by_session (dict[str, dict[str, np.ndarray]]):
            ROI masks per session
    """

    aligned_with_rois = {}
    masks_by_session = {}

    for session_id, df in aligned_sessions.items():
        if session_id not in rois_by_session:
            print(f"Skipping {session_id}: no ROIs collected")
            aligned_with_rois[session_id] = df
            continue

        polys = load_roi_polygons_for_session(rois_by_session[session_id])

        # add boolean + signed distance features (your existing function)
        df2 = add_roi_features(df, polys, ref_x=ref_x, ref_y=ref_y)

        # ---- NEW: arena center distance ----
        if "arena" in polys:
            cx, cy = polygon_centroid(polys["arena"])
            df2["arena_center_x"] = cx
            df2["arena_center_y"] = cy

            x = df2[ref_x].to_numpy(dtype=float, copy=False)
            y = df2[ref_y].to_numpy(dtype=float, copy=False)
            df2["dist_to_arena_center_px"] = np.sqrt((x - cx)**2 + (y - cy)**2)

            # optional: normalize by arena "radius" (max vertex distance from centroid)
            rmax = polygon_max_radius(polys["arena"], cx, cy)
            df2["dist_to_arena_center_norm"] = df2["dist_to_arena_center_px"] / rmax if rmax > 0 else np.nan
        else:
            df2["arena_center_x"] = np.nan
            df2["arena_center_y"] = np.nan
            df2["dist_to_arena_center_px"] = np.nan
            df2["dist_to_arena_center_norm"] = np.nan
        # ---- /NEW ----

        # build masks (needs H,W from video)
        video_path = Path(df[video_col].iloc[0])
        H, W = get_video_hw(video_path)
        masks = build_roi_masks(polys, H, W)

        aligned_with_rois[session_id] = df2
        masks_by_session[session_id] = masks

        print(f"{session_id}: added ROI features for {list(polys.keys())} (H,W=({H},{W}))")

    return aligned_with_rois, masks_by_session

