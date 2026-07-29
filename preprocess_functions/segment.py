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
