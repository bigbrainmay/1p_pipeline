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
    startbox_col="in_startbox_L",
    arena_only_col="arena_only",
    fps=30.0,
    max_gap_s=3.0,
    dwell_frames=1,      # start with 1; increase to 3–6 after it works
    min_trial_s=0.5,
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
       
    sb = df[startbox_col].astype(bool).to_numpy()
    ar = df[arena_only_col].astype(bool).to_numpy()
    gap = (~sb) & (~ar)

    max_gap = int(round(max_gap_s * fps))
    min_trial = int(round(min_trial_s * fps))

    def stable(mask, i):
        return mask[i:i+dwell_frames].sum() >= dwell_frames

    trials = []
    i = 0
    n = len(df)

    while i < n:
        # find a startbox moment
        while i < n and not stable(sb, i):
            i += 1
        if i >= n: break

        # leave startbox
        j = i
        while j < n and sb[j]:
            j += 1
        if j >= n: break

        # find arena entry within window after leaving startbox
        win_end = min(n, j + max_gap)
        k = j
        s = None
        while k < win_end:
            if stable(ar, k):
                s = k
                break
            k += 1
        if s is None:
            i = j
            continue

        # find startbox re-entry within window after being in arena at least once
        m = s + 1
        e = None
        while m < n:
            if stable(sb, m):
                # require we saw arena within the last max_gap frames
                lo = max(0, m - max_gap)
                if ar[lo:m].any():
                    e = m
                    break
            m += 1
        if e is None:
            break

        if (e - s) >= min_trial:
            trials.append((int(s), int(e)))

        i = e + 1

    return trials
