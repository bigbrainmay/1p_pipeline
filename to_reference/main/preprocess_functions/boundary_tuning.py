import re

import numpy as np
import pandas as pd


CELL_COL_RE = re.compile(r"^cell_(\d+)$")


def cell_idx_from_col(cell_col: str) -> int:
    match = CELL_COL_RE.fullmatch(str(cell_col))
    if not match:
        raise ValueError(f"Expected a cell column like 'cell_17', got {cell_col!r}")
    return int(match.group(1))


def cell_col_from_idx(cell_idx: int | str) -> str:
    if isinstance(cell_idx, str) and CELL_COL_RE.fullmatch(cell_idx):
        return cell_idx
    return f"cell_{int(cell_idx)}"


def get_cell_columns(df: pd.DataFrame) -> list[str]:
    cell_cols = [col for col in df.columns if CELL_COL_RE.fullmatch(str(col))]
    return sorted(cell_cols, key=cell_idx_from_col)


def _arena_mask_array(arena_mask, mask_name: str = "arena") -> np.ndarray:
    if isinstance(arena_mask, dict):
        if mask_name not in arena_mask:
            raise KeyError(
                f"Mask dictionary is missing {mask_name!r}. "
                f"Found masks: {list(arena_mask.keys())}"
            )
        arena_mask = arena_mask[mask_name]

    mask = np.asarray(arena_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"arena_mask must be 2D, got shape {mask.shape}")
    return mask


def _numeric_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Missing column {col!r}. Found: {df.columns.tolist()}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _boolean_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Missing column {col!r}. Found: {df.columns.tolist()}")

    values = df[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy() != 0

    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "t"]).to_numpy(dtype=bool)


def _points_inside_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inside = np.zeros(len(x), dtype=bool)

    finite = np.isfinite(x) & np.isfinite(y)
    finite_idx = np.where(finite)[0]
    if finite_idx.size == 0:
        return inside

    xi = np.rint(x[finite_idx]).astype(int)
    yi = np.rint(y[finite_idx]).astype(int)
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    checked_idx = finite_idx[in_bounds]
    inside[checked_idx] = mask[yi[in_bounds], xi[in_bounds]]
    return inside


def compute_head_directed_boundary_distance(
    df: pd.DataFrame,
    arena_mask,
    x_col: str = "ear_mid_x",
    y_col: str = "ear_mid_y",
    nose_x_col: str = "nose.x",
    nose_y_col: str = "nose.y",
    step_px: float = 1.0,
    max_ray_px: float | None = None,
    mask_name: str = "arena",
) -> np.ndarray:
    """
    Measure distance from ear midpoint to the arena boundary along the
    ear-midpoint-to-nose direction.

    Returns one distance per dataframe row. Invalid rows are NaN.
    """
    if step_px <= 0:
        raise ValueError("step_px must be > 0")

    mask = _arena_mask_array(arena_mask, mask_name=mask_name)
    h, w = mask.shape

    x = _numeric_array(df, x_col)
    y = _numeric_array(df, y_col)
    nose_x = _numeric_array(df, nose_x_col)
    nose_y = _numeric_array(df, nose_y_col)

    dx = nose_x - x
    dy = nose_y - y
    norm = np.hypot(dx, dy)

    inside = _points_inside_mask(mask, x, y)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(nose_x)
        & np.isfinite(nose_y)
        & np.isfinite(norm)
        & (norm > 0)
        & inside
    )

    distances = np.full(len(df), np.nan, dtype=float)
    active = np.where(valid)[0]
    if active.size == 0:
        return distances

    ux = np.zeros(len(df), dtype=float)
    uy = np.zeros(len(df), dtype=float)
    ux[valid] = dx[valid] / norm[valid]
    uy[valid] = dy[valid] / norm[valid]

    if max_ray_px is None:
        max_ray_px = float(np.hypot(h, w))

    for step in np.arange(step_px, max_ray_px + step_px, step_px):
        if active.size == 0:
            break

        xi = np.rint(x[active] + ux[active] * step).astype(int)
        yi = np.rint(y[active] + uy[active] * step).astype(int)
        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

        still_inside = np.zeros(active.size, dtype=bool)
        still_inside[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]

        exited = ~still_inside
        if exited.any():
            distances[active[exited]] = step

        active = active[still_inside]

    return distances


def assign_boundary_proximity_bins(
    head_directed_boundary_dist_px: np.ndarray,
    valid_mask: np.ndarray | None = None,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign equal-occupancy bins ordered center -> boundary.

    Larger raw head-directed boundary distances are assigned to low bin numbers.
    Smaller raw distances, closer to the boundary, are assigned to high bin
    numbers.
    """
    distances = np.asarray(head_directed_boundary_dist_px, dtype=float)
    if valid_mask is None:
        valid_mask = np.ones(len(distances), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_idx = np.where(valid_mask & np.isfinite(distances))[0]
    if valid_idx.size < n_bins:
        raise ValueError(
            f"Need at least {n_bins} valid frames for {n_bins} bins, "
            f"got {valid_idx.size}"
        )

    ordered_idx = valid_idx[np.argsort(distances[valid_idx])[::-1]]
    chunks = np.array_split(ordered_idx, n_bins)

    bin_idx = np.full(len(distances), -1, dtype=int)
    bin_median_distance_px = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i, chunk in enumerate(chunks):
        bin_idx[chunk] = i
        bin_counts[i] = len(chunk)
        bin_median_distance_px[i] = np.nanmedian(distances[chunk])

    return bin_idx, bin_median_distance_px, bin_counts


def add_boundary_tuning_columns(
    df: pd.DataFrame,
    arena_mask,
    n_bins: int = 20,
    distance_col: str = "head_directed_boundary_dist_px",
    bin_col: str = "boundary_proximity_bin",
    cell_dropped_col: str = "cell_dropped",
    **distance_kwargs,
) -> pd.DataFrame:
    out = df.copy()
    distances = compute_head_directed_boundary_distance(
        out,
        arena_mask=arena_mask,
        **distance_kwargs,
    )

    valid = np.isfinite(distances)
    if cell_dropped_col in out.columns:
        valid &= ~_boolean_array(out, cell_dropped_col)

    bin_idx, _, _ = assign_boundary_proximity_bins(
        distances,
        valid_mask=valid,
        n_bins=n_bins,
    )

    out[distance_col] = distances
    out[bin_col] = bin_idx
    return out


def _mean_by_bin(values: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    tuning = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        mask = (bin_idx == i) & np.isfinite(values)
        if mask.any():
            tuning[i] = np.nanmean(values[mask])
    return tuning


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)

    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    valid_p = p[valid]
    m = len(valid_p)
    if m == 0:
        return q

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    ranked_q = ranked_p * m / ranks
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q_valid = np.empty_like(valid_p)
    q_valid[order] = ranked_q
    q[valid] = q_valid
    return q


def _resolve_cell_cols(df: pd.DataFrame, cell_cols: list[str] | None) -> list[str]:
    if cell_cols is None:
        cell_cols = get_cell_columns(df)
    missing = [col for col in cell_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing cell columns: {missing}")
    if not cell_cols:
        raise ValueError("No cell_* columns found")
    return cell_cols


def compute_boundary_tuning_results(
    df: pd.DataFrame,
    arena_mask,
    cell_cols: list[str] | None = None,
    n_bins: int = 20,
    n_shuffles: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 0,
    distance_col: str = "head_directed_boundary_dist_px",
    bin_col: str = "boundary_proximity_bin",
    cell_dropped_col: str = "cell_dropped",
    step_px: float = 1.0,
    **distance_kwargs,
) -> pd.DataFrame:
    """
    Compute positive cell firing tuning to boundary proximity.

    Positive Spearman rho means mean raw cell activity increases from center
    bins toward boundary bins. Significance is one-sided against circular
    time-shift shuffles and FDR-corrected across cells.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")

    cell_cols = _resolve_cell_cols(df, cell_cols)
    rng = np.random.default_rng(random_state)

    distances = compute_head_directed_boundary_distance(
        df,
        arena_mask=arena_mask,
        step_px=step_px,
        **distance_kwargs,
    )

    behavior_valid = np.isfinite(distances)
    if cell_dropped_col in df.columns:
        behavior_valid &= ~_boolean_array(df, cell_dropped_col)

    bin_idx, bin_median_distance_px, bin_counts = assign_boundary_proximity_bins(
        distances,
        valid_mask=behavior_valid,
        n_bins=n_bins,
    )

    behavior_idx = np.where(bin_idx >= 0)[0]
    rows = []

    for cell_col in cell_cols:
        cell_values = _numeric_array(df, cell_col)
        observed_tuning = _mean_by_bin(cell_values, bin_idx, n_bins)

        observed_valid_idx = behavior_idx[np.isfinite(cell_values[behavior_idx])]
        observed_rho = _spearman_r(
            bin_idx[observed_valid_idx],
            cell_values[observed_valid_idx],
        )

        shuffle_rhos = np.full(n_shuffles, np.nan, dtype=float)
        shuffle_tunings = np.full((n_shuffles, n_bins), np.nan, dtype=float)

        for shuffle_idx in range(n_shuffles):
            shift = int(rng.integers(1, len(df))) if len(df) > 1 else 0
            shifted_values = np.roll(cell_values, shift)
            shifted_valid_idx = behavior_idx[np.isfinite(shifted_values[behavior_idx])]

            shuffle_rhos[shuffle_idx] = _spearman_r(
                bin_idx[shifted_valid_idx],
                shifted_values[shifted_valid_idx],
            )
            shuffle_tunings[shuffle_idx] = _mean_by_bin(
                shifted_values,
                bin_idx,
                n_bins,
            )

        valid_shuffle_rhos = shuffle_rhos[np.isfinite(shuffle_rhos)]
        if np.isfinite(observed_rho) and valid_shuffle_rhos.size:
            p_value = (
                1 + np.sum(valid_shuffle_rhos >= observed_rho)
            ) / (valid_shuffle_rhos.size + 1)
        else:
            p_value = np.nan

        rows.append(
            {
                "cell_idx": cell_idx_from_col(cell_col),
                "cell_col": cell_col,
                "rho": observed_rho,
                "p": p_value,
                "n_valid_frames": int(observed_valid_idx.size),
                "n_bins": int(n_bins),
                "alpha": float(alpha),
                "distance_col": distance_col,
                "bin_col": bin_col,
                "bin_counts": bin_counts.copy(),
                "bin_median_distance_px": bin_median_distance_px.copy(),
                "tuning": observed_tuning,
                "shuffle_rhos": shuffle_rhos,
                "shuffle_tuning_mean": np.nanmean(shuffle_tunings, axis=0),
                "shuffle_tuning_low": np.nanpercentile(shuffle_tunings, 2.5, axis=0),
                "shuffle_tuning_high": np.nanpercentile(shuffle_tunings, 97.5, axis=0),
            }
        )

    results = pd.DataFrame(rows)
    results["q"] = _fdr_bh(results["p"].to_numpy())
    results["significant"] = (results["rho"] > 0) & (results["q"] < alpha)
    results.attrs[distance_col] = distances
    results.attrs[bin_col] = bin_idx
    return results


def get_significant_boundary_cell_indices(results: pd.DataFrame) -> np.ndarray:
    return results.loc[results["significant"], "cell_idx"].to_numpy(dtype=int)


def get_significant_boundary_cell_columns(results: pd.DataFrame) -> list[str]:
    return results.loc[results["significant"], "cell_col"].tolist()


def _result_row(
    results: pd.DataFrame,
    cell_idx: int | str | None = None,
    cell_col: str | None = None,
):
    if cell_col is None:
        if cell_idx is None:
            raise ValueError("Provide cell_idx or cell_col")
        cell_col = cell_col_from_idx(cell_idx)

    matches = results.loc[results["cell_col"] == cell_col]
    if matches.empty:
        raise ValueError(f"No result found for {cell_col!r}")
    return matches.iloc[0]


def plot_boundary_tuning_cell(
    results: pd.DataFrame,
    cell_idx: int | str | None = None,
    cell_col: str | None = None,
    ax=None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    row = _result_row(results, cell_idx=cell_idx, cell_col=cell_col)
    tuning = np.asarray(row["tuning"], dtype=float)
    x = np.arange(len(tuning))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x, tuning, marker="o")
    ax.set_xlabel("boundary proximity bin (center -> boundary)")
    ax.set_ylabel("mean raw cell trace")
    ax.set_title(
        f"{row['cell_col']} tuning "
        f"(rho={row['rho']:.3f}, p={row['p']:.4f}, q={row['q']:.4f})"
    )

    if show:
        plt.show()
    return ax


def plot_boundary_tuning_null_rho(
    results: pd.DataFrame,
    cell_idx: int | str | None = None,
    cell_col: str | None = None,
    ax=None,
    bins: int = 40,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    row = _result_row(results, cell_idx=cell_idx, cell_col=cell_col)
    shuffle_rhos = np.asarray(row["shuffle_rhos"], dtype=float)
    observed_rho = float(row["rho"])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.hist(shuffle_rhos[np.isfinite(shuffle_rhos)], bins=bins, alpha=0.75)
    ax.axvline(observed_rho, color="red", linewidth=2, label="observed rho")
    ax.set_xlabel("shuffled Spearman rho")
    ax.set_ylabel("count")
    ax.set_title(
        f"{row['cell_col']} circular-shift null "
        f"(p={row['p']:.4f}, q={row['q']:.4f})"
    )
    ax.legend()

    if show:
        plt.show()
    return ax


def plot_boundary_tuning_shuffle_band(
    results: pd.DataFrame,
    cell_idx: int | str | None = None,
    cell_col: str | None = None,
    ax=None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    row = _result_row(results, cell_idx=cell_idx, cell_col=cell_col)
    tuning = np.asarray(row["tuning"], dtype=float)
    shuffle_mean = np.asarray(row["shuffle_tuning_mean"], dtype=float)
    shuffle_low = np.asarray(row["shuffle_tuning_low"], dtype=float)
    shuffle_high = np.asarray(row["shuffle_tuning_high"], dtype=float)
    x = np.arange(len(tuning))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(
        x,
        shuffle_low,
        shuffle_high,
        color="gray",
        alpha=0.25,
        label="shuffle 95% band",
    )
    ax.plot(x, shuffle_mean, color="gray", linewidth=2, label="shuffle mean")
    ax.plot(x, tuning, marker="o", color="tab:blue", label="observed")
    ax.set_xlabel("boundary proximity bin (center -> boundary)")
    ax.set_ylabel("mean raw cell trace")
    ax.set_title(f"{row['cell_col']} observed tuning vs shuffle band")
    ax.legend()

    if show:
        plt.show()
    return ax


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def arena_boundary_points(arena_mask, mask_name: str = "arena", stride: int = 1) -> np.ndarray:
    """
    Return arena mask boundary pixels as an array of (x, y) points.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    mask = _arena_mask_array(arena_mask, mask_name=mask_name)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    neighbors = np.zeros(mask.shape, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbors += padded[
                1 + dy : 1 + dy + mask.shape[0],
                1 + dx : 1 + dx + mask.shape[1],
            ]

    boundary = mask & (neighbors < 8)
    y, x = np.nonzero(boundary)
    points = np.column_stack([x, y]).astype(float)

    if points.size == 0:
        raise ValueError("Arena mask has no boundary pixels")

    return points[::stride]


def _resolve_cell_col_arg(cell_idx: int | str | None = None, cell_col: str | None = None) -> str:
    if cell_col is not None:
        return cell_col
    if cell_idx is None:
        raise ValueError("Provide cell_idx or cell_col")
    return cell_col_from_idx(cell_idx)


def compute_egocentric_boundary_map(
    df: pd.DataFrame,
    arena_mask,
    cell_idx: int | str | None = None,
    cell_col: str | None = None,
    angle_bins: int = 36,
    distance_bins: int = 20,
    distance_max_px: float | None = None,
    boundary_stride: int = 1,
    frame_stride: int = 1,
    chunk_size: int = 128,
    x_col: str = "ear_mid_x",
    y_col: str = "ear_mid_y",
    nose_x_col: str = "nose.x",
    nose_y_col: str = "nose.y",
    cell_dropped_col: str = "cell_dropped",
    mask_name: str = "arena",
) -> dict:
    """
    Compute an egocentric boundary map for one cell.

    Angle 0 means the boundary is straight ahead of the animal. Positive angles
    are to the animal's left after converting image y coordinates into a
    Cartesian y-up convention.
    """
    if angle_bins < 1 or distance_bins < 1:
        raise ValueError("angle_bins and distance_bins must be >= 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    cell_col = _resolve_cell_col_arg(cell_idx=cell_idx, cell_col=cell_col)
    mask = _arena_mask_array(arena_mask, mask_name=mask_name)
    boundary_xy = arena_boundary_points(mask, stride=boundary_stride)

    x = _numeric_array(df, x_col)
    y = _numeric_array(df, y_col)
    nose_x = _numeric_array(df, nose_x_col)
    nose_y = _numeric_array(df, nose_y_col)
    activity = _numeric_array(df, cell_col)

    head_dx = nose_x - x
    head_dy_cart = -(nose_y - y)
    head_norm = np.hypot(head_dx, head_dy_cart)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(nose_x)
        & np.isfinite(nose_y)
        & np.isfinite(activity)
        & np.isfinite(head_norm)
        & (head_norm > 0)
        & _points_inside_mask(mask, x, y)
    )
    if cell_dropped_col in df.columns:
        valid &= ~_boolean_array(df, cell_dropped_col)

    frame_idx = np.where(valid)[0][::frame_stride]
    if frame_idx.size == 0:
        raise ValueError("No valid frames for egocentric boundary map")

    angle_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
    if distance_max_px is None:
        distance_max_px = float(np.hypot(*mask.shape))
    distance_edges = np.linspace(0, distance_max_px, distance_bins + 1)

    occupancy = np.zeros((distance_bins, angle_bins), dtype=float)
    activity_sum = np.zeros((distance_bins, angle_bins), dtype=float)

    boundary_x = boundary_xy[:, 0]
    boundary_y = boundary_xy[:, 1]

    for start in range(0, len(frame_idx), chunk_size):
        chunk = frame_idx[start : start + chunk_size]

        rel_x = boundary_x[None, :] - x[chunk, None]
        rel_y_cart = -(boundary_y[None, :] - y[chunk, None])
        boundary_angle = np.arctan2(rel_y_cart, rel_x)
        head_angle = np.arctan2(head_dy_cart[chunk], head_dx[chunk])[:, None]

        ego_angle = _wrap_angle(boundary_angle - head_angle)
        distance = np.hypot(rel_x, rel_y_cart)

        angle_idx = np.searchsorted(angle_edges, ego_angle, side="right") - 1
        distance_idx = np.searchsorted(distance_edges, distance, side="right") - 1
        distance_idx = np.minimum(distance_idx, distance_bins - 1)

        valid_pairs = (
            (angle_idx >= 0)
            & (angle_idx < angle_bins)
            & (distance >= distance_edges[0])
            & (distance <= distance_edges[-1])
            & (distance_idx >= 0)
            & (distance_idx < distance_bins)
        )

        flat_idx = distance_idx * angle_bins + angle_idx
        flat_valid = flat_idx[valid_pairs]
        pair_activity = np.broadcast_to(
            activity[chunk, None],
            flat_idx.shape,
        )[valid_pairs]

        occupancy += np.bincount(
            flat_valid,
            minlength=distance_bins * angle_bins,
        ).reshape(distance_bins, angle_bins)
        activity_sum += np.bincount(
            flat_valid,
            weights=pair_activity,
            minlength=distance_bins * angle_bins,
        ).reshape(distance_bins, angle_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_activity = activity_sum / occupancy
    mean_activity[occupancy == 0] = np.nan

    return {
        "cell_col": cell_col,
        "cell_idx": cell_idx_from_col(cell_col),
        "mean_activity": mean_activity,
        "activity_sum": activity_sum,
        "occupancy": occupancy,
        "angle_edges": angle_edges,
        "distance_edges": distance_edges,
        "frame_indices": frame_idx,
        "boundary_points": boundary_xy,
        "angle_bins": angle_bins,
        "distance_bins": distance_bins,
    }


def plot_egocentric_boundary_map(
    egocentric_map: dict,
    ax=None,
    cmap: str = "viridis",
    occupancy_min: int = 1,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    mean_activity = np.asarray(egocentric_map["mean_activity"], dtype=float).copy()
    occupancy = np.asarray(egocentric_map["occupancy"], dtype=float)
    mean_activity[occupancy < occupancy_min] = np.nan

    angle_edges_deg = np.degrees(egocentric_map["angle_edges"])
    distance_edges = egocentric_map["distance_edges"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        mean_activity,
        origin="lower",
        aspect="auto",
        extent=[
            angle_edges_deg[0],
            angle_edges_deg[-1],
            distance_edges[0],
            distance_edges[-1],
        ],
        cmap=cmap,
    )
    ax.axvline(0, color="white", linewidth=1, alpha=0.8)
    ax.set_xlabel("egocentric boundary bearing (degrees; left +)")
    ax.set_ylabel("boundary distance (px)")
    ax.set_title(f"{egocentric_map['cell_col']} egocentric boundary map")
    plt.colorbar(im, ax=ax, label="mean raw cell trace")

    if show:
        plt.show()
    return ax


def plot_egocentric_boundary_occupancy(
    egocentric_map: dict,
    ax=None,
    cmap: str = "magma",
    show: bool = True,
):
    import matplotlib.pyplot as plt

    angle_edges_deg = np.degrees(egocentric_map["angle_edges"])
    distance_edges = egocentric_map["distance_edges"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        egocentric_map["occupancy"],
        origin="lower",
        aspect="auto",
        extent=[
            angle_edges_deg[0],
            angle_edges_deg[-1],
            distance_edges[0],
            distance_edges[-1],
        ],
        cmap=cmap,
    )
    ax.axvline(0, color="white", linewidth=1, alpha=0.8)
    ax.set_xlabel("egocentric boundary bearing (degrees; left +)")
    ax.set_ylabel("boundary distance (px)")
    ax.set_title("egocentric boundary occupancy")
    plt.colorbar(im, ax=ax, label="boundary samples")

    if show:
        plt.show()
    return ax


def plot_egocentric_boundary_polar(
    egocentric_map: dict,
    ax=None,
    cmap: str = "viridis",
    occupancy_min: int = 1,
    radius_max: float | None = None,
    smooth_sigma: float | tuple[float, float] | None = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    mean_activity = np.asarray(egocentric_map["mean_activity"], dtype=float).copy()
    occupancy = np.asarray(egocentric_map["occupancy"], dtype=float)
    mean_activity[occupancy < occupancy_min] = np.nan

    if smooth_sigma is not None:
        from scipy.ndimage import gaussian_filter

        valid = np.isfinite(mean_activity)
        values = np.where(valid, mean_activity, 0)
        weights = valid.astype(float)

        smooth_values = gaussian_filter(
            values,
            sigma=smooth_sigma,
            mode=("nearest", "wrap"),
        )
        smooth_weights = gaussian_filter(
            weights,
            sigma=smooth_sigma,
            mode=("nearest", "wrap"),
        )
        mean_activity = smooth_values / np.maximum(smooth_weights, 1e-12)

    radius_edges = egocentric_map["distance_edges"]
    angle_edges = np.asarray(egocentric_map["angle_edges"], dtype=float)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    full_circle_order = np.argsort(angle_centers % (2 * np.pi))

    mean_activity = mean_activity[:, full_circle_order]
    theta_edges = np.linspace(0, 2 * np.pi, mean_activity.shape[1] + 1)
    theta_grid, radius_grid = np.meshgrid(theta_edges, radius_edges)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    im = ax.pcolormesh(
        theta_grid,
        radius_grid,
        mean_activity,
        shading="auto",
        cmap=cmap,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)

    if radius_max is None:
        occupied_distance_bins = np.where(np.nansum(occupancy, axis=1) >= occupancy_min)[0]
        if occupied_distance_bins.size:
            radius_max = radius_edges[occupied_distance_bins[-1] + 1]
        else:
            radius_max = radius_edges[-1]
    ax.set_ylim(0, radius_max)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_title(f"{egocentric_map['cell_col']} egocentric boundary map")
    plt.colorbar(im, ax=ax, label="mean raw cell trace")

    if show:
        plt.show()
    return ax
