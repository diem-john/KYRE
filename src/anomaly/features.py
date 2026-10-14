"""Feature extraction for PCA-space trajectories.

Motion magnitudes describe changes in PCA space. They are not physical speed,
acceleration, or jerk unless the source coordinates have a physical meaning.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def resolve_position_columns(
    frame: pd.DataFrame,
    prefix: str = "z_",
    position_columns: Sequence[str] | None = None,
) -> list[str]:
    """Resolve position columns and sort numeric suffixes naturally."""
    if position_columns is not None:
        columns = list(position_columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Position columns are missing: {missing}")
    else:
        columns = [column for column in frame.columns if column.startswith(prefix)]
        columns.sort(
            key=lambda name: (0, int(name[len(prefix):]))
            if name[len(prefix):].isdigit() else (1, name)
        )
    if not columns:
        raise ValueError(f"No position columns found for prefix {prefix!r}")
    return columns


def build_timestep_features(
    frame: pd.DataFrame,
    *,
    position_columns: Sequence[str] | None = None,
    pca_prefix: str = "z_",
    trajectory_col: str | None = None,
    cluster_col: str = "final_cluster_label",
    year_col: str = "year",
    week_col: str = "week",
) -> pd.DataFrame:
    """Build one feature row per source observation without crossing trajectories."""
    if frame.empty:
        raise ValueError("Cannot extract anomaly features from an empty dataframe")
    positions = resolve_position_columns(frame, pca_prefix, position_columns)
    trajectory_key = trajectory_col or cluster_col
    for column in (trajectory_key, year_col, week_col):
        if column not in frame.columns:
            raise ValueError(f"Required column {column!r} is missing")

    work = frame.copy().reset_index(drop=True)
    work["_source_row"] = np.arange(len(work), dtype=np.int64)
    position_values = work[positions].to_numpy(dtype=np.float64)
    if not np.isfinite(position_values).all():
        raise ValueError("Position columns contain NaN or infinite values")
    work = work.sort_values(
        [trajectory_key, year_col, week_col, "_source_row"], kind="stable"
    )

    parts = []
    for trajectory_id, group in work.groupby(trajectory_key, sort=False, dropna=False):
        z = group[positions].to_numpy(dtype=np.float64)
        n_rows, n_dims = z.shape
        velocity = np.zeros_like(z)
        if n_rows > 1:
            velocity[1:] = np.diff(z, axis=0)
        acceleration = np.zeros_like(z)
        if n_rows > 2:
            acceleration[2:] = np.diff(velocity, axis=0)[1:]
        jerk = np.zeros_like(z)
        if n_rows > 3:
            jerk[3:] = np.diff(acceleration, axis=0)[2:]

        part = pd.DataFrame({
            "_source_row": group["_source_row"].to_numpy(),
            "trajectory_id": [trajectory_id] * n_rows,
            "cluster": group[cluster_col].to_numpy()
            if cluster_col in group.columns else [trajectory_id] * n_rows,
            "year": group[year_col].to_numpy(),
            "week": group[week_col].to_numpy(),
            "velocity_norm": np.linalg.norm(velocity, axis=1),
            "acceleration_norm": np.linalg.norm(acceleration, axis=1),
            "jerk_norm": np.linalg.norm(jerk, axis=1),
        })
        for index in range(n_dims):
            part[f"position_{index}"] = z[:, index]
            part[f"velocity_{index}"] = velocity[:, index]
            part[f"acceleration_{index}"] = acceleration[:, index]
        parts.append(part)

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("_source_row", kind="stable")
        .reset_index(drop=True)
    )


def build_window_features(
    timestep_features: pd.DataFrame,
    *,
    window_size: int = 12,
    stride: int = 1,
) -> pd.DataFrame:
    """Summarize windows without crossing trajectory boundaries.

    Trajectories shorter than the configured window yield one partial window
    when they contain at least two observations.
    """
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be >= 1")
    if timestep_features.empty:
        raise ValueError("Cannot build windows from an empty feature frame")

    position_columns = [
        column for column in timestep_features.columns if column.startswith("position_")
    ]
    value_columns = position_columns + [
        column for column in (
            "velocity_norm", "acceleration_norm", "jerk_norm", "motion_score",
            "neural_prediction_error",
        )
        if column in timestep_features.columns
    ]
    records = []
    for trajectory_id, group in timestep_features.groupby("trajectory_id", sort=False, dropna=False):
        group = group.sort_values(["year", "week", "_source_row"], kind="stable")
        n_rows = len(group)
        if n_rows < 2:
            continue
        if n_rows < window_size:
            starts = [0]
        else:
            starts = list(range(0, n_rows - window_size + 1, stride))
            last_start = n_rows - window_size
            if starts[-1] != last_start:
                starts.append(last_start)

        for start in starts:
            window = group.iloc[start:min(start + window_size, n_rows)]
            if len(window) < 2:
                continue
            record = {
                "trajectory_id": trajectory_id,
                "cluster": window["cluster"].iloc[0],
                "year_start": window["year"].iloc[0],
                "week_start": window["week"].iloc[0],
                "year_end": window["year"].iloc[-1],
                "week_end": window["week"].iloc[-1],
                "window_start_row": int(window["_source_row"].iloc[0]),
                "window_end_row": int(window["_source_row"].iloc[-1]),
                "window_length": int(len(window)),
            }
            for column in value_columns:
                values = window[column].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                if not len(finite):
                    mean_value = std_value = max_abs_value = np.nan
                else:
                    mean_value = float(np.mean(finite))
                    std_value = float(np.std(finite))
                    max_abs_value = float(np.max(np.abs(finite)))
                record[f"{column}_mean"] = mean_value
                record[f"{column}_std"] = std_value
                record[f"{column}_max_abs"] = max_abs_value
            records.append(record)
    if not records:
        raise ValueError("No windows could be constructed; trajectories need >= 2 rows")
    return pd.DataFrame.from_records(records)
