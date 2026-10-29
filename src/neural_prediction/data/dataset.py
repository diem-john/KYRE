from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.trajectory_prediction.utils.preprocessing import embedding_cols, sort_time


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 12
    horizon: int = 6
    stride: int = 1
    include_velocity: bool = True
    include_acceleration: bool = True


class TrajectoryWindowDataset(Dataset):
    """Temporal windows that never cross cluster boundaries."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in self.samples[index].items()}


def _features(position: np.ndarray, config: WindowConfig) -> np.ndarray:
    parts = [position]
    velocity = np.diff(position, axis=0)
    if config.include_velocity:
        parts.append(np.vstack([np.zeros((1, position.shape[1]), dtype=np.float32), velocity]))
    if config.include_acceleration:
        acceleration = np.diff(velocity, axis=0)
        parts.append(np.vstack([np.zeros((2, position.shape[1]), dtype=np.float32), acceleration]))
    return np.concatenate(parts, axis=1)


def _make_sample(values, start, config):
    end = start + config.lookback + config.horizon
    return {
        "x": _features(values[start:start + config.lookback], config),
        "y": values[start + config.lookback:end],
    }


def build_trajectory_windows(df, config, embedding_prefix="z_", cluster_col="final_cluster_label"):
    """Build training windows entirely within the supplied chronological split."""
    if config.lookback < 2 or config.horizon < 1 or config.stride < 1:
        raise ValueError("lookback >= 2, horizon >= 1, and stride >= 1 are required")
    cols = embedding_cols(df, embedding_prefix)
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {embedding_prefix!r}")
    samples = []
    for _, group in df.groupby(cluster_col, sort=False):
        values = sort_time(group)[cols].to_numpy(dtype=np.float32)
        minimum = config.lookback + config.horizon
        for start in range(0, max(0, len(values) - minimum + 1), config.stride):
            samples.append(_make_sample(values, start, config))
    if not samples:
        raise ValueError("No temporal windows could be constructed from the supplied data")
    return TrajectoryWindowDataset(samples)


def build_validation_windows(history_df, target_df, config, embedding_prefix="z_", cluster_col="final_cluster_label"):
    """Build validation windows using historical context plus validation targets."""
    cols = embedding_cols(history_df, embedding_prefix)
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {embedding_prefix!r}")
    samples = []
    for cluster, target_group in target_df.groupby(cluster_col, sort=False):
        history_group = history_df[history_df[cluster_col] == cluster]
        if history_group.empty:
            continue
        combined = sort_time(pd.concat([history_group, target_group], ignore_index=True))
        values = combined[cols].to_numpy(dtype=np.float32)
        history_end = len(history_group)
        for target_start in range(history_end, len(values) - config.horizon + 1, config.horizon):
            input_start = target_start - config.lookback
            if input_start >= 0:
                samples.append(_make_sample(values, input_start, config))
    if not samples:
        raise ValueError("No validation windows could be constructed")
    return TrajectoryWindowDataset(samples)


def feature_dimension(embedding_dim: int, config: WindowConfig) -> int:
    return embedding_dim * (1 + int(config.include_velocity) + int(config.include_acceleration))
