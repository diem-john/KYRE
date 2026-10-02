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


def _derivatives(sequence: np.ndarray):
    velocity = np.diff(sequence, axis=0)
    acceleration = np.diff(velocity, axis=0)
    return velocity, acceleration


def build_trajectory_windows(
    df: pd.DataFrame,
    config: WindowConfig,
    embedding_prefix: str = "z_",
    cluster_col: str = "final_cluster_label",
) -> TrajectoryWindowDataset:
    """Convert model-space centroid histories into supervised sequences."""
    if config.lookback < 2 or config.horizon < 1 or config.stride < 1:
        raise ValueError("lookback >= 2, horizon >= 1, and stride >= 1 are required")

    cols = embedding_cols(df, embedding_prefix)
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {embedding_prefix!r}")

    samples = []
    for _, group in df.groupby(cluster_col, sort=False):
        values = sort_time(group)[cols].to_numpy(dtype=np.float32)
        minimum = config.lookback + config.horizon
        if len(values) < minimum:
            continue
        for start in range(0, len(values) - minimum + 1, config.stride):
            x_position = values[start:start + config.lookback]
            y = values[start + config.lookback:start + minimum]
            x_parts = [x_position]
            if config.include_velocity:
                velocity, _ = _derivatives(x_position)
                x_parts.append(np.vstack([np.zeros((1, values.shape[1]), dtype=np.float32), velocity]))
            if config.include_acceleration:
                _, acceleration = _derivatives(x_position)
                x_parts.append(np.vstack([np.zeros((2, values.shape[1]), dtype=np.float32), acceleration]))
            samples.append({"x": np.concatenate(x_parts, axis=1), "y": y})

    if not samples:
        raise ValueError("No temporal windows could be constructed from the supplied data")
    return TrajectoryWindowDataset(samples)


def feature_dimension(embedding_dim: int, config: WindowConfig) -> int:
    return embedding_dim * (1 + int(config.include_velocity) + int(config.include_acceleration))
