"""Validation helpers for anomaly detectors without ground-truth labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_position_spike(
    frame: pd.DataFrame,
    *,
    row_index: int,
    position_columns: list[str],
    magnitude: float = 8.0,
    dimension: int = 0,
) -> pd.DataFrame:
    """Return a copy with one controlled PCA-space spike injected.

    row_index is a positional row index, not a dataframe index label.
    """
    if not 0 <= row_index < len(frame):
        raise IndexError("row_index is outside the dataframe")
    if not position_columns:
        raise ValueError("position_columns cannot be empty")
    if not 0 <= dimension < len(position_columns):
        raise IndexError("dimension is outside position_columns")
    column = position_columns[dimension]
    if column not in frame.columns:
        raise ValueError(f"Position column {column!r} is missing")
    result = frame.copy(deep=True)
    values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    scale = float(np.std(finite)) if len(finite) else 1.0
    if scale <= 1e-12:
        scale = 1.0
    result.iloc[row_index, result.columns.get_loc(column)] = (
        float(result.iloc[row_index][column]) + magnitude * scale
    )
    return result


def ranking_summary(scores: np.ndarray, injected_indices: list[int]) -> dict[str, float]:
    """Summarize ranks of known injected anomaly indices."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not injected_indices:
        raise ValueError("injected_indices cannot be empty")
    if any(index < 0 or index >= len(scores) for index in injected_indices):
        raise IndexError("An injected index is outside the score array")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    percentiles = ranks[injected_indices] / len(scores)
    return {
        "injected_count": float(len(injected_indices)),
        "mean_injected_percentile": float(np.mean(percentiles)),
        "minimum_injected_percentile": float(np.min(percentiles)),
    }
