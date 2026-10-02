from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.neural_prediction.evaluation.metrics import trajectory_metrics, horizon_metrics


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 12
    horizon: int = 6
    stride: int = 6
    min_history: int = 24


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1][None, :], horizon, axis=0)


def rolling_origin_windows(
    trajectory: np.ndarray,
    config: BacktestConfig,
):
    """Yield historical context and known future targets for rolling-origin evaluation."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape [time, dimensions]")
    first_origin = max(config.min_history, config.lookback)
    last_origin = len(trajectory) - config.horizon
    for origin in range(first_origin, last_origin + 1, config.stride):
        history = trajectory[:origin]
        target = trajectory[origin:origin + config.horizon]
        yield origin, history, target


def backtest_forecaster(
    trajectories: dict[str, np.ndarray],
    predictor_factory,
    config: BacktestConfig,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a common-origin evaluation for a predictor factory.

    predictor_factory(history, horizon) must return [horizon, dimensions].
    """
    rows = []
    for cluster, trajectory in trajectories.items():
        for origin, history, target in rolling_origin_windows(trajectory, config):
            prediction = np.asarray(predictor_factory(history, config.horizon), dtype=float)
            if prediction.shape != target.shape:
                raise ValueError(f"{model_name} returned {prediction.shape}; expected {target.shape}")
            aggregate = trajectory_metrics(target[None, ...], prediction[None, ...])
            rows.append({
                "cluster": cluster,
                "origin": origin,
                "model": model_name,
                **aggregate,
            })
            for row in horizon_metrics(target[None, ...], prediction[None, ...]):
                rows.append({
                    "cluster": cluster,
                    "origin": origin,
                    "model": model_name,
                    "row_type": "horizon",
                    **row,
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result, result
    horizon_df = result[result.get("row_type", pd.Series(index=result.index, dtype=object)).eq("horizon")].copy()
    summary = (
        horizon_df.groupby(["model", "horizon"], as_index=False)
        .agg(
            n_forecasts=("cluster", "count"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            cosine_similarity=("cosine_similarity", "mean"),
            velocity_error=("velocity_error", "mean"),
            displacement_cosine_similarity=("displacement_cosine_similarity", "mean"),
            displacement_magnitude_error=("displacement_magnitude_error", "mean"),
        )
    )
    return result, summary\n\n\ndef compare_backtests(result_frames: list[pd.DataFrame]) -> pd.DataFrame:\n    """Combine horizon-level backtest results from multiple models."""\n    frames = [frame for frame in result_frames if frame is not None and not frame.empty]\n    if not frames:\n        return pd.DataFrame()\n    horizon_frames = []\n    for frame in frames:\n        if "row_type" in frame.columns:\n            horizon_frames.append(frame[frame["row_type"] == "horizon"])\n    if not horizon_frames:\n        return pd.DataFrame()\n    combined = pd.concat(horizon_frames, ignore_index=True)\n    return (combined.groupby(["model", "horizon"], as_index=False)\n            .agg(n_forecasts=("cluster", "count"), rmse=("rmse", "mean"), mae=("mae", "mean"),\n                 cosine_similarity=("cosine_similarity", "mean"), velocity_error=("velocity_error", "mean"),\n                 displacement_cosine_similarity=("displacement_cosine_similarity", "mean"),\n                 displacement_magnitude_error=("displacement_magnitude_error", "mean")))
