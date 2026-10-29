
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from src.neural_prediction.evaluation.metrics import trajectory_metrics


@dataclass
class BacktestConfig:
    """Configuration for rolling-origin trajectory backtesting."""

    lookback: int = 12
    horizon: int = 6
    stride: int = 6
    min_history: int = 24


def persistence_forecast(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Persistence baseline.

    Repeats the most recent observed embedding for all future steps.
    """
    history = np.asarray(history, dtype=np.float32)

    if history.ndim != 2:
        raise ValueError(
            "history must have shape [time, dimensions]"
        )

    if len(history) == 0:
        raise ValueError("history cannot be empty")

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    last_value = history[-1]

    return np.repeat(
        last_value[None, :],
        horizon,
        axis=0,
    ).astype(np.float32)


def rolling_origin_windows(
    trajectory: np.ndarray,
    config: BacktestConfig,
) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate rolling-origin evaluation windows.

    Returns
    -------
    history:
        Observations available before the forecast origin.

    target:
        Future observations to forecast.

    origin:
        Index of the forecast origin.
    """
    trajectory = np.asarray(
        trajectory,
        dtype=np.float32,
    )

    if trajectory.ndim != 2:
        raise ValueError(
            "trajectory must have shape [time, dimensions]"
        )

    if config.lookback < 1:
        raise ValueError("lookback must be at least 1")

    if config.horizon < 1:
        raise ValueError("horizon must be at least 1")

    if config.stride < 1:
        raise ValueError("stride must be at least 1")

    if config.min_history < 1:
        raise ValueError("min_history must be at least 1")

    n_samples = len(trajectory)

    if n_samples < config.min_history + config.horizon:
        return

    first_origin = max(
        config.min_history,
        config.lookback,
    )

    last_origin = n_samples - config.horizon

    for origin in range(
        first_origin,
        last_origin + 1,
        config.stride,
    ):
        history = trajectory[:origin]

        target = trajectory[
            origin:origin + config.horizon
        ]

        yield history, target, origin


def _prediction_to_numpy(prediction) -> np.ndarray:
    """
    Convert a model prediction to a NumPy float32 array.

    Supports NumPy arrays, CPU tensors, and GPU tensors.
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    return np.asarray(
        prediction,
        dtype=np.float32,
    )


def _normalize_prediction_shape(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Normalize a prediction to match the target shape.

    A single-sample model output may have shape
    [1, horizon, dimensions], while the backtest target
    has shape [horizon, dimensions].

    Only the leading batch dimension is removed when its
    size is exactly one. Other shape mismatches are rejected.
    """
    prediction = _prediction_to_numpy(prediction)

    # Remove the batch dimension for a single trajectory.
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]

    if prediction.shape != target.shape:
        raise ValueError(
            f"Predictor returned shape {prediction.shape}, "
            f"expected {target.shape}. "
            "Check the checkpoint forecast horizon, "
            "embedding dimensions, and predictor output format."
        )

    return prediction


def backtest_forecaster(
    trajectories: dict[str, np.ndarray],
    predictor_factory: Callable[[], object],
    config: BacktestConfig,
    model_name: str = "model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a frozen-checkpoint rolling-origin backtest.

    Parameters
    ----------
    trajectories:
        Mapping from cluster/entity ID to trajectory arrays
        of shape [time, dimensions].

    predictor_factory:
        Callable returning a predictor object exposing
        predict(history).

    config:
        Backtest configuration.

    model_name:
        Name used in the result tables.

    Returns
    -------
    detailed:
        Metrics for every cluster and forecast origin.

    summary:
        Metrics aggregated by model.
    """
    detailed_rows = []

    for cluster_id, trajectory in trajectories.items():
        trajectory = np.asarray(
            trajectory,
            dtype=np.float32,
        )

        for history, target, origin in rolling_origin_windows(
            trajectory,
            config,
        ):
            predictor = predictor_factory()

            raw_prediction = predictor.predict(history)

            prediction = _normalize_prediction_shape(
                raw_prediction,
                target,
            )

            # Add a sample dimension: [horizon, dimensions]
            # becomes [samples=1, horizon, dimensions].
            metrics = trajectory_metrics(
                prediction[np.newaxis, :, :],
                target[np.newaxis, :, :],
            )

            detailed_rows.append(
                {
                    "model": model_name,
                    "cluster": cluster_id,
                    "origin": origin,
                    **metrics,
                }
            )

    detailed = pd.DataFrame(detailed_rows)

    if detailed.empty:
        return detailed, pd.DataFrame()

    summary_columns = [
        column
        for column in [
            "rmse",
            "mae",
            "cosine_similarity",
            "velocity_error",
            "displacement_cosine",
            "displacement_magnitude_error",
        ]
        if column in detailed.columns
    ]

    if summary_columns:
        summary = (
            detailed.groupby("model")[summary_columns]
            .mean()
            .reset_index()
        )
    else:
        summary = detailed[["model"]].drop_duplicates().reset_index(
            drop=True
        )

    return detailed, summary


def compare_backtests(
    results: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine backtest summary tables from multiple models.
    """
    valid_results = [
        result
        for result in results
        if result is not None and not result.empty
    ]

    if not valid_results:
        return pd.DataFrame()

    return pd.concat(
        valid_results,
        ignore_index=True,
    )
