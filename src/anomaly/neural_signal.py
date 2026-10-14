"""Leakage-aware one-step-ahead neural prediction residuals for anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ordered_position_columns(frame: pd.DataFrame, prefix: str = "z_") -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    columns.sort(
        key=lambda name: (0, int(name[len(prefix):]))
        if name[len(prefix):].isdigit() else (1, name)
    )
    if not columns:
        raise ValueError(f"No PCA position columns found with prefix {prefix!r}")
    return columns


def _sort_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["year", "week"], kind="stable")


class NeuralPredictionSignal:
    """Compute one-step-ahead forecast errors without using the target as input.

    For each target week, the predictor receives only earlier observations from
    the same cluster. The first forecast step is compared with the actual target.
    """

    def __init__(
        self,
        predictor: Any,
        *,
        cluster_col: str = "final_cluster_label",
        pca_prefix: str = "z_",
        year_col: str = "year",
        week_col: str = "week",
    ) -> None:
        self.predictor = predictor
        self.cluster_col = cluster_col
        self.pca_prefix = pca_prefix
        self.year_col = year_col
        self.week_col = week_col

    @property
    def lookback(self) -> int:
        return int(self.predictor.window_config.lookback)

    def score(
        self,
        target_data: pd.DataFrame,
        *,
        history_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return source-row-aligned prediction errors for target observations.

        Rows without enough preceding history receive NaN; they are not silently
        classified as normal. Target observations are appended to the history only
        after their own predictions are made, providing rolling one-step-ahead
        evaluation.
        """
        if target_data.empty:
            return pd.DataFrame(columns=["_source_row", "neural_prediction_error"])
        required = {self.cluster_col, self.year_col, self.week_col}
        missing = sorted(required.difference(target_data.columns))
        if missing:
            raise ValueError(f"Target data is missing required columns: {missing}")

        positions = _ordered_position_columns(target_data, self.pca_prefix)
        if history_data is not None:
            missing_history = sorted(
                (required | set(positions)).difference(history_data.columns)
            )
            if missing_history:
                raise ValueError(f"History data is missing required columns: {missing_history}")

        target = target_data.copy().reset_index(drop=True)
        target["_source_row"] = np.arange(len(target), dtype=np.int64)
        errors = np.full(len(target), np.nan, dtype=np.float64)
        context = history_data.copy() if history_data is not None else target_data.iloc[0:0].copy()

        for cluster, target_group in target.groupby(self.cluster_col, sort=False, dropna=False):
            target_group = _sort_trajectory(target_group)
            if context.empty:
                history_group = target_group.iloc[0:0]
            else:
                history_group = context[context[self.cluster_col] == cluster]
                history_group = _sort_trajectory(history_group)
                first_target_time = (
                    target_group[self.year_col].iloc[0],
                    target_group[self.week_col].iloc[0],
                )
                # Only keep context before the earliest target. This also makes
                # calls safe when the caller passes a broader frame by mistake.
                history_group = history_group[
                    (history_group[self.year_col] < first_target_time[0])
                    | (
                        (history_group[self.year_col] == first_target_time[0])
                        & (history_group[self.week_col] < first_target_time[1])
                    )
                ]

            prior_values = history_group[positions].to_numpy(dtype=np.float32)
            prior_times = list(zip(history_group[self.year_col].tolist(), history_group[self.week_col].tolist()))
            for _, row in target_group.iterrows():
                target_time = (row[self.year_col], row[self.week_col])
                # Only context observations strictly earlier than this target are valid.
                if len(prior_times):
                    eligible = [
                        idx for idx, timestamp in enumerate(prior_times)
                        if timestamp < target_time
                    ]
                    history_values = prior_values[eligible]
                else:
                    history_values = prior_values

                if len(history_values) >= self.lookback:
                    prediction = np.asarray(
                        self.predictor.predict(history_values), dtype=np.float64
                    )
                    actual = row[positions].to_numpy(dtype=np.float64)
                    if prediction.ndim != 2 or prediction.shape[1] != len(positions):
                        raise ValueError(
                            "Neural predictor output must have shape "
                            f"[horizon, {len(positions)}]; received {prediction.shape}"
                        )
                    if prediction.shape[0] < 1:
                        raise ValueError("Neural predictor returned an empty forecast horizon")
                    errors[int(row["_source_row"])] = float(
                        np.sqrt(np.mean(np.square(actual - prediction[0])))
                    )

                # Once this target has been evaluated, it becomes available to
                # subsequent target predictions (online/rolling-origin semantics).
                prior_values = np.vstack([prior_values, row[positions].to_numpy(dtype=np.float32)])
                prior_times.append(target_time)

        return pd.DataFrame({
            "_source_row": np.arange(len(target), dtype=np.int64),
            "neural_prediction_error": errors,
        })
