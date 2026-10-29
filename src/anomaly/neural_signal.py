
"""Leakage-aware one-step-ahead neural prediction residuals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ordered_position_columns(
    frame: pd.DataFrame,
    prefix: str = "z_",
) -> list[str]:
    """Return PCA position columns in numeric dimension order."""

    columns = [
        column for column in frame.columns
        if column.startswith(prefix)
    ]

    columns.sort(
        key=lambda name: (
            (0, int(name[len(prefix):]))
            if name[len(prefix):].isdigit()
            else (1, name)
        )
    )

    if not columns:
        raise ValueError(
            f"No PCA position columns found with prefix {prefix!r}"
        )

    return columns


def _sort_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort observations chronologically."""

    return frame.sort_values(
        ["year", "week"],
        kind="stable",
    )


class NeuralPredictionSignal:
    """
    Compute rolling one-step-ahead forecast errors without target leakage.

    For each target week, the predictor receives only earlier observations
    from the same cluster. The first predicted step is compared against
    the target's actual PCA position.

    Rows without sufficient historical context receive NaN.
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
        """Read the required lookback from the loaded predictor."""

        window_config = getattr(
            self.predictor, "window_config", None
        )

        if window_config is None:
            raise AttributeError(
                "The neural predictor must expose window_config. "
                "Ensure the checkpoint is loaded using the updated "
                "NeuralTrajectoryPredictor implementation."
            )

        lookback = getattr(window_config, "lookback", None)
        if lookback is None or int(lookback) < 1:
            raise ValueError(
                "The predictor's window_config.lookback must be "
                "a positive integer."
            )

        return int(lookback)

    def score(
        self,
        target_data: pd.DataFrame,
        *,
        history_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return source-row-aligned neural prediction errors.

        Parameters
        ----------
        target_data:
            Observations to score.
        history_data:
            Optional earlier observations used to initialize each
            cluster's rolling history. Only observations strictly earlier
            than the first target timestamp for that cluster are retained.

        Returns
        -------
        pandas.DataFrame
            Contains `_source_row` and `neural_prediction_error`.
        """

        if target_data.empty:
            return pd.DataFrame(
                columns=[
                    "_source_row",
                    "neural_prediction_error",
                ]
            )

        required = {
            self.cluster_col,
            self.year_col,
            self.week_col,
        }

        missing = sorted(required.difference(target_data.columns))
        if missing:
            raise ValueError(
                f"Target data is missing required columns: {missing}"
            )

        positions = _ordered_position_columns(
            target_data, self.pca_prefix
        )

        if history_data is not None:
            missing_history = sorted(
                (required | set(positions))
                .difference(history_data.columns)
            )
            if missing_history:
                raise ValueError(
                    "History data is missing required columns: "
                    f"{missing_history}"
                )

        # Preserve target input order so the scores can be joined back
        # to the original observations.
        target = target_data.copy().reset_index(drop=True)
        target["_source_row"] = np.arange(
            len(target), dtype=np.int64
        )

        errors = np.full(
            len(target), np.nan, dtype=np.float64
        )
        predictions = np.full(
            (len(target), len(positions)), np.nan, dtype=np.float64
        )

        context = (
            history_data.copy()
            if history_data is not None
            else target_data.iloc[0:0].copy()
        )

        lookback = self.lookback

        for cluster, target_group in target.groupby(
            self.cluster_col,
            sort=False,
            dropna=False,
        ):
            target_group = _sort_trajectory(target_group)

            if context.empty:
                history_group = target_group.iloc[0:0]
            else:
                if pd.isna(cluster):
                    history_group = context[
                        context[self.cluster_col].isna()
                    ]
                else:
                    history_group = context[
                        context[self.cluster_col] == cluster
                    ]

                history_group = _sort_trajectory(history_group)

                first_target_time = (
                    target_group[self.year_col].iloc[0],
                    target_group[self.week_col].iloc[0],
                )

                # Exclude same-week and future context rows.
                history_group = history_group[
                    (
                        history_group[self.year_col]
                        < first_target_time[0]
                    )
                    | (
                        (
                            history_group[self.year_col]
                            == first_target_time[0]
                        )
                        & (
                            history_group[self.week_col]
                            < first_target_time[1]
                        )
                    )
                ]

            prior_values = history_group[positions].to_numpy(
                dtype=np.float32
            )

            prior_times = list(
                zip(
                    history_group[self.year_col].tolist(),
                    history_group[self.week_col].tolist(),
                )
            )

            for _, row in target_group.iterrows():
                target_time = (
                    row[self.year_col],
                    row[self.week_col],
                )

                # Only observations strictly earlier than the target
                # can be used to predict it.
                eligible = [
                    index
                    for index, timestamp in enumerate(prior_times)
                    if timestamp < target_time
                ]

                history_values = prior_values[eligible]

                if len(history_values) >= lookback:
                    prediction_output = self.predictor.predict(
                        history_values
                    )

                    # Support both torch.Tensor and NumPy predictor APIs.
                    if hasattr(prediction_output, "detach"):
                        prediction_output = (
                            prediction_output.detach()
                            .cpu()
                            .numpy()
                        )

                    prediction = np.asarray(
                        prediction_output,
                        dtype=np.float64,
                    )

                    # Normalize a single-batch result if a predictor
                    # returns [1, horizon, dimensions].
                    if (
                        prediction.ndim == 3
                        and prediction.shape[0] == 1
                    ):
                        prediction = prediction[0]

                    if prediction.ndim != 2:
                        raise ValueError(
                            "Neural predictor output must have shape "
                            "[horizon, embedding_dimensions]; received "
                            f"{prediction.shape}."
                        )

                    if prediction.shape[1] != len(positions):
                        raise ValueError(
                            "Neural predictor output dimension mismatch: "
                            f"expected {len(positions)}, received "
                            f"{prediction.shape[1]}."
                        )

                    if prediction.shape[0] < 1:
                        raise ValueError(
                            "Neural predictor returned an empty "
                            "forecast horizon."
                        )

                    actual = row[positions].to_numpy(
                        dtype=np.float64
                    )

                    # First forecast step is the next-week estimate.
                    source_row = int(row["_source_row"])
                    predictions[source_row] = prediction[0]
                    errors[source_row] = float(
                        np.sqrt(
                            np.mean(
                                np.square(actual - prediction[0])
                            )
                        )
                    )

                # Make the target available only after its own score
                # has been computed, enabling rolling-origin evaluation.
                prior_values = np.vstack(
                    [
                        prior_values,
                        row[positions].to_numpy(dtype=np.float32),
                    ]
                )
                prior_times.append(target_time)

        # Keep one-step-ahead forecasts aligned with the target rows so the
        # dashboard can plot actual vs predicted PCA coordinates and mark anomalies.
        forecast_columns: dict[str, Any] = {
            "_source_row": np.arange(len(target), dtype=np.int64),
            "neural_prediction_error": errors,
        }
        for dim, position_col in enumerate(positions):
            actual_values = target[position_col].to_numpy(dtype=np.float64)
            forecast_columns[f"actual_{position_col}"] = actual_values
            forecast_columns[f"predicted_{position_col}"] = predictions[:, dim]

        return pd.DataFrame(forecast_columns)
