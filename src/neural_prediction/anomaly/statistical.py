"""Robust statistics for interpretable motion anomaly scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MOTION_COLUMNS = ("velocity_norm", "acceleration_norm", "jerk_norm")


class RobustMotionScorer:
    """Score motion features using median and MAD fitted on reference data."""

    def __init__(self, columns: tuple[str, ...] = DEFAULT_MOTION_COLUMNS, epsilon: float = 1e-8):
        self.columns = tuple(columns)
        self.epsilon = float(epsilon)
        self.medians_: dict[str, float] = {}
        self.scales_: dict[str, float] = {}
        self.fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "RobustMotionScorer":
        available = [column for column in self.columns if column in frame.columns]
        if not available:
            raise ValueError("No motion columns available for robust scoring")
        for column in available:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                raise ValueError(f"Motion feature {column!r} has no finite reference values")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = 1.4826 * mad
            if scale <= self.epsilon:
                q25, q75 = np.quantile(values, [0.25, 0.75])
                scale = float((q75 - q25) / 1.349)
            if scale <= self.epsilon:
                scale = float(np.std(values))
            self.medians_[column] = median
            self.scales_[column] = max(scale, self.epsilon)
        self.columns_ = tuple(available)
        self.fitted_ = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("RobustMotionScorer must be fitted before scoring")
        output = frame.copy()
        component_columns = []
        for column in self.columns_:
            if column not in output.columns:
                raise ValueError(f"Required motion feature {column!r} is missing")
            score_column = f"{column}_robust_z"
            output[score_column] = (
                output[column].astype(float) - self.medians_[column]
            ).abs() / self.scales_[column]
            component_columns.append(score_column)
        values = output[component_columns].to_numpy(dtype=float)
        output["motion_score"] = np.max(values, axis=1)
        output["motion_top_contributor"] = output[component_columns].idxmax(axis=1).str.replace(
            "_robust_z", "", regex=False
        )
        return output

    def to_dict(self) -> dict:
        if not self.fitted_:
            raise RuntimeError("Cannot serialize an unfitted scorer")
        return {
            "columns": list(self.columns_),
            "epsilon": self.epsilon,
            "medians": self.medians_,
            "scales": self.scales_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RobustMotionScorer":
        scorer = cls(tuple(payload["columns"]), payload.get("epsilon", 1e-8))
        scorer.columns_ = tuple(payload["columns"])
        scorer.medians_ = {key: float(value) for key, value in payload["medians"].items()}
        scorer.scales_ = {key: float(value) for key, value in payload["scales"].items()}
        scorer.fitted_ = True
        return scorer
