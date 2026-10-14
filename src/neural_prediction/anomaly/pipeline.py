"""Fit and score timestep- and trajectory-window anomalies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.neural_prediction.anomaly.config import AnomalyConfig
from src.neural_prediction.anomaly.features import build_timestep_features, build_window_features
from src.neural_prediction.anomaly.statistical import RobustMotionScorer


class AnomalyDetectionPipeline:
    """Unsupervised anomaly detection with robust motion and Isolation Forest scores.

    Fit once on a designated reference period. Scoring does not refit models or
    thresholds, so it can be called safely from interactive dashboard filters.
    """

    ARTIFACT_VERSION = 1

    def __init__(self, config: AnomalyConfig | None = None):
        self.config = config or AnomalyConfig()
        self.motion_scorer = RobustMotionScorer()
        self.timestep_model: IsolationForest | None = None
        self.window_model: IsolationForest | None = None
        self.timestep_feature_columns_: list[str] = []
        self.window_feature_columns_: list[str] = []
        self.reference_timestep_scores_: dict[str, np.ndarray] = {}
        self.reference_window_scores_: dict[str, np.ndarray] = {}
        self.thresholds_: dict[str, dict[str, float]] = {}
        self.fitted_ = False

    def _timestep_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return build_timestep_features(
            frame,
            pca_prefix=self.config.pca_prefix,
            trajectory_col=self.config.trajectory_col,
            cluster_col=self.config.cluster_col,
            year_col=self.config.year_col,
            week_col=self.config.week_col,
        )

    def _window_features(self, scored_timesteps: pd.DataFrame) -> pd.DataFrame:
        return build_window_features(
            scored_timesteps,
            window_size=self.config.window_size,
            stride=self.config.window_stride,
        )

    @staticmethod
    def _numeric_features(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
        columns = []
        for column in frame.columns:
            if column in exclude or not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = frame[column].to_numpy(dtype=float)
            if np.isfinite(values).all():
                columns.append(column)
        if not columns:
            raise ValueError("No finite numeric features available for anomaly detection")
        return columns

    @staticmethod
    def _percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=float)
        reference = reference[np.isfinite(reference)]
        if not len(reference):
            return np.full(len(values), 0.5, dtype=float)
        return np.searchsorted(np.sort(reference), values, side="right") / len(reference)

    def _labels(self, scores: np.ndarray, level: str) -> np.ndarray:
        thresholds = self.thresholds_[level]
        return np.where(
            scores >= thresholds["high"],
            "High anomaly",
            np.where(scores >= thresholds["warning"], "Warning", "Normal"),
        )

    def fit(self, reference_data: pd.DataFrame) -> "AnomalyDetectionPipeline":
        """Fit both detectors and calibrate thresholds using reference data only."""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")
        timesteps = self.motion_scorer.fit(self._timestep_features(reference_data)).score(
            self._timestep_features(reference_data)
        )
        timestep_exclude = {
            "_source_row", "trajectory_id", "cluster", "year", "week",
            "motion_top_contributor",
        }
        self.timestep_feature_columns_ = self._numeric_features(timesteps, timestep_exclude)
        if len(timesteps) < 2:
            raise ValueError("At least two timestep observations are required")
        self.timestep_model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        ).fit(timesteps[self.timestep_feature_columns_])
        timestep_if = -self.timestep_model.score_samples(timesteps[self.timestep_feature_columns_])
        timestep_motion = timesteps["motion_score"].to_numpy(dtype=float)
        timestep_combined = np.maximum(
            self._percentile_rank(timestep_if, timestep_if),
            self._percentile_rank(timestep_motion, timestep_motion),
        )
        self.reference_timestep_scores_ = {
            "isolation": timestep_if,
            "motion": timestep_motion,
            "combined": timestep_combined,
        }
        self.thresholds_["timestep"] = {
            "warning": float(np.quantile(timestep_combined, self.config.warning_quantile)),
            "high": float(np.quantile(timestep_combined, self.config.high_quantile)),
        }

        windows = self._window_features(timesteps)
        window_exclude = {
            "trajectory_id", "cluster", "year_start", "week_start", "year_end",
            "week_end", "window_start_row", "window_end_row",
        }
        self.window_feature_columns_ = self._numeric_features(windows, window_exclude)
        if len(windows) < 2:
            raise ValueError("At least two trajectory windows are required")
        self.window_model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        ).fit(windows[self.window_feature_columns_])
        window_if = -self.window_model.score_samples(windows[self.window_feature_columns_])
        motion_column = "motion_score_max_abs"
        window_motion = (
            windows[motion_column].to_numpy(dtype=float)
            if motion_column in windows.columns else np.zeros(len(windows), dtype=float)
        )
        window_combined = np.maximum(
            self._percentile_rank(window_if, window_if),
            self._percentile_rank(window_motion, window_motion),
        )
        self.reference_window_scores_ = {
            "isolation": window_if,
            "motion": window_motion,
            "combined": window_combined,
        }
        self.thresholds_["window"] = {
            "warning": float(np.quantile(window_combined, self.config.warning_quantile)),
            "high": float(np.quantile(window_combined, self.config.high_quantile)),
        }
        self.fitted_ = True
        return self

    def score_timesteps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Score observations and return one result per source row."""
        self._require_fitted()
        timesteps = self.motion_scorer.score(self._timestep_features(data))
        isolation = -self.timestep_model.score_samples(timesteps[self.timestep_feature_columns_])
        motion_rank = self._percentile_rank(
            timesteps["motion_score"].to_numpy(dtype=float),
            self.reference_timestep_scores_["motion"],
        )
        isolation_rank = self._percentile_rank(
            isolation, self.reference_timestep_scores_["isolation"]
        )
        combined = np.maximum(motion_rank, isolation_rank)
        timesteps["isolation_score"] = isolation
        timesteps["motion_rank"] = motion_rank
        timesteps["isolation_rank"] = isolation_rank
        timesteps["anomaly_score"] = combined
        timesteps["anomaly_label"] = self._labels(combined, "timestep")
        return timesteps.sort_values("_source_row").reset_index(drop=True)

    def score_windows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Score trajectory windows and return their source ranges and labels."""
        self._require_fitted()
        timesteps = self.score_timesteps(data)
        windows = self._window_features(timesteps)
        isolation = -self.window_model.score_samples(windows[self.window_feature_columns_])
        motion_column = "motion_score_max_abs"
        motion = (
            windows[motion_column].to_numpy(dtype=float)
            if motion_column in windows.columns else np.zeros(len(windows), dtype=float)
        )
        motion_rank = self._percentile_rank(motion, self.reference_window_scores_["motion"])
        isolation_rank = self._percentile_rank(
            isolation, self.reference_window_scores_["isolation"]
        )
        combined = np.maximum(motion_rank, isolation_rank)
        windows["isolation_score"] = isolation
        windows["motion_rank"] = motion_rank
        windows["isolation_rank"] = isolation_rank
        windows["anomaly_score"] = combined
        windows["anomaly_label"] = self._labels(combined, "window")
        return windows

    def summarize_trajectories(
        self,
        timestep_results: pd.DataFrame,
        window_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate point and window evidence into one row per trajectory."""
        if timestep_results.empty:
            return pd.DataFrame(columns=[
                "trajectory_id", "timestep_count", "flagged_timesteps", "max_timestep_score"
            ])
        summary = timestep_results.groupby("trajectory_id", dropna=False).agg(
            timestep_count=("anomaly_score", "size"),
            flagged_timesteps=("anomaly_label", lambda values: int((values != "Normal").sum())),
            max_timestep_score=("anomaly_score", "max"),
        )
        if window_results.empty:
            summary["window_count"] = 0
            summary["flagged_windows"] = 0
            summary["max_window_score"] = np.nan
        else:
            window_summary = window_results.groupby("trajectory_id", dropna=False).agg(
                window_count=("anomaly_score", "size"),
                flagged_windows=("anomaly_label", lambda values: int((values != "Normal").sum())),
                max_window_score=("anomaly_score", "max"),
            )
            summary = summary.join(window_summary, how="left")
            summary[["window_count", "flagged_windows"]] = summary[
                ["window_count", "flagged_windows"]
            ].fillna(0).astype(int)
        return summary.reset_index()

    def _require_fitted(self) -> None:
        if not self.fitted_ or self.timestep_model is None or self.window_model is None:
            raise RuntimeError("Fit or load the anomaly detection pipeline before scoring")

    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline to a joblib artifact."""
        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "artifact_version": self.ARTIFACT_VERSION,
            "config": self.config.to_dict(),
            "pipeline": self,
        }, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetectionPipeline":
        """Load a fitted pipeline and validate its artifact version."""
        artifact: dict[str, Any] = joblib.load(Path(path))
        if artifact.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("Unsupported anomaly pipeline artifact version")
        pipeline = artifact.get("pipeline")
        if not isinstance(pipeline, cls) or not pipeline.fitted_:
            raise ValueError("Artifact does not contain a fitted anomaly pipeline")
        return pipeline
