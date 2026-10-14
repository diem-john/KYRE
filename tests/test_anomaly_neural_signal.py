from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.anomaly import AnomalyConfig, AnomalyDetectionPipeline
from src.anomaly.neural_signal import NeuralPredictionSignal


class PersistencePredictor:
    def __init__(self, lookback=4, horizon=2):
        self.window_config = SimpleNamespace(lookback=lookback, horizon=horizon)

    def predict(self, history):
        return np.repeat(history[-1][None, :], self.window_config.horizon, axis=0)


def trajectory_frame(start_week, count, cluster="A", spike_at=None):
    rows = []
    for offset in range(count):
        week = start_week + offset
        value = float(np.sin(week / 4.0))
        if spike_at is not None and week == spike_at:
            value += 5.0
        rows.append({
            "final_cluster_label": cluster,
            "year": 2025,
            "week": week,
            "z_0": value,
            "z_1": float(np.cos(week / 6.0)),
        })
    return pd.DataFrame(rows)


def test_neural_signal_scores_one_step_ahead_without_target_leakage():
    predictor = PersistencePredictor()
    signal = NeuralPredictionSignal(predictor)
    history = trajectory_frame(1, 12)
    target = trajectory_frame(13, 3, spike_at=14)

    scores = signal.score(target, history_data=history)

    assert scores["neural_prediction_error"].notna().all()
    assert scores.loc[1, "neural_prediction_error"] > scores.loc[0, "neural_prediction_error"]


def test_pipeline_fuses_neural_residual_at_timestep_and_window_levels():
    train = pd.concat([
        trajectory_frame(1, 20, "A"),
        trajectory_frame(1, 20, "B"),
    ], ignore_index=True)
    validation = pd.concat([
        trajectory_frame(21, 6, "A", spike_at=23),
        trajectory_frame(21, 6, "B"),
    ], ignore_index=True)
    test = pd.concat([
        trajectory_frame(27, 4, "A", spike_at=28),
        trajectory_frame(27, 4, "B"),
    ], ignore_index=True)

    pipeline = AnomalyDetectionPipeline(
        AnomalyConfig(window_size=4, window_stride=1, n_estimators=10),
        neural_predictor=PersistencePredictor(),
    ).fit(train, calibration_data=validation)

    points = pipeline.score_timesteps(test, history_data=pd.concat([train, validation], ignore_index=True))
    windows = pipeline.score_windows(test, history_data=pd.concat([train, validation], ignore_index=True))

    assert "neural_prediction_error" in points
    assert "neural_rank" in points
    assert "neural_prediction_error" in windows
    assert "neural_rank" in windows
    assert np.isfinite(points["anomaly_score"]).all()
    assert set(points["anomaly_label"]).issubset({"Normal", "Warning", "High anomaly"})


def test_neural_enabled_pipeline_requires_held_out_calibration_data():
    train = trajectory_frame(1, 20)
    pipeline = AnomalyDetectionPipeline(
        AnomalyConfig(window_size=4, n_estimators=10),
        neural_predictor=PersistencePredictor(),
    )
    try:
        pipeline.fit(train)
    except ValueError as exc:
        assert "calibration_data is required" in str(exc)
    else:
        raise AssertionError("Neural-enabled pipeline should require held-out calibration data")
