
import numpy as np
import pandas as pd

from src.anomaly import AnomalyConfig, AnomalyDetectionPipeline
from src.anomaly.features import build_timestep_features, build_window_features


def sample_data(n=36):
    """Create two synthetic trajectories for anomaly-detection tests."""
    rows = []
    rng = np.random.default_rng(17)

    for cluster in ("A", "B"):
        for i in range(n):
            rows.append(
                {
                    "final_cluster_label": cluster,
                    "year": 2025 + (i // 52),
                    "week": (i % 52) + 1,
                    "z_0": float(np.sin(i / 5) + rng.normal(0, 0.03)),
                    "z_1": float(np.cos(i / 7) + rng.normal(0, 0.03)),
                    "z_2": float(i / n + rng.normal(0, 0.02)),
                }
            )

    return pd.DataFrame(rows)


def test_timestep_features_keep_rows_and_do_not_cross_clusters():
    frame = sample_data(8)

    features = build_timestep_features(frame)

    assert len(features) == len(frame)
    assert features["_source_row"].tolist() == list(range(len(frame)))
    assert np.allclose(
        features.groupby("trajectory_id", sort=False).head(1)["velocity_norm"],
        0.0,
    )


def test_window_features_do_not_cross_trajectory_boundaries():
    frame = sample_data(8)

    timesteps = build_timestep_features(frame)
    windows = build_window_features(timesteps, window_size=4, stride=2)

    assert set(windows["trajectory_id"]) == {"A", "B"}
    assert (windows["window_length"] == 4).all()


def test_pipeline_fit_score_and_round_trip(tmp_path):
    frame = sample_data(30)
    config = AnomalyConfig(window_size=8, window_stride=2, n_estimators=20)

    pipeline = AnomalyDetectionPipeline(config).fit(frame)
    points = pipeline.score_timesteps(frame)
    windows = pipeline.score_windows(frame)
    summary = pipeline.summarize_trajectories(points, windows)

    assert len(points) == len(frame)
    assert not windows.empty
    assert set(points["anomaly_label"]).issubset(
        {"Normal", "Warning", "High anomaly"}
    )
    assert set(summary["trajectory_id"]) == {"A", "B"}

    path = pipeline.save(tmp_path / "pipeline.joblib")
    loaded = AnomalyDetectionPipeline.load(path)
    reloaded_points = loaded.score_timesteps(frame)

    assert np.allclose(
        points["anomaly_score"], reloaded_points["anomaly_score"]
    )



def test_neural_calibration_split_requires_strictly_later_period():
    reference = sample_data(10)
    calibration = sample_data(10).assign(year=2026)

    AnomalyDetectionPipeline._validate_chronological_split(
        reference, calibration
    )


def test_neural_calibration_split_rejects_overlapping_periods():
    reference = sample_data(10)
    calibration = sample_data(10)

    try:
        AnomalyDetectionPipeline._validate_chronological_split(
            reference, calibration
        )
    except ValueError as exc:
        assert "strictly later" in str(exc)
    else:
        raise AssertionError("Expected overlapping calibration period to be rejected")


def test_neural_calibration_split_allows_unseen_clusters():
    """Unseen clusters are valid for statistical scoring; neural history may be unavailable."""
    reference = sample_data(10)
    calibration = sample_data(10).assign(year=2026)
    calibration["final_cluster_label"] = "UNSEEN"

    # Chronology is enforced, but cluster novelty alone is not leakage or invalid data.
    AnomalyDetectionPipeline._validate_chronological_split(
        reference, calibration
    )
