# KYRE Anomaly Detection

Standalone unsupervised anomaly-detection module for KYRE. This package is kept separate from `src/neural_prediction/` and does not modify the neural prediction pipeline.

## Scope

- Timestep-level motion anomaly scoring using robust median/MAD statistics.
- Timestep-level Isolation Forest scoring.
- Fixed-length trajectory-window features and a separate window-level Isolation Forest.
- Reference-set quantile thresholds and trajectory summaries.
- Save/load support and helpers for controlled synthetic-spike validation.

The module does not require anomaly labels. Scores describe statistical unusualness, not confirmed faults or harmful behavior.

## Basic usage

Fit the pipeline on an explicitly selected reference period:

    from pathlib import Path
    from src.anomaly import AnomalyConfig, AnomalyDetectionPipeline

    config = AnomalyConfig(
        pca_prefix='z_',
        cluster_col='final_cluster_label',
        trajectory_col=None,
        window_size=12,
        window_stride=1,
        random_state=42,
    )
    pipeline = AnomalyDetectionPipeline(config).fit(reference_df)
    pipeline.save(Path('projects') / project_name / 'anomaly_models' / 'anomaly_pipeline.joblib')

Score a separate dataset without refitting:

    timestep_results = pipeline.score_timesteps(evaluation_df)
    window_results = pipeline.score_windows(evaluation_df)
    trajectory_summary = pipeline.summarize_trajectories(timestep_results, window_results)

## Input expectations

- Numeric PCA/position columns matching pca_prefix (default: z_).
- A trajectory identifier, defaulting to final_cluster_label.
- Chronological year and week columns.

Rows are sorted within trajectories before finite differences are calculated. Derivatives never cross trajectory boundaries. Non-finite PCA values are rejected explicitly rather than silently imputed.

## Interpretation

- motion_score: maximum robust deviation across velocity, acceleration, and jerk magnitudes.
- isolation_score: negative Isolation Forest score_samples; larger values are more unusual relative to the fitted model.
- anomaly_score: maximum of empirical reference ranks for motion and Isolation Forest signals, ranging from 0 to 1.
- anomaly_label: Normal, Warning, or High anomaly, based on reference-score quantiles.

These scores are not probabilities. With no ground-truth labels, use injected-anomaly tests, stability checks, temporal holdout evaluation, and expert review. Fit preprocessing and thresholds on the reference period only.

## Integration status

This package is intentionally independent of the neural prediction package and is not wired into the Streamlit app yet. Validate against KYRE's actual anomaly-detection data schema before adding any dashboard integration.
