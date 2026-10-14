from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure imports resolve when invoked as: python scripts/validate_anomaly_real_data.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CLUSTER_COL, EMB_PREFIX, PCA_PREFIX, SEED
from src.anomaly import AnomalyConfig, AnomalyDetectionPipeline
from src.neural_prediction.inference import NeuralTrajectoryPredictor
from src.trajectory_prediction.utils.pca_preprocessing import add_pca_features
from src.trajectory_prediction.utils.preprocessing import (
    load_centroid_weekly_embeddings,
    preprocess_centroids,
    split_train_validation_test_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KYRE anomaly detection on the real centroid trajectory dataset."
    )
    parser.add_argument(
        "--project-name",
        default="trajectory_prediction_data_hitachi",
        help="Project folder under projects/ containing cluster_embeddings/centroids_with_usage_labels/.",
    )
    parser.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    parser.add_argument("--pca-dims", type=int, default=32)
    parser.add_argument("--min-weeks", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--validation-weeks", type=int, default=12)
    parser.add_argument("--test-weeks", type=int, default=24)
    parser.add_argument("--max-staleness-weeks", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--spike-magnitude", type=float, default=8.0)
    parser.add_argument("--max-injection-trajectories", type=int, default=5)
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        default=None,
        help="Optional trained neural_prediction .pth checkpoint. When supplied, validation data calibrates the neural residual signal.",
    )
    parser.add_argument(
        "--neural-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for neural inference when --neural-checkpoint is supplied.",
    )
    return parser.parse_args()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def label_distribution(frame: pd.DataFrame, split: str) -> dict:
    subset = frame[frame["sequence_split"] == split]
    if subset.empty:
        return {
            "rows": 0,
            "mean_anomaly_score": None,
            "warning_or_high_count": 0,
            "warning_or_high_rate": None,
            "label_counts": {},
        }
    counts = subset["anomaly_label"].value_counts().to_dict()
    flagged = int((subset["anomaly_label"] != "Normal").sum())
    return {
        "rows": int(len(subset)),
        "mean_anomaly_score": float(subset["anomaly_score"].mean()),
        "median_anomaly_score": float(subset["anomaly_score"].median()),
        "warning_or_high_count": flagged,
        "warning_or_high_rate": float(flagged / len(subset)),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }


def main() -> int:
    args = parse_args()
    project_dir = args.projects_root / args.project_name
    centroid_dir = project_dir / "cluster_embeddings" / "centroids_with_usage_labels"
    output_dir = project_dir / "anomaly_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not centroid_dir.is_dir():
        raise FileNotFoundError(
            f"Centroid directory not found: {centroid_dir}\n"
            "Pass --project-name with the folder under projects/ that contains "
            "cluster_embeddings/centroids_with_usage_labels/."
        )

    parquet_files = sorted(centroid_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {centroid_dir}")

    print(f"[1/7] Loading centroid data from {centroid_dir}")
    raw = load_centroid_weekly_embeddings(centroid_dir)
    required = {CLUSTER_COL, "year", "week"}
    missing = sorted(required.difference(raw.columns))
    embedding_cols = sorted(
        [column for column in raw.columns if column.startswith(EMB_PREFIX)],
        key=lambda column: (
            0, int(column[len(EMB_PREFIX):])
        ) if column[len(EMB_PREFIX):].isdigit() else (1, column),
    )
    if missing:
        raise ValueError(f"Raw centroid data is missing required columns: {missing}")
    if not embedding_cols:
        raise ValueError(f"No embedding columns with prefix {EMB_PREFIX!r} were found.")
    if "n_users" not in raw.columns:
        print("WARNING: 'n_users' is absent; preprocessing will skip user-weight gap filling.")

    print(f"  Parquet files: {len(parquet_files)}")
    print(f"  Raw rows after existing loader's outlier-cluster filter: {len(raw):,}")
    print(f"  Embedding dimensions: {len(embedding_cols)}")
    print(f"  Raw time range: year={raw['year'].min()}..{raw['year'].max()}")

    print("[2/7] Reusing the neural_prediction centroid preprocessing")
    clean, cluster_summary = preprocess_centroids(
        raw,
        min_weeks=args.min_weeks,
        max_gap=args.max_gap,
        max_staleness_weeks=args.max_staleness_weeks,
        emb_prefix=EMB_PREFIX,
        cluster_col=CLUSTER_COL,
        user_weight_col="n_users",
    )
    if clean.empty:
        raise ValueError(
            "No trajectories survived preprocessing. Inspect cluster_summary and adjust "
            "--min-weeks or --max-staleness-weeks only if appropriate for the dataset."
        )

    print("[3/7] Reusing the neural_prediction chronological train/validation/test split")
    train, validation, test, split_summary = split_train_validation_test_sequences(
        clean,
        validation_weeks=args.validation_weeks,
        test_weeks=args.test_weeks,
        min_train_weeks=args.min_weeks,
        cluster_col=CLUSTER_COL,
        project_name=None,
        is_save_split_csv=False,
    )
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "At least one split is empty. Check sequence_split_summary and whether enough "
            "clusters have min_train_weeks + validation_weeks + test_weeks observations."
        )

    print("[4/7] Fitting scaler/PCA on train only; transforming validation and test")
    train_m, pca, scaler, variance = add_pca_features(
        train, n_dims=args.pca_dims, input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX, seed=SEED,
    )
    validation_m, _, _, _ = add_pca_features(
        validation, n_dims=args.pca_dims, input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX, seed=SEED, pca_model=pca, scaler_model=scaler,
    )
    test_m, _, _, _ = add_pca_features(
        test, n_dims=args.pca_dims, input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX, seed=SEED, pca_model=pca, scaler_model=scaler,
    )

    # The current neural_prediction prepare_data() references an undefined clean_m
    # in its return dict. This script avoids that path and explicitly combines the
    # split outputs after applying the train-fitted PCA/scaler to each split.
    all_model = pd.concat([train_m, validation_m, test_m], ignore_index=True)
    all_model = all_model.sort_values(
        [CLUSTER_COL, "year", "week"], kind="stable"
    ).reset_index(drop=True)
    if "sequence_split" not in all_model.columns:
        raise RuntimeError("Temporal split labels were not preserved by preprocessing.")

    config = AnomalyConfig(
        pca_prefix=PCA_PREFIX,
        cluster_col=CLUSTER_COL,
        trajectory_col=None,
        year_col="year",
        week_col="week",
        window_size=args.window_size,
        window_stride=args.window_stride,
        n_estimators=args.estimators,
        random_state=SEED,
    )

    neural_predictor = None
    checkpoint_path = None
    if args.neural_checkpoint is not None:
        checkpoint_path = args.neural_checkpoint
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Neural checkpoint not found: {checkpoint_path}")
        print(f"Loading neural prediction checkpoint: {checkpoint_path}")
        neural_predictor = NeuralTrajectoryPredictor.load(
            checkpoint_path, device=args.neural_device
        )
        checkpoint_dims = getattr(
            getattr(neural_predictor.model, "config", None), "embedding_dim", None
        )
        if checkpoint_dims != args.pca_dims:
            raise ValueError(
                f"Checkpoint embedding_dim={checkpoint_dims} does not match "
                f"--pca-dims={args.pca_dims}. Use the PCA dimensionality and fitted "
                "PCA/scaler basis used when this checkpoint was trained."
            )

    print("[5/7] Fitting anomaly pipeline on train rows; validation calibrates neural fusion when enabled")
    pipeline = AnomalyDetectionPipeline(
        config, neural_predictor=neural_predictor
    ).fit(
        train_m,
        calibration_data=validation_m if neural_predictor is not None else None,
    )

    print("[6/7] Scoring trajectories while preserving causal temporal context")
    # Scoring the combined chronological frame preserves motion-feature context.
    # Neural residuals are generated one step ahead: for this full-frame pass the
    # signal builds history incrementally, without passing future rows as context.
    empty_history = all_model.iloc[0:0].copy()
    timestep_scores = pipeline.score_timesteps(
        all_model, history_data=empty_history
    )
    timestep_splits = all_model["sequence_split"].to_numpy()
    timestep_scores["sequence_split"] = timestep_splits[
        timestep_scores["_source_row"].astype(int).to_numpy()
    ]

    window_scores = pipeline.score_windows(
        all_model, history_data=empty_history
    )
    split_labels = all_model["sequence_split"].to_numpy()
    window_scores["start_split"] = split_labels[
        window_scores["window_start_row"].astype(int).to_numpy()
    ]
    window_scores["end_split"] = split_labels[
        window_scores["window_end_row"].astype(int).to_numpy()
    ]
    window_scores["sequence_split"] = np.where(
        window_scores["start_split"] == window_scores["end_split"],
        window_scores["start_split"],
        "cross_split_context",
    )

    trajectory_summaries = []
    for split in ("validation", "test"):
        points = timestep_scores[timestep_scores["sequence_split"] == split]
        windows = window_scores[window_scores["sequence_split"] == split]
        if not points.empty:
            summary = pipeline.summarize_trajectories(points, windows)
            summary.insert(0, "sequence_split", split)
            trajectory_summaries.append(summary)

    summary_df = (
        pd.concat(trajectory_summaries, ignore_index=True)
        if trajectory_summaries else pd.DataFrame()
    )

    # Controlled spike sensitivity check on held-out test data. This is not a
    # substitute for labeled ground truth and does not affect model fitting.
    print("[7/7] Running controlled spike-injection sensitivity checks")
    test_for_injection = test_m.sort_values(
        [CLUSTER_COL, "year", "week"], kind="stable"
    ).reset_index(drop=True)
    prior_test_history = pd.concat([train_m, validation_m], ignore_index=True)
    baseline_test_scores = pipeline.score_timesteps(
        test_for_injection, history_data=prior_test_history
    )
    z_cols = [column for column in test_for_injection.columns if column.startswith(PCA_PREFIX)]
    injection_records = []
    if z_cols and args.max_injection_trajectories > 0:
        chosen = []
        for cluster, group in test_for_injection.groupby(CLUSTER_COL, sort=False):
            if len(group) < 3:
                continue
            chosen.append((cluster, int(group.index[len(group) // 2])))
            if len(chosen) >= args.max_injection_trajectories:
                break

        for cluster, row_idx in chosen:
            column = z_cols[0]
            scale = float(train_m[column].std(ddof=0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            injected = test_for_injection.copy()
            original_value = float(injected.at[row_idx, column])
            injected.at[row_idx, column] = original_value + args.spike_magnitude * scale
            injected_scores = pipeline.score_timesteps(
                injected, history_data=prior_test_history
            )
            baseline_row = baseline_test_scores.loc[
                baseline_test_scores["_source_row"] == row_idx
            ].iloc[0]
            injected_row = injected_scores.loc[
                injected_scores["_source_row"] == row_idx
            ].iloc[0]
            injection_records.append({
                "trajectory_id": str(cluster),
                "row_index": row_idx,
                "injected_column": column,
                "spike_magnitude_in_train_std": float(args.spike_magnitude),
                "baseline_score": float(baseline_row["anomaly_score"]),
                "injected_score": float(injected_row["anomaly_score"]),
                "score_increased": bool(
                    injected_row["anomaly_score"] > baseline_row["anomaly_score"]
                ),
                "injected_label": str(injected_row["anomaly_label"]),
            })

    injection_df = pd.DataFrame(injection_records)
    pipeline.save(output_dir / "anomaly_pipeline.joblib")
    cluster_summary.to_csv(output_dir / "cluster_preprocessing_summary.csv", index=False)
    split_summary.to_csv(output_dir / "sequence_split_summary.csv", index=False)
    timestep_scores.to_csv(output_dir / "timestep_scores.csv", index=False)
    window_scores.to_csv(output_dir / "window_scores.csv", index=False)
    summary_df.to_csv(output_dir / "trajectory_summary.csv", index=False)
    injection_df.to_csv(output_dir / "spike_injection_sensitivity.csv", index=False)
    variance.to_csv(output_dir / "pca_variance.csv", index=False)

    report = {
        "project_name": args.project_name,
        "centroid_directory": str(centroid_dir),
        "parquet_files": [str(path) for path in parquet_files],
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "embedding_dimensions": int(len(embedding_cols)),
        "pca_dimensions": int(args.pca_dims),
        "position_columns": z_cols,
        "cluster_count_raw": int(raw[CLUSTER_COL].nunique()),
        "cluster_count_kept": int(clean[CLUSTER_COL].nunique()),
        "split_rows": {
            "train": int(len(train_m)),
            "validation": int(len(validation_m)),
            "test": int(len(test_m)),
        },
        "split_clusters": {
            "train": int(train_m[CLUSTER_COL].nunique()),
            "validation": int(validation_m[CLUSTER_COL].nunique()),
            "test": int(test_m[CLUSTER_COL].nunique()),
        },
        "pca_explained_variance_ratio_sum": (
            float(pca.explained_variance_ratio_.sum())
            if hasattr(pca, "explained_variance_ratio_") else None
        ),
        "neural_prediction_signal": {
            "enabled": neural_predictor is not None,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "device": args.neural_device if neural_predictor is not None else None,
            "lookback_weeks": (
                int(neural_predictor.window_config.lookback)
                if neural_predictor is not None else None
            ),
            "forecast_horizon_weeks": (
                int(neural_predictor.window_config.horizon)
                if neural_predictor is not None else None
            ),
            "threshold_calibration_split": "validation" if neural_predictor is not None else "train",
        },
        "anomaly_label_distribution": {
            split: label_distribution(timestep_scores, split)
            for split in ("train", "validation", "test")
        },
        "test_window_label_counts": {
            str(k): int(v) for k, v in
            window_scores.loc[window_scores["sequence_split"] == "test", "anomaly_label"]
            .value_counts().to_dict().items()
        },
        "injection_checks": {
            "count": int(len(injection_df)),
            "score_increased_count": int(injection_df["score_increased"].sum())
            if not injection_df.empty else 0,
            "score_increased_rate": float(injection_df["score_increased"].mean())
            if not injection_df.empty else None,
        },
        "limitations": [
            "Anomaly scores are unsupervised rankings, not probabilities or ground-truth accuracy.",
            "Spike injection tests sensitivity to artificial PCA-space outliers only.",
            "When neural prediction scoring is enabled, its residual reference and fused thresholds are calibrated on held-out validation data.",
            "Rows without enough preceding history have no neural residual; the neural rank is neutral for those rows.",
            "Flagged real trajectories require domain review before being treated as genuine anomalies.",
        ],
        "output_directory": str(output_dir),
    }
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)

    print("\nValidation run complete.")
    print(f"Outputs: {output_dir}")
    print(f"Rows: train={len(train_m):,}, validation={len(validation_m):,}, test={len(test_m):,}")
    print("Validation/test anomaly-label distributions:")
    for split in ("validation", "test"):
        print(f"  {split}: {report['anomaly_label_distribution'][split]}")
    print(f"Spike sensitivity: {report['injection_checks']}")
    print("Read validation_report.json before interpreting anomaly rates as model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
