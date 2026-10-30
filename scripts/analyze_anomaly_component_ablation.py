#!/usr/bin/env python3
"""Component-level ablation using one already-scored fused validation run.

This diagnostic holds the fitted component scores fixed and compares:
  A) motion + Isolation Forest,
  B) neural residual rank only,
  C) motion + Isolation Forest + neural residual rank.

Warning/high thresholds are calibrated on validation scores only, then applied
unchanged to train/validation/test. Neural-only calibration excludes rows where
the neural residual is unavailable. This is a score-level ablation, not a
refit/retrain ablation: it isolates the contribution of score components under
the same fitted models and preprocessing basis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = {
    "statistical": ["motion_rank", "isolation_rank"],
    "neural_only": ["neural_rank"],
    "fused": ["motion_rank", "isolation_rank", "neural_rank"],
}
SPLITS = ("train", "validation", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory containing timestep_scores.csv and window_scores.csv from the fused run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <run-dir>/component_ablation).",
    )
    parser.add_argument("--warning-quantile", type=float, default=0.95)
    parser.add_argument("--high-quantile", type=float, default=0.99)
    return parser.parse_args()


def score_frame(frame: pd.DataFrame, level: str, warning_q: float, high_q: float):
    required = {"sequence_split", "motion_rank", "isolation_rank"}
    if level == "window":
        # Window records should already have split labels assigned by the validator.
        required.add("trajectory_id")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{level} CSV is missing required columns: {missing}")

    outputs = []
    threshold_report = {}
    for config, columns in CONFIGS.items():
        absent = [c for c in columns if c not in frame.columns]
        if absent:
            raise ValueError(f"{level} CSV lacks {config} components: {absent}")

        matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
        values = matrix.max(axis=1, skipna=False).to_numpy(dtype=float)
        valid_calibration = frame["sequence_split"].eq("validation").to_numpy()
        if config == "neural_only":
            if "neural_prediction_error" not in frame.columns:
                raise ValueError(
                    f"{level} CSV has no neural_prediction_error column; cannot "
                    "exclude unavailable neural residuals from calibration."
                )
            residual = pd.to_numeric(
                frame["neural_prediction_error"], errors="coerce"
            ).to_numpy(dtype=float)
            valid_calibration &= np.isfinite(residual)

        calibration = values[valid_calibration & np.isfinite(values)]
        if not len(calibration):
            raise ValueError(
                f"No finite validation scores available to calibrate {config} ({level})."
            )
        warning = float(np.quantile(calibration, warning_q))
        high = float(np.quantile(calibration, high_q))
        if high < warning:
            raise ValueError(f"Invalid threshold order for {config} ({level}).")

        labels = np.full(len(frame), "Normal", dtype=object)
        finite = np.isfinite(values)
        labels[finite & (values >= warning)] = "Warning"
        labels[finite & (values >= high)] = "High anomaly"

        result = frame.copy()
        result["ablation_config"] = config
        result["ablation_score"] = values
        result["ablation_label"] = labels
        result["calibration_warning_threshold"] = warning
        result["calibration_high_threshold"] = high
        outputs.append(result)
        threshold_report[config] = {
            "components": columns,
            "calibration_rows": int(len(calibration)),
            "warning_threshold": warning,
            "high_threshold": high,
            "calibration_warning_rate_or_higher": float(
                np.mean(calibration >= warning)
            ),
            "calibration_high_rate": float(np.mean(calibration >= high)),
        }
    return pd.concat(outputs, ignore_index=True), threshold_report


def summarize(scored: pd.DataFrame, level: str):
    records = []
    for (config, split), group in scored.groupby(
        ["ablation_config", "sequence_split"], dropna=False, sort=True
    ):
        labels = group["ablation_label"]
        records.append({
            "level": level,
            "config": config,
            "split": split,
            "n": int(len(group)),
            "mean_score": float(group["ablation_score"].mean()),
            "median_score": float(group["ablation_score"].median()),
            "warning_or_high_count": int(labels.ne("Normal").sum()),
            "warning_or_high_rate": float(labels.ne("Normal").mean()),
            "warning_count": int(labels.eq("Warning").sum()),
            "high_count": int(labels.eq("High anomaly").sum()),
        })
    return pd.DataFrame(records)


def cluster_summary(scored: pd.DataFrame, level: str):
    cluster_col = next(
        (c for c in ("trajectory_id", "cluster", "final_cluster_label") if c in scored.columns),
        None,
    )
    if cluster_col is None:
        raise ValueError(f"{level} CSV needs trajectory_id or cluster for per-cluster diagnostics.")
    grouped = scored.groupby(
        ["ablation_config", "sequence_split", cluster_col], dropna=False, sort=True
    )
    out = grouped.agg(
        n=("ablation_score", "size"),
        mean_score=("ablation_score", "mean"),
        median_score=("ablation_score", "median"),
        flagged_count=("ablation_label", lambda s: int(s.ne("Normal").sum())),
        high_count=("ablation_label", lambda s: int(s.eq("High anomaly").sum())),
    ).reset_index().rename(columns={cluster_col: "trajectory_or_cluster"})
    out["flagged_rate"] = out["flagged_count"] / out["n"]
    out.insert(0, "level", level)
    return out


def main():
    args = parse_args()
    if not (0 < args.warning_quantile < 1 and 0 < args.high_quantile < 1):
        raise SystemExit("Quantiles must be strictly between 0 and 1.")
    if args.warning_quantile > args.high_quantile:
        raise SystemExit("--warning-quantile must be <= --high-quantile.")

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "component_ablation")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_clusters = []
    threshold_report = {}
    input_files = {}
    for level in ("timestep", "window"):
        path = run_dir / f"{level}_scores.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
        frame = pd.read_csv(path)
        scored, thresholds = score_frame(
            frame, level, args.warning_quantile, args.high_quantile
        )
        scored.to_csv(output_dir / f"{level}_component_ablation.csv", index=False)
        all_summaries.append(summarize(scored, level))
        all_clusters.append(cluster_summary(scored, level))
        threshold_report[level] = thresholds
        input_files[level] = path.name

    summary = pd.concat(all_summaries, ignore_index=True)
    clusters = pd.concat(all_clusters, ignore_index=True)
    summary.to_csv(output_dir / "component_ablation_summary.csv", index=False)
    clusters.to_csv(output_dir / "component_ablation_by_cluster.csv", index=False)

    report = {
        "method": "fixed-score component ablation; no model refitting",
        "input_run_directory": run_dir.as_posix(),
        "warning_quantile": args.warning_quantile,
        "high_quantile": args.high_quantile,
        "inputs": input_files,
        "configurations": {
            "statistical": "max(motion_rank, isolation_rank)",
            "neural_only": "neural_rank; validation calibration excludes rows with unavailable neural residuals",
            "fused": "max(motion_rank, isolation_rank, neural_rank)",
        },
        "thresholds": threshold_report,
        "interpretation_limits": [
            "This is a fixed-score ablation using component scores from the same fused run; models and preprocessing are not refit for each configuration.",
            "Warning/high thresholds are estimated on validation scores only and applied unchanged to all splits.",
            "Neural-only scores are calibrated only where neural_prediction_error is finite; rows without residuals retain the pipeline's neutral neural rank.",
            "Overlapping trajectory windows are correlated and must not be treated as independent observations.",
            "Without ground-truth anomaly labels, flagged rates measure detector behavior, not accuracy or precision.",
        ],
    }
    with (output_dir / "component_ablation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Component ablation complete: {output_dir}")
    print(summary[summary["split"].isin(["validation", "test"])].to_string(index=False))


if __name__ == "__main__":
    main()
