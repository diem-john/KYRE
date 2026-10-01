import numpy as np
import pandas as pd
import ast
import json
from pathlib import Path

from src.trajectory_prediction.utils.preprocessing import embedding_cols, sort_time
from src.behavioral_anomaly_detector.utils import build_decoded_behavior_column
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def to_original_space(Y, pca = None, scaler=None):
    """
    Convert PCA reduced embeddings back to original space embedding input space.
    """
    if pca is not None:
        Y = pca.inverse_transform(Y)
    if scaler is not None:
        Y = scaler.inverse_transform(Y)
    return Y

def _safe_cosine(a, b):
    """
    Cosine similarity between two vectors.

    Returns NaN if either vector has zero magnitude.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return np.nan

    return np.dot(a, b) / denom

def _weighted_mean(values, weights):
    """
    Weighted mean that safely handles missing values and zero weights.
    """
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce")

    valid = values.notna() & weights.notna() & (weights > 0)

    if not valid.any():
        return np.nan

    return np.average(values[valid], weights=weights[valid])


def _get_target_n_users(group, target_idx, weight_col):
    """
    Get n_users for the actual target week used in backtesting.

    Falls back to NaN if the column does not exist.
    """
    if weight_col not in group.columns:
        return np.nan

    if target_idx >= len(group):
        return np.nan

    return pd.to_numeric(
        pd.Series([group.iloc[target_idx][weight_col]]),
        errors="coerce",
    ).iloc[0]

def check_historical_and_prediction_centroid_overlap(predicted_behavior_change_df, historical_behavior_change_df):
    """
    Checks if there's overlap with the historical and predicted centroid positions.
    If there's overlap, evaluation of decoded elapsed time accurary and feature used matching can proceed.
    """
    
    dup_check_col = ["year","week","final_cluster_label"]
    
    return predicted_behavior_change_df[dup_check_col].set_index(dup_check_col).index.isin(
            historical_behavior_change_df[dup_check_col].set_index(dup_check_col).index
            ).any()
    
def median_historical_displacement(df, emb_prefix, cluster_col):
    """
    Median week-to-week centroid movement in the original embedding space.

    This is used as the RMSE cutoff for reliable horizons:
    a forecast should have lower RMSE than the typical historical centroid move.
    """
    emb_cols = embedding_cols(df, emb_prefix)
    displacements = []

    for _, group in df.groupby(cluster_col, sort=False):
        group = sort_time(group)
        X = group[emb_cols].to_numpy(dtype=float)

        if len(X) < 2:
            continue

        displacements.extend(np.linalg.norm(np.diff(X, axis=0), axis=1))

    if len(displacements) == 0:
        return np.nan

    return float(np.nanmedian(displacements))

def compute_metrics(y_true, y_pred, y_reference=None):
    """
    Computes position-level metrics and direction-of-change metrics.

    Position-level metrics:
    - mse
    - rmse
    - cosine_similarity

    Direction/change metrics:
    - displacement_cosine_similarity
    - displacement_magnitude_error
    - persistence_rmse
    - rmse_improvement_vs_persistence

    If y_reference is provided, the direction/change metrics compare:

        y_true - y_reference
        y_pred - y_reference

    This tells us if the predicted centroid moved in the same direction,
    by the right amount, and whether it beats the persistence baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    cosine = cosine_similarity(y_true, y_pred).diagonal().mean()

    displacement_cosine = np.nan
    displacement_magnitude_error = np.nan
    persistence_rmse = np.nan
    rmse_improvement_vs_persistence = np.nan

    if y_reference is not None:
        y_reference = np.asarray(y_reference, dtype=float)

        true_delta = y_true - y_reference
        pred_delta = y_pred - y_reference

        displacement_cosines = [
            _safe_cosine(true_delta[i], pred_delta[i])
            for i in range(len(true_delta))
        ]

        true_magnitudes = np.linalg.norm(true_delta, axis=1)
        pred_magnitudes = np.linalg.norm(pred_delta, axis=1)

        displacement_cosine = np.nanmean(displacement_cosines)
        displacement_magnitude_error = np.nanmean(
            np.abs(true_magnitudes - pred_magnitudes)
        )

        persistence_mse = mean_squared_error(y_true, y_reference)
        persistence_rmse = np.sqrt(persistence_mse)

        if persistence_rmse > 0:
            rmse_improvement_vs_persistence = 1 - (rmse / persistence_rmse)

    return {
        "mse": mse,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "displacement_cosine_similarity": displacement_cosine,
        "displacement_magnitude_error": displacement_magnitude_error,
        "persistence_rmse": persistence_rmse,
        "rmse_improvement_vs_persistence": rmse_improvement_vs_persistence,
    }

def iter_cluster_arrays(df, emb_prefix, cluster_col):
    """
    The function is an iterator that creates one time series per cluster.
    """
    emb_cols = embedding_cols(df, emb_prefix)
    for cluster, group in df.groupby(cluster_col, sort=False):
        group = sort_time(group)
        yield cluster, group, group[emb_cols].to_numpy(dtype=float), emb_cols

def print_prediction_fit_metrics(
    logger,
    reliable_horizon_summary_df
):
    metrics_to_print = [
    "mean_rmse_original",
    "mean_cosine_similarity_original",
    "mean_displacement_cosine_similarity_original",
    "mean_displacement_magnitude_error_original",
    ]

    labels = {
        "mean_rmse_original": "Mean RMSE",
        "mean_cosine_similarity_original": "Mean Cosine Similarity",
        "mean_displacement_cosine_similarity_original": "Mean Displacement Cosine Similarity",
        "mean_displacement_magnitude_error_original": "Mean Displacement Magnitude Error",
    }
    
    median_historical_displacement = reliable_horizon_summary_df['median_historical_displacement'].unique()[0]
    
    logger.log(f"Reference Median Historical Displacement: {median_historical_displacement}", level="STATUS")

    for split in ["test_holdout", "test_backtest", "validation_backtest", "train_backtest"]:
        logger.log(f"=== Eval Split: {split} ===", level="STATUS")

        row = reliable_horizon_summary_df[reliable_horizon_summary_df["eval_split"] == split]

        if row.empty:
            logger.log("No rows found.")
        else:
            for metric in metrics_to_print:
                value = row[metric].iloc[0]
                logger.log(f"{labels[metric]}: {value}", level="STATUS")
                
    logger.log(f"Detailed metric results are saved in the approach's 'metric' folder.", level="INFO")
    
def _parse_decoded_behavior(value):
    """
    Convert decoded_behavior into:
        {feature_name: elapsed_time}

    used_prob is intentionally ignored.
    """
    if value is None:
        return {}

    if isinstance(value, float) and np.isnan(value):
        return {}

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return {}

        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}

    if not isinstance(value, list):
        return {}

    decoded = {}

    for item in value:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature")
        elapsed = item.get(
            "elapsed",
            item.get("elapsed_time", 0.0),
        )

        if feature is None:
            continue

        feature = str(feature).strip()

        if not feature:
            continue

        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            elapsed = 0.0

        if not np.isfinite(elapsed):
            elapsed = 0.0

        # Sum values when the same feature appears more than once.
        decoded[feature] = (
            decoded.get(feature, 0.0)
            + max(elapsed, 0.0)
        )

    return decoded


def _add_prediction_horizon(
    predicted_df,
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    horizon_col="horizon",
):
    """
    Add horizon = 1, 2, 3, ... per cluster based on chronological order.

    If the horizon column already exists, it is retained.
    """
    df = predicted_df.copy()

    if horizon_col in df.columns:
        return df

    required_cols = [
        cluster_col,
        year_col,
        week_col,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Cannot calculate horizon because predicted_df "
            f"is missing columns: {missing_cols}"
        )

    df["_prediction_week_date"] = pd.to_datetime(
        df[year_col].astype(str)
        + "-W"
        + df[week_col]
        .astype(int)
        .astype(str)
        .str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    if df["_prediction_week_date"].isna().any():
        invalid_rows = df.loc[
            df["_prediction_week_date"].isna(),
            [cluster_col, year_col, week_col],
        ].head()

        raise ValueError(
            "Some year-week values could not be converted "
            f"to ISO week dates:\n{invalid_rows}"
        )

    df = df.sort_values(
        [cluster_col, "_prediction_week_date"]
    ).copy()

    df[horizon_col] = (
        df.groupby(cluster_col)
        .cumcount()
        .add(1)
    )

    return (
        df.drop(columns="_prediction_week_date")
        .reset_index(drop=True)
    )


def calculate_decoded_alignment_metrics(
    historical_decoded_behavior,
    predicted_decoded_behavior,
):
    """
    Compare one predicted decoded centroid with the corresponding
    historical decoded centroid.

    Missing features receive elapsed time = 0.
    """
    historical = _parse_decoded_behavior(
        historical_decoded_behavior
    )
    predicted = _parse_decoded_behavior(
        predicted_decoded_behavior
    )

    historical_features = set(historical)
    predicted_features = set(predicted)

    matched_features = (
        historical_features & predicted_features
    )
    union_features = (
        historical_features | predicted_features
    )

    # ----------------------------------------------------------
    # Feature alignment
    # ----------------------------------------------------------
    feature_precision = (
        len(matched_features) / len(predicted_features)
        if predicted_features
        else float(not historical_features)
    )

    feature_recall = (
        len(matched_features) / len(historical_features)
        if historical_features
        else float(not predicted_features)
    )

    feature_f1 = (
        2 * feature_precision * feature_recall
        / (feature_precision + feature_recall)
        if feature_precision + feature_recall > 0
        else 0.0
    )

    feature_jaccard = (
        len(matched_features) / len(union_features)
        if union_features
        else 1.0
    )

    # ----------------------------------------------------------
    # Elapsed-time alignment
    # ----------------------------------------------------------
    if union_features:
        ordered_features = sorted(union_features)

        historical_elapsed = np.array(
            [
                historical.get(feature, 0.0)
                for feature in ordered_features
            ],
            dtype=float,
        )

        predicted_elapsed = np.array(
            [
                predicted.get(feature, 0.0)
                for feature in ordered_features
            ],
            dtype=float,
        )

        elapsed_absolute_error = np.abs(
            predicted_elapsed - historical_elapsed
        )

        elapsed_mae = float(
            elapsed_absolute_error.mean()
        )

        historical_total = float(
            historical_elapsed.sum()
        )
        predicted_total = float(
            predicted_elapsed.sum()
        )

        mean_historical_elapsed = float(
            historical_elapsed.mean()
        )

        elapsed_nmae = (
            elapsed_mae / mean_historical_elapsed
            if mean_historical_elapsed > 0
            else float(elapsed_mae > 0)
        )

        total_elapsed_error = abs(
            predicted_total - historical_total
        )

        total_elapsed_percentage_error = (
            total_elapsed_error / historical_total
            if historical_total > 0
            else float(total_elapsed_error > 0)
        )

        # Compare the proportion of elapsed time assigned
        # to each decoded feature.
        if historical_total > 0 and predicted_total > 0:
            historical_share = (
                historical_elapsed / historical_total
            )
            predicted_share = (
                predicted_elapsed / predicted_total
            )

            elapsed_distribution_similarity = (
                1.0
                - np.abs(
                    historical_share - predicted_share
                ).sum()
                / 2.0
            )
        else:
            elapsed_distribution_similarity = float(
                historical_total == predicted_total
            )

    else:
        historical_total = 0.0
        predicted_total = 0.0
        elapsed_mae = 0.0
        elapsed_nmae = 0.0
        total_elapsed_error = 0.0
        total_elapsed_percentage_error = 0.0
        elapsed_distribution_similarity = 1.0

    return {
        "n_historical_features": len(historical_features),
        "n_predicted_features": len(predicted_features),
        "n_matched_features": len(matched_features),

        "feature_precision": feature_precision,
        "feature_recall": feature_recall,
        "feature_f1": feature_f1,
        "feature_jaccard": feature_jaccard,

        "historical_total_elapsed": historical_total,
        "predicted_total_elapsed": predicted_total,
        "elapsed_mae": elapsed_mae,
        "elapsed_nmae": elapsed_nmae,
        "elapsed_distribution_similarity":
            elapsed_distribution_similarity,
        "total_elapsed_error": total_elapsed_error,
        "total_elapsed_percentage_error":
            total_elapsed_percentage_error,
    }


def evaluate_decoded_predictions_per_horizon(
    historical_df,
    predicted_df,
    match_cols=(
        "year",
        "week",
        "final_cluster_label",
    ),
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    horizon_col="horizon",
    decoded_col="decoded_behavior",
):
    """
    Evaluate decoded centroid prediction alignment.

    Returns
    -------
    row_metrics_df
        Metrics for every matched cluster-week prediction.

    horizon_summary_df
        Mean and median metrics for each forecast horizon.

    overall_summary_df
        Overall descriptive summary across all predictions.
    """
    historical_df = historical_df.copy()

    predicted_df = _add_prediction_horizon(
        predicted_df=predicted_df,
        cluster_col=cluster_col,
        year_col=year_col,
        week_col=week_col,
        horizon_col=horizon_col,
    )

    historical_required = (
        list(match_cols) + [decoded_col]
    )
    predicted_required = (
        list(match_cols)
        + [horizon_col, decoded_col]
    )

    missing_historical = [
        col for col in historical_required
        if col not in historical_df.columns
    ]
    missing_predicted = [
        col for col in predicted_required
        if col not in predicted_df.columns
    ]

    if missing_historical:
        raise ValueError(
            "historical_df is missing columns: "
            f"{missing_historical}"
        )

    if missing_predicted:
        raise ValueError(
            "predicted_df is missing columns: "
            f"{missing_predicted}"
        )

    historical = (
        historical_df[historical_required]
        .rename(
            columns={
                decoded_col:
                    "historical_decoded_behavior"
            }
        )
        .copy()
    )

    predicted = (
        predicted_df[predicted_required]
        .rename(
            columns={
                decoded_col:
                    "predicted_decoded_behavior"
            }
        )
        .copy()
    )

    if historical.duplicated(
        subset=list(match_cols)
    ).any():
        duplicate_rows = historical.loc[
            historical.duplicated(
                subset=list(match_cols),
                keep=False,
            ),
            list(match_cols),
        ].head()

        raise ValueError(
            "historical_df contains duplicate rows for "
            f"the matching columns:\n{duplicate_rows}"
        )

    matched_df = predicted.merge(
        historical,
        on=list(match_cols),
        how="inner",
        validate="many_to_one",
    )

    metric_rows = []

    for row in matched_df.to_dict("records"):
        metrics = calculate_decoded_alignment_metrics(
            historical_decoded_behavior=row[
                "historical_decoded_behavior"
            ],
            predicted_decoded_behavior=row[
                "predicted_decoded_behavior"
            ],
        )

        metric_rows.append({
            **{
                col: row[col]
                for col in match_cols
            },
            horizon_col: row[horizon_col],
            **metrics,
        })

    row_metrics_df = pd.DataFrame(metric_rows)

    if row_metrics_df.empty:
        return (
            row_metrics_df,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    metric_cols = [
        "feature_precision",
        "feature_recall",
        "feature_f1",
        "feature_jaccard",
        "elapsed_mae",
        "elapsed_nmae",
        "elapsed_distribution_similarity",
        "total_elapsed_error",
        "total_elapsed_percentage_error",
    ]

    # ----------------------------------------------------------
    # Per-horizon summary
    # ----------------------------------------------------------
    horizon_summary_df = (
        row_metrics_df
        .groupby(horizon_col)[metric_cols]
        .agg(["mean", "median"])
    )

    horizon_summary_df.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in horizon_summary_df.columns
    ]

    horizon_summary_df = (
        horizon_summary_df
        .reset_index()
    )

    horizon_counts = (
        row_metrics_df
        .groupby(horizon_col)
        .size()
        .rename("n_evaluated_predictions")
        .reset_index()
    )

    horizon_summary_df = (
        horizon_summary_df
        .merge(
            horizon_counts,
            on=horizon_col,
            how="left",
        )
        .sort_values(horizon_col)
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Overall summary
    # ----------------------------------------------------------
    overall_summary_df = (
        row_metrics_df[metric_cols]
        .agg([
            "mean",
            "median",
            "std",
            "min",
            "max",
        ])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    overall_summary_df.insert(
        1,
        "n_evaluated_predictions",
        len(row_metrics_df),
    )

    return (
        row_metrics_df,
        horizon_summary_df,
        overall_summary_df,
    )
    
def evaluate_decoded_behavior_similarity(
    logger,
    historical_behavior_change_df,
    predicted_behavior_change_df,
    decoded_weekly_centroid_emb,
    project_name=None,
    model_name=None,
):
    """
    This function evaluates how close the decoded behavior of the predicted centroid positions are relative to the decoded
    historical centroid positions the model is predicting.
    
    The following metric CSVs are generated and saved:
        `decoded_row_metrics_df`: The cluster-week level metric results.
        `decoded_horizon_metrics_df`: The aggregated horizon-level metric results.
        `compact_overall_summary_df`: The overall decoded behavior metric result.
    """
    logger.log("Evaluating decoded behavior...", level="STATUS")
    
    historical_behavior_change_df, predicted_behavior_change_df = build_decoded_behavior_column(
        historical_behavior_change_df,
        predicted_behavior_change_df,
        decoded_weekly_centroid_emb
    )
    
    decoded_row_metrics_df, decoded_horizon_metrics_df, decoded_overall_summary_df = evaluate_decoded_predictions_per_horizon(
        historical_df=historical_behavior_change_df,
        predicted_df=predicted_behavior_change_df,
        match_cols=(
            "year",
            "week",
            "final_cluster_label",
        ),
        decoded_col="decoded_behavior",
    )
    
    essential_metrics = [
        "feature_f1",
        "feature_jaccard",
        "elapsed_nmae",
        "elapsed_distribution_similarity",
        "total_elapsed_percentage_error",
    ]

    compact_overall_summary_df = (
        decoded_overall_summary_df[
            decoded_overall_summary_df["metric"].isin(
                essential_metrics
            )
        ]
        .reset_index(drop=True)
    )
    
    output_dir = Path("projects") / project_name / "centroid_trajectory_predictions" / model_name / "decoded_behavior_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Saves decoded behavior metric results to 
    decoded_row_metrics_df.to_csv(
        output_dir / f"{model_name}_decoded_row_metrics.csv",
        index=False,
    )
    decoded_horizon_metrics_df.to_csv(
        output_dir / f"{model_name}_decoded_horizon_metrics.csv",
        index=False,
    )
    
    compact_overall_summary_df.to_csv(
        output_dir / f"{model_name}_overall_summary_df.csv",
        index=False,
    )

    return compact_overall_summary_df

def print_decoded_behavior_prediction_fit_metrics(
    logger,
    compact_overall_summary_df,
    model_name,
):
    """
    A function that logs the fit of the predicted and historical decoded behavior.
    """
    
    metrics_to_print = [
        "feature_f1",
        "feature_jaccard",
        "elapsed_nmae",
        "elapsed_distribution_similarity",
        "total_elapsed_percentage_error"
    ]

    labels = {
        "feature_f1": "Feature F1",
        "feature_jaccard": "Feature Jaccard",
        "elapsed_nmae": "Normalized Mean Squared Error Elapsed Time",
        "elapsed_distribution_similarity": "Elapsed Time Distribution Similarity",
        "total_elapsed_percentage_error": "Total Elapsed Time Percentage Error",
    }
    
    logger.log(f"=== Decoded Behavior Metrics for {str(model_name).capitalize()} ===", level="STATUS")

    for index, metric in enumerate(metrics_to_print):
        mean_score = compact_overall_summary_df.iloc[index]['mean']
        median_score = compact_overall_summary_df.iloc[index]['median']
        logger.log(f"{labels[metric]}: Mean = {mean_score} | Median = {median_score}", level="STATUS")
        
    logger.log(f"Detailed metric results are saved in the approach's 'decoded_behavior_metrics' folder.", level="INFO")