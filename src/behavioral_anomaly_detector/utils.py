import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from glob import glob

def rolling_zscore(s, window=8, min_periods=4):
    """
    Rolling z-score using only previous weeks. The function excludes the current week in its own anomaly baseline.
    This calculates a weekly z-score that checks if the behavioral and centroid movement change is typical within each cluster.
    """
    past = s.shift(1)
    
    # instead of mean in the typically used z-scores, we use median since we're expecting that there are weeks with
    # huge week movements.
    # MAD = Median Absolute Deviation. An outlier-resistant measure of variability.
    rolling_median = past.rolling(
        window=window,
        min_periods=min_periods
    ).median()

    rolling_mad = (
        (past - rolling_median)
        .abs()
        .rolling(window=window, min_periods=min_periods)
        .median()
    )

    denom = 1.4826 * rolling_mad.replace(0, np.nan) # 1.4826 is used to scale the z-score so that it would be comparable to standard z-score (using standard deviation)

    return (s - rolling_median) / denom


def add_week_index(df, year_col="year", week_col="week"):
    """
    Adds a sortable year-week index.
    This is used only for ordering and forecast-start detection.
    """
    out = df.copy()
    out["_week_index"] = (
        out[year_col].astype(int) * 100
        + out[week_col].astype(int)
    )
    return out


def build_detection_timelines(
    df,
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    is_prediction_col="is_prediction",
    timeline_col="timeline_type",
):
    """
    Builds two clean timelines per cluster:

    1. actual timeline:
       actual historical rows only.
       Baseline = historical + historical.

    2. prediction timeline:
       actual historical context before the first predicted week
       + predicted rows from the first predicted week onward.
       Baseline = historical + predicted.

    This prevents overlapping actual and predicted rows for the same week from
    contaminating one another's rolling baselines.
    """
    if is_prediction_col not in df.columns:
        raise ValueError(f"Missing required column: {is_prediction_col}")

    work = add_week_index(df, year_col=year_col, week_col=week_col)
    work[is_prediction_col] = work[is_prediction_col].astype(bool)

    timeline_parts = []

    for _, g in work.groupby(cluster_col, sort=False):
        g = g.sort_values("_week_index").copy()

        actual_g = g[~g[is_prediction_col]].copy()
        pred_g = g[g[is_prediction_col]].copy()

        # Actual timeline: only actual historical rows.
        if len(actual_g) > 0:
            actual_timeline = actual_g.copy()
            actual_timeline[timeline_col] = "actual"
            timeline_parts.append(actual_timeline)

        # Prediction timeline: actual context before forecast start + predicted continuation.
        if len(pred_g) > 0:
            forecast_start = pred_g["_week_index"].min()

            prediction_context = actual_g[
                actual_g["_week_index"] < forecast_start
            ].copy()

            prediction_timeline = pd.concat(
                [prediction_context, pred_g],
                ignore_index=True,
            )

            prediction_timeline[timeline_col] = "prediction"
            timeline_parts.append(prediction_timeline)

    if not timeline_parts:
        return work.assign(**{timeline_col: "actual"})

    out = pd.concat(timeline_parts, ignore_index=True)

    return out.sort_values(
        [timeline_col, cluster_col, "_week_index"]
    ).reset_index(drop=True)
    
def build_feature_used(rows):
    """
    Create one record per feature, including its usage probability
    and elapsed-value information.
    """
    return (
        rows[
            ["feature", "pred_used_prob", "pred_elapsed"]
        ]
        .drop_duplicates()
        .rename(columns={
            "pred_used_prob": "used_prob",
            "pred_elapsed": "elapsed",
        })
        .to_dict("records")
    )
    
def build_decoded_behavior_column(
    historical_behavior_change_df,
    predicted_behavior_change_df,
    decoded_weekly_centroid_emb,
):
    """
    Builds a decoded behavior column containing the centroid's
    feature used, probabity feature used, and estimated elapsed time.
    """
    
    # One row per cluster/year/week/prediction type
    weekly_feature_usage = (
        decoded_weekly_centroid_emb
        .groupby(
            ["final_cluster_label", "year", "week", "is_prediction"],
            dropna=False,
        )
        .apply(build_feature_used, include_groups=False)
        .reset_index(name="decoded_behavior")
    )

    # Attaches the built decoded behavior column to the weekly records.

    # Separate historical and predicted records
    historical_feature_usage = weekly_feature_usage.loc[
        ~weekly_feature_usage["is_prediction"]
    ].drop(columns="is_prediction")

    predicted_feature_usage = weekly_feature_usage.loc[
        weekly_feature_usage["is_prediction"]
    ].drop(columns="is_prediction")


    merge_keys = ["final_cluster_label", "year", "week"]

    historical_behavior_change_df = historical_behavior_change_df.merge(
        historical_feature_usage,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    predicted_behavior_change_df = predicted_behavior_change_df.merge(
        predicted_feature_usage,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    
    return historical_behavior_change_df, predicted_behavior_change_df

def scale_and_fit_centroid_embeddings(
    historical_behavior_change_df,
    predicted_behavior_change_df,
    seed
):
    """
    The function fits (Scaler and PCA model) and transforms the centroid embeddings for visualization.
    """
    # Identify embedding columns and keep them in numeric order:
    # emb_0, emb_1, emb_2, ..., emb_n
    embedding_cols = sorted(
        [
            col
            for col in historical_behavior_change_df.columns
            if col.startswith("emb_")
        ],
        key=lambda col: int(col.split("_")[-1]),
    )


    # Create copies so the original dfs are not modified unexpectedly
    historical_behavior_change_df = historical_behavior_change_df.copy()
    predicted_behavior_change_df = predicted_behavior_change_df.copy()


    # Ensures correct embedding dtype and extract embeddings
    historical_embeddings = (
        historical_behavior_change_df[embedding_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    predicted_embeddings = (
        predicted_behavior_change_df[embedding_cols]
        .apply(pd.to_numeric, errors="coerce")
    )


    # Fit both Scaler and PCA models using historical data only and transform only the rest.
    embedding_scaler = StandardScaler()

    historical_embeddings_scaled = embedding_scaler.fit_transform(
        historical_embeddings
    )

    predicted_embeddings_scaled = embedding_scaler.transform(
        predicted_embeddings
    )

    embedding_pca = PCA(
        n_components=3,
        random_state=seed,
    )

    historical_pca = embedding_pca.fit_transform(
        historical_embeddings_scaled
    )

    predicted_pca = embedding_pca.transform(
        predicted_embeddings_scaled
    )


    # Add the shared 3D PCA coordinates
    pca_cols = ["pca_0", "pca_1", "pca_2"]

    historical_behavior_change_df[pca_cols] = historical_pca
    predicted_behavior_change_df[pca_cols] = predicted_pca
    
    return historical_behavior_change_df, predicted_behavior_change_df

def prepare_historical_predicted_embeddings_for_decoding(
    historical_weekly_centroid_emb,
    predictive_model,
    project_name,
    project_dir    
):
    """
    The function prepares the historical and predicted embeddings before decoding by:
    - Filtering the essential columns need for decoding.
    - Loading the predicted centroid embeddings provided the model/prediction approach name.
    - Concatinating the historical and predicted centroid embeddings for visualization. (`concat_weekly_centroid_emb`)
    """
    
    # filtering out columns and retaining embeddings and cluster + year-week identifiers
    emb_cols = [col for col in historical_weekly_centroid_emb.columns if col.startswith('emb')]
    cols_to_retain = ['year', 'week', 'final_cluster_label'] + emb_cols
    historical_weekly_centroid_emb = historical_weekly_centroid_emb[cols_to_retain]

    # Adding a column to denote that the embedding is historical embedding and not a prediction.
    # This is done since it will be concatinated along with the predicted embeddings
    historical_weekly_centroid_emb['is_prediction'] = False
    
    
    # Loading of predicted embeddings
    PREDICTIVE_MODEL_DIR = f'projects/{project_name}/centroid_trajectory_predictions/{predictive_model}'
    PREDICTION_CSV_DIR = [Path(p) for p in glob(f'{PREDICTIVE_MODEL_DIR}/*.csv')][-1]
    predicted_weekly_centroid_emb = pd.read_csv(PREDICTION_CSV_DIR)

    predicted_weekly_centroid_emb = predicted_weekly_centroid_emb[cols_to_retain]

    # Adding a column to denote that the embedding is a predicted embedding.
    predicted_weekly_centroid_emb['is_prediction'] = True
    
    # Concatinates and saves the historical and predicted weekly centroid embeddings
    concat_weekly_centroid_emb = pd.concat([historical_weekly_centroid_emb, predicted_weekly_centroid_emb], ignore_index=True)
    HISTORICAL_AND_PREDICTED_CENTROID_EMB_DIR = project_dir / "centroid_trajectory_predictions"
    concat_weekly_centroid_emb.to_csv(f'{HISTORICAL_AND_PREDICTED_CENTROID_EMB_DIR}/historical_and_predicted_centroids.csv', index=False)
    
    return historical_weekly_centroid_emb, predicted_weekly_centroid_emb, concat_weekly_centroid_emb