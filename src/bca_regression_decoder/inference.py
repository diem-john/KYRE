import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.bca_regression_decoder.model import FeatureConditionedDecoderMultiTask


@torch.no_grad()
def predict_for_feature_ids(
    model,
    device,
    z_week: np.ndarray,
    feature_ids: np.ndarray,
):
    """
    Predict used probability and elapsed time for selected features.

    This function answers:

        For this one user-week embedding,
        what is the predicted usage for these specific feature IDs?

    Args:
        model:
            Trained FeatureConditionedDecoderMultiTask model.

        device:
            PyTorch device.

        z_week:
            One user-week embedding.

            Shape:
                (embedding_dim,)

        feature_ids:
            Feature IDs to predict.

            Shape:
                (num_selected_features,)

    Returns:
        used_prob:
            Probability that each selected feature was used.

            Shape:
                (num_selected_features,)

        time_pred:
            Predicted elapsed_time for each selected feature.

            Shape:
                (num_selected_features,)
    """

    model.eval()

    z_week = np.asarray(z_week, dtype=np.float32)
    feature_ids = np.asarray(feature_ids, dtype=np.int64)

    if feature_ids.size == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    # Repeat the same user-week embedding once for each feature ID.
    #
    # Example:
    #   z_week shape:      (256,)
    #   feature_ids shape: (100,)
    #
    # After repeat:
    #   z_rows shape:      (100, 256)
    z_rows = np.repeat(
        z_week[None, :],
        repeats=len(feature_ids),
        axis=0,
    )

    z_tensor = torch.tensor(
        z_rows,
        dtype=torch.float32,
        device=device,
    )

    f_tensor = torch.tensor(
        feature_ids,
        dtype=torch.long,
        device=device,
    )

    used_logit, time_pred = model(z_tensor, f_tensor)

    used_prob = torch.sigmoid(used_logit).detach().cpu().numpy()
    time_pred = time_pred.detach().cpu().numpy()

    return used_prob, time_pred


@torch.no_grad()
def predict_all_features_for_week(
    model,
    device,
    z_week: np.ndarray,
    num_features: int,
    batch: int = 2048,
):
    """
    Predict all possible features for one user-week embedding.

    This function loops through every feature ID:

        0, 1, 2, ..., num_features - 1

    and predicts:
        - used probability
        - elapsed_time

    Args:
        model:
            Trained decoder model.

        device:
            PyTorch device.

        z_week:
            One user-week embedding.

            Shape:
                (embedding_dim,)

        num_features:
            Total number of possible features.

        batch:
            Number of feature IDs to score at once.

    Returns:
        used_prob_all:
            Used probability for every feature.

            Shape:
                (num_features,)

        time_pred_all:
            Predicted elapsed_time for every feature.

            Shape:
                (num_features,)
    """

    model.eval()

    used_prob_all = np.zeros(
        shape=(num_features,),
        dtype=np.float32,
    )

    time_pred_all = np.zeros(
        shape=(num_features,),
        dtype=np.float32,
    )

    for start in range(0, num_features, batch):
        end = min(start + batch, num_features)

        feature_ids = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        used_prob_batch, time_pred_batch = predict_for_feature_ids(
            model=model,
            device=device,
            z_week=z_week,
            feature_ids=feature_ids,
        )

        used_prob_all[start:end] = used_prob_batch
        time_pred_all[start:end] = time_pred_batch

    return used_prob_all, time_pred_all


def infer_to_long_df(
    model,
    device,
    emb_df_in: pd.DataFrame,
    emb_cols: list,
    feature_names: list,
    prob_thresh: float = 0.5,
    top_k: Optional[int] = None,
    batch: int = 2048,
    id_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Decode embedding rows into long-format predicted behavior.

    Each input row represents one embedding vector. Depending on the input,
    this can be:
        - one user-week embedding
        - one cluster-week centroid embedding
        - one forecasted trajectory centroid embedding

    For each input row:
        1. predict used probability and elapsed time for every feature_id
        2. keep features where pred_used_prob >= prob_thresh
        3. optionally keep only the top_k highest-probability kept features
        4. output one row per predicted feature

    Metadata columns:
        id_cols controls which non-embedding columns are copied into the
        output.

        If id_cols is None, the function automatically preserves any
        available columns from:
            username, final_cluster_label, year, week

        If id_cols is provided, all requested columns must exist in the
        input dataframe.

    Args:
        model:
            Trained decoder model.

        device:
            PyTorch device.

        emb_df_in:
            DataFrame containing embedding columns and optional metadata
            columns.

        emb_cols:
            List of embedding columns to feed into the decoder.

        feature_names:
            List mapping feature_id -> feature name.

        prob_thresh:
            Minimum predicted-used probability required to keep a feature.

        top_k:
            Optional maximum number of predicted features per input row.
            Applied after threshold filtering.

        batch:
            Number of feature IDs scored at once.

        id_cols:
            Optional list of metadata columns to preserve in the output.

    Returns:
        Long-format dataframe containing metadata columns plus:
            feature_id
            feature
            pred_used_prob
            pred_elapsed
    """

    rows = []
    num_features = len(feature_names)

    missing_emb_cols = [
        col
        for col in emb_cols
        if col not in emb_df_in.columns
    ]

    if missing_emb_cols:
        raise ValueError(
            "Input embedding dataframe is missing required embedding columns: "
            f"{missing_emb_cols[:10]}"
        )
    
    if id_cols is None:
        default_id_cols = ["username", "final_cluster_label", "year", "week"]
        id_cols = [
            col
            for col in default_id_cols
            if col in emb_df_in.columns
        ]

    missing_id_cols = [
        col
        for col in id_cols
        if col not in emb_df_in.columns
    ]

    if missing_id_cols:
        raise ValueError(
            "Input embedding dataframe is missing requested id columns: "
            f"{missing_id_cols}"
        )

    for _, row in tqdm(
        emb_df_in.iterrows(),
        total=len(emb_df_in),
        desc="Decoding embeddings",
    ):
        z_week = row[emb_cols].to_numpy(dtype=np.float32)

        used_prob_all, time_pred_all = predict_all_features_for_week(
            model=model,
            device=device,
            z_week=z_week,
            num_features=num_features,
            batch=batch,
        )

        # Keep all features above the chosen probability threshold.
        keep = np.where(used_prob_all >= prob_thresh)[0]

        # If top_k is set, keep only the highest-probability features.
        if top_k is not None and len(keep) > top_k:
            sorted_keep = np.argsort(used_prob_all[keep])[::-1]
            keep = keep[sorted_keep[:top_k]]

        id_values = {
            col: row[col]
            for col in id_cols
        }

        for feature_id in keep:
            out_row = dict(id_values)

            out_row.update(
                {
                    "feature_id": int(feature_id),
                    "feature": feature_names[int(feature_id)],
                    "pred_used_prob": float(used_prob_all[feature_id]),
                    "pred_elapsed": float(time_pred_all[feature_id]),
                }
            )

            rows.append(out_row)

    columns = id_cols + [
        "feature_id",
        "feature",
        "pred_used_prob",
        "pred_elapsed",
    ]

    return pd.DataFrame(rows, columns=columns)


def decode_embeddings_to_behavior_csv(
    model,
    device,
    emb_csv_or_df,
    out_path: str | Path,
    emb_cols: list,
    feature_names: list,
    prob_thresh: float = 0.5,
    top_k: Optional[int] = None,
    batch: int = 2048,
    id_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Decode embeddings into predicted behavior and save the result to CSV.

    This is used by inference mode. It can decode either user-week
    embeddings or trajectory centroid embeddings as long as the input
    contains the expected embedding columns.

    The output is long-format:
        one row per input embedding row per predicted feature.

    Args:
        model:
            Trained decoder model.

        device:
            PyTorch device.

        emb_csv_or_df:
            Either a path to an embedding CSV or an already-loaded dataframe.

        out_path:
            Output CSV path.

        emb_cols:
            Embedding columns expected by the model.

        feature_names:
            Feature name list where index = feature_id.

        prob_thresh:
            Probability threshold for keeping predicted features.

        top_k:
            Optional top-k predicted features per input row.

        batch:
            Feature scoring batch size.

        id_cols:
            Optional metadata columns to preserve in the output.

    Returns:
        Decoded prediction dataframe.
    """

    if isinstance(emb_csv_or_df, (str, Path)):
        emb_df_in = pd.read_csv(emb_csv_or_df)
    else:
        emb_df_in = emb_csv_or_df.copy()

    pred_long = infer_to_long_df(
        model=model,
        device=device,
        emb_df_in=emb_df_in,
        emb_cols=emb_cols,
        feature_names=feature_names,
        prob_thresh=prob_thresh,
        top_k=top_k,
        batch=batch,
        id_cols=id_cols,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_long.to_csv(out_path, index=False)

    return pred_long