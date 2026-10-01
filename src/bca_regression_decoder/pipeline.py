from pathlib import Path

import torch

from src.trajectory_prediction.utils.preprocessing import embedding_cols
from src.bca_regression_decoder.decoder_assets import (
    load_feature_names,
    load_best_threshold,
    build_and_load_decoder,
)
from src.bca_regression_decoder.inference import decode_embeddings_to_behavior_csv


def get_centroid_trajectory_dir(project_name):
    return (
        Path("projects")
        / project_name
        / "centroid_trajectory_predictions"
    )


def get_decoder_best_model_dir(project_name):
    return (
        get_centroid_trajectory_dir(project_name)
        / "behavioral_decoder"
        / "best_model"
    )


def get_decoder_output_dir(project_name):
    output_dir = (
        get_centroid_trajectory_dir(project_name)
        / "behavioral_decoder"
        / "decoded_centroids"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_decoder_checkpoint_paths(project_name):
    best_model_dir = get_decoder_best_model_dir(project_name)

    return {
        "best_model_dir": best_model_dir,
        "best_model_path": best_model_dir / "best_decoder_state_dict.pt",
        "best_threshold_path": best_model_dir / "best_threshold.txt",
        "model_metadata_path": best_model_dir / "model_metadata.json",
        "feature_mapping_candidates": [
            best_model_dir / "feature_mapping.csv",
            best_model_dir / "feature_embeddings.csv",
        ],
    }


def validate_centroid_decoder_input(
    centroid_df,
    label,
    emb_prefix="emb_",
    cluster_col="final_cluster_label",
):
    """
    Validate that the centroid dataframe has the columns needed by the decoder.
    """
    emb_cols = embedding_cols(centroid_df, emb_prefix)

    if not emb_cols:
        raise ValueError(
            f"No embedding columns found in {label} dataframe "
            f"using emb_prefix='{emb_prefix}'."
        )

    required_cols = [
        cluster_col,
        "year",
        "week",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in centroid_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{label} dataframe missing required columns: {missing_cols}"
        )

    return emb_cols


def run_bca_centroid_decoder(
    project_name,
    historical_centroid_df,
    predicted_centroid_df,
    emb_prefix="emb_",
    cluster_col="final_cluster_label",
    top_k=None,
    batch=2048,
    device=None,
    logger=None,
    model_name=None,
):
    """
    Decode historical and predicted centroid embeddings.

    This function assumes historical_centroid_df and predicted_centroid_df
    are already prepared by the trajectory pipeline.

    Loads decoder model from:
        projects/<project_name>/centroid_trajectory_predictions/
            behavioral_decoder/best_model/

    Saves decoded outputs to:
        projects/<project_name>/centroid_trajectory_predictions/
            behavioral_decoder/decoded_centroids/
    """

    checkpoint_paths = get_decoder_checkpoint_paths(project_name)
    decoded_output_dir = get_decoder_output_dir(project_name)

    historical_emb_cols = validate_centroid_decoder_input(
        centroid_df=historical_centroid_df,
        label="historical",
        emb_prefix=emb_prefix,
        cluster_col=cluster_col,
    )

    predicted_emb_cols = validate_centroid_decoder_input(
        centroid_df=predicted_centroid_df,
        label="predicted",
        emb_prefix=emb_prefix,
        cluster_col=cluster_col,
    )

    if historical_emb_cols != predicted_emb_cols:
        raise ValueError(
            "Historical and predicted centroid embedding columns do not match."
        )

    emb_cols = historical_emb_cols

    historical_input = historical_centroid_df.copy()
    predicted_input = predicted_centroid_df.copy()

    historical_input["is_prediction"] = False
    predicted_input["is_prediction"] = True

    device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    feature_names, feature_path = load_feature_names(
        checkpoint_paths["feature_mapping_candidates"]
    )

    best_threshold = load_best_threshold(
        checkpoint_paths["best_threshold_path"],
        default_threshold=0.5,
    )

    decoder_model, metadata = build_and_load_decoder(
        best_model_path=checkpoint_paths["best_model_path"],
        model_metadata_path=checkpoint_paths["model_metadata_path"],
        feature_names=feature_names,
        emb_cols=emb_cols,
        device=device,
    )

    id_cols = [
        cluster_col,
        "year",
        "week",
        "is_prediction",
    ]

    decoded_historical_csv = (
        decoded_output_dir / model_name / "decoded_historical_weekly_centroid_emb.csv"
    )

    decoded_predicted_csv = (
        decoded_output_dir / model_name / "decoded_predicted_weekly_centroid_emb.csv"
    )

    if logger is not None:
        logger.log(
            msg=f"Using decoder best model directory: {checkpoint_paths['best_model_dir']}",
            level="INFO",
        )
        logger.log(
            msg=f"Loaded decoder feature mapping: {feature_path}",
            level="INFO",
        )
        logger.log(
            msg=f"Using saved decoder best threshold: {best_threshold}",
            level="INFO",
        )

    decoded_historical_df = decode_embeddings_to_behavior_csv(
        model=decoder_model,
        device=device,
        emb_csv_or_df=historical_input,
        out_path=decoded_historical_csv,
        emb_cols=emb_cols,
        feature_names=feature_names,
        prob_thresh=best_threshold,
        top_k=top_k,
        batch=batch,
        id_cols=id_cols,
    )

    decoded_predicted_df = decode_embeddings_to_behavior_csv(
        model=decoder_model,
        device=device,
        emb_csv_or_df=predicted_input,
        out_path=decoded_predicted_csv,
        emb_cols=emb_cols,
        feature_names=feature_names,
        prob_thresh=best_threshold,
        top_k=top_k,
        batch=batch,
        id_cols=id_cols,
    )

    if logger is not None:
        logger.log(
            msg=f"Saved decoded historical: {decoded_historical_csv}",
            level="INFO",
        )
        logger.log(
            msg=f"Saved decoded predicted: {decoded_predicted_csv}",
            level="INFO",
        )

    return {
        "decoded_historical_df": decoded_historical_df,
        "decoded_predicted_df": decoded_predicted_df,
        "decoded_historical_csv": decoded_historical_csv,
        "decoded_predicted_csv": decoded_predicted_csv,
        "threshold": best_threshold,
        "metadata": metadata,
    }