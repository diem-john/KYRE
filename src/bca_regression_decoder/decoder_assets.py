import json
import importlib
from pathlib import Path

import pandas as pd
import torch

from src.bca_regression_decoder.model import FeatureConditionedDecoderMultiTask


def load_feature_names(feature_mapping_candidates):
    """
    Loads feature names where index position = feature_id.

    Expected input:
        feature_mapping_candidates:
            list of possible paths to feature_mapping.csv or feature_embeddings.csv

    Expected file columns:
        feature
    """
    feature_path = None

    for candidate in feature_mapping_candidates:
        candidate = Path(candidate)

        if candidate.exists():
            feature_path = candidate
            break

    if feature_path is None:
        raise FileNotFoundError(
            "Could not find feature mapping file. Tried:\n"
            + "\n".join(str(p) for p in feature_mapping_candidates)
        )

    feature_df = pd.read_csv(feature_path)

    if "feature" not in feature_df.columns:
        raise ValueError(
            f"{feature_path} must contain a 'feature' column."
        )

    feature_names = (
        feature_df["feature"]
        .astype(str)
        .tolist()
    )

    return feature_names, feature_path


def load_model_metadata(model_metadata_path):
    """
    Loads decoder model metadata.

    Expected metadata keys:
        z_dim
        num_features
        emb_dim
        hid_dim
        dropout
    """
    model_metadata_path = Path(model_metadata_path)

    if not model_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model metadata: {model_metadata_path}"
        )

    with open(model_metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_threshold(best_threshold_path, default_threshold=0.5):
    """
    Loads best validation threshold.

    If best_threshold.txt is missing, fallback to default_threshold.
    """
    best_threshold_path = Path(best_threshold_path)

    if not best_threshold_path.exists():
        return default_threshold

    with open(best_threshold_path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def build_and_load_decoder(
    best_model_path,
    model_metadata_path,
    feature_names,
    emb_cols,
    device,
):
    """
    Builds the decoder model from model_metadata.json and loads the checkpoint.
    """
    best_model_path = Path(best_model_path)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Missing decoder checkpoint: {best_model_path}"
        )

    metadata = load_model_metadata(model_metadata_path)

    z_dim = len(emb_cols)
    num_features = len(feature_names)

    expected_z_dim = int(metadata["z_dim"])
    expected_num_features = int(metadata["num_features"])

    if z_dim != expected_z_dim:
        raise RuntimeError(
            "Embedding dimension mismatch.\n"
            f"Input centroid z_dim={z_dim}, "
            f"but decoder checkpoint expects z_dim={expected_z_dim}."
        )

    if num_features != expected_num_features:
        raise RuntimeError(
            "Feature count mismatch.\n"
            f"Loaded feature mapping has num_features={num_features}, "
            f"but decoder checkpoint expects num_features={expected_num_features}."
        )

    model = FeatureConditionedDecoderMultiTask(
        z_dim=z_dim,
        num_features=num_features,
        feat_emb_dim=int(metadata["emb_dim"]),
        hidden=int(metadata["hid_dim"]),
        dropout=float(metadata["dropout"]),
    ).to(device)

    # Helps avoid torch._utils checkpoint loading issues in some environments.
    try:
        torch._utils = importlib.import_module("torch._utils")
    except Exception:
        pass

    try:
        state_dict = torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    except Exception:
        state_dict = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata