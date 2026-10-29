"""Persistence helpers for fitted trajectory preprocessing artifacts.

The PCA and scaler are stored separately from torch checkpoints because
inference loads checkpoints with weights_only=True. Artifacts are written
atomically so an interrupted write does not leave a partially-written file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ARTIFACT_VERSION = 1


def save_preprocessing_artifacts(
    path: str | Path,
    *,
    pca: Any,
    scaler: Any,
    embedding_columns: list[str],
    input_prefix: str,
    output_prefix: str,
    n_components: int,
    seed: int,
) -> Path:
    """Persist the exact train-fitted scaler/PCA and their input schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "pca": pca,
        "scaler": scaler,
        "embedding_columns": list(embedding_columns),
        "input_prefix": str(input_prefix),
        "output_prefix": str(output_prefix),
        "n_components": int(n_components),
        "seed": int(seed),
        "input_dimension": int(getattr(scaler, "n_features_in_", len(embedding_columns))),
        "output_dimension": int(getattr(pca, "n_components_", n_components)),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_preprocessing_artifacts(path: str | Path) -> dict[str, Any]:
    """Load and validate a persisted preprocessing artifact."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Preprocessing artifact not found: {source}")
    payload = joblib.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid preprocessing artifact: {source}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported preprocessing artifact version "
            f"{payload.get('artifact_version')!r}; expected {ARTIFACT_VERSION}."
        )
    required = {
        "pca", "scaler", "embedding_columns", "input_dimension",
        "output_dimension", "input_prefix", "output_prefix",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "Preprocessing artifact is missing required fields: "
            + ", ".join(missing)
        )
    columns = payload["embedding_columns"]
    if len(columns) != int(payload["input_dimension"]):
        raise ValueError(
            "Preprocessing artifact schema mismatch: embedding column count "
            "does not match the recorded input dimension."
        )
    return payload


def transform_embeddings(
    embeddings: np.ndarray,
    artifact: dict[str, Any],
) -> np.ndarray:
    """Apply the saved scaler and PCA to a 2-D raw embedding matrix."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"Expected raw embeddings [time, dimensions], got {values.shape}."
        )
    expected = int(artifact["input_dimension"])
    if values.shape[1] != expected:
        raise ValueError(
            "Raw embedding dimension mismatch: preprocessing artifact expects "
            f"{expected} dimensions, received {values.shape[1]}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Raw embeddings contain NaN or infinite values.")
    transformed = artifact["pca"].transform(
        artifact["scaler"].transform(values)
    )
    return np.asarray(transformed, dtype=np.float32)
