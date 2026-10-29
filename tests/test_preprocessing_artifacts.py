from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.neural_prediction.preprocessing_artifacts import (
    load_preprocessing_artifacts,
    save_preprocessing_artifacts,
    transform_embeddings,
)


def test_preprocessing_artifact_round_trip(tmp_path: Path):
    rng = np.random.default_rng(17)
    train = rng.normal(size=(40, 8))
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=3, random_state=17).fit(scaler.transform(train))
    path = tmp_path / "trajectory.preprocessing.joblib"

    save_preprocessing_artifacts(
        path,
        pca=pca,
        scaler=scaler,
        embedding_columns=[f"embedding_{i}" for i in range(8)],
        input_prefix="embedding_",
        output_prefix="pca_",
        n_components=3,
        seed=17,
    )

    loaded = load_preprocessing_artifacts(path)
    actual = transform_embeddings(train[:5], loaded)
    expected = pca.transform(scaler.transform(train[:5])).astype(np.float32)

    assert actual.shape == (5, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_transform_rejects_wrong_input_dimension(tmp_path: Path):
    rng = np.random.default_rng(4)
    train = rng.normal(size=(20, 5))
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=2, random_state=4).fit(scaler.transform(train))
    path = tmp_path / "trajectory.preprocessing.joblib"

    save_preprocessing_artifacts(
        path,
        pca=pca,
        scaler=scaler,
        embedding_columns=[f"embedding_{i}" for i in range(5)],
        input_prefix="embedding_",
        output_prefix="pca_",
        n_components=2,
        seed=4,
    )
    loaded = load_preprocessing_artifacts(path)

    try:
        transform_embeddings(np.ones((3, 4)), loaded)
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected an input dimension validation error")
