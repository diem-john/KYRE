from src.trajectory_prediction.utils.preprocessing import embedding_cols
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def add_pca_features(
    df,
    n_dims,
    input_prefix,
    output_prefix,
    seed,
    pca_model=None,
    scaler_model=None,
):
    """
    Add PCA/model-space features.

    If scaler_model and pca_model are provided, this transforms with the fitted
    train objects. It does not refit on validation/test data.
    """
    emb_cols = embedding_cols(df, input_prefix)

    if df.empty:
        out = df.copy()
        z_cols = [f"{output_prefix}{i}" for i in range(n_dims)]
        for col in z_cols:
            out[col] = []
        variance_df = pd.DataFrame()
        return out, pca_model, scaler_model, variance_df

    X = df[emb_cols].to_numpy(dtype=float)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Embedding matrix has NaN or inf values after preprocessing.")
    if n_dims > len(emb_cols):
        raise ValueError(f"PCA_DIMS={n_dims} is larger than embedding dims={len(emb_cols)}")

    if scaler_model is None:
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(X)
    else:
        X_scaled = scaler_model.transform(X)

    if pca_model is None:
        pca_model = PCA(n_components=n_dims, random_state=seed)
        Z = pca_model.fit_transform(X_scaled)
    else:
        Z = pca_model.transform(X_scaled)

    out = df.copy()
    z_cols = [f"{output_prefix}{i}" for i in range(n_dims)]
    out[z_cols] = Z

    variance_df = pd.DataFrame()
    if hasattr(pca_model, "explained_variance_ratio_"):
        variance_df = pd.DataFrame({
            "component": z_cols,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca_model.explained_variance_ratio_),
        })

    return out, pca_model, scaler_model, variance_df