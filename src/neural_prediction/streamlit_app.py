from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from config import EMB_PREFIX, CLUSTER_COL, PCA_PREFIX, PROJECT_NAME, SEED
from src.trajectory_prediction.utils.preprocessing import (
    load_centroid_weekly_embeddings,
    preprocess_centroids,
    split_train_validation_test_sequences,
)
from src.trajectory_prediction.utils.pca_preprocessing import add_pca_features
from src.neural_prediction.data.dataset import WindowConfig, build_trajectory_windows, build_validation_windows, feature_dimension
from src.neural_prediction.models.physics_lstm import ModelConfig, PhysicsLSTM
from src.neural_prediction.losses.physics_loss import LossWeights, PhysicsLoss
from src.neural_prediction.training.trainer import Trainer, TrainingConfig
from src.neural_prediction.inference import NeuralTrajectoryPredictor
from src.neural_prediction.models.registry import list_checkpoints, read_checkpoint_metadata
from src.neural_prediction.evaluation.metrics import trajectory_metrics, horizon_metrics


st.set_page_config(page_title="KYRE Neural Prediction Lab", layout="wide")
st.title("KYRE — Neural Trajectory Prediction Lab")
st.caption("Physics/dynamics-regularized neural forecasting for centroid trajectories")

if "prepared" not in st.session_state:
    st.session_state.prepared = None
if "training_history" not in st.session_state:
    st.session_state.training_history = None
if "model" not in st.session_state:
    st.session_state.model = None


def prepare_data(project_name: str, pca_dims: int, min_weeks: int, max_gap: int, validation_weeks: int, test_weeks: int, max_staleness: int):
    project_dir = Path("projects") / project_name
    centroid_dir = project_dir / "cluster_embeddings" / "centroids_with_usage_labels"
    raw = load_centroid_weekly_embeddings(centroid_dir)
    clean, summary = preprocess_centroids(
        raw,
        min_weeks=min_weeks,
        max_gap=max_gap,
        max_staleness_weeks=max_staleness,
        emb_prefix=EMB_PREFIX,
        cluster_col=CLUSTER_COL,
        user_weight_col="n_users",
    )
    train, validation, test, split_summary = split_train_validation_test_sequences(
        clean,
        validation_weeks=validation_weeks,
        test_weeks=test_weeks,
        min_train_weeks=min_weeks,
        cluster_col=CLUSTER_COL,
        project_name=project_name,
        is_save_split_csv=False,
    )
    train_m, pca, scaler, variance = add_pca_features(
        train, n_dims=pca_dims, input_prefix=EMB_PREFIX, output_prefix=PCA_PREFIX, seed=SEED
    )
    val_m, _, _, _ = add_pca_features(
        validation, pca_model=pca, scaler_model=scaler, n_dims=pca_dims,
        input_prefix=EMB_PREFIX, output_prefix=PCA_PREFIX, seed=SEED
    )
    test_m, _, _, _ = add_pca_features(
        test, pca_model=pca, scaler_model=scaler, n_dims=pca_dims,
        input_prefix=EMB_PREFIX, output_prefix=PCA_PREFIX, seed=SEED
    )
    return {"raw": raw, "clean": clean, "clean_model": clean_m, "summary": summary, "train": train_m, "validation": val_m, "test": test_m, "split_summary": split_summary, "pca": pca, "scaler": scaler, "variance": variance}


with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", PROJECT_NAME)
    pca_dims = st.number_input("PCA dimensions", 2, 128, 32)
    min_weeks = st.number_input("Minimum weeks", 4, 104, 8)
    max_gap = st.number_input("Maximum internal gap", 0, 12, 2)
    validation_weeks = st.number_input("Validation weeks", 1, 52, 12)
    test_weeks = st.number_input("Test weeks", 1, 52, 24)
    max_staleness = st.number_input("Maximum staleness", 0, 20, 3)

extract_tab, dataset_tab, physics_tab, train_tab, validation_tab, inference_tab, model_tab = st.tabs([
    "1 · Extraction", "2 · Dataset", "3 · Physics", "4 · Training", "5 · Validation", "6 · Inference", "7 · Models"
])

with extract_tab:
    st.subheader("Existing KYRE extraction and PCA pipeline")
    st.write("This tab intentionally reuses the current centroid loading, preprocessing, temporal split, and PCA implementation.")
    if st.button("Load and prepare data", type="primary"):
        with st.spinner("Loading centroid data and fitting PCA on the training split..."):
            st.session_state.prepared = prepare_data(project_name, pca_dims, min_weeks, max_gap, validation_weeks, test_weeks, max_staleness)
        st.success("Data prepared.")
    if st.session_state.prepared:
        d = st.session_state.prepared
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw rows", len(d["raw"]))
        c2.metric("Clean rows", len(d["clean"]))
        c3.metric("Clusters kept", d["summary"]["kept"].sum())
        c4.metric("PCA dimensions", pca_dims)
        st.dataframe(d["split_summary"].head(20), use_container_width=True)

with dataset_tab:
    st.subheader("Temporal window construction")
    lookback = st.number_input("Lookback weeks", 2, 104, 12)
    horizon = st.number_input("Forecast horizon", 1, 52, 6)
    stride = st.number_input("Window stride", 1, 20, 1)
    include_velocity = st.checkbox("Include velocity features", True)
    include_acceleration = st.checkbox("Include acceleration features", True)
    window_cfg = WindowConfig(int(lookback), int(horizon), int(stride), include_velocity, include_acceleration)
    if st.session_state.prepared:
        d = st.session_state.prepared
        try:
            train_ds = build_trajectory_windows(d["train"], window_cfg, PCA_PREFIX, CLUSTER_COL)
            val_ds = build_validation_windows(d["train"], d["validation"], window_cfg, PCA_PREFIX, CLUSTER_COL)
            st.session_state.train_dataset = train_ds
            st.session_state.validation_dataset = val_ds
            st.success(f"Training windows: {len(train_ds):,} · Validation windows: {len(val_ds):,}")
            st.write(f"Input features per timestep: **{feature_dimension(int(pca_dims), window_cfg)}**")
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.info("Load data in Extraction first.")

with physics_tab:
    st.subheader("Dynamics and physics-inspired constraints")
    st.write("The first implementation treats physical foundations as discrete dynamical constraints on latent centroid motion.")
    w_pos = st.number_input("Position loss weight", 0.0, 10.0, 1.0, 0.05)
    w_vel = st.number_input("Velocity loss weight", 0.0, 10.0, 0.25, 0.05)
    w_acc = st.number_input("Acceleration loss weight", 0.0, 10.0, 0.10, 0.05)
    w_cont = st.number_input("Continuity loss weight", 0.0, 10.0, 0.05, 0.05)
    st.latex(r"L = \\lambda_z L_z + \\lambda_v L_v + \\lambda_a L_a + \\lambda_c L_c")
    st.session_state.loss_weights = LossWeights(w_pos, w_vel, w_acc, w_cont)

with train_tab:
    st.subheader("Train Physics-Regularized LSTM")
    hidden = st.number_input("Hidden dimension", 16, 1024, 128, 16)
    layers = st.number_input("LSTM layers", 1, 6, 2)
    dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.05)
    epochs = st.number_input("Epochs", 1, 1000, 50)
    batch = st.number_input("Batch size", 1, 1024, 64)
    lr = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f")
    patience = st.number_input("Early stopping patience", 1, 200, 10)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    if st.button("Train model", type="primary"):
        if "train_dataset" not in st.session_state or "validation_dataset" not in st.session_state:
            st.error("Build the Dataset first.")
        else:
            mc = ModelConfig(
                input_dim=feature_dimension(int(pca_dims), window_cfg),
                embedding_dim=int(pca_dims), hidden_dim=int(hidden), num_layers=int(layers),
                dropout=float(dropout), horizon=int(horizon),
            )
            model = PhysicsLSTM(mc)
            loss = PhysicsLoss(getattr(st.session_state, "loss_weights", LossWeights()))
            tc = TrainingConfig(\n                epochs=int(epochs), batch_size=int(batch), learning_rate=float(lr),\n                patience=int(patience), device=device,\n                physics_warmup_epochs=int(warmup_epochs), physics_ramp_epochs=int(ramp_epochs),\n            )
            trainer = Trainer(model, loss, tc)
            project_dir = Path("projects") / project_name / "neural_models"
            checkpoint = project_dir / "physics_lstm_best.pth"
            with st.spinner("Training..."):
                history = trainer.fit(st.session_state.train_dataset, st.session_state.validation_dataset, checkpoint, window_config=window_cfg)
            st.session_state.model = model
            st.session_state.training_history = pd.DataFrame(history)
            st.session_state.last_checkpoint = checkpoint
            st.success(f"Training complete. Best checkpoint: {checkpoint}")
    if st.session_state.training_history is not None:
        st.line_chart(st.session_state.training_history.set_index("epoch")[ [c for c in st.session_state.training_history.columns if c.endswith("total")] ])

with validation_tab:
    st.subheader("Validation")
    if st.session_state.training_history is not None:
        history = st.session_state.training_history
        st.dataframe(history.tail(10), use_container_width=True)
        best = history.loc[history["val_total"].idxmin()] if "val_total" in history else history.iloc[-1]
        st.metric("Best validation loss", f"{best.get('val_total', best.get('train_total')):.6f}")
    else:
        st.info("Train a model first.")

with inference_tab:
    st.subheader("Inference")
    st.write("The inference integration will consume the saved checkpoint and emit trajectories compatible with the existing decoder pipeline.")
    if st.session_state.model is not None:
        st.success("A trained model is available in the current session.")
    else:
        st.info("Train or load a model first.")

with model_tab:
    st.subheader("Model artifacts")
    st.write("Checkpoints contain model state, optimizer state, model/loss configuration, and training configuration.")
    if "last_checkpoint" in st.session_state:
        st.code(str(st.session_state.last_checkpoint))
        if st.session_state.last_checkpoint.exists():
            st.download_button("Download .pth checkpoint", st.session_state.last_checkpoint.read_bytes(), file_name=st.session_state.last_checkpoint.name)
