from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ACTUAL TRAJECTORY DATA ROOT
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "projects"
    / "trajectory_prediction_data_hitachi"
)


# ============================================================
# KYRE IMPORTS
# ============================================================

from config import (
    EMB_PREFIX,
    PROJECT_NAME,
    SEED,
)

from src.trajectory_prediction.utils.preprocessing import (
    load_centroid_weekly_embeddings,
    preprocess_centroids,
    split_train_validation_test_sequences,
)

from src.trajectory_prediction.utils.pca_preprocessing import (
    add_pca_features,
)

from src.neural_prediction.data.dataset import (
    WindowConfig,
    build_trajectory_windows,
    build_validation_windows,
    feature_dimension,
)

from config import CLUSTER_COL, PCA_PREFIX

from src.neural_prediction.models.physics_lstm import (
    ModelConfig,
    PhysicsLSTM,
)

from src.neural_prediction.models.hybrid_cnn_lstm_attention import (
    HybridModelConfig,
    ConvLSTMAttention,
)

from src.neural_prediction.losses.physics_loss import (
    LossWeights,
    PhysicsLoss,
)

from src.neural_prediction.training.trainer import (
    Trainer,
    TrainingConfig,
)

from src.neural_prediction.inference import (
    NeuralTrajectoryPredictor,
)

from src.neural_prediction.models.registry import (
    list_checkpoints,
    read_checkpoint_metadata,
)

from src.neural_prediction.evaluation.backtest import (
    BacktestConfig,
    backtest_forecaster,
    persistence_forecast,
    compare_backtests,
)


@st.cache_resource
def load_cached_predictor(checkpoint_path: str, modified_time: float):
    return NeuralTrajectoryPredictor.load(Path(checkpoint_path))



# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

# Page configuration is managed by app_unified.py when this module is
# opened as part of the unified Streamlit application.

st.title("Kinetic Yield and Resource Embeddings: KYRE Prediction Lab")
st.subheader("© Open iT Inc.,  2026")

st.caption(
    "Physics/dynamics-regularized neural forecasting "
    "for centroid trajectories"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prepared" not in st.session_state:
    st.session_state.prepared = None

if "training_history" not in st.session_state:
    st.session_state.training_history = None

if "model" not in st.session_state:
    st.session_state.model = None

if "last_checkpoint" not in st.session_state:
    st.session_state.last_checkpoint = None

if "train_dataset" not in st.session_state:
    st.session_state.train_dataset = None

if "validation_dataset" not in st.session_state:
    st.session_state.validation_dataset = None

if "loss_weights" not in st.session_state:
    st.session_state.loss_weights = LossWeights()

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None


# ============================================================
# PATH HELPERS
# ============================================================

def find_centroid_directory() -> Path:
    """
    Locate the existing KYRE centroid directory.

    We deliberately DO NOT assume that PROJECT_NAME is a
    physical directory under DATA_ROOT.

    Expected target directory:

        .../cluster_embeddings/
            centroids_with_usage_labels/

    The function first checks the direct location and then
    recursively searches under DATA_ROOT.
    """

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            "Trajectory data root does not exist:\n"
            f"{DATA_ROOT}"
        )

    # --------------------------------------------------------
    # Case 1:
    # DATA_ROOT/cluster_embeddings/centroids_with_usage_labels
    # --------------------------------------------------------

    direct = (
        DATA_ROOT
        / "cluster_embeddings"
        / "centroids_with_usage_labels"
    )

    if direct.is_dir():
        return direct

    # --------------------------------------------------------
    # Case 2:
    # Search underneath DATA_ROOT.
    #
    # This handles cases where the actual project/data
    # directory has a different name.
    # --------------------------------------------------------

    matches = list(
        DATA_ROOT.rglob(
            "centroids_with_usage_labels"
        )
    )

    matches = [
        path
        for path in matches
        if path.is_dir()
    ]

    if not matches:
        raise FileNotFoundError(
            "Could not find a "
            "'centroids_with_usage_labels' directory under:\n"
            f"{DATA_ROOT}\n\n"
            "Searched recursively under the trajectory "
            "data root."
        )

    # Prefer a directory containing parquet files.
    parquet_matches = [
        path
        for path in matches
        if any(path.glob("*.parquet"))
    ]

    if parquet_matches:
        return parquet_matches[0]

    # If the directory exists but is currently empty,
    # return the first discovered location so that the
    # subsequent error identifies the exact location.
    return matches[0]


def get_model_directory() -> Path:
    """
    Neural model artifacts are stored separately from the
    existing centroid data.

    Location:

        projects/
            trajectory_prediction_data_hitachi/
                neural_models/
    """

    return DATA_ROOT / "neural_models"


def get_parquet_files(
    centroid_directory: Path,
) -> list[Path]:
    """
    Return parquet files in the centroid directory.
    """

    return sorted(
        centroid_directory.glob("*.parquet")
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(
    project_name: str,
    pca_dims: int,
    min_weeks: int,
    max_gap: int,
    validation_weeks: int,
    test_weeks: int,
    max_staleness: int,
):
    """
    Reuse the existing KYRE extraction, preprocessing,
    temporal split, and PCA pipeline.

    Important:

    PROJECT_NAME is treated as a logical project identifier
    for the existing KYRE preprocessing code.

    It is NOT assumed to be the name of a physical directory.
    """

    # --------------------------------------------------------
    # Locate actual centroid data.
    # --------------------------------------------------------

    centroid_dir = find_centroid_directory()

    # --------------------------------------------------------
    # Validate parquet files.
    # --------------------------------------------------------

    parquet_files = get_parquet_files(
        centroid_dir
    )

    if not parquet_files:
        raise FileNotFoundError(
            "No parquet files found in:\n"
            f"{centroid_dir}"
        )

    # --------------------------------------------------------
    # Load existing centroid embeddings.
    # --------------------------------------------------------

    raw = load_centroid_weekly_embeddings(
        centroid_dir
    )

    # --------------------------------------------------------
    # Existing KYRE preprocessing.
    # --------------------------------------------------------

    clean, summary = preprocess_centroids(
        raw,
        min_weeks=min_weeks,
        max_gap=max_gap,
        max_staleness_weeks=max_staleness,
        emb_prefix=EMB_PREFIX,
        cluster_col=CLUSTER_COL,
        user_weight_col="n_users",
    )

    # --------------------------------------------------------
    # Existing train / validation / test split.
    # --------------------------------------------------------

    (
        train,
        validation,
        test,
        split_summary,
    ) = split_train_validation_test_sequences(
        clean,
        validation_weeks=validation_weeks,
        test_weeks=test_weeks,
        min_train_weeks=min_weeks,
        cluster_col=CLUSTER_COL,
        project_name=project_name,
        is_save_split_csv=False,
    )

    # --------------------------------------------------------
    # Fit PCA ONLY on training data.
    # --------------------------------------------------------

    (
        train_m,
        pca,
        scaler,
        variance,
    ) = add_pca_features(
        train,
        n_dims=pca_dims,
        input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Apply fitted PCA to validation.
    # --------------------------------------------------------

    val_m, _, _, _ = add_pca_features(
        validation,
        pca_model=pca,
        scaler_model=scaler,
        n_dims=pca_dims,
        input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Apply fitted PCA to test.
    # --------------------------------------------------------

    test_m, _, _, _ = add_pca_features(
        test,
        pca_model=pca,
        scaler_model=scaler,
        n_dims=pca_dims,
        input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Apply fitted PCA to the complete cleaned dataset.
    #
    # This fixes the previous undefined `clean_m`.
    # --------------------------------------------------------

    clean_m, _, _, _ = add_pca_features(
        clean,
        pca_model=pca,
        scaler_model=scaler,
        n_dims=pca_dims,
        input_prefix=EMB_PREFIX,
        output_prefix=PCA_PREFIX,
        seed=SEED,
    )

    return {
        "raw": raw,
        "clean": clean,
        "clean_model": clean_m,
        "summary": summary,
        "train": train_m,
        "validation": val_m,
        "test": test_m,
        "split_summary": split_summary,
        "pca": pca,
        "scaler": scaler,
        "variance": variance,
        "centroid_dir": centroid_dir,
    }


# ============================================================
# BACKTEST ADAPTERS
# ============================================================

class PersistencePredictor:
    """
    Adapter for the persistence baseline.

    The backtest engine expects an object with:

        predict(history)
    """

    def __init__(
        self,
        horizon: int,
    ):
        self.horizon = int(horizon)

    def predict(
        self,
        history,
    ):
        return persistence_forecast(
            history,
            self.horizon,
        )



class FrozenNeuralPredictor:
    """Adapter that prepares PCA history for neural inference."""

    def __init__(
        self,
        predictor: NeuralTrajectoryPredictor,
        include_velocity: bool = True,
        include_acceleration: bool = True,
    ):
        self.predictor = predictor
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration

    @staticmethod
    def build_features(
        history: np.ndarray,
        include_velocity: bool = True,
        include_acceleration: bool = True,
    ) -> np.ndarray:
        """Build position, velocity and acceleration features."""

        position = np.asarray(history, dtype=np.float32)

        if position.ndim != 2:
            raise ValueError(
                f"Expected [time, dimensions], got {position.shape}"
            )

        if position.shape[0] == 0:
            raise ValueError("History cannot be empty.")

        parts = [position]

        # First differences; the first velocity is zero.
        velocity = np.diff(
            position,
            axis=0,
            prepend=position[:1],
        ).astype(np.float32)

        if include_velocity:
            parts.append(velocity)

        # Second differences; the first acceleration is zero.
        acceleration = np.diff(
            velocity,
            axis=0,
            prepend=velocity[:1],
        ).astype(np.float32)

        if include_acceleration:
            parts.append(acceleration)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def predict(self, history):
        features = self.build_features(
            history,
            include_velocity=self.include_velocity,
            include_acceleration=self.include_acceleration,
        )

        return self.predictor.predict(features)


# ============================================================
# TRAJECTORY CONSTRUCTION
# ============================================================

def build_trajectories(
    clean_model: pd.DataFrame,
    cluster_col: str,
    horizon: int,
    min_history: int,
):
    """
    Convert the PCA-transformed dataframe into a dictionary:

        {
            cluster_id: numpy trajectory
        }

    Each trajectory has shape:

        [time, PCA dimensions]
    """

    trajectories = {}

    for cluster, group in clean_model.groupby(
        cluster_col,
        sort=False,
    ):

        group = group.sort_values(
            ["year", "week"]
        )

        z_cols = [
            column
            for column in group.columns
            if column.startswith(PCA_PREFIX)
        ]

        if not z_cols:
            continue

        values = group[
            z_cols
        ].to_numpy(
            dtype=np.float32
        )

        if len(values) >= (
            min_history + horizon
        ):
            trajectories[
                str(cluster)
            ] = values

    return trajectories


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project")

    # This is the logical KYRE project identifier.
    #
    # It is NOT used to construct the filesystem path.
    project_name = st.text_input(
        "Project name",
        PROJECT_NAME,
    )

    st.caption(
        "Trajectory data root:"
    )

    st.code(
        str(DATA_ROOT)
    )

    pca_dims = st.number_input(
        "PCA dimensions",
        min_value=2,
        max_value=128,
        value=32,
        step=1,
    )

    min_weeks = st.number_input(
        "Minimum weeks",
        min_value=4,
        max_value=104,
        value=8,
        step=1,
    )

    max_gap = st.number_input(
        "Maximum internal gap",
        min_value=0,
        max_value=12,
        value=2,
        step=1,
    )

    validation_weeks = st.number_input(
        "Validation weeks",
        min_value=1,
        max_value=52,
        value=12,
        step=1,
    )

    test_weeks = st.number_input(
        "Test weeks",
        min_value=1,
        max_value=52,
        value=24,
        step=1,
    )

    max_staleness = st.number_input(
        "Maximum staleness",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
    )


# ============================================================
# TABS
# ============================================================


(
    extract_tab,
    dataset_tab,
    physics_tab,
    train_tab,
    validation_tab,
    backtest_tab,
    visualization_tab,
    inference_tab,
    model_tab,
) = st.tabs(
    [
        "1 · Extraction",
        "2 · Dataset",
        "3 · Physics",
        "4 · Training",
        "5 · Validation",
        "6 · Backtest",
        "7 · 3D Visualization",
        "8 · Inference",
        "9 · Models",
    ]
)



# ============================================================
# EXTRACTION TAB
# ============================================================

with extract_tab:

    st.subheader(
        "Existing KYRE extraction and PCA pipeline"
    )

    st.write(
        "This tab reuses the existing centroid loading, "
        "preprocessing, temporal split, and PCA implementation."
    )

    # --------------------------------------------------------
    # Display actual discovered path.
    # --------------------------------------------------------

    try:

        centroid_dir_preview = (
            find_centroid_directory()
        )

        st.success(
            "Centroid directory detected:"
        )

        st.code(
            str(centroid_dir_preview)
        )

        parquet_preview = get_parquet_files(
            centroid_dir_preview
        )

        st.write(
            f"Parquet files detected: "
            f"**{len(parquet_preview)}**"
        )

    except FileNotFoundError as exc:

        st.warning(
            str(exc)
        )

    # --------------------------------------------------------
    # Prepare button.
    # --------------------------------------------------------

    if st.button(
        "Load and prepare data",
        type="primary",
    ):

        with st.spinner(
            "Loading centroid data and fitting PCA "
            "on the training split..."
        ):

            try:

                st.session_state.prepared = (
                    prepare_data(
                        project_name=project_name,
                        pca_dims=int(pca_dims),
                        min_weeks=int(min_weeks),
                        max_gap=int(max_gap),
                        validation_weeks=int(
                            validation_weeks
                        ),
                        test_weeks=int(
                            test_weeks
                        ),
                        max_staleness=int(
                            max_staleness
                        ),
                    )
                )

                st.success(
                    "Data prepared successfully."
                )

            except Exception as exc:

                st.exception(exc)

    # --------------------------------------------------------
    # Display prepared data.
    # --------------------------------------------------------

    if st.session_state.prepared:

        d = st.session_state.prepared

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Raw rows",
            len(d["raw"]),
        )

        c2.metric(
            "Clean rows",
            len(d["clean"]),
        )

        c3.metric(
            "Clusters kept",
            int(
                d["summary"]["kept"].sum()
            ),
        )

        c4.metric(
            "PCA dimensions",
            pca_dims,
        )

        st.write(
            "Centroid source:"
        )

        st.code(
            str(d["centroid_dir"])
        )

        st.dataframe(
            d["split_summary"].head(20),
            use_container_width=True,
        )


# ============================================================
# DATASET TAB
# ============================================================

with dataset_tab:

    st.subheader(
        "Temporal window construction"
    )

    lookback = st.number_input(
        "Lookback weeks",
        min_value=2,
        max_value=104,
        value=12,
        step=1,
    )

    horizon = st.number_input(
        "Forecast horizon",
        min_value=1,
        max_value=52,
        value=6,
        step=1,
    )

    stride = st.number_input(
        "Window stride",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
    )

    include_velocity = st.checkbox(
        "Include velocity features",
        value=True,
    )

    include_acceleration = st.checkbox(
        "Include acceleration features",
        value=True,
    )

    window_cfg = WindowConfig(
        int(lookback),
        int(horizon),
        int(stride),
        include_velocity,
        include_acceleration,
    )

    if st.session_state.prepared:

        d = st.session_state.prepared

        try:

            train_ds = build_trajectory_windows(
                d["train"],
                window_cfg,
                PCA_PREFIX,
                CLUSTER_COL,
            )

            val_ds = build_validation_windows(
                d["train"],
                d["validation"],
                window_cfg,
                PCA_PREFIX,
                CLUSTER_COL,
            )

            st.session_state.train_dataset = (
                train_ds
            )

            st.session_state.validation_dataset = (
                val_ds
            )

            st.success(
                f"Training windows: "
                f"{len(train_ds):,} "
                f"· Validation windows: "
                f"{len(val_ds):,}"
            )

            st.write(
                "Input features per timestep: "
                f"**{feature_dimension(int(pca_dims), window_cfg)}**"
            )

        except Exception as exc:

            st.exception(exc)

    else:

        st.info(
            "Load data in Extraction first."
        )


# ============================================================
# PHYSICS TAB
# ============================================================

with physics_tab:

    st.subheader(
        "Dynamics and physics-inspired constraints"
    )

    st.write(
        "The current implementation treats physical "
        "foundations as discrete dynamical constraints "
        "on latent centroid motion."
    )

    w_pos = st.number_input(
        "Position loss weight",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.05,
    )

    w_vel = st.number_input(
        "Velocity loss weight",
        min_value=0.0,
        max_value=10.0,
        value=0.25,
        step=0.05,
    )

    w_acc = st.number_input(
        "Acceleration loss weight",
        min_value=0.0,
        max_value=10.0,
        value=0.10,
        step=0.05,
    )

    w_cont = st.number_input(
        "Continuity loss weight",
        min_value=0.0,
        max_value=10.0,
        value=0.05,
        step=0.05,
    )

    st.latex(
        r"L = \lambda_z L_z + "
        r"\lambda_v L_v + "
        r"\lambda_a L_a + "
        r"\lambda_c L_c"
    )

    st.session_state.loss_weights = (
        LossWeights(
            w_pos,
            w_vel,
            w_acc,
            w_cont,
        )
    )


# ============================================================
# TRAINING TAB
# ============================================================

with train_tab:

    st.subheader(
        "Train Neural Prediction Engine"
    )

    st.caption(
        "Train a neural trajectory predictor with "
        "physics/dynamics-aware loss constraints."
    )

    # ========================================================
    # MODEL ARCHITECTURE
    # ========================================================

    st.markdown(
        "### Model Architecture"
    )

    architecture = st.selectbox(
        "Architecture",
        [
            "Hybrid CNN + LSTM + Attention",
            "Physics-Regularized LSTM",
        ],
        key="training_architecture",
    )

    col_arch_1, col_arch_2 = st.columns(2)

    with col_arch_1:

        hidden = st.number_input(
            "Hidden dimension",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
            key="training_hidden",
        )

        layers = st.number_input(
            "LSTM layers",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key="training_layers",
        )

        dropout = st.slider(
            "Dropout",
            min_value=0.0,
            max_value=0.8,
            value=0.2,
            step=0.05,
            key="training_dropout",
        )

    with col_arch_2:

        attention_heads = st.number_input(
            "Attention heads",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
            key="training_attention_heads",
            help=(
                "Number of self-attention heads used by "
                "the Hybrid CNN + LSTM + Attention model."
            ),
        )

        conv_channels = st.number_input(
            "Convolution channels",
            min_value=8,
            max_value=512,
            value=128,
            step=8,
            key="training_conv_channels",
            help=(
                "Number of temporal convolution channels "
                "used by the Hybrid model."
            ),
        )

    # ========================================================
    # TRAINING CONFIGURATION
    # ========================================================

    st.markdown(
        "### Training Configuration"
    )

    col_train_1, col_train_2, col_train_3 = st.columns(3)

    with col_train_1:

        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=50,
            step=1,
            key="training_epochs",
        )

        batch = st.number_input(
            "Batch size",
            min_value=1,
            max_value=1024,
            value=64,
            step=1,
            key="training_batch",
        )

    with col_train_2:

        lr = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-1,
            value=1e-3,
            format="%.6f",
            key="training_lr",
        )

        patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            key="training_patience",
        )

    with col_train_3:

        device = st.selectbox(
            "Device",
            [
                "auto",
                "cpu",
                "cuda",
            ],
            key="training_device",
        )

        st.write("")

        st.info(
            f"Forecast horizon: **{int(horizon)}**\n\n"
            f"PCA dimensions: **{int(pca_dims)}**"
        )

    # ========================================================
    # PHYSICS CURRICULUM
    # ========================================================

    st.markdown(
        "### Physics-Loss Curriculum"
    )

    col_phys_1, col_phys_2 = st.columns(2)

    with col_phys_1:

        warmup_epochs = st.number_input(
            "Physics Warm-up Epochs",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            key="training_warmup",
            help=(
                "Number of initial epochs during which "
                "physics-loss terms are disabled."
            ),
        )

    with col_phys_2:

        ramp_epochs = st.number_input(
            "Physics Ramp Epochs",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            key="training_ramp",
            help=(
                "Number of epochs over which physics-loss "
                "terms are gradually increased from 0 to 1."
            ),
        )

    st.caption(
        "Physics curriculum: "
        f"{int(warmup_epochs)} warm-up epochs → "
        f"{int(ramp_epochs)} ramp epochs → "
        "full physics regularization."
    )

    # ========================================================
    # DATASET CHECK
    # ========================================================

    dataset_ready = (
        st.session_state.train_dataset is not None
        and
        st.session_state.validation_dataset is not None
    )

    if not dataset_ready:

        st.warning(
            "Build the Dataset in Tab 2 before starting training."
        )

    else:

        train_count = len(
            st.session_state.train_dataset
        )

        validation_count = len(
            st.session_state.validation_dataset
        )

        col_data_1, col_data_2, col_data_3 = st.columns(3)

        col_data_1.metric(
            "Training windows",
            f"{train_count:,}",
        )

        col_data_2.metric(
            "Validation windows",
            f"{validation_count:,}",
        )

        col_data_3.metric(
            "Input features / timestep",
            f"{feature_dimension(int(pca_dims), window_cfg)}",
        )

    # ========================================================
    # REALTIME TELEMETRY CONTAINERS
    # ========================================================

    st.markdown(
        "### Realtime Training Telemetry"
    )

    telemetry_status = st.empty()

    progress_bar = st.progress(
        0,
        text="Waiting to start training...",
    )

    # --------------------------------------------------------
    # Realtime metrics
    # --------------------------------------------------------

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = (
        st.columns(5)
    )

    epoch_metric = metric_col_1.empty()
    train_metric = metric_col_2.empty()
    validation_metric = metric_col_3.empty()
    physics_metric = metric_col_4.empty()
    learning_rate_metric = metric_col_5.empty()

    # --------------------------------------------------------
    # Plot placeholders
    # --------------------------------------------------------

    convergence_placeholder = st.empty()

    loss_components_placeholder = st.empty()

    curriculum_placeholder = st.empty()

    # ========================================================
    # TRAIN BUTTON
    # ========================================================

    if st.button(
        "Train model",
        type="primary",
        use_container_width=True,
        key="train_neural_model",
    ):

        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        if not dataset_ready:

            st.error(
                "Build the Dataset first."
            )

        else:

            # =================================================
            # MODEL INPUT DIMENSION
            # =================================================

            input_dim = feature_dimension(
                int(pca_dims),
                window_cfg,
            )

            # =================================================
            # BUILD SELECTED ARCHITECTURE
            # =================================================

            if architecture.startswith(
                "Hybrid"
            ):

                model_config = (
                    HybridModelConfig(
                        input_dim=input_dim,
                        embedding_dim=int(
                            pca_dims
                        ),
                        hidden_dim=int(
                            hidden
                        ),
                        lstm_layers=int(
                            layers
                        ),
                        dropout=float(
                            dropout
                        ),
                        horizon=int(
                            horizon
                        ),
                        attention_heads=int(
                            attention_heads
                        ),
                        conv_channels=int(
                            conv_channels
                        ),
                    )
                )

                model = ConvLSTMAttention(
                    model_config
                )

                model_type = "hybrid"

            else:

                model_config = ModelConfig(
                    input_dim=input_dim,
                    embedding_dim=int(
                        pca_dims
                    ),
                    hidden_dim=int(
                        hidden
                    ),
                    num_layers=int(
                        layers
                    ),
                    dropout=float(
                        dropout
                    ),
                    horizon=int(
                        horizon
                    ),
                )

                model = PhysicsLSTM(
                    model_config
                )

                model_type = "physics_lstm"

            # =================================================
            # PHYSICS LOSS
            # =================================================

            loss = PhysicsLoss(
                st.session_state.loss_weights
            )

            # =================================================
            # TRAINING CONFIGURATION
            # =================================================

            training_config = TrainingConfig(
                epochs=int(
                    epochs
                ),
                batch_size=int(
                    batch
                ),
                learning_rate=float(
                    lr
                ),
                patience=int(
                    patience
                ),
                device=device,
                physics_warmup_epochs=int(
                    warmup_epochs
                ),
                physics_ramp_epochs=int(
                    ramp_epochs
                ),
            )

            # =================================================
            # CREATE TRAINER
            # =================================================

            trainer = Trainer(
                model,
                loss,
                training_config,
            )

            # =================================================
            # CHECKPOINT DIRECTORY
            #
            # IMPORTANT:
            # No PROJECT_NAME is appended.
            # =================================================

            model_dir = get_model_directory()

            model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if model_type == "hybrid":

                model_slug = (
                    "hybrid_cnn_lstm_attention"
                )

            else:

                model_slug = (
                    "physics_lstm"
                )

            checkpoint = (
                model_dir
                / f"{model_slug}_best.pth"
            )

            # =================================================
            # LIVE TRAINING HISTORY
            # =================================================

            live_history = []

            # =================================================
            # REALTIME CALLBACK
            # =================================================

            def update_training_telemetry(
                metrics,
            ):

                # ---------------------------------------------
                # Store current epoch
                # ---------------------------------------------

                live_history.append(
                    dict(metrics)
                )

                live_df = pd.DataFrame(
                    live_history
                )

                current_epoch = int(
                    metrics.get(
                        "epoch",
                        len(live_history),
                    )
                )

                total_epochs = int(
                    epochs
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                progress = min(
                    1.0,
                    current_epoch
                    / max(
                        1,
                        total_epochs,
                    ),
                )

                progress_bar.progress(
                    progress,
                    text=(
                        f"Epoch "
                        f"{current_epoch}/"
                        f"{total_epochs}"
                        f" · Physics scale "
                        f"{metrics.get('physics_scale', 0.0):.2f}"
                    ),
                )

                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                telemetry_status.info(
                    "Training neural prediction model · "
                    f"Epoch {current_epoch}/{total_epochs}"
                )

                # =================================================
                # METRIC CARDS
                # =================================================

                epoch_metric.metric(
                    "Epoch",
                    (
                        f"{current_epoch}/"
                        f"{total_epochs}"
                    ),
                )

                train_total = metrics.get(
                    "train_total",
                    float("nan"),
                )

                val_total = metrics.get(
                    "val_total",
                    float("nan"),
                )

                physics_scale = metrics.get(
                    "physics_scale",
                    0.0,
                )

                current_lr = metrics.get(
                    "learning_rate",
                    float("nan"),
                )

                epoch_metric.metric(
                    "Epoch",
                    (
                        f"{current_epoch}/"
                        f"{total_epochs}"
                    ),
                )

                train_metric.metric(
                    "Train Loss",
                    f"{float(train_total):.6f}",
                )

                validation_metric.metric(
                    "Validation Loss",
                    f"{float(val_total):.6f}",
                )

                physics_metric.metric(
                    "Physics Scale",
                    f"{float(physics_scale):.2f}",
                )

                learning_rate_metric.metric(
                    "Learning Rate",
                    f"{float(current_lr):.2e}",
                )

                # =================================================
                # GRAPH 1
                # MODEL CONVERGENCE
                # =================================================

                fig_convergence = go.Figure()

                if (
                    "train_total"
                    in live_df.columns
                ):

                    fig_convergence.add_trace(
                        go.Scatter(
                            x=live_df["epoch"],
                            y=live_df["train_total"],
                            mode="lines+markers",
                            name="Train Loss",
                        )
                    )

                if (
                    "val_total"
                    in live_df.columns
                ):

                    fig_convergence.add_trace(
                        go.Scatter(
                            x=live_df["epoch"],
                            y=live_df["val_total"],
                            mode="lines+markers",
                            name="Validation Loss",
                        )
                    )

                fig_convergence.update_layout(
                    title=(
                        "Model Convergence"
                    ),
                    xaxis_title="Epoch",
                    yaxis_title="Total Loss",
                    hovermode="x unified",
                    height=420,
                    margin=dict(
                        l=50,
                        r=30,
                        t=60,
                        b=50,
                    ),
                )

                convergence_placeholder.plotly_chart(
                    fig_convergence,
                    use_container_width=True,
                )

                # =================================================
                # GRAPH 2
                # PHYSICS LOSS COMPONENTS
                # =================================================

                fig_components = go.Figure()

                component_definitions = [
                    (
                        "train_position",
                        "Position Loss",
                    ),
                    (
                        "train_velocity",
                        "Velocity Loss",
                    ),
                    (
                        "train_acceleration",
                        "Acceleration Loss",
                    ),
                    (
                        "train_continuity",
                        "Continuity Loss",
                    ),
                ]

                for column, label in (
                    component_definitions
                ):

                    if column in live_df.columns:

                        fig_components.add_trace(
                            go.Scatter(
                                x=live_df["epoch"],
                                y=live_df[column],
                                mode="lines+markers",
                                name=label,
                            )
                        )

                fig_components.update_layout(
                    title=(
                        "Physics / Dynamics Loss Components"
                    ),
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode="x unified",
                    height=420,
                    margin=dict(
                        l=50,
                        r=30,
                        t=60,
                        b=50,
                    ),
                )

                loss_components_placeholder.plotly_chart(
                    fig_components,
                    use_container_width=True,
                )

                # =================================================
                # GRAPH 3
                # PHYSICS CURRICULUM + LEARNING RATE
                # =================================================

                fig_curriculum = go.Figure()

                if (
                    "physics_scale"
                    in live_df.columns
                ):

                    fig_curriculum.add_trace(
                        go.Scatter(
                            x=live_df["epoch"],
                            y=live_df["physics_scale"],
                            mode="lines+markers",
                            name="Physics Scale",
                            yaxis="y",
                        )
                    )

                if (
                    "learning_rate"
                    in live_df.columns
                ):

                    fig_curriculum.add_trace(
                        go.Scatter(
                            x=live_df["epoch"],
                            y=live_df["learning_rate"],
                            mode="lines+markers",
                            name="Learning Rate",
                            yaxis="y2",
                        )
                    )

                fig_curriculum.update_layout(
                    title=(
                        "Physics Curriculum and Learning Rate"
                    ),
                    xaxis_title="Epoch",
                    yaxis=dict(
                        title="Physics Scale",
                        range=[
                            0,
                            1.05,
                        ],
                    ),
                    yaxis2=dict(
                        title="Learning Rate",
                        overlaying="y",
                        side="right",
                        type="log",
                    ),
                    hovermode="x unified",
                    height=420,
                    margin=dict(
                        l=50,
                        r=70,
                        t=60,
                        b=50,
                    ),
                )

                curriculum_placeholder.plotly_chart(
                    fig_curriculum,
                    use_container_width=True,
                )

            # =================================================
            # START TRAINING
            # =================================================

            try:

                telemetry_status.info(
                    "Initializing neural prediction model..."
                )

                progress_bar.progress(
                    0,
                    text=(
                        "Initializing training..."
                    ),
                )

                history = trainer.fit(
                    st.session_state.train_dataset,
                    st.session_state.validation_dataset,
                    checkpoint,
                    window_config=window_cfg,
                    epoch_callback=(
                        update_training_telemetry
                    ),
                )

                # =================================================
                # SAVE TRAINING STATE
                # =================================================

                st.session_state.model = (
                    model
                )

                st.session_state.training_history = (
                    pd.DataFrame(
                        history
                    )
                )

                st.session_state.last_checkpoint = (
                    checkpoint
                )

                # =================================================
                # FINAL STATUS
                # =================================================

                progress_bar.progress(
                    1.0,
                    text=(
                        "Training complete."
                    ),
                )

                telemetry_status.success(
                    "Neural prediction model "
                    "training completed successfully."
                )

                st.success(
                    "Training complete."
                )

                st.success(
                    "Best checkpoint:"
                )

                st.code(
                    str(checkpoint)
                )

            except Exception as exc:

                telemetry_status.error(
                    "Training failed."
                )

                st.exception(
                    exc
                )

    # ========================================================
    # FINAL TRAINING HISTORY
    # ========================================================

    if (
        st.session_state.training_history
        is not None
    ):

        st.markdown(
            "### Training History"
        )

        history_df = (
            st.session_state.training_history
        )

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------

        if len(history_df) > 0:

            summary_col_1, summary_col_2, summary_col_3 = (
                st.columns(3)
            )

            if "val_total" in history_df.columns:

                best_index = (
                    history_df[
                        "val_total"
                    ].idxmin()
                )

                best_row = (
                    history_df.loc[
                        best_index
                    ]
                )

                summary_col_1.metric(
                    "Best Validation Loss",
                    (
                        f"{float(best_row['val_total']):.6f}"
                    ),
                )

                summary_col_2.metric(
                    "Best Epoch",
                    int(
                        best_row["epoch"]
                    ),
                )

            elif "train_total" in history_df.columns:

                best_index = (
                    history_df[
                        "train_total"
                    ].idxmin()
                )

                best_row = (
                    history_df.loc[
                        best_index
                    ]
                )

                summary_col_1.metric(
                    "Best Training Loss",
                    (
                        f"{float(best_row['train_total']):.6f}"
                    ),
                )

                summary_col_2.metric(
                    "Best Epoch",
                    int(
                        best_row["epoch"]
                    ),
                )

            summary_col_3.metric(
                "Completed Epochs",
                len(history_df),
            )

        # ----------------------------------------------------
        # Final convergence graph
        # ----------------------------------------------------

        total_columns = [
            column
            for column in history_df.columns
            if column.endswith(
                "total"
            )
        ]

        if total_columns:

            final_convergence = go.Figure()

            for column in total_columns:

                label = (
                    column
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                final_convergence.add_trace(
                    go.Scatter(
                        x=history_df[
                            "epoch"
                        ],
                        y=history_df[
                            column
                        ],
                        mode="lines+markers",
                        name=label,
                    )
                )

            final_convergence.update_layout(
                title=(
                    "Final Training Convergence"
                ),
                xaxis_title="Epoch",
                yaxis_title="Total Loss",
                hovermode="x unified",
                height=450,
            )

            st.plotly_chart(
                final_convergence,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Complete telemetry table
        # ----------------------------------------------------

        with st.expander(
            "View complete training telemetry"
        ):

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# VALIDATION TAB
# ============================================================

with validation_tab:

    st.subheader(
        "Validation"
    )

    if (
        st.session_state.training_history
        is not None
    ):

        history = (
            st.session_state.training_history
        )

        st.dataframe(
            history.tail(10),
            use_container_width=True,
        )

        if "val_total" in history.columns:

            best = history.loc[
                history["val_total"].idxmin()
            ]

            st.metric(
                "Best validation loss",
                f"{best['val_total']:.6f}",
            )

        elif "train_total" in history.columns:

            best = history.loc[
                history["train_total"].idxmin()
            ]

            st.metric(
                "Best training loss",
                f"{best['train_total']:.6f}",
            )

    else:

        st.info(
            "Train a model first."
        )


# ============================================================
# BACKTEST TAB
# ============================================================

with backtest_tab:

    st.subheader(
        "Rolling-Origin Backtest"
    )

    st.caption(
        "This first backtest evaluates a frozen checkpoint "
        "at multiple historical origins. Retraining at every "
        "origin is a separate experimental protocol."
    )

    if st.session_state.prepared is None:

        st.info(
            "Load data in Extraction first."
        )

    else:

        bt_lookback = st.number_input(
            "Backtest lookback",
            min_value=2,
            max_value=104,
            value=int(lookback),
            step=1,
            key="bt_lookback",
        )

        bt_horizon = st.number_input(
            "Backtest horizon",
            min_value=1,
            max_value=52,
            value=int(horizon),
            step=1,
            key="bt_horizon",
        )

        bt_stride = st.number_input(
            "Backtest origin stride",
            min_value=1,
            max_value=52,
            value=int(horizon),
            step=1,
            key="bt_stride",
        )

        bt_min_history = st.number_input(
            "Minimum history",
            min_value=4,
            max_value=208,
            value=max(
                24,
                int(lookback),
            ),
            step=1,
            key="bt_min_history",
        )

        model_dir = get_model_directory()

        checkpoints = list_checkpoints(
            model_dir
        )

        if checkpoints:

            selected_bt = st.selectbox(
                "Backtest checkpoint",
                checkpoints,
                format_func=lambda p: p.name,
                key="bt_checkpoint",
            )

            metadata = read_checkpoint_metadata(
                selected_bt
            )

            if metadata:

                with st.expander(
                    "Checkpoint metadata"
                ):

                    st.json(
                        metadata
                    )

            if st.button(
                "Run frozen-checkpoint backtest",
                type="primary",
            ):

                try:

                    predictor = (
                        NeuralTrajectoryPredictor.load(
                            selected_bt
                        )
                    )

                    clean_model = (
                        st.session_state.prepared[
                            "clean_model"
                        ]
                    )

                    trajectories = (
                        build_trajectories(
                            clean_model=clean_model,
                            cluster_col=CLUSTER_COL,
                            horizon=int(
                                bt_horizon
                            ),
                            min_history=int(
                                bt_min_history
                            ),
                        )
                    )

                    if not trajectories:

                        st.warning(
                            "No trajectories satisfy "
                            "the minimum-history requirement."
                        )

                    else:

                        backtest_config = (
                            BacktestConfig(
                                lookback=int(
                                    bt_lookback
                                ),
                                horizon=int(
                                    bt_horizon
                                ),
                                stride=int(
                                    bt_stride
                                ),
                                min_history=int(
                                    bt_min_history
                                ),
                            )
                        )

                        # ------------------------------------------------
                        # Neural model.
                        # ------------------------------------------------

                        neural_predictor = (
                            FrozenNeuralPredictor(
                                predictor
                            )
                        )

                        (
                            neural_results,
                            neural_summary,
                        ) = backtest_forecaster(
                            trajectories,
                            lambda: neural_predictor,
                            backtest_config,
                            selected_bt.stem,
                        )

                        # ------------------------------------------------
                        # Persistence baseline.
                        # ------------------------------------------------

                        persistence_predictor = (
                            PersistencePredictor(
                                int(bt_horizon)
                            )
                        )

                        (
                            persistence_results,
                            persistence_summary,
                        ) = backtest_forecaster(
                            trajectories,
                            lambda: (
                                persistence_predictor
                            ),
                            backtest_config,
                            "persistence",
                        )

                        # ------------------------------------------------
                        # Compare.
                        # ------------------------------------------------

                        comparison = (
                            compare_backtests(
                                [
                                    neural_summary,
                                    persistence_summary,
                                ]
                            )
                        )

                        st.session_state.backtest_results = (
                            comparison
                        )

                        st.subheader(
                            "Backtest Summary"
                        )

                        st.dataframe(
                            comparison,
                            use_container_width=True,
                        )

                        st.subheader(
                            "Detailed Neural Results"
                        )

                        st.dataframe(
                            neural_results,
                            use_container_width=True,
                        )

                        st.subheader(
                            "Detailed Persistence Results"
                        )

                        st.dataframe(
                            persistence_results,
                            use_container_width=True,
                        )

                        st.success(
                            "Backtest completed on "
                            f"{len(trajectories)} trajectories."
                        )

                except Exception as exc:

                    st.exception(exc)

        else:

            st.info(
                "No trained checkpoints found. "
                "Train a model first."
            )


# ============================================================
# INFERENCE TAB
# ============================================================

with inference_tab:

    st.subheader(
        "Inference"
    )

    st.write(
        "The inference integration consumes the saved "
        "checkpoint and emits trajectories compatible "
        "with the existing latent-space prediction pipeline."
    )

    if (
        st.session_state.model
        is not None
    ):

        st.success(
            "A trained model is available "
            "in the current session."
        )

    elif (
        st.session_state.last_checkpoint
        is not None
    ):

        st.info(
            "Last checkpoint:"
        )

        st.code(
            str(
                st.session_state.last_checkpoint
            )
        )

    else:

        st.info(
            "Train or load a model first."
        )



# ============================================================
# 3D TRAJECTORY VISUALIZATION TAB
# ============================================================



with visualization_tab:
    st.subheader("3D Trajectory Explorer")

    st.caption(
        "Explore weekly centroid trajectories in the first three "
        "PCA dimensions. Forecast errors update automatically "
        "when the selected timestep changes."
    )

    prepared = st.session_state.prepared

    if prepared is None:
        st.info(
            "Load and prepare data in the Extraction tab first."
        )

    else:
        clean_model = prepared["clean_model"]

        # ========================================================
        # 1. IDENTIFY PCA DIMENSIONS AND REQUIRED COLUMNS
        # ========================================================
        pca_cols = [
            column
            for column in clean_model.columns
            if column.startswith(PCA_PREFIX)
        ]

        required_cols = (
            pca_cols
            + ["year", "week", CLUSTER_COL]
        )

        missing_cols = [
            column
            for column in required_cols
            if column not in clean_model.columns
        ]

        if missing_cols:
            st.error(
                f"Required columns are missing: {missing_cols}"
            )

        elif len(pca_cols) < 3:
            st.warning(
                "At least three PCA dimensions are required "
                "for 3D visualization."
            )

        else:
            x_col, y_col, z_col = pca_cols[:3]

            # ====================================================
            # 2. SELECT CLUSTER
            # ====================================================
            clusters = sorted(
                clean_model[CLUSTER_COL]
                .dropna()
                .unique()
                .tolist(),
                key=str,
            )

            if not clusters:
                st.warning("No clusters are available.")

            else:
                selected_cluster = st.selectbox(
                    "Select cluster",
                    options=clusters,
                    format_func=str,
                    key="trajectory_3d_cluster",
                )

                # ================================================
                # 3. PREPARE THE SELECTED TRAJECTORY
                # ================================================
                trajectory_df = clean_model.loc[
                    clean_model[CLUSTER_COL] == selected_cluster
                ].copy()

                trajectory_df = trajectory_df.sort_values(
                    ["year", "week"]
                ).reset_index(drop=True)

                # All PCA dimensions are needed for model evaluation.
                trajectory_df = trajectory_df.dropna(
                    subset=pca_cols + ["year", "week"]
                ).reset_index(drop=True)

                if trajectory_df.empty:
                    st.warning(
                        "This cluster has no complete PCA "
                        "coordinates to visualize."
                    )

                else:
                    n_points = len(trajectory_df)

                    positions = trajectory_df[
                        pca_cols
                    ].to_numpy(dtype=np.float32)

                    # ============================================
                    # 4. MODEL CHECKPOINT SELECTION
                    # ============================================
                    checkpoints = list_checkpoints(
                        get_model_directory()
                    )

                    selected_checkpoint = None

                    if checkpoints:
                        selected_checkpoint = st.selectbox(
                            "Forecast evaluation model",
                            options=checkpoints,
                            format_func=lambda path: path.name,
                            key="visualization_error_checkpoint",
                        )

                    # Use the existing Dataset tab configuration.
                    # These variables are defined in the dataset tab.
                    model_lookback = int(lookback)
                    model_horizon = int(horizon)

                    use_velocity = bool(include_velocity)
                    use_acceleration = bool(include_acceleration)

                    # ============================================
                    # 5. TIMESTEP SLIDER
                    # ============================================
                    st.markdown("### Weekly trajectory")

                    selected_step = st.slider(
                        "Trajectory timestep",
                        min_value=0,
                        max_value=n_points - 1,
                        value=0,
                        step=1,
                        key=f"trajectory_step_{selected_cluster}",
                        help=(
                            "Move the slider to update the selected "
                            "position and automatically recalculate "
                            "forecast errors."
                        ),
                    )

                    x = trajectory_df[x_col].to_numpy()
                    y = trajectory_df[y_col].to_numpy()
                    z = trajectory_df[z_col].to_numpy()

                    current = trajectory_df.iloc[selected_step]

                    year = int(current["year"])
                    week = int(current["week"])

                    # ============================================
                    # 6. CURRENT TRAJECTORY METRICS
                    # ============================================
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Cluster",
                        str(selected_cluster),
                    )

                    col2.metric(
                        "Week",
                        week,
                    )

                    col3.metric(
                        "Year",
                        year,
                    )

                    col4.metric(
                        "Trajectory progress",
                        f"{selected_step + 1} / {n_points}",
                    )

                    # ============================================
                    # 7. AUTOMATIC FORECAST ERROR ANALYSIS
                    #    Displayed BEFORE the 3D visualization
                    # ============================================
                    st.divider()
                    st.markdown(
                        "### Forecast errors at selected timestep"
                    )

                    st.caption(
                        "The selected timestep is the forecast origin. "
                        "The model uses observations up to and including "
                        "this point to predict subsequent PCA embeddings."
                    )

                    if selected_checkpoint is None:
                        st.info(
                            "No model checkpoints found. Train and save "
                            "a model to enable automatic error evaluation."
                        )

                    elif n_points < 2:
                        st.info(
                            "At least two observations are required "
                            "for forecast evaluation."
                        )

                    else:
                        try:
                            checkpoint_path = Path(
                                selected_checkpoint
                            )

                            predictor = load_cached_predictor(
                                str(checkpoint_path),
                                checkpoint_path.stat().st_mtime,
                            )

                            # ------------------------------------
                            # Prepare history and future targets
                            # ------------------------------------
                            history_positions = positions[
                                :selected_step + 1
                            ]

                            actual_future = positions[
                                selected_step + 1:
                            ]

                            # Neural models are usually trained with
                            # a fixed lookback. At early timesteps,
                            # left-pad with the first observation so
                            # the input length matches training.
                            #
                            # This makes early forecasts possible, but
                            # those forecasts use synthetic repeated
                            # observations and should be interpreted
                            # cautiously.
                            if len(history_positions) < model_lookback:
                                padding_count = (
                                    model_lookback
                                    - len(history_positions)
                                )

                                padding = np.repeat(
                                    history_positions[[0]],
                                    padding_count,
                                    axis=0,
                                )

                                model_history = np.concatenate(
                                    [
                                        padding,
                                        history_positions,
                                    ],
                                    axis=0,
                                )
                            else:
                                model_history = history_positions[
                                    -model_lookback:
                                ]

                            # ------------------------------------
                            # Construct model input features
                            # Matches dataset.py _features()
                            # ------------------------------------

                            # Construct model input features.
                            # Matches dataset.py _features() exactly.
                            feature_parts = [model_history]

                            # Compute velocity from the original positions.
                            raw_velocity = np.diff(model_history, axis=0)

                            if use_velocity:
                                velocity_features = np.vstack(
                                    [
                                        np.zeros(
                                            (1, model_history.shape[1]),
                                            dtype=np.float32,
                                        ),
                                        raw_velocity,
                                    ]
                                )
                                feature_parts.append(velocity_features)

                            if use_acceleration:
                                # Compute acceleration from the unpadded velocity.
                                raw_acceleration = np.diff(raw_velocity, axis=0)

                                acceleration_features = np.vstack(
                                    [
                                        np.zeros(
                                            (2, model_history.shape[1]),
                                            dtype=np.float32,
                                        ),
                                        raw_acceleration,
                                    ]
                                )
                                feature_parts.append(acceleration_features)

                            model_features = np.concatenate(
                                feature_parts,
                                axis=1,
                            ).astype(np.float32)

                            # ------------------------------------
                            # Predict future PCA embeddings
                            # ------------------------------------
                            prediction = predictor.predict(
                                model_features
                            )

                            # Support PyTorch tensors, including CUDA.
                            if hasattr(prediction, "detach"):
                                prediction = (
                                    prediction.detach()
                                    .cpu()
                                    .numpy()
                                )

                            prediction = np.asarray(prediction)

                            # Normalize [1, horizon, dimensions]
                            # into [horizon, dimensions].
                            if (
                                prediction.ndim == 3
                                and prediction.shape[0] == 1
                            ):
                                prediction = prediction[0]

                            if prediction.ndim != 2:
                                raise ValueError(
                                    "Unexpected prediction shape: "
                                    f"{prediction.shape}. Expected "
                                    "[horizon, PCA dimensions] or "
                                    "[1, horizon, PCA dimensions]."
                                )

                            if (
                                prediction.shape[1]
                                != len(pca_cols)
                            ):
                                raise ValueError(
                                    "Model output dimensions do not match "
                                    "the number of PCA columns. "
                                    f"Model output: {prediction.shape[1]}; "
                                    f"PCA dimensions: {len(pca_cols)}. "
                                    "Check the selected checkpoint and "
                                    "prepared dataset."
                                )

                            # ------------------------------------
                            # Compare predictions with actual future
                            # ------------------------------------
                            n_eval = min(
                                len(prediction),
                                len(actual_future),
                            )

                            if n_eval == 0:
                                st.info(
                                    "There are no future observations "
                                    "after this timestep to evaluate."
                                )

                            else:
                                predicted = prediction[:n_eval]
                                actual = actual_future[:n_eval]

                                residual = predicted - actual

                                # RMSE and MAE across PCA dimensions
                                # for each forecast timestep.
                                per_step_rmse = np.sqrt(
                                    np.mean(
                                        residual ** 2,
                                        axis=1,
                                    )
                                )

                                per_step_mae = np.mean(
                                    np.abs(residual),
                                    axis=1,
                                )

                                # Cosine similarity per forecast step.
                                pred_norm = np.linalg.norm(
                                    predicted,
                                    axis=1,
                                )

                                actual_norm = np.linalg.norm(
                                    actual,
                                    axis=1,
                                )

                                denominator = (
                                    pred_norm * actual_norm
                                )

                                cosine_similarity = np.divide(
                                    np.sum(
                                        predicted * actual,
                                        axis=1,
                                    ),
                                    denominator,
                                    out=np.full(
                                        n_eval,
                                        np.nan,
                                        dtype=float,
                                    ),
                                    where=denominator > 1e-12,
                                )

                                # Align each forecast step with its
                                # actual year and week.
                                target_rows = trajectory_df.iloc[
                                    selected_step + 1:
                                    selected_step + 1 + n_eval
                                ]

                                error_df = pd.DataFrame(
                                    {
                                        "Forecast Timestep": np.arange(
                                            1,
                                            n_eval + 1,
                                        ),
                                        "Year": target_rows[
                                            "year"
                                        ].to_numpy(),
                                        "Week": target_rows[
                                            "week"
                                        ].to_numpy(),
                                        "RMSE": per_step_rmse,
                                        "MAE": per_step_mae,
                                        "Cosine Similarity": (
                                            cosine_similarity
                                        ),
                                    }
                                )

                                # Overall errors across all evaluated
                                # forecast steps and PCA dimensions.
                                overall_rmse = float(
                                    np.sqrt(
                                        np.mean(residual ** 2)
                                    )
                                )

                                overall_mae = float(
                                    np.mean(
                                        np.abs(residual)
                                    )
                                )

                                valid_cosine = cosine_similarity[
                                    np.isfinite(cosine_similarity)
                                ]

                                mean_cosine = (
                                    float(np.mean(valid_cosine))
                                    if len(valid_cosine)
                                    else float("nan")
                                )

                                # --------------------------------
                                # Summary metrics
                                # --------------------------------
                                st.caption(
                                    f"Forecast origin: Year {year}, "
                                    f"Week {week} · "
                                    f"Model horizon: {model_horizon} · "
                                    f"Evaluated future steps: {n_eval}"
                                )

                                if (
                                    selected_step + 1
                                    < model_lookback
                                ):
                                    st.warning(
                                        "This forecast uses repeated "
                                        "initial observations to fill "
                                        "the model's lookback window. "
                                        "Errors at early timesteps may "
                                        "not be directly comparable with "
                                        "forecasts using a full history."
                                    )

                                error_col1, error_col2, error_col3, error_col4 = (
                                    st.columns(4)
                                )

                                error_col1.metric(
                                    "Overall RMSE",
                                    f"{overall_rmse:.6f}",
                                )

                                error_col2.metric(
                                    "Overall MAE",
                                    f"{overall_mae:.6f}",
                                )

                                error_col3.metric(
                                    "Mean cosine similarity",
                                    (
                                        f"{mean_cosine:.4f}"
                                        if np.isfinite(mean_cosine)
                                        else "N/A"
                                    ),
                                )

                                error_col4.metric(
                                    "Evaluated timesteps",
                                    n_eval,
                                )

                                st.caption(
                                    "RMSE and MAE measure errors in PCA "
                                    "embedding space; they are not "
                                    "percentages. Lower values are better. "
                                    "Cosine similarity closer to 1 "
                                    "indicates better vector alignment."
                                )

                                # --------------------------------
                                # Per-timestep error table
                                # --------------------------------
                                st.markdown(
                                    "#### Error by forecast timestep"
                                )

                                st.dataframe(
                                    error_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # --------------------------------
                                # RMSE and MAE chart
                                # --------------------------------
                                error_fig = go.Figure()

                                error_fig.add_trace(
                                    go.Scatter(
                                        x=error_df[
                                            "Forecast Timestep"
                                        ],
                                        y=error_df["RMSE"],
                                        mode="lines+markers",
                                        name="RMSE",
                                    )
                                )

                                error_fig.add_trace(
                                    go.Scatter(
                                        x=error_df[
                                            "Forecast Timestep"
                                        ],
                                        y=error_df["MAE"],
                                        mode="lines+markers",
                                        name="MAE",
                                    )
                                )

                                error_fig.update_layout(
                                    title=(
                                        "Forecast Error by Future Timestep"
                                    ),
                                    xaxis_title="Forecast Timestep",
                                    yaxis_title="Error in PCA space",
                                    height=350,
                                    hovermode="x unified",
                                    margin=dict(
                                        l=20,
                                        r=20,
                                        t=60,
                                        b=20,
                                    ),
                                )

                                st.plotly_chart(
                                    error_fig,
                                    use_container_width=True,
                                    key="automatic_forecast_error_plot",
                                )

                                # --------------------------------
                                # Cosine similarity chart
                                # --------------------------------
                                with st.expander(
                                    "View cosine similarity by timestep"
                                ):
                                    cosine_fig = go.Figure()

                                    cosine_fig.add_trace(
                                        go.Scatter(
                                            x=error_df[
                                                "Forecast Timestep"
                                            ],
                                            y=error_df[
                                                "Cosine Similarity"
                                            ],
                                            mode="lines+markers",
                                            name="Cosine similarity",
                                        )
                                    )

                                    cosine_fig.update_layout(
                                        title=(
                                            "Predicted vs Actual "
                                            "Embedding Alignment"
                                        ),
                                        xaxis_title="Forecast Timestep",
                                        yaxis_title="Cosine similarity",
                                        yaxis=dict(
                                            range=[-1, 1],
                                        ),
                                        height=300,
                                        margin=dict(
                                            l=20,
                                            r=20,
                                            t=60,
                                            b=20,
                                        ),
                                    )

                                    st.plotly_chart(
                                        cosine_fig,
                                        use_container_width=True,
                                        key="automatic_forecast_cosine_plot",
                                    )

                                # --------------------------------
                                # Download the error table
                                # --------------------------------
                                st.download_button(
                                    "Download timestep errors as CSV",
                                    data=error_df.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name=(
                                        f"cluster_{selected_cluster}_"
                                        f"year_{year}_week_{week}_"
                                        "forecast_errors.csv"
                                    ),
                                    mime="text/csv",
                                    key=(
                                        "download_visualization_errors_"
                                        f"{selected_cluster}_{selected_step}"
                                    ),
                                )

                        except Exception as exc:
                            st.error(
                                "Automatic forecast evaluation failed: "
                                f"{exc}"
                            )

                    # ============================================
                    # 8. 3D TRAJECTORY VISUALIZATION
                    # ============================================
                    st.divider()
                    st.markdown("### 3D trajectory visualization")

                    fig = go.Figure()

                    # Full trajectory
                    fig.add_trace(
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode="lines",
                            name="Full trajectory",
                            line=dict(
                                color="lightgray",
                                width=3,
                                dash="dash",
                            ),
                            opacity=0.6,
                            hovertemplate=(
                                "PCA 1: %{x:.3f}<br>"
                                "PCA 2: %{y:.3f}<br>"
                                "PCA 3: %{z:.3f}"
                                "<extra>Full trajectory</extra>"
                            ),
                        )
                    )

                    # Observed path through the selected timestep
                    fig.add_trace(
                        go.Scatter3d(
                            x=x[:selected_step + 1],
                            y=y[:selected_step + 1],
                            z=z[:selected_step + 1],
                            mode="lines+markers",
                            name="Observed path",
                            line=dict(width=5),
                            marker=dict(size=3),
                            customdata=trajectory_df[
                                ["year", "week"]
                            ].to_numpy()[:selected_step + 1],
                            hovertemplate=(
                                "Year: %{customdata[0]}<br>"
                                "Week: %{customdata[1]}<br>"
                                "PCA 1: %{x:.3f}<br>"
                                "PCA 2: %{y:.3f}<br>"
                                "PCA 3: %{z:.3f}"
                                "<extra>Observed path</extra>"
                            ),
                        )
                    )

                    # Starting point
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x[0]],
                            y=[y[0]],
                            z=[z[0]],
                            mode="markers",
                            name="Start",
                            marker=dict(
                                size=7,
                                symbol="diamond",
                            ),
                            hovertemplate=(
                                "Starting point<extra></extra>"
                            ),
                        )
                    )

                    # Current point
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x[selected_step]],
                            y=[y[selected_step]],
                            z=[z[selected_step]],
                            mode="markers",
                            name="Selected timestep",
                            marker=dict(
                                size=9,
                                symbol="circle",
                                line=dict(width=2),
                            ),
                            customdata=[[year, week]],
                            hovertemplate=(
                                "Year: %{customdata[0]}<br>"
                                "Week: %{customdata[1]}<br>"
                                "PCA 1: %{x:.3f}<br>"
                                "PCA 2: %{y:.3f}<br>"
                                "PCA 3: %{z:.3f}"
                                "<extra>Current position</extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        title=(
                            f"Cluster {selected_cluster} "
                            "— Weekly Trajectory"
                        ),
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col,
                            aspectmode="data",
                        ),
                        height=700,
                        margin=dict(
                            l=0,
                            r=0,
                            t=60,
                            b=0,
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="trajectory_3d_plot",
                    )

                    with st.expander(
                        "View selected timestep data"
                    ):
                        st.dataframe(
                            trajectory_df.iloc[[selected_step]],
                            use_container_width=True,
                            hide_index=True,
                        )




# ============================================================
# MODELS TAB
# ============================================================

with model_tab:

    st.subheader(
        "Model artifacts"
    )

    st.write(
        "Checkpoints contain model state, optimizer state, "
        "model/loss configuration, training configuration, "
        "and window configuration."
    )

    model_dir = get_model_directory()

    st.write(
        "Checkpoint directory:"
    )

    st.code(
        str(model_dir)
    )

    checkpoints = list_checkpoints(
        model_dir
    )

    if checkpoints:

        st.write(
            f"Found {len(checkpoints)} checkpoint(s)."
        )

        for checkpoint in checkpoints:

            with st.expander(
                checkpoint.name
            ):

                st.code(
                    str(checkpoint)
                )

                metadata = (
                    read_checkpoint_metadata(
                        checkpoint
                    )
                )

                if metadata:

                    st.json(
                        metadata
                    )

                if checkpoint.exists():

                    st.download_button(
                        "Download .pth checkpoint",
                        checkpoint.read_bytes(),
                        file_name=checkpoint.name,
                        mime=(
                            "application/octet-stream"
                        ),
                        key=(
                            f"download_{checkpoint.name}"
                        ),
                    )

    else:

        st.info(
            "No model checkpoints found."
        )


# ============================================================
# DEBUG INFORMATION
# ============================================================

with st.sidebar:

    with st.expander(
        "Debug paths"
    ):

        st.write(
            "Repository root:"
        )

        st.code(
            str(PROJECT_ROOT)
        )

        st.write(
            "Trajectory data root:"
        )

        st.code(
            str(DATA_ROOT)
        )

        st.write(
            "Logical project name:"
        )

        st.code(
            str(project_name)
        )

        st.write(
            "Detected centroid directory:"
        )

        try:

            detected_centroid_dir = (
                find_centroid_directory()
            )

            st.code(
                str(
                    detected_centroid_dir
                )
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

        st.write(
            "Neural model directory:"
        )

        st.code(
            str(
                get_model_directory()
            )
        )