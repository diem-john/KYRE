# KYRE Neural Prediction Lab
> Kinetic Yield and Resource Embeddings

A Streamlit-based research and experimentation environment for neural forecasting of KYRE cluster-centroid trajectories. The feature reuses the existing centroid preprocessing and PCA pipeline, constructs chronological trajectory windows, trains neural sequence models with dynamics-informed loss terms, evaluates validation behavior, and compares frozen-checkpoint forecasts against a persistence baseline through rolling-origin backtesting.

> This README documents the `src/neural_prediction` feature. It does not describe the separate `src/anomaly` or `src/behavioral_anomaly_detector` modules.

> Developed for Open iT Inc.
>
> Copyright (c) Open iT Inc., 2026

## Overview

The Neural Prediction Lab investigates the following question:

**Given a cluster's historical weekly trajectory in PCA space, how accurately can a neural model forecast its future centroid positions while respecting observed motion dynamics?**

The implementation is organized around eight Streamlit tabs:

1. Extraction
2. Dataset
3. Physics
4. Training
5. Validation
6. Backtest
7. Inference
8. Models

The workflow separates data preparation, temporal-window construction, model optimization, evaluation, and checkpoint handling. The feature is designed to work with KYRE's existing centroid embeddings and PCA representation rather than replacing the upstream extraction pipeline.

## Key Features

- Reuses the existing weekly centroid-loading and preprocessing utilities
- Filters and prepares centroid trajectories using configurable minimum-history, gap, and staleness settings
- Splits sequences chronologically into training, validation, and test periods
- Fits PCA on the training split and applies the fitted transform to validation and test data
- Builds sliding temporal windows without crossing cluster boundaries
- Supports position, velocity, and acceleration input features
- Provides two neural architectures:
  - Physics-Regularized LSTM
  - Hybrid CNN + LSTM + Multi-Head Attention
- Uses a composite objective with position, velocity, acceleration, and continuity terms
- Supports configurable optimizer settings, early stopping, and gradient clipping
- Records training and validation loss components
- Saves model checkpoints with model, loss, training, and window configuration metadata
- Computes trajectory-level and horizon-level forecasting metrics
- Evaluates frozen checkpoints using rolling-origin backtesting
- Compares neural forecasts with a persistence baseline
- Supports CPU execution and automatic CUDA selection through PyTorch

## Application Workflow

### 1. Extraction

The Extraction tab reuses the existing KYRE centroid data pipeline.

The preparation flow:

1. Loads weekly centroid embeddings from the selected project directory.
2. Preprocesses centroid records using the configured minimum-history, maximum-gap, and maximum-staleness settings.
3. Creates chronological training, validation, and test splits.
4. Fits PCA using the training split.
5. Applies the same fitted PCA transformation to validation and test data.
6. Stores the prepared data and split summaries in Streamlit session state.

The expected source directory is:

```text
projects/<project_name>/cluster_embeddings/centroids_with_usage_labels/
```

The default project name is configured in the repository's root `config.py`. Set the Project name in the app sidebar to use another project directory.

The application uses the existing configuration conventions, including:

- `emb_` — source embedding feature prefix
- `final_cluster_label` — cluster identifier
- `z_` — PCA-space feature prefix
- `n_users` — user-count weighting field used by preprocessing

The Extraction tab reports raw and cleaned row counts, retained-cluster summaries, and the configured PCA dimensionality.

### 2. Dataset Construction

The Dataset tab creates chronological input/target windows independently for each cluster.

The `WindowConfig` parameters are:

| Parameter | Meaning |
|---|---|
| `lookback` | Number of historical weekly steps supplied to the model |
| `horizon` | Number of future weekly steps predicted |
| `stride` | Step between consecutive training-window start positions |
| `include_velocity` | Include first-difference features in the model input |
| `include_acceleration` | Include second-difference features in the model input |

The default settings in the interface are a 12-week lookback and a 6-week forecast horizon.

Each training sample contains:

- `x` — historical position features, optionally augmented with velocity and acceleration
- `y` — future PCA-space positions over the configured horizon

Velocity and acceleration are computed from the historical positions. Their initial undefined values are padded with zeros so the feature arrays remain time-aligned.

Windows are generated within each cluster and within the supplied temporal split. They do not cross cluster boundaries. Validation windows use historical context and validation-period targets.

The total input feature dimension is:

```text
PCA dimensions × (1 + velocity enabled + acceleration enabled)
```

For example, with 32 PCA dimensions and both derivative features enabled, the model receives 96 input features per timestep.

### 3. Physics and Dynamics-Informed Loss

The Physics tab configures the weights for the composite training objective. The current implementation uses discrete trajectory dynamics as regularization terms in PCA space; these should be understood as dynamics-informed constraints, not as a full governing-equation solver.

The total objective is:

\[
L = \lambda_z L_z + \lambda_v L_v + \lambda_a L_a + \lambda_c L_c
\]

where:

- \(L_z\) — position prediction loss, using Smooth L1 loss
- \(L_v\) — velocity prediction loss based on first differences
- \(L_a\) — acceleration prediction loss based on second differences
- \(L_c\) — continuity penalty for predicted speeds that exceed a threshold derived from target-trajectory speed statistics

The interface exposes separate weights for position, velocity, acceleration, and continuity. The default values in the current implementation are 1.0, 0.25, 0.10, and 0.05, respectively.

These terms encourage the forecast to match future positions while discouraging implausible changes in motion. Their usefulness must be evaluated empirically for each dataset and forecast horizon.

### 4. Model Training

The Training tab supports two model architectures.

#### Physics-Regularized LSTM

Implemented in `src/neural_prediction/models/physics_lstm.py`.

The model:

1. Encodes the historical input sequence using an LSTM.
2. Uses the final encoded state as the trajectory representation.
3. Passes the representation through a normalized fully connected prediction head.
4. Produces a multi-step forecast with shape `[batch, horizon, embedding_dim]`.

The LSTM architecture provides a recurrent baseline for learning temporal evolution in the PCA-space trajectories.

#### Hybrid CNN + LSTM + Attention

Implemented in `src/neural_prediction/models/hybrid_cnn_lstm_attention.py`.

The model combines:

1. **Temporal convolution** to extract local patterns from the input sequence.
2. **LSTM encoding** to represent ordered trajectory evolution.
3. **Multi-head self-attention** to reweight information across the encoded history.
4. **Residual and feed-forward blocks** to refine the temporal representation.
5. **A prediction head** to generate the future PCA-space trajectory.

The hybrid architecture is intended to combine short-range temporal pattern extraction, recurrent sequence modeling, and attention-based temporal weighting.

#### Training Configuration

The interface exposes the following settings:

- Architecture
- Hidden dimension
- Number of LSTM layers
- Dropout
- Attention heads
- Convolution channels
- Epoch count
- Batch size
- Learning rate
- Early-stopping patience
- Compute device (`auto`, `cpu`, or `cuda`)

The trainer uses AdamW, gradient clipping, and validation-loss monitoring when a validation dataset is available. The best checkpoint is saved when the monitored loss improves. Training history is displayed in the interface.

### 5. Validation

The Validation tab displays recent training-history rows and the best recorded validation loss when validation results are available.

The training history tracks the total loss and its components, including:

- Position loss
- Velocity loss
- Acceleration loss
- Continuity loss

Validation loss is useful for monitoring optimization and selecting checkpoints, but it is not a substitute for a final held-out evaluation.

### 6. Rolling-Origin Backtest

The Backtest tab evaluates a selected saved checkpoint at multiple historical forecast origins. The checkpoint remains frozen across origins; the model is not retrained separately at each origin.

The default `BacktestConfig` values are:

| Parameter | Default | Meaning |
|---|---:|---|
| `lookback` | 12 | Historical sequence length used by the predictor |
| `horizon` | 6 | Future steps evaluated at each origin |
| `stride` | 6 | Number of time steps between forecast origins |
| `min_history` | 24 | Minimum historical length before the first origin |

At each origin, the evaluator provides historical context to the forecasting function and compares its prediction with the known future trajectory. It records aggregate and horizon-level results.

A persistence baseline repeats the latest observed PCA-space position across the forecast horizon:

```text
Last observed position -> Repeat for each future step
```

The dashboard compares the selected neural checkpoint with this baseline. This comparison helps determine whether the model adds value beyond assuming that the centroid remains at its most recent position.

#### Evaluation Metrics

The evaluation utilities calculate:

- **RMSE** — root mean squared error in PCA coordinates
- **MAE** — mean absolute error in PCA coordinates
- **Cosine similarity** — directional similarity between predicted and actual position vectors
- **Velocity error** — difference between predicted and actual trajectory increments
- **Displacement cosine similarity** — directional agreement between predicted and actual net displacement over the horizon
- **Displacement magnitude error** — absolute difference between predicted and actual displacement magnitudes

Metrics are available at the overall trajectory level and separately for each forecast horizon. Since the model operates in PCA space, these errors are measured in the transformed coordinate system, not directly in the original embedding coordinates.

### 7. Inference

The intended inference workflow is to use a saved checkpoint to generate future trajectories compatible with the existing downstream decoder pipeline.

The current Streamlit Inference tab is an integration placeholder: it indicates whether a trained model is available in the current session, but does not yet provide a complete standalone checkpoint-selection and prediction workflow. Treat inference as partially integrated until that UI path is completed and verified for each supported architecture.

The lower-level predictor utility is defined in `src/neural_prediction/inference.py`. Before relying on checkpoint loading in production, verify that the loader reconstructs the correct architecture from checkpoint metadata and that the expected PCA transformation and feature schema are supplied consistently.

### 8. Model Checkpoints

Checkpoints are written under:

```text
projects/<project_name>/neural_models/
```

The current training interface uses these filenames:

```text
hybrid_cnn_lstm_attention_best.pth
physics_lstm_best.pth
```

A checkpoint includes model weights, optimizer state, epoch and best-loss information, model configuration, loss configuration, training configuration, and window configuration.

The model registry utilities in `src/neural_prediction/models/registry.py` list `.pth` checkpoints and read their metadata.

Checkpoints are local artifacts. Keep them backed up separately if they are required for reproducibility, and do not assume they include the raw dataset or the fitted PCA transformer unless those assets are explicitly saved alongside them.

## Project Structure

The neural prediction feature is organized under `src/neural_prediction/`:

```text
src/neural_prediction/
├── streamlit_app.py
├── data/
│   └── dataset.py
├── models/
│   ├── physics_lstm.py
│   ├── hybrid_cnn_lstm_attention.py
│   ├── factory.py
│   └── registry.py
├── losses/
│   └── physics_loss.py
├── physics/
│   └── dynamics.py
├── training/
│   └── trainer.py
├── evaluation/
│   ├── metrics.py
│   └── backtest.py
└── inference.py
```

Other relevant repository components include:

- `config.py` — project defaults, feature-column names, and shared settings
- `src/trajectory_prediction/utils/preprocessing.py` — centroid loading, temporal sorting, and preprocessing utilities
- `src/trajectory_prediction/utils/pca_preprocessing.py` — PCA feature preparation
- `projects/` — project-specific source data and local neural-model checkpoints

## Core Components

### `streamlit_app.py`

Provides the eight-tab research workflow, exposes configuration controls, manages prepared data and training history in Streamlit session state, and presents validation, backtest, and checkpoint information.

### `data/dataset.py`

Defines `WindowConfig`, `TrajectoryWindowDataset`, temporal-window construction, validation-window construction, derivative features, and input-dimension calculation.

### `models/physics_lstm.py`

Defines the recurrent Physics-Regularized LSTM trajectory predictor.

### `models/hybrid_cnn_lstm_attention.py`

Defines the hybrid temporal CNN, LSTM, and multi-head attention model.

### `losses/physics_loss.py`

Implements the composite position, velocity, acceleration, and continuity objective.

### `training/trainer.py`

Implements training and validation loops, AdamW optimization, gradient clipping, early stopping, and checkpoint saving.

### `evaluation/metrics.py`

Calculates aggregate and horizon-specific trajectory metrics.

### `evaluation/backtest.py`

Implements rolling-origin evaluation, persistence forecasting, and model-result aggregation.

### `models/registry.py`

Lists available checkpoints and reads checkpoint metadata.

### `inference.py`

Provides the lower-level neural trajectory prediction interface for checkpoint-based forecasting. The Streamlit inference workflow still requires further integration and verification.

## Installation

### Requirements

Use the repository's `requirements.txt` and a compatible Python environment. Core dependencies include:

- Python
- Streamlit
- PyTorch
- NumPy
- Pandas
- scikit-learn
- PyArrow

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If using Conda, activate the project's environment before installing dependencies or running the app.

## Running the Application

From the repository root:

```bash
python -m streamlit run src/neural_prediction/streamlit_app.py
```

The application opens in Streamlit in your browser.

## Typical Usage

### Step 1 — Prepare Data

1. Open **1 · Extraction**.
2. Enter the project name.
3. Configure PCA dimensions, minimum weeks, maximum internal gap, validation weeks, test weeks, and maximum staleness.
4. Select **Load and prepare data**.
5. Review the row counts and split summary.

### Step 2 — Build Temporal Windows

1. Open **2 · Dataset**.
2. Set the lookback, forecast horizon, and stride.
3. Enable or disable velocity and acceleration features.
4. Review the resulting training and validation window counts.

### Step 3 — Configure the Loss

1. Open **3 · Physics**.
2. Set the position, velocity, acceleration, and continuity weights.
3. Keep the weights fixed during an experiment when comparing model architectures.

### Step 4 — Train a Model

1. Open **4 · Training**.
2. Select Physics-Regularized LSTM or Hybrid CNN + LSTM + Attention.
3. Set model dimensions, dropout, optimizer-related parameters, epochs, batch size, patience, and device.
4. Select **Train model**.
5. Review the training history and saved checkpoint path.

### Step 5 — Review Validation

Open **5 · Validation** to inspect the recent training history and best recorded validation loss.

### Step 6 — Run Backtesting

1. Open **6 · Backtest**.
2. Set the backtest lookback, horizon, origin stride, and minimum history.
3. Select a checkpoint.
4. Select **Run frozen-checkpoint backtest**.
5. Compare horizon-level metrics against the persistence baseline.

### Step 7 — Inspect Checkpoints

Open **8 · Models** to view the most recently trained checkpoint path and download the checkpoint when available.

The **7 · Inference** tab is currently a placeholder for the complete interactive inference workflow.

## Evaluation and Reproducibility

For meaningful model comparisons:

- Keep chronological data splits fixed across experiments.
- Fit PCA on the training period only and reuse that transform for validation and test periods.
- Keep lookback, horizon, and split definitions consistent when comparing architectures.
- Compare neural predictions with the persistence baseline.
- Inspect per-horizon metrics rather than relying only on a single aggregate score.
- Record the selected project, PCA dimensionality, window settings, loss weights, model configuration, and checkpoint.
- Treat results in PCA coordinates as latent-space metrics; they are not automatically equivalent to errors in the original embedding space.
- Evaluate on an untouched test period before making claims about generalization.

The rolling-origin backtest currently uses a frozen checkpoint. It does not represent a retrain-at-each-origin evaluation protocol.

## Current Limitations

- The Streamlit inference tab is not yet a complete standalone inference workflow.
- Checkpoint loading and model reconstruction should be verified for each architecture before routine inference.
- Checkpoints do not, by themselves, guarantee recovery of the raw input data or fitted PCA transformation.
- Backtesting evaluates a frozen checkpoint rather than retraining at every historical origin.
- The current physics-inspired objective operates on discrete derivatives in PCA space and is not a full physical governing-equation model.
- PCA-space metrics do not directly express errors in original embedding coordinates.
- Forecast quality depends on data coverage, centroid stability, cluster history, and the selected horizon.
- Longer forecast horizons may amplify prediction error.

## Future Extensions

### Inference and Deployment

- Complete checkpoint selection and loading in the Streamlit Inference tab
- Validate architecture reconstruction for every supported checkpoint type
- Persist and restore preprocessing metadata, including PCA transformation details
- Add batch inference across clusters
- Export predicted trajectories for downstream decoding

### Evaluation

- Add explicit test-set evaluation to the UI
- Add retraining-per-origin backtesting as a separate protocol
- Export detailed and summary metrics
- Add comparisons across multiple checkpoints and configurations
- Track experiment settings alongside metrics

### Model Development

- Evaluate alternative temporal architectures
- Tune physics-loss weights systematically
- Investigate uncertainty estimates and prediction intervals
- Test robustness to short, stale, or irregular cluster trajectories

## Development Philosophy

### 1. Preserve Temporal Boundaries

Training windows must not cross cluster boundaries or the intended chronological split.

### 2. Separate Architecture from Dynamics

The neural model predicts future positions; the loss module independently expresses the dynamics-informed constraints used during optimization.

### 3. Compare Against a Baseline

A neural forecaster should be evaluated against a simple persistence forecast to establish whether added model complexity provides measurable benefit.

### 4. Keep Evaluation Protocols Explicit

A frozen-checkpoint rolling-origin backtest and a retrain-at-each-origin backtest answer different questions and should not be treated as equivalent.

## Maintainer

Daryll John Medina <br>
Lead Data Scientist
