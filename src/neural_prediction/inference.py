
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models.physics_lstm import ModelConfig, PhysicsLSTM
from .models.hybrid_cnn_lstm_attention import (
    HybridModelConfig,
    ConvLSTMAttention,
)
from .data.dataset import WindowConfig, _features
from .preprocessing_artifacts import (
    load_preprocessing_artifacts,
    transform_embeddings,
)


class NeuralTrajectoryPredictor:
    """
    Loads a saved neural trajectory checkpoint and performs inference.

    Supports:
    - Physics-Regularized LSTM
    - Hybrid CNN + LSTM + Attention

    The predictor accepts either:
    1. Raw PCA position history [weeks, embedding_dimensions].
       Velocity and acceleration features are constructed according to
       the checkpoint's saved WindowConfig.
    2. Model-ready history [weeks, input_features].

    Predictions are returned as a CPU-independent PyTorch tensor on
    the model's device, with shape [horizon, embedding_dimensions]
    for a single trajectory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.checkpoint = checkpoint or {}

        self.preprocessing_artifacts = None
        artifact_path = self.checkpoint.get("preprocessing_artifact_path")
        if artifact_path:
            self.preprocessing_artifacts = load_preprocessing_artifacts(
                artifact_path
            )

        saved_window_config = self.checkpoint.get("window_config", {})
        if not isinstance(saved_window_config, dict):
            saved_window_config = {
                "lookback": getattr(saved_window_config, "lookback", 12),
                "horizon": getattr(saved_window_config, "horizon", None),
                "stride": getattr(saved_window_config, "stride", 1),
                "include_velocity": getattr(
                    saved_window_config, "include_velocity", True
                ),
                "include_acceleration": getattr(
                    saved_window_config, "include_acceleration", True
                ),
            }

        model_config = self.checkpoint.get("model_config", {})
        model_horizon = (
            model_config.get("horizon", 6)
            if isinstance(model_config, dict)
            else 6
        )

        self.window_config = WindowConfig(
            lookback=int(saved_window_config.get("lookback", 12)),
            horizon=int(
                saved_window_config.get("horizon") or model_horizon
            ),
            stride=int(saved_window_config.get("stride", 1)),
            include_velocity=bool(
                saved_window_config.get("include_velocity", True)
            ),
            include_acceleration=bool(
                saved_window_config.get("include_acceleration", True)
            ),
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> "NeuralTrajectoryPredictor":
        """Load a checkpoint and reconstruct its model architecture."""

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        if device == "auto":
            resolved_device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        else:
            resolved_device = device

        map_location = torch.device(resolved_device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

        if isinstance(checkpoint, dict):
            # Prefer the stored path, but support moving a checkpoint and its
            # companion artifact together to another machine/directory.
            artifact_value = checkpoint.get("preprocessing_artifact_path")
            if artifact_value:
                artifact_path = Path(artifact_value)
                if not artifact_path.is_file():
                    companion_path = checkpoint_path.parent / artifact_path.name
                    if companion_path.is_file():
                        checkpoint["preprocessing_artifact_path"] = str(
                            companion_path.resolve()
                        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid checkpoint: expected a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint is missing 'model_state_dict'."
            )

        if "model_config" not in checkpoint:
            raise KeyError(
                "Checkpoint is missing 'model_config'."
            )

        model_config = checkpoint["model_config"]
        if not isinstance(model_config, dict):
            raise TypeError(
                "Checkpoint 'model_config' must be a dictionary."
            )

        model_type = checkpoint.get("model_type", "physics_lstm")

        if model_type == "hybrid":
            config = HybridModelConfig(**model_config)
            model = ConvLSTMAttention(config)
        elif model_type == "physics_lstm":
            config = ModelConfig(**model_config)
            model = PhysicsLSTM(config)
        else:
            raise ValueError(
                f"Unsupported checkpoint model_type: {model_type!r}"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        return cls(
            model=model,
            device=map_location,
            checkpoint=checkpoint,
        )

    @torch.inference_mode()
    def predict(
        self,
        history: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future trajectory values from historical observations.

        Parameters
        ----------
        history:
            Raw position history [lookback, embedding_dimensions] or
            model-ready features [lookback, input_features].
            A batched tensor [batch, lookback, features] is also accepted
            when it is already model-ready.

        Returns
        -------
        torch.Tensor
            For a single trajectory, predictions have shape
            [horizon, embedding_dimensions]. The result stays on the
            model's device.
        """

        if torch.is_tensor(history):
            history_array = history.detach().cpu().numpy()
        else:
            history_array = np.asarray(history, dtype=np.float32)

        history_array = np.asarray(
            history_array, dtype=np.float32
        )

        if history_array.ndim not in (2, 3):
            raise ValueError(
                "Expected history shaped [weeks, features] or "
                "[batch, weeks, features]. "
                f"Received shape {history_array.shape}."
            )

        if history_array.shape[-2] < self.window_config.lookback:
            raise ValueError(
                "Insufficient history: expected at least "
                f"{self.window_config.lookback} weeks, received "
                f"{history_array.shape[-2]}."
            )

        # Use only the most recent lookback window.
        history_array = history_array[
            ..., -self.window_config.lookback:, :
        ]

        # If the caller supplies raw embeddings, apply the exact scaler/PCA
        # fitted during training before constructing temporal features.
        # PCA-space input is left untouched to avoid a double transform.
        if self.preprocessing_artifacts is not None:
            raw_dim = int(
                self.preprocessing_artifacts["input_dimension"]
            )
            pca_dim = int(
                self.preprocessing_artifacts["output_dimension"]
            )
            supplied_dim = int(history_array.shape[-1])
            if supplied_dim == raw_dim and raw_dim != pca_dim:
                if history_array.ndim != 2:
                    raise ValueError(
                        "Raw embedding batches must be transformed per "
                        "trajectory; supply a single [weeks, dimensions] "
                        "history or pre-transformed batched features."
                    )
                history_array = transform_embeddings(
                    history_array, self.preprocessing_artifacts
                )
            elif supplied_dim not in (raw_dim, pca_dim):
                raise ValueError(
                    "Input dimensions do not match the saved preprocessing "
                    f"schema: expected raw={raw_dim} or PCA={pca_dim}, "
                    f"received {supplied_dim}."
                )

        expected_features = getattr(
            getattr(self.model, "config", None),
            "input_dim",
            None,
        )

        # If the input is raw PCA positions, construct the same
        # position/velocity/acceleration features used during training.
        if (
            expected_features is not None
            and history_array.shape[-1] != expected_features
        ):
            if history_array.ndim != 2:
                raise ValueError(
                    "Batched raw-position histories are not supported "
                    "for automatic feature construction. Supply "
                    "model-ready batched features instead."
                )

            history_array = _features(
                history_array,
                self.window_config,
            )
            history_array = np.asarray(
                history_array, dtype=np.float32
            )

        if (
            expected_features is not None
            and history_array.shape[-1] != expected_features
        ):
            raise ValueError(
                "Input feature dimension mismatch after preprocessing: "
                f"checkpoint expects {expected_features}, received "
                f"{history_array.shape[-1]}."
            )

        tensor = torch.as_tensor(
            history_array,
            dtype=torch.float32,
            device=self.device,
        )

        single_trajectory = tensor.ndim == 2
        if single_trajectory:
            tensor = tensor.unsqueeze(0)

        self.model.eval()
        prediction = self.model(tensor)

        if not torch.is_tensor(prediction):
            raise TypeError(
                "The model must return a torch.Tensor; received "
                f"{type(prediction).__name__}."
            )

        if prediction.ndim != 3:
            raise ValueError(
                "Expected model output shaped "
                "[batch, horizon, embedding_dimensions]; received "
                f"{tuple(prediction.shape)}."
            )

        if prediction.shape[1] < 1:
            raise ValueError("The model returned an empty forecast.")

        if single_trajectory:
            prediction = prediction.squeeze(0)

        return prediction
