from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.neural_prediction.models.factory import build_model
from src.neural_prediction.models.physics_lstm import ModelConfig
from src.neural_prediction.models.hybrid_cnn_lstm_attention import HybridModelConfig
from src.neural_prediction.data.dataset import WindowConfig, _features


class NeuralTrajectoryPredictor:
    """Load a trained KYRE checkpoint and forecast a single trajectory."""

    def __init__(self, model, window_config, device="cpu"):
        self.model = model.to(device).eval()
        self.window_config = window_config
        self.device = torch.device(device)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", "physics_lstm")
        if model_type == "hybrid":
            mc = HybridModelConfig(**checkpoint["model_config"])
            model = build_model("hybrid", mc)
        else:
            mc = ModelConfig(**checkpoint["model_config"])
            model = build_model("physics_lstm", mc)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_window_config = checkpoint.get("window_config", {})
        if not isinstance(saved_window_config, dict):
            saved_window_config = {
                "lookback": getattr(saved_window_config, "lookback", 12),
                "stride": getattr(saved_window_config, "stride", 1),
                "include_velocity": getattr(saved_window_config, "include_velocity", True),
                "include_acceleration": getattr(saved_window_config, "include_acceleration", True),
            }
        window_config = WindowConfig(
            lookback=saved_window_config.get("lookback", 12),
            horizon=mc.horizon,
            stride=saved_window_config.get("stride", 1),
            include_velocity=saved_window_config.get("include_velocity", True),
            include_acceleration=saved_window_config.get("include_acceleration", True),
        )
        return cls(model, window_config, device)

    @torch.no_grad()
    def predict(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or history.shape[0] < self.window_config.lookback:
            raise ValueError(f"history must have shape [weeks, dimensions] with at least {self.window_config.lookback} weeks")
        x_position = history[-self.window_config.lookback:]
        x = _features(x_position, self.window_config)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze(0).cpu().numpy()
