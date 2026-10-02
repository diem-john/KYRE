from __future__ import annotations

from .physics_lstm import ModelConfig, PhysicsLSTM
from .hybrid_cnn_lstm_attention import HybridModelConfig, ConvLSTMAttention


def build_model(name: str, config):
    name = name.lower()
    if name == "physics_lstm":
        return PhysicsLSTM(config)
    if name == "hybrid":
        return ConvLSTMAttention(config)
    raise ValueError(f"Unknown neural model: {name}")
