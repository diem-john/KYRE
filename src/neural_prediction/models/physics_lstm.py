from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    embedding_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    horizon: int = 6


class PhysicsLSTM(nn.Module):
    """Multi-step LSTM trajectory predictor.

    The network predicts future centroid embeddings; dynamical constraints are
    deliberately implemented in the loss module so architecture and dynamics
    assumptions remain independently configurable.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.horizon * config.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        state = encoded[:, -1, :]
        output = self.head(state)
        return output.view(x.shape[0], self.config.horizon, self.config.embedding_dim)

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
