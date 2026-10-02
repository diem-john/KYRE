from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridModelConfig:
    input_dim: int
    embedding_dim: int
    horizon: int = 6
    conv_channels: int = 128
    conv_kernel_size: int = 3
    hidden_dim: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.2


class ConvLSTMAttention(nn.Module):
    """Hybrid temporal encoder: local convolution -> recurrent dynamics -> attention.

    The three blocks have distinct roles:
    * Conv1d extracts short-range temporal motifs.
    * LSTM models ordered state evolution.
    * Multi-head self-attention reweights the encoded history before decoding.
    """

    def __init__(self, config: HybridModelConfig):
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.config = config
        padding = config.conv_kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_dim, config.conv_channels, config.conv_kernel_size, padding=padding),
            nn.BatchNorm1d(config.conv_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.conv_channels, config.hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        lstm_dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.horizon * config.embedding_dim),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # [B, T, F] -> [B, F, T]
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        recurrent, _ = self.lstm(conv)
        recurrent = self.norm(recurrent)
        attended, weights = self.attention(recurrent, recurrent, recurrent, need_weights=return_attention)
        encoded = self.norm(attended + recurrent)
        encoded = self.norm(encoded + self.ffn(encoded))
        # Attention-weighted temporal pooling; this keeps the complete history
        # available instead of using only the final recurrent state.
        state = encoded.mean(dim=1)
        output = self.decoder(state).view(x.shape[0], self.config.horizon, self.config.embedding_dim)
        if return_attention:
            return output, weights
        return output

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
