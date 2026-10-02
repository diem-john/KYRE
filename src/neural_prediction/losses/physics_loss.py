from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.neural_prediction.physics.dynamics import trajectory_derivatives


@dataclass(frozen=True)
class LossWeights:
    position: float = 1.0
    velocity: float = 0.25
    acceleration: float = 0.10
    continuity: float = 0.05


class PhysicsLoss(nn.Module):
    """Composite trajectory loss using discrete dynamical constraints."""

    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights\n        self.curriculum_scale = 1.0

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        position = F.smooth_l1_loss(prediction, target)
        pred_velocity, pred_acceleration = trajectory_derivatives(prediction)
        true_velocity, true_acceleration = trajectory_derivatives(target)

        velocity = F.smooth_l1_loss(pred_velocity, true_velocity) if pred_velocity.numel() else prediction.new_tensor(0.0)
        acceleration = F.smooth_l1_loss(pred_acceleration, true_acceleration) if pred_acceleration.numel() else prediction.new_tensor(0.0)

        # Adaptive continuity constraint: penalize predicted speeds that are
        # substantially larger than the observed target dynamics in the sample.
        target_speed = torch.linalg.vector_norm(true_velocity, dim=-1)
        pred_speed = torch.linalg.vector_norm(pred_velocity, dim=-1)
        mean_speed = target_speed.detach().mean(dim=1, keepdim=True)
        std_speed = target_speed.detach().std(dim=1, keepdim=True, unbiased=False)
        speed_limit = mean_speed + 2.0 * std_speed + 1e-8
        continuity = torch.relu(pred_speed - speed_limit).pow(2).mean() if pred_speed.numel() else prediction.new_tensor(0.0)

        total = (
            self.weights.position * position
            + self.weights.velocity * velocity
            + self.weights.acceleration * acceleration
            + self.weights.continuity * continuity
        )
        return total, {
            "total": total,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "continuity": continuity,
        }

    def set_curriculum_scale(self, scale: float) -> None:\n        self.curriculum_scale = max(0.0, min(1.0, float(scale)))\n\n    def config(self) -> dict:
        return asdict(self.weights)
