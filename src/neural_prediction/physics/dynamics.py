from __future__ import annotations

import torch


def trajectory_derivatives(trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete velocity and acceleration along the time dimension."""
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape [batch, time, dimension]")
    velocity = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    return velocity, acceleration


def dynamics_statistics(trajectory: torch.Tensor) -> dict[str, float]:
    velocity, acceleration = trajectory_derivatives(trajectory)
    return {
        "mean_speed": float(torch.linalg.vector_norm(velocity, dim=-1).mean().detach().cpu()),
        "max_speed": float(torch.linalg.vector_norm(velocity, dim=-1).max().detach().cpu()),
        "mean_acceleration": float(torch.linalg.vector_norm(acceleration, dim=-1).mean().detach().cpu()) if acceleration.numel() else 0.0,
        "max_acceleration": float(torch.linalg.vector_norm(acceleration, dim=-1).max().detach().cpu()) if acceleration.numel() else 0.0,
    }
