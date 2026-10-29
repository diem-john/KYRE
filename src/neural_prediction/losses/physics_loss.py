from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """
    Relative weights for the trajectory loss components.
    """

    position: float = 1.0
    velocity: float = 0.25
    acceleration: float = 0.10
    continuity: float = 0.05


class PhysicsLoss(nn.Module):
    """
    Physics/dynamics-aware trajectory loss.

    The returned ``total`` loss is always a differentiable
    torch.Tensor connected to the model prediction.

    Individual loss components are also returned for logging.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
    ) -> None:

        super().__init__()

        self.weights = (
            weights
            if weights is not None
            else LossWeights()
        )

        # Physics curriculum multiplier.
        #
        # 0.0 -> physics terms disabled
        # 1.0 -> full physics regularization
        self.physics_scale = 1.0

    # =========================================================
    # CURRICULUM
    # =========================================================

    def set_curriculum_scale(
        self,
        scale: float,
    ) -> None:
        """
        Set the multiplier applied to physics/dynamics terms.
        """

        self.physics_scale = float(
            max(
                0.0,
                min(
                    1.0,
                    scale,
                ),
            )
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _ensure_tensor(
        value: Any,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a value into a tensor on the same device/dtype
        as the reference tensor.

        This should primarily be used for zero/default values.
        """
        if torch.is_tensor(value):
            return value

        return torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _smooth_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth L1 regression loss.
        """

        return F.smooth_l1_loss(
            prediction,
            target,
            reduction="mean",
        )

    # =========================================================
    # DERIVATIVES
    # =========================================================

    @staticmethod
    def _velocity(
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate discrete velocity from consecutive positions.

        Expected shape:

            [batch, horizon, dimensions]
        """

        if positions.shape[1] < 2:
            return positions[:, :0, :]

        return (
            positions[:, 1:, :]
            - positions[:, :-1, :]
        )

    @staticmethod
    def _acceleration(
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate discrete acceleration from consecutive positions.
        """

        velocity = PhysicsLoss._velocity(
            positions
        )

        if velocity.shape[1] < 2:
            return velocity[:, :0, :]

        return (
            velocity[:, 1:, :]
            - velocity[:, :-1, :]
        )

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """
        Calculate trajectory and physics-aware losses.

        Returns
        -------
        dict
            ``total`` is guaranteed to be a differentiable
            torch.Tensor.

            Other values are tensors where possible and are safe
            for conversion to Python floats by the Trainer.
        """

        if not torch.is_tensor(prediction):
            raise TypeError(
                "PhysicsLoss prediction must be a torch.Tensor."
            )

        if not torch.is_tensor(target):
            raise TypeError(
                "PhysicsLoss target must be a torch.Tensor."
            )

        # -----------------------------------------------------
        # Position loss
        # -----------------------------------------------------

        position_loss = self._smooth_l1(
            prediction,
            target,
        )

        # -----------------------------------------------------
        # Start total from the actual differentiable loss.
        #
        # DO NOT initialize this as:
        #
        #     total = 0.0
        #
        # because that can result in a Python float when
        # physics components are disabled.
        # -----------------------------------------------------

        total = (
            float(self.weights.position)
            * position_loss
        )

        # -----------------------------------------------------
        # Velocity loss
        # -----------------------------------------------------

        predicted_velocity = self._velocity(
            prediction
        )

        target_velocity = self._velocity(
            target
        )

        if predicted_velocity.numel() > 0:

            velocity_loss = self._smooth_l1(
                predicted_velocity,
                target_velocity,
            )

        else:

            # Keep this connected to prediction's graph.
            velocity_loss = (
                prediction.sum() * 0.0
            )

        # -----------------------------------------------------
        # Acceleration loss
        # -----------------------------------------------------

        predicted_acceleration = self._acceleration(
            prediction
        )

        target_acceleration = self._acceleration(
            target
        )

        if predicted_acceleration.numel() > 0:

            acceleration_loss = self._smooth_l1(
                predicted_acceleration,
                target_acceleration,
            )

        else:

            acceleration_loss = (
                prediction.sum() * 0.0
            )

        # -----------------------------------------------------
        # Continuity / excessive-speed penalty
        # -----------------------------------------------------

        if predicted_velocity.numel() > 0:

            predicted_speed = torch.linalg.vector_norm(
                predicted_velocity,
                dim=-1,
            )

            target_speed = torch.linalg.vector_norm(
                target_velocity,
                dim=-1,
            )

            # Use target-derived speed as the physical reference.
            #
            # A small tolerance prevents penalizing normal
            # numerical variation.
            speed_limit = (
                target_speed.detach()
                * 1.10
                + 1e-6
            )

            excess_speed = F.relu(
                predicted_speed
                - speed_limit
            )

            continuity_loss = (
                excess_speed.pow(2).mean()
            )

        else:

            continuity_loss = (
                prediction.sum() * 0.0
            )

        # -----------------------------------------------------
        # Physics curriculum
        # -----------------------------------------------------

        physics_scale = float(
            self.physics_scale
        )

        physics_terms = (
            float(self.weights.velocity)
            * velocity_loss
            +
            float(self.weights.acceleration)
            * acceleration_loss
            +
            float(self.weights.continuity)
            * continuity_loss
        )

        total = (
            total
            +
            physics_scale
            * physics_terms
        )

        # -----------------------------------------------------
        # Safety guarantee
        # -----------------------------------------------------

        if not torch.is_tensor(total):
            raise RuntimeError(
                "PhysicsLoss produced a non-tensor total loss."
            )

        # NOTE:
        # Do NOT require total.requires_grad here.
        #
        # During validation the Trainer correctly uses
        # torch.no_grad(), so the loss will not have a
        # gradient graph. That is expected.
        #
        # The Trainer is responsible for requiring a
        # differentiable loss only during the training pass.

        return {
            "total": total,
            "position": position_loss,
            "velocity": velocity_loss,
            "acceleration": acceleration_loss,
            "continuity": continuity_loss,
        }

    # =========================================================
    # CHECKPOINT CONFIGURATION
    # =========================================================

    def checkpoint_config(self) -> dict[str, Any]:
        """
        Return serializable loss configuration for checkpoints.
        """

        return {
            "weights": {
                "position": float(
                    self.weights.position
                ),
                "velocity": float(
                    self.weights.velocity
                ),
                "acceleration": float(
                    self.weights.acceleration
                ),
                "continuity": float(
                    self.weights.continuity
                ),
            },
            "physics_scale": float(
                self.physics_scale
            ),
        }