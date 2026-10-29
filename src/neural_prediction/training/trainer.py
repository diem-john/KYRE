from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip: float = 1.0
    device: str = "auto"

    # Physics-loss curriculum
    physics_warmup_epochs: int = 5
    physics_ramp_epochs: int = 10


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """
    Neural trajectory model trainer.

    Supports:
    - AdamW optimization
    - gradient clipping
    - validation
    - early stopping
    - checkpointing
    - physics-loss curriculum
    - realtime epoch callback for Streamlit telemetry

    Important gradient behavior
    ----------------------------
    Training:
        Gradients are enabled and total loss MUST be connected
        to the model computation graph.

    Validation:
        Gradients are disabled intentionally. The validation loss
        does NOT need requires_grad=True.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        if config.device == "cuda":

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        elif config.device == "cpu":

            self.device = torch.device(
                "cpu"
            )

        else:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        self.model.to(self.device)

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _numeric_value(
        value: Any,
    ) -> float:
        """
        Safely convert tensor / numpy scalar / Python numeric
        to float.

        Supports:
        - torch.Tensor
        - numpy.ndarray
        - numpy scalar
        - Python float/int
        """

        if torch.is_tensor(value):

            return float(
                value.detach()
                .cpu()
                .item()
            )

        if isinstance(
            value,
            np.ndarray,
        ):

            return float(
                value.item()
            )

        if isinstance(
            value,
            np.generic,
        ):

            return float(
                value.item()
            )

        return float(value)

    # ---------------------------------------------------------

    @staticmethod
    def _move_batch(
        batch: Any,
        device: torch.device,
    ):
        """
        Move tensors in a batch to the selected device.

        Supports:
        - tuple
        - list
        - dictionary
        - tensor
        """

        if isinstance(
            batch,
            (list, tuple),
        ):

            return tuple(
                item.to(device)
                if torch.is_tensor(item)
                else item
                for item in batch
            )

        if isinstance(
            batch,
            dict,
        ):

            return {
                key: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }

        if torch.is_tensor(batch):

            return batch.to(device)

        return batch

    # =========================================================
    # PHYSICS CURRICULUM
    # =========================================================

    def _set_physics_scale(
        self,
        epoch: int,
    ) -> float:
        """
        Physics curriculum.

        warmup:
            scale = 0

        ramp:
            scale increases linearly from 0 -> 1

        after ramp:
            scale = 1

        Epoch is zero-indexed.
        """

        warmup = max(
            0,
            int(
                self.config.physics_warmup_epochs
            ),
        )

        ramp = max(
            0,
            int(
                self.config.physics_ramp_epochs
            ),
        )

        # -----------------------------------------------------
        # Warm-up
        # -----------------------------------------------------

        if epoch < warmup:

            scale = 0.0

        # -----------------------------------------------------
        # No ramp
        # -----------------------------------------------------

        elif ramp == 0:

            scale = 1.0

        # -----------------------------------------------------
        # Ramp
        # -----------------------------------------------------

        else:

            ramp_position = (
                epoch
                - warmup
                + 1
            )

            scale = min(
                1.0,
                ramp_position
                / float(ramp),
            )

        # -----------------------------------------------------
        # Update loss function
        # -----------------------------------------------------

        if hasattr(
            self.loss_fn,
            "set_curriculum_scale",
        ):

            self.loss_fn.set_curriculum_scale(
                scale
            )

        return scale

    # =========================================================
    # EPOCH
    # =========================================================

    def _epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> dict[str, float]:
        """
        Execute one training or validation epoch.

        Parameters
        ----------
        loader:
            DataLoader containing the dataset.

        training:
            True  -> training mode with gradients/backpropagation.
            False -> validation mode with no gradients.

        Returns
        -------
        dict[str, float]
            Averaged loss components.
        """

        # -----------------------------------------------------
        # Model mode
        # -----------------------------------------------------

        if training:

            self.model.train()

        else:

            self.model.eval()

        # -----------------------------------------------------
        # Telemetry accumulators
        # -----------------------------------------------------

        totals: dict[str, float] = {}

        total_samples = 0

        # -----------------------------------------------------
        # Explicit gradient context
        #
        # This is the important fix.
        #
        # Training:
        #     gradients enabled
        #
        # Validation:
        #     gradients disabled
        # -----------------------------------------------------

        with torch.set_grad_enabled(
            training
        ):

            for batch in loader:

                # =================================================
                # MOVE BATCH TO DEVICE
                # =================================================

                batch = self._move_batch(
                    batch,
                    self.device,
                )

                # =================================================
                # DATASET FORMAT
                # =================================================

                if isinstance(
                    batch,
                    dict,
                ):

                    if "x" not in batch:
                        raise KeyError(
                            "Training batch dictionary must "
                            "contain key 'x'."
                        )

                    if "y" not in batch:
                        raise KeyError(
                            "Training batch dictionary must "
                            "contain key 'y'."
                        )

                    x = batch["x"]
                    y = batch["y"]

                    extra = {
                        key: value
                        for key, value in batch.items()
                        if key not in {
                            "x",
                            "y",
                        }
                    }

                else:

                    if not isinstance(
                        batch,
                        (tuple, list),
                    ):
                        raise TypeError(
                            "Training batch must be a tuple, "
                            "list, or dictionary."
                        )

                    if len(batch) < 2:
                        raise ValueError(
                            "Training batch must contain at "
                            "least x and y."
                        )

                    x = batch[0]
                    y = batch[1]

                    rest = batch[2:]

                    extra = {}

                    if rest:
                        extra["extra"] = rest

                # =================================================
                # BASIC TENSOR VALIDATION
                # =================================================

                if not torch.is_tensor(x):
                    raise TypeError(
                        "Input x must be a torch.Tensor."
                    )

                if not torch.is_tensor(y):
                    raise TypeError(
                        "Target y must be a torch.Tensor."
                    )

                # =================================================
                # ZERO GRADIENTS
                # =================================================

                if training:

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                # =================================================
                # FORWARD
                # =================================================

                prediction = self.model(x)

                # =================================================
                # LOSS
                # =================================================

                try:

                    loss_output = self.loss_fn(
                        prediction,
                        y,
                    )

                except TypeError:

                    # Allows future physics losses that accept
                    # additional batch information.
                    loss_output = self.loss_fn(
                        prediction,
                        y,
                        **extra,
                    )

                # =================================================
                # NORMALIZE LOSS OUTPUT
                # =================================================

                if isinstance(
                    loss_output,
                    dict,
                ):

                    losses = loss_output

                    if "total" not in losses:

                        raise ValueError(
                            "PhysicsLoss must return a "
                            "dictionary containing 'total'."
                        )

                    total_loss = losses["total"]

                else:

                    total_loss = loss_output

                    losses = {
                        "total": total_loss
                    }

                # =================================================
                # TRAINING-ONLY LOSS VALIDATION
                # =================================================

                if training:

                    # ---------------------------------------------
                    # Total must be a tensor
                    # ---------------------------------------------

                    if not torch.is_tensor(
                        total_loss
                    ):

                        raise TypeError(
                            "The total training loss must be "
                            "a torch.Tensor so that backward() "
                            "can be called. Individual logging "
                            "components may be Python floats."
                        )

                    # ---------------------------------------------
                    # Total must be connected to the model graph
                    # ---------------------------------------------

                    if not total_loss.requires_grad:

                        raise RuntimeError(
                            "The total training loss is not "
                            "connected to the model computation "
                            "graph. Check the model forward pass "
                            "and loss implementation."
                        )

                    # ---------------------------------------------
                    # Backpropagation
                    # ---------------------------------------------

                    total_loss.backward()

                    # ---------------------------------------------
                    # Gradient clipping
                    # ---------------------------------------------

                    if self.config.grad_clip is not None:

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.grad_clip,
                        )

                    # ---------------------------------------------
                    # Optimizer step
                    # ---------------------------------------------

                    self.optimizer.step()

                # =================================================
                # DETERMINE BATCH SIZE
                # =================================================

                if x.ndim > 0:

                    n = int(
                        x.shape[0]
                    )

                elif y.ndim > 0:

                    n = int(
                        y.shape[0]
                    )

                else:

                    n = 1

                total_samples += n

                # =================================================
                # AGGREGATE TELEMETRY
                # =================================================

                for key, value in losses.items():

                    numeric_value = (
                        self._numeric_value(
                            value
                        )
                    )

                    totals[key] = (
                        totals.get(
                            key,
                            0.0,
                        )
                        + numeric_value * n
                    )

        # =====================================================
        # EMPTY DATASET
        # =====================================================

        if total_samples == 0:

            return {
                key: 0.0
                for key in totals
            }

        # =====================================================
        # SAMPLE-WEIGHTED AVERAGES
        # =====================================================

        return {
            key: value / total_samples
            for key, value in totals.items()
        }

    # =========================================================
    # FIT
    # =========================================================

    def fit(
        self,
        train_dataset,
        validation_dataset,
        checkpoint_path: Path,
        window_config=None,
        epoch_callback: Optional[
            Callable[
                [dict[str, Any]],
                None,
            ]
        ] = None,
    ) -> list[dict[str, Any]]:
        """
        Train the model.

        Parameters
        ----------
        train_dataset:
            Training dataset.

        validation_dataset:
            Validation dataset.

        checkpoint_path:
            Destination for the best checkpoint.

        window_config:
            Optional trajectory window configuration.

        epoch_callback:
            Optional callback executed after every epoch.

            Intended for Streamlit realtime telemetry.
        """

        # =====================================================
        # DATA LOADERS
        # =====================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # =====================================================
        # CHECKPOINT PATH
        # =====================================================

        checkpoint_path = Path(
            checkpoint_path
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # HISTORY
        # =====================================================

        history: list[
            dict[str, Any]
        ] = []

        # =====================================================
        # EARLY STOPPING STATE
        # =====================================================

        best_val_loss = float(
            "inf"
        )

        best_epoch = -1

        epochs_without_improvement = 0

        # =====================================================
        # TRAINING LOOP
        # =====================================================

        for epoch in range(
            int(self.config.epochs)
        ):

            # =================================================
            # PHYSICS CURRICULUM
            # =================================================

            physics_scale = (
                self._set_physics_scale(
                    epoch
                )
            )

            # =================================================
            # TRAIN
            # =================================================

            train_metrics = self._epoch(
                train_loader,
                training=True,
            )

            # =================================================
            # VALIDATION
            # =================================================

            val_metrics = self._epoch(
                validation_loader,
                training=False,
            )

            # =================================================
            # EPOCH RECORD
            # =================================================

            record: dict[
                str,
                Any,
            ] = {

                "epoch": epoch + 1,

                "physics_scale": (
                    physics_scale
                ),

                "learning_rate": (
                    self.optimizer
                    .param_groups[0]["lr"]
                ),
            }

            # -------------------------------------------------
            # Training metrics
            # -------------------------------------------------

            for key, value in train_metrics.items():

                record[
                    f"train_{key}"
                ] = float(value)

            # -------------------------------------------------
            # Validation metrics
            # -------------------------------------------------

            for key, value in val_metrics.items():

                record[
                    f"val_{key}"
                ] = float(value)

            history.append(
                record
            )

            # =================================================
            # VALIDATION OBJECTIVE
            # =================================================

            val_total = float(
                val_metrics.get(
                    "total",
                    np.inf,
                )
            )

            # =================================================
            # BEST MODEL
            # =================================================

            if val_total < best_val_loss:

                best_val_loss = val_total

                best_epoch = epoch + 1

                epochs_without_improvement = 0

                self._save_checkpoint(
                    checkpoint_path,
                    epoch=epoch + 1,
                    best_loss=best_val_loss,
                    window_config=window_config,
                    physics_scale=physics_scale,
                )

            else:

                epochs_without_improvement += 1

            # =================================================
            # REALTIME CALLBACK
            # =================================================

            if epoch_callback is not None:

                telemetry = {
                    **record,

                    "best_val_loss": (
                        best_val_loss
                    ),

                    "best_epoch": (
                        best_epoch
                    ),

                    "epochs_without_improvement": (
                        epochs_without_improvement
                    ),
                }

                epoch_callback(
                    telemetry
                )

            # =================================================
            # EARLY STOPPING
            # =================================================

            if (
                epochs_without_improvement
                >= self.config.patience
            ):

                break

        return history

    # =========================================================
    # CHECKPOINT
    # =========================================================

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_loss: float,
        window_config=None,
        physics_scale: float = 1.0,
    ):
        """
        Save the best model checkpoint.

        The checkpoint contains:
        - model state
        - optimizer state
        - epoch
        - best validation loss
        - model configuration
        - loss configuration
        - training configuration
        - window configuration
        - physics curriculum
        - model type
        """

        # =====================================================
        # MODEL CONFIGURATION
        # =====================================================

        if hasattr(
            self.model,
            "checkpoint_config",
        ):

            model_config = (
                self.model.checkpoint_config()
            )

        elif hasattr(
            self.model,
            "config",
        ):

            model_config = asdict(
                self.model.config
            )

        else:

            model_config = {}

        # =====================================================
        # LOSS CONFIGURATION
        # =====================================================

        if hasattr(
            self.loss_fn,
            "checkpoint_config",
        ):

            loss_config = (
                self.loss_fn.checkpoint_config()
            )

        elif hasattr(
            self.loss_fn,
            "config",
        ):

            loss_config = asdict(
                self.loss_fn.config
            )

        else:

            loss_config = {}

        # =====================================================
        # CHECKPOINT
        # =====================================================

        checkpoint = {

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "epoch": epoch,

            "best_loss": best_loss,

            "model_config": model_config,

            "loss_config": loss_config,

            "training_config": asdict(
                self.config
            ),

            "window_config": (
                asdict(window_config)
                if window_config is not None
                else None
            ),

            "physics_curriculum": {

                "current_scale": (
                    physics_scale
                ),

                "warmup_epochs": (
                    self.config
                    .physics_warmup_epochs
                ),

                "ramp_epochs": (
                    self.config
                    .physics_ramp_epochs
                ),
            },
        }

        # =====================================================
        # MODEL TYPE
        # =====================================================

        model_name = (
            self.model
            .__class__
            .__name__
            .lower()
        )

        if (
            "conv" in model_name
            or "hybrid" in model_name
        ):

            checkpoint[
                "model_type"
            ] = "hybrid"

        else:

            checkpoint[
                "model_type"
            ] = "physics_lstm"

        # =====================================================
        # SAVE
        # =====================================================

        torch.save(
            checkpoint,
            path,
        )