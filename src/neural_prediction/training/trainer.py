from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    device: str = "auto"
    grad_clip: float = 1.0


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class Trainer:
    def __init__(self, model, loss_fn, config: TrainingConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def _epoch(self, loader, training: bool):
        self.model.train(training)
        totals = {"total": 0.0, "position": 0.0, "velocity": 0.0, "acceleration": 0.0, "continuity": 0.0}
        count = 0
        for batch in loader:
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(x)
            loss, parts = self.loss_fn(prediction, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            n = x.shape[0]
            count += n
            for key, value in parts.items():
                totals[key] += float(value.detach().cpu()) * n
        if not count:
            raise ValueError("DataLoader is empty")
        return {k: v / count for k, v in totals.items()}

    def fit(self, train_dataset, validation_dataset=None, checkpoint_path=None):
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False) if validation_dataset is not None else None
        history, best, bad = [], float("inf"), 0
        for epoch in range(1, self.config.epochs + 1):
            train = self._epoch(train_loader, True)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}}
            if val_loader is not None:
                with torch.no_grad():
                    val = self._epoch(val_loader, False)
                row.update({f"val_{k}": v for k, v in val.items()})
                monitor = val["total"]
            else:
                monitor = train["total"]
            history.append(row)
            if monitor < best:
                best, bad = monitor, 0
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, epoch, best)
            else:
                bad += 1
                if bad >= self.config.patience:
                    break
        return history

    def save_checkpoint(self, path, epoch, best_loss):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": self.model.checkpoint_config(),
            "loss_config": self.loss_fn.config(),
            "training_config": asdict(self.config),
        }, path)
