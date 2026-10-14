from __future__ import annotations

import json
from pathlib import Path


def list_checkpoints(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)


def read_checkpoint_metadata(path: str | Path) -> dict:
    """Read lightweight metadata without requiring model construction."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_loss": checkpoint.get("best_loss"),
        "model_config": checkpoint.get("model_config", {}),
        "loss_config": checkpoint.get("loss_config", {}),
        "training_config": checkpoint.get("training_config", {}),
    }
