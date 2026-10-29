from __future__ import annotations

import numpy as np


def trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Aggregate trajectory metrics over [samples, horizon, dimensions]."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError("y_true and y_pred must have identical shape [samples, horizon, dimensions]")

    error = y_pred - y_true
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    denom = np.linalg.norm(true_flat, axis=1) * np.linalg.norm(pred_flat, axis=1)
    cosine = np.divide(np.sum(true_flat * pred_flat, axis=1), denom, out=np.ones_like(denom), where=denom > 1e-12)

    true_velocity = np.diff(y_true, axis=1)
    pred_velocity = np.diff(y_pred, axis=1)
    velocity_error = float(np.mean(np.linalg.norm(pred_velocity - true_velocity, axis=-1))) if true_velocity.size else 0.0

    true_displacement = y_true[:, -1] - y_true[:, 0]
    pred_displacement = y_pred[:, -1] - y_pred[:, 0]
    ddenom = np.linalg.norm(true_displacement, axis=1) * np.linalg.norm(pred_displacement, axis=1)
    displacement_cosine = np.divide(np.sum(true_displacement * pred_displacement, axis=1), ddenom, out=np.ones_like(ddenom), where=ddenom > 1e-12)

    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "cosine_similarity": float(np.mean(cosine)),
        "velocity_error": velocity_error,
        "displacement_cosine_similarity": float(np.mean(displacement_cosine)),
        "displacement_magnitude_error": float(np.mean(np.abs(np.linalg.norm(pred_displacement, axis=1) - np.linalg.norm(true_displacement, axis=1)))),
    }


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    """Calculate metrics independently at each forecast horizon."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for h in range(y_true.shape[1]):
        yt, yp = y_true[:, h:h + 1, :], y_pred[:, h:h + 1, :]
        row = trajectory_metrics(yt, yp)
        row["horizon"] = h + 1
        rows.append(row)
    return rows
