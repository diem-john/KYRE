#!/usr/bin/env python3
"""CLI evaluation for KYRE trajectory forecasts versus persistence baseline.

Run from the KYRE repository root:
python scripts/evaluate_trajectory_cli.py --checkpoint path/to/best_model.pt --train-csv path/to/train.csv --validation-csv path/to/validation.csv --output-dir outputs/trajectory_eval

Expected columns: z_0...z_n, final_cluster_label, year, week.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--output-dir", default="outputs/trajectory_eval")
    p.add_argument("--pca-prefix", default="z_")
    p.add_argument("--cluster-col", default="final_cluster_label")
    p.add_argument("--year-col", default="year")
    p.add_argument("--week-col", default="week")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-origins", type=int, default=0, help="Maximum forecast origins; 0 means all")
    p.add_argument("--plot-dimensions", type=int, nargs="*", default=[0, 1, 2])
    return p.parse_args()


def position_columns(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


def metrics(actual, pred):
    a, p = np.asarray(actual, dtype=float).ravel(), np.asarray(pred, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if not len(a):
        return {"n": 0, "mae": None, "rmse": None, "mse": None, "bias": None}
    e = p - a
    return {"n": int(len(a)), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mse": float(np.mean(e ** 2)),
            "bias": float(np.mean(e))}


def sort_time(df, year, week):
    return df.sort_values([year, week], kind="stable").reset_index(drop=True)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root)) if str(repo_root) not in sys.path else None
    from src.neural_prediction.inference import NeuralTrajectoryPredictor

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, val = pd.read_csv(args.train_csv), pd.read_csv(args.validation_csv)
    required = {args.cluster_col, args.year_col, args.week_col}
    for label, frame in (("train", train), ("validation", val)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} CSV missing columns: {missing}")
    cols, val_cols = position_columns(train, args.pca_prefix), position_columns(val, args.pca_prefix)
    if not cols or cols != val_cols:
        raise ValueError(f"Training/validation PCA columns must match; found {cols} vs {val_cols}")
    for frame in (train, val):
        frame[args.year_col] = pd.to_numeric(frame[args.year_col], errors="raise")
        frame[args.week_col] = pd.to_numeric(frame[args.week_col], errors="raise")
        frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce")
    train = train.dropna(subset=[args.cluster_col, args.year_col, args.week_col] + cols)
    val = val.dropna(subset=[args.cluster_col, args.year_col, args.week_col] + cols)

    predictor = NeuralTrajectoryPredictor.load(args.checkpoint, device=args.device)
    lookback, horizon = predictor.window_config.lookback, predictor.window_config.horizon
    print(f"Device={predictor.device}; lookback={lookback}; horizon={horizon}; PCA dims={len(cols)}")
    print(f"Train rows={len(train)}; validation rows={len(val)}")

    train_groups = {k: sort_time(g, args.year_col, args.week_col) for k, g in train.groupby(args.cluster_col, sort=False)}
    val_groups = {k: sort_time(g, args.year_col, args.week_col) for k, g in val.groupby(args.cluster_col, sort=False)}
    rows, skipped, origins = [], 0, 0
    for cluster, vg in val_groups.items():
        tg = train_groups.get(cluster)
        if tg is None or tg.empty:
            print(f"WARNING: no training history for cluster {cluster!r}; skipping")
            continue
        first = (vg.iloc[0][args.year_col], vg.iloc[0][args.week_col])
        tg = tg[(tg[args.year_col] < first[0]) | ((tg[args.year_col] == first[0]) & (tg[args.week_col] < first[1]))]
        history = tg[cols].to_numpy(dtype=np.float32)
        times = list(zip(tg[args.year_col], tg[args.week_col]))
        targets = vg[cols].to_numpy(dtype=np.float32)
        target_times = list(zip(vg[args.year_col], vg[args.week_col]))
        if len(history) < lookback:
            skipped += len(vg)
            continue
        for i, (year, week) in enumerate(target_times):
            prior_idx = [j for j, t in enumerate(times) if t < (year, week)]
            prior = history[prior_idx]
            if len(prior) < lookback:
                skipped += 1
                history = np.vstack((history, targets[i]))
                times.append((year, week))
                continue
            pred = predictor.predict(prior[-lookback:])
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred, dtype=float)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]
            if pred.ndim != 2 or pred.shape[1] != len(cols):
                raise ValueError(f"Checkpoint returned {pred.shape}; expected [horizon, {len(cols)}]")
            origins += 1
            last = prior[-1].astype(float)
            for step in range(1, min(horizon, len(targets) - i) + 1):
                j = i + step - 1
                row = {"cluster": cluster, "origin_year": year, "origin_week": week,
                       "target_year": target_times[j][0], "target_week": target_times[j][1],
                       "horizon_step": step}
                for d, col in enumerate(cols):
                    row[f"actual_{col}"] = float(targets[j, d])
                    row[f"neural_pred_{col}"] = float(pred[step - 1, d])
                    row[f"persistence_pred_{col}"] = float(last[d])
                row["neural_euclidean_error"] = float(np.linalg.norm(pred[step - 1] - targets[j]))
                row["persistence_euclidean_error"] = float(np.linalg.norm(last - targets[j]))
                rows.append(row)
            history = np.vstack((history, targets[i]))
            times.append((year, week))
            if args.max_origins and origins >= args.max_origins:
                break
        if args.max_origins and origins >= args.max_origins:
            break
    if not rows:
        raise RuntimeError("No validation forecasts produced. Check cluster IDs, chronology, and available history.")

    df = pd.DataFrame(rows)
    df.to_csv(out / "rolling_predictions.csv", index=False)
    actual = df[[f"actual_{c}" for c in cols]].to_numpy()
    neural = df[[f"neural_pred_{c}" for c in cols]].to_numpy()
    persistence = df[[f"persistence_pred_{c}" for c in cols]].to_numpy()
    nm, pm = metrics(actual, neural), metrics(actual, persistence)
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "lookback": lookback, "forecast_horizon": horizon,
               "pca_columns": cols, "training_rows": len(train), "validation_rows": len(val),
               "forecast_origin_count": origins, "skipped_short_history_count": skipped,
               "neural": nm, "persistence": pm, "neural_beats_persistence_on_mae": nm["mae"] < pm["mae"]}
    (out / "overall_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    horizon_rows = []
    for step, g in df.groupby("horizon_step", sort=True):
        horizon_rows.append({"horizon_step": int(step),
                             "neural_mae": metrics(g[[f"actual_{c}" for c in cols]], g[[f"neural_pred_{c}" for c in cols]])["mae"],
                             "neural_rmse": metrics(g[[f"actual_{c}" for c in cols]], g[[f"neural_pred_{c}" for c in cols]])["rmse"],
                             "persistence_mae": metrics(g[[f"actual_{c}" for c in cols]], g[[f"persistence_pred_{c}" for c in cols]])["mae"],
                             "persistence_rmse": metrics(g[[f"actual_{c}" for c in cols]], g[[f"persistence_pred_{c}" for c in cols]])["rmse"]})
    pd.DataFrame(horizon_rows).to_csv(out / "metrics_by_horizon.csv", index=False)
    dimension_rows = []
    for c in cols:
        n, p = metrics(df[f"actual_{c}"], df[f"neural_pred_{c}"]), metrics(df[f"actual_{c}"], df[f"persistence_pred_{c}"])
        dimension_rows.append({"dimension": c, "neural_mae": n["mae"], "neural_rmse": n["rmse"],
                               "neural_bias": n["bias"], "persistence_mae": p["mae"],
                               "persistence_rmse": p["rmse"], "persistence_bias": p["bias"]})
    pd.DataFrame(dimension_rows).to_csv(out / "metrics_by_dimension.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hm = pd.read_csv(out / "metrics_by_horizon.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(hm.horizon_step, hm.neural_mae, marker="o", label="Neural model MAE")
        ax.plot(hm.horizon_step, hm.persistence_mae, marker="o", label="Persistence MAE")
        ax.set(xlabel="Forecast horizon step", ylabel="MAE across PCA dimensions", title="Forecast error by horizon")
        ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out / "mae_by_horizon.png", dpi=160); plt.close(fig)
        one = df[df.horizon_step == 1].sort_values(["target_year", "target_week"], kind="stable")
        dims = [d for d in args.plot_dimensions if f"{args.pca_prefix}{d}" in cols]
        if dims:
            fig, axes = plt.subplots(len(dims), 1, figsize=(12, max(3.2, 2.8 * len(dims))), squeeze=False)
            for ax, d in zip(axes[:, 0], dims):
                c = f"{args.pca_prefix}{d}"
                ax.plot(np.arange(len(one)), one[f"actual_{c}"], label="Observed", marker=".")
                ax.plot(np.arange(len(one)), one[f"neural_pred_{c}"], label="Neural prediction", marker=".")
                ax.plot(np.arange(len(one)), one[f"persistence_pred_{c}"], label="Persistence baseline", linestyle="--")
                ax.set(title=f"One-step-ahead forecast: {c}", xlabel="Validation observation (chronological order)", ylabel=c)
                ax.grid(True, alpha=.25); ax.legend()
            fig.tight_layout(); fig.savefig(out / "observed_vs_predicted.png", dpi=160); plt.close(fig)
    except ImportError:
        print("WARNING: matplotlib unavailable; plots skipped.")

    print("\n=== Overall validation metrics ===")
    print(f"Neural MAE: {nm['mae']:.6f} | RMSE: {nm['rmse']:.6f}")
    print(f"Persistence MAE: {pm['mae']:.6f} | RMSE: {pm['rmse']:.6f}")
    print(f"Neural beats persistence on MAE: {summary['neural_beats_persistence_on_mae']}")
    print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
