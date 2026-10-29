import pandas as pd
from pathlib import Path

import numpy as np

from src.trajectory_prediction.metrics import _weighted_mean, compute_metrics, to_original_space, iter_cluster_arrays, median_historical_displacement
from src.trajectory_prediction.utils.preprocessing import embedding_cols, sort_time, iso_week_start
from src.trajectory_prediction.statistical_predictors import make_model, recursive_predictions, rolling_predictions


def _add_space_suffix(stats: dict, suffix: str) -> dict:
    return {f"{key}_{suffix}": value for key, value in stats.items()}


def _average_metric_rows(rows, group_cols, weight_col):
    """
    Average raw metric rows while keeping one row per requested group.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    excluded_cols = set(group_cols + [
        "backtest_window",
        "cutoff",
        "is_latest_backtest",
        "target_year",
        "target_week",
    ])

    metric_cols = [
        col for col in df.columns
        if col not in excluded_cols and col != weight_col
    ]

    agg_dict = {col: (col, "mean") for col in metric_cols}
    agg_dict["n_backtests"] = ("backtest_window", "count")

    if weight_col in df.columns:
        agg_dict[weight_col] = (weight_col, "mean")

    return (
        df
        .groupby(group_cols, as_index=False)
        .agg(**agg_dict)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def summarize_reliable_horizon(
    horizon_metrics_df,
    median_historical_displacement,
    min_cosine=0.95,
    min_rmse_improvement=0.00,
    min_displacement_cosine=0.25,
    weight_col=None,
    cluster_col=None,
):
    """
    Summarize reliable forecast horizons separately for train/test metric splits.
    """
    if horizon_metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "rmse_original",
        "mse_original",
        "cosine_similarity_original",
        "rmse_improvement_vs_persistence_original",
        "persistence_rmse_original",
        "displacement_cosine_similarity_original",
        "displacement_magnitude_error_original",
    ]

    group_cols = ["eval_split", "model", "horizon"] if "eval_split" in horizon_metrics_df.columns else ["model", "horizon"]

    horizon_summary = (
        horizon_metrics_df
        .groupby(group_cols, as_index=False)
        .agg(
            n_forecasts=(cluster_col, "count"),
            n_backtests=("n_backtests", "sum"),
            **{f"mean_{col}": (col, "mean") for col in metric_cols},
        )
    )

    weighted_summary = (
        horizon_metrics_df
        .groupby(group_cols)
        .apply(
            lambda group: pd.Series({
                "total_n_users": group[weight_col].sum() if weight_col in group.columns else np.nan,
                **{
                    f"weighted_mean_{col}": _weighted_mean(
                        group[col],
                        group[weight_col] if weight_col in group.columns else pd.Series(np.ones(len(group))),
                    )
                    for col in metric_cols
                },
            })
        )
        .reset_index()
    )

    horizon_summary = horizon_summary.merge(weighted_summary, on=group_cols, how="left")
    horizon_summary["median_historical_displacement"] = median_historical_displacement

    horizon_summary["is_reliable"] = (
        (horizon_summary["mean_rmse_original"] < horizon_summary["median_historical_displacement"])
        & (horizon_summary["weighted_mean_rmse_original"] < horizon_summary["median_historical_displacement"])
        & (horizon_summary["mean_cosine_similarity_original"] >= min_cosine)
        & (horizon_summary["weighted_mean_cosine_similarity_original"] >= min_cosine)
        & (horizon_summary["mean_displacement_cosine_similarity_original"] > min_displacement_cosine)
        & (horizon_summary["weighted_mean_displacement_cosine_similarity_original"] > min_displacement_cosine)
        & (horizon_summary["mean_displacement_magnitude_error_original"] < horizon_summary["median_historical_displacement"])
        & (horizon_summary["weighted_mean_displacement_magnitude_error_original"] < horizon_summary["median_historical_displacement"])
    )

    horizon_summary["has_good_direction"] = (
        (horizon_summary["mean_displacement_cosine_similarity_original"] > min_displacement_cosine)
        & (horizon_summary["weighted_mean_displacement_cosine_similarity_original"] > min_displacement_cosine)
    )

    # Backward-compatible aliases for display/save code.
    alias_map = {
        "cosine_similarity": "mean_cosine_similarity_original",
        "rmse": "mean_rmse_original",
        "mse": "mean_mse_original",
        "rmse_improvement_vs_persistence": "mean_rmse_improvement_vs_persistence_original",
        "displacement_cosine_similarity": "mean_displacement_cosine_similarity_original",
        "displacement_magnitude_error": "mean_displacement_magnitude_error_original",
    }

    for alias, source in alias_map.items():
        if source in horizon_summary.columns:
            horizon_summary[alias] = horizon_summary[source]

    rows = []
    selector_cols = ["eval_split", "model"] if "eval_split" in horizon_summary.columns else ["model"]

    for keys, group in horizon_summary.groupby(selector_cols, sort=False):
        group = group.sort_values("horizon")
        reliable = group[group["is_reliable"]]
        good_direction = group[group["has_good_direction"]]

        row = dict(zip(selector_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "reliable_horizon": int(reliable["horizon"].max()) if not reliable.empty else 0,
            "n_reliable_horizons": int(reliable["is_reliable"].sum()),
            "good_direction_horizon": int(good_direction["horizon"].max()) if not good_direction.empty else 0,
            "n_good_direction_horizons": int(good_direction["has_good_direction"].sum()),
        })

        first_horizon = group[group["horizon"] == 1]
        if not first_horizon.empty:
            for col in horizon_summary.columns:
                if col not in selector_cols + ["horizon"]:
                    row[col] = first_horizon.iloc[0][col]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(selector_cols).reset_index(drop=True)


def save_prediction_and_metrics(
    pred_df,
    horizon_metrics_df,
    reliable_horizon_summary_df,
    one_step_metrics_df,
    n_weeks,
    project_name,
    cluster_col,
    user_weight_col
):
    """
    Save prediction rows and metric tables.

    pred_df can contain either:
    - unknown_future predictions after the latest available week, or
    - known_holdout predictions for the latest known weeks.
    """
    if pred_df.empty:
        return

    model_name = pred_df.iloc[0]["model"]
    prediction_mode = pred_df.iloc[0].get("prediction_mode", "unknown_future")

    output_dir = Path("projects") / project_name / "centroid_trajectory_predictions" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reliable_horizon_summary_df.empty:
        first_summary = reliable_horizon_summary_df.iloc[0]
        cosine_similarity_mean = np.round(first_summary.get("mean_cosine_similarity_original", np.nan), 4)
        reliable_horizon = first_summary.get("reliable_horizon", 0)
    else:
        cosine_similarity_mean = np.nan
        reliable_horizon = 0

    metric_cols = [
        cluster_col,
        "eval_split",
        "horizon",
        "n_backtests",
        user_weight_col,
        "cosine_similarity_original",
        "rmse_original",
        "mse_original",
        "displacement_cosine_similarity_original",
        "displacement_magnitude_error_original",
        "persistence_rmse_original",
        "rmse_improvement_vs_persistence_original",
        "cosine_similarity_pca",
        "rmse_pca",
        "mse_pca",
        "displacement_cosine_similarity_pca",
        "displacement_magnitude_error_pca",
        "persistence_rmse_pca",
        "rmse_improvement_vs_persistence_pca",
    ]
    metric_cols = [col for col in metric_cols if col in horizon_metrics_df.columns]

    metric_eval_split = "test_holdout" if prediction_mode == "known_holdout" else "test_backtest"
    metrics_for_file = horizon_metrics_df
    if "eval_split" in metrics_for_file.columns:
        metrics_for_file = metrics_for_file[metrics_for_file["eval_split"] == metric_eval_split]

    clean_pred_df = pred_df.drop(
        columns=[
            col for col in pred_df.columns
            if col in {"is_known_target", "global_latest_date"}
            or col.endswith("_actual")
            or col.endswith("_model_space_actual")
        ],
        errors="ignore",
    )

    metrics_merge_cols = [col for col in metric_cols if col != "eval_split"]
    metrics_for_merge = metrics_for_file[metrics_merge_cols].copy()
    if user_weight_col in metrics_for_merge.columns:
        metrics_for_merge = metrics_for_merge.rename(columns={user_weight_col: "metric_n_users"})

    pred_df_with_metrics = clean_pred_df.merge(
        metrics_for_merge,
        how="left",
        on=[cluster_col, "horizon"],
    )

    pred_df_with_metrics.to_csv(
        output_dir / f"{model_name}_{prediction_mode}_{n_weeks}_weeks.csv",
        index=False,
    )

    output_metrics_dir = output_dir / "metrics"
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    horizon_metrics_df.to_csv(output_metrics_dir / "horizon_metrics.csv", index=False)
    reliable_horizon_summary_df.to_csv(output_metrics_dir / "reliable_horizon_summary.csv", index=False)
    one_step_metrics_df.to_csv(output_metrics_dir / "one_step_metrics.csv", index=False)


def _metric_rows_for_forecast_window(
    cluster,
    train_group,
    test_group,
    model_name,
    model_config,
    emb_prefix,
    pca,
    scaler,
    eval_split,
    backtest_window,
    cutoff,
    user_weight_col,
    cluster_col,
    lag,
    degree,
    q_pos,
    q_vel,
    r,
    velocity_decay,
    anchor_n_weeks,
    anchor_weight,
):
    """
    Compute recursive-horizon and one-step metrics for one forecast window.
    """
    emb_cols = embedding_cols(train_group, emb_prefix)
    y_train = train_group[emb_cols].to_numpy(dtype=float)
    y_test = test_group[emb_cols].to_numpy(dtype=float)

    if len(y_train) < 2 or len(y_test) == 0:
        return [], []

    model = make_model(model_name,
                       dim=y_train.shape[1],
                       model_config=model_config,
                       lag=lag,
                       degree=degree,
                       q_pos=q_pos,
                       q_vel=q_vel,
                       r=r,
                       velocity_decay=velocity_decay,
                       anchor_n_weeks=anchor_n_weeks,
                       anchor_weight=anchor_weight,
                       )
    horizon_preds = recursive_predictions(model, y_train, len(y_test))

    model = make_model(model_name,
                       dim=y_train.shape[1],
                       model_config=model_config,
                       lag=lag,
                       degree=degree,
                       q_pos=q_pos,
                       q_vel=q_vel,
                       r=r,
                       velocity_decay=velocity_decay,
                       anchor_n_weeks=anchor_n_weeks,
                       anchor_weight=anchor_weight,
                       )
    one_step_preds = rolling_predictions(model, y_train, y_test)

    horizon_rows = []
    one_step_rows = []

    for horizon in range(1, len(y_test) + 1):
        y_true_pca = y_test[horizon - 1:horizon]
        y_pred_pca = horizon_preds[horizon - 1:horizon]
        reference_pca = y_train[-1:].copy()

        pca_stats = compute_metrics(
            y_true_pca,
            y_pred_pca,
            y_reference=reference_pca,
        )

        original_stats = compute_metrics(
            to_original_space(y_true_pca, pca, scaler),
            to_original_space(y_pred_pca, pca, scaler),
            y_reference=to_original_space(reference_pca, pca, scaler),
        )

        target_row = test_group.iloc[horizon - 1]
        target_n_users = pd.to_numeric(
            pd.Series([target_row.get(user_weight_col, np.nan)]),
            errors="coerce",
        ).iloc[0]

        horizon_rows.append({
            cluster_col: cluster,
            "model": model_name,
            "eval_split": eval_split,
            "horizon": horizon,
            "backtest_window": backtest_window,
            "cutoff": cutoff,
            "is_latest_backtest": True,
            "target_year": int(target_row["year"]),
            "target_week": int(target_row["week"]),
            user_weight_col: target_n_users,
            **_add_space_suffix(original_stats, "original"),
            **_add_space_suffix(pca_stats, "pca"),
        })

    one_step_reference_pca = np.vstack([y_train[-1:], y_test[:-1]])

    one_step_pca_stats = compute_metrics(
        y_test,
        one_step_preds,
        y_reference=one_step_reference_pca,
    )

    one_step_original_stats = compute_metrics(
        to_original_space(y_test, pca, scaler),
        to_original_space(one_step_preds, pca, scaler),
        y_reference=to_original_space(one_step_reference_pca, pca, scaler),
    )

    if user_weight_col in test_group.columns:
        test_n_users = pd.to_numeric(test_group[user_weight_col], errors="coerce").mean()
    else:
        test_n_users = np.nan

    one_step_rows.append({
        cluster_col: cluster,
        "model": model_name,
        "eval_split": eval_split,
        "backtest_window": backtest_window,
        "cutoff": cutoff,
        "is_latest_backtest": True,
        "n_train": len(y_train),
        "n_test": len(y_test),
        user_weight_col: test_n_users,
        **_add_space_suffix(one_step_original_stats, "original"),
        **_add_space_suffix(one_step_pca_stats, "pca"),
    })

    return horizon_rows, one_step_rows


def _prediction_rows_for_window(
    cluster,
    history_group,
    target_group,
    model_name,
    model_config,
    emb_prefix,
    pca,
    scaler,
    n_weeks,
    prediction_mode,
    original_emb_cols,
    model_space_cols,
    cluster_col,
    user_weight_col,
    lag,
    degree,
    q_pos,
    q_vel,
    r,
    velocity_decay,
    anchor_n_weeks,
    anchor_weight,
    base_n_weeks=None,
    global_latest_date=None,
    
):
    """
    Create prediction rows for either unknown future weeks or latest known weeks.

    If target_group is None, forecast n_weeks after the history end.
    If target_group is provided, forecast len(target_group) known weeks and attach actual coordinates.

    base_n_weeks is the originally requested forecast length. When stale clusters
    need extra generated steps, horizons beyond base_n_weeks are flagged.
    """
    history_group = sort_time(history_group).copy()
    emb_cols = embedding_cols(history_group, emb_prefix)
    history = history_group[emb_cols].to_numpy(dtype=float)

    if len(history) < 2:
        return []

    if target_group is not None:
        target_group = sort_time(target_group).head(n_weeks).copy()
        horizon_count = len(target_group)
        if horizon_count == 0:
            return []
    else:
        horizon_count = n_weeks

    last_row = history_group.iloc[-1]
    last_date = (
        pd.to_datetime(last_row["week_date"])
        if "week_date" in history_group.columns
        else iso_week_start(
            pd.Series([last_row["year"]]),
            pd.Series([last_row["week"]]),
        ).iloc[0]
    )

    base_n_weeks = n_weeks if base_n_weeks is None else int(base_n_weeks)
    global_latest_date = pd.to_datetime(global_latest_date) if global_latest_date is not None else None
    staleness_weeks = (
        max(0, int((global_latest_date - last_date).days // 7))
        if global_latest_date is not None
        else 0
    )

    model = make_model(model_name,
                       dim=history.shape[1],
                       model_config=model_config,
                       lag=lag,
                       degree=degree,
                       q_pos=q_pos,
                       q_vel=q_vel,
                       r=r,
                       velocity_decay=velocity_decay,
                       anchor_n_weeks=anchor_n_weeks,
                       anchor_weight=anchor_weight,
                       )
    rows = []

    if model_name == "kalman":
        all_predictions_model_space = model.predict_path(
            history=history,
            n_steps=horizon_count,
        )
    else:
        all_predictions_model_space = recursive_predictions(
            model=model,
            history=history,
            n_steps=horizon_count,
        )

    for horizon in range(1, horizon_count + 1):
        pred_model_space = all_predictions_model_space[
            horizon - 1:
            horizon
        ]

        pred_original_space = to_original_space(
            pred_model_space,
            pca,
            scaler,
        )[0]

        if target_group is None:
            forecast_date = last_date + pd.DateOffset(weeks=horizon)
            iso = forecast_date.isocalendar()
            target_row = None
            year = int(iso.year)
            week = int(iso.week)
        else:
            target_row = target_group.iloc[horizon - 1]
            forecast_date = (
                pd.to_datetime(target_row["week_date"])
                if "week_date" in target_group.columns
                else iso_week_start(
                    pd.Series([target_row["year"]]),
                    pd.Series([target_row["week"]]),
                ).iloc[0]
            )
            year = int(target_row["year"])
            week = int(target_row["week"])

        row = {
            cluster_col: cluster,
            "model": model_name,
            "prediction_mode": prediction_mode,
            "horizon": horizon,
            "forecast_origin_year": int(last_row["year"]),
            "forecast_origin_week": int(last_row["week"]),
            "future_date": forecast_date,
            "year": year,
            "week": week,
            "staleness_weeks": staleness_weeks,
            "stale_additional_forecasted_weeks": max(0, horizon - base_n_weeks),
        }

        if user_weight_col in last_row.index:
            row["origin_n_users"] = pd.to_numeric(
                pd.Series([last_row[user_weight_col]]),
                errors="coerce",
            ).iloc[0]

        # Save the inverse-transformed prediction in the original embedding space
        # using plain emb_0 ... emb_n columns for downstream decoder compatibility.
        row.update({f"emb_{idx}": value for idx, value in enumerate(pred_original_space)})

        # Retain the reduced/model-space prediction columns for PCA trajectory checks.
        row.update({f"{col}_model_space_pred": value for col, value in zip(model_space_cols, pred_model_space[0])})

        if target_row is not None and user_weight_col in target_row.index:
            row["target_n_users"] = pd.to_numeric(
                pd.Series([target_row[user_weight_col]]),
                errors="coerce",
            ).iloc[0]

        rows.append(row)

    return rows


def predict_future_weeks_with_metrics(
    train_df,
    validation_df=None,
    test_df=None,
    n_weeks=None,
    forecast_n_known_weeks=None,
    model_name="drift",
    emb_prefix=None,
    pca=None,
    scaler=None,
    min_cosine=None,
    min_rmse_improvement=None,
    min_displacement_cosine=None,
    max_backtest_windows=None,
    min_train_weeks=None,
    PROJECT_NAME=None,
    model_config=None,
    lag=None,
    degree=None,
    q_pos=None,
    q_vel=None,
    r=None,
    velocity_decay=None,
    anchor_n_weeks=None,
    anchor_weight=None,
    cluster_col=None,
    user_weight_col=None,
    clean_df=None
):
    """
    Create predictions and compute train/test metrics.

    Prediction toggle:
    - forecast_n_known_weeks=None:
        Predict n_weeks unknown future weeks after the latest available week.
    - forecast_n_known_weeks=N:
        Forecast the latest N known weeks. Actual centroid coordinates are attached
        to the prediction rows so the predicted path can be visually checked.

    Metrics:
    - train_backtest: rolling windows inside train_df.
    - validation_backtest: rolling-origin windows over validation_df using train history.
    - test_holdout:
        * if forecast_n_known_weeks is set, uses all rows before the latest N known
          weeks to forecast those latest N known weeks;
        * otherwise, uses train + validation history to forecast the explicit test split.
    - test_backtest: rolling-origin windows over test_df using train + validation history.
    """
    if n_weeks <= 0:
        raise ValueError("n_weeks must be greater than 0.")
    if forecast_n_known_weeks is not None and forecast_n_known_weeks <= 0:
        raise ValueError("forecast_n_known_weeks must be None or a positive integer.")
    if forecast_n_known_weeks is not None and forecast_n_known_weeks > 0:
        # In known-week checking mode, the holdout length is also the forecast length.
        n_weeks = int(forecast_n_known_weeks)

    validation_df = validation_df if validation_df is not None else train_df.iloc[0:0].copy()
    test_df = test_df if test_df is not None else train_df.iloc[0:0].copy()

    resolved_model_config = {
        "lag": lag,
        "degree": degree,
        "q_pos": q_pos,
        "q_vel": q_vel,
        "r": r,
        "velocity_decay": velocity_decay,
        "anchor_n_weeks": anchor_n_weeks,
        "anchor_weight": anchor_weight,
    }
    if model_config is not None:
        resolved_model_config.update(model_config)

    prediction_rows = []
    horizon_metric_rows = []
    one_step_metric_rows = []

    all_known_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    model_space_cols = embedding_cols(all_known_df, emb_prefix)
    original_emb_cols = embedding_cols(clean_df, emb_prefix)

    if "week_date" in all_known_df.columns:
        global_latest_date = pd.to_datetime(all_known_df["week_date"]).max()
    else:
        global_latest_date = iso_week_start(all_known_df["year"], all_known_df["week"]).max()

    # --------------------------------------------------
    # 1. Prediction output.
    # --------------------------------------------------
    if forecast_n_known_weeks is None:
        prediction_mode = "unknown_future"

        for cluster, group, Y, emb_cols in iter_cluster_arrays(all_known_df, emb_prefix, cluster_col):
            group = sort_time(group)
            last_row = group.iloc[-1]
            last_date = (
                pd.to_datetime(last_row["week_date"])
                if "week_date" in group.columns
                else iso_week_start(
                    pd.Series([last_row["year"]]),
                    pd.Series([last_row["week"]]),
                ).iloc[0]
            )
            staleness_weeks = max(0, int((global_latest_date - last_date).days // 7))
            forecast_steps = n_weeks + staleness_weeks

            prediction_rows.extend(
                _prediction_rows_for_window(
                    cluster=cluster,
                    history_group=group,
                    target_group=None,
                    model_name=model_name,
                    model_config=resolved_model_config,
                    emb_prefix=emb_prefix,
                    pca=pca,
                    scaler=scaler,
                    n_weeks=forecast_steps,
                    prediction_mode=prediction_mode,
                    original_emb_cols=original_emb_cols,
                    model_space_cols=model_space_cols,
                    base_n_weeks=n_weeks,
                    global_latest_date=global_latest_date,
                    cluster_col=cluster_col,
                    user_weight_col=user_weight_col,
                    lag=lag,
                    degree=degree,
                    q_pos=q_pos,
                    q_vel=q_vel,
                    r=r,
                    velocity_decay=velocity_decay,
                    anchor_n_weeks=anchor_n_weeks,
                    anchor_weight=anchor_weight,
                )
            )

    else:
        prediction_mode = "known_holdout"
        known_horizon = forecast_n_known_weeks

        for cluster, group, Y, emb_cols in iter_cluster_arrays(all_known_df, emb_prefix, cluster_col):
            group = sort_time(group)

            if len(group) < min_train_weeks + known_horizon:
                continue

            history_group = group.iloc[:-known_horizon]
            target_group = group.iloc[-known_horizon:]

            prediction_rows.extend(
                _prediction_rows_for_window(
                    cluster=cluster,
                    history_group=history_group,
                    target_group=target_group,
                    model_name=model_name,
                    model_config=resolved_model_config,
                    emb_prefix=emb_prefix,
                    pca=pca,
                    scaler=scaler,
                    n_weeks=known_horizon,
                    prediction_mode=prediction_mode,
                    original_emb_cols=original_emb_cols,
                    model_space_cols=model_space_cols,
                    base_n_weeks=n_weeks,
                    global_latest_date=None,
                    cluster_col=cluster_col,
                    user_weight_col=user_weight_col,
                    lag=lag,
                    degree=degree,
                    q_pos=q_pos,
                    q_vel=q_vel,
                    r=r,
                    velocity_decay=velocity_decay,
                    anchor_n_weeks=anchor_n_weeks,
                    anchor_weight=anchor_weight,
                )
            )


    # --------------------------------------------------
    # 2. Train rolling backtest metrics.
    # --------------------------------------------------
    for cluster, group, Y, emb_cols in iter_cluster_arrays(train_df, emb_prefix, cluster_col):
        group = sort_time(group)
        n_obs = len(Y)

        if n_obs < min_train_weeks + n_weeks:
            continue

        possible_cutoffs = list(range(min_train_weeks, n_obs - n_weeks + 1))
        selected_cutoffs = possible_cutoffs[-max_backtest_windows:]

        for backtest_window, cutoff in enumerate(selected_cutoffs, start=1):
            train_group = group.iloc[:cutoff]
            test_group = group.iloc[cutoff:cutoff + n_weeks]

            rows, one_step_rows = _metric_rows_for_forecast_window(
                cluster=cluster,
                train_group=train_group,
                test_group=test_group,
                model_name=model_name,
                model_config=resolved_model_config,
                emb_prefix=emb_prefix,
                pca=pca,
                scaler=scaler,
                eval_split="train_backtest",
                backtest_window=backtest_window,
                cutoff=cutoff,
                lag=lag,
                degree=degree,
                q_pos=q_pos,
                q_vel=q_vel,
                r=r,
                velocity_decay=velocity_decay,
                anchor_n_weeks=anchor_n_weeks,
                anchor_weight=anchor_weight,
                user_weight_col=user_weight_col,
                cluster_col=cluster_col
            )
            horizon_metric_rows.extend(rows)
            one_step_metric_rows.extend(one_step_rows)

    # --------------------------------------------------
    # 3. Validation rolling backtest metrics.
    # --------------------------------------------------
    for cluster, validation_group in validation_df.groupby(cluster_col, sort=False):
        base_history_group = sort_time(train_df[train_df[cluster_col] == cluster])
        validation_group = sort_time(validation_group)

        if len(base_history_group) < min_train_weeks or validation_group.empty:
            continue

        possible_origins = list(range(len(validation_group)))
        selected_origins = possible_origins[:max_backtest_windows]

        for backtest_window, origin in enumerate(selected_origins, start=1):
            train_group = pd.concat(
                [base_history_group, validation_group.iloc[:origin]],
                ignore_index=True,
            )
            test_group = validation_group.iloc[origin:origin + n_weeks]

            rows, one_step_rows = _metric_rows_for_forecast_window(
                cluster=cluster,
                train_group=train_group,
                test_group=test_group,
                model_name=model_name,
                model_config=resolved_model_config,
                emb_prefix=emb_prefix,
                pca=pca,
                scaler=scaler,
                eval_split="validation_backtest",
                backtest_window=backtest_window,
                cutoff=len(train_group),
                lag=lag,
                degree=degree,
                q_pos=q_pos,
                q_vel=q_vel,
                r=r,
                velocity_decay=velocity_decay,
                anchor_n_weeks=anchor_n_weeks,
                anchor_weight=anchor_weight,
                user_weight_col=user_weight_col,
                cluster_col=cluster_col
            )
            horizon_metric_rows.extend(rows)
            one_step_metric_rows.extend(one_step_rows)

    # --------------------------------------------------
    # 4. Single-origin test holdout metrics.
    # --------------------------------------------------
    if forecast_n_known_weeks is not None:
        # In known-week mode, evaluate the same latest N known weeks used by
        # the generated known-holdout prediction.
        for cluster, group, _, _ in iter_cluster_arrays(
            all_known_df, emb_prefix, cluster_col
        ):
            group = sort_time(group)

            if len(group) < min_train_weeks + n_weeks:
                continue

            train_group = group.iloc[:-n_weeks]
            test_group = group.iloc[-n_weeks:]

            rows, one_step_rows = _metric_rows_for_forecast_window(
                cluster=cluster,
                train_group=train_group,
                test_group=test_group,
                model_name=model_name,
                model_config=resolved_model_config,
                emb_prefix=emb_prefix,
                pca=pca,
                scaler=scaler,
                eval_split="test_holdout",
                backtest_window=1,
                cutoff=len(train_group),
                lag=lag,
                degree=degree,
                q_pos=q_pos,
                q_vel=q_vel,
                r=r,
                velocity_decay=velocity_decay,
                anchor_n_weeks=anchor_n_weeks,
                anchor_weight=anchor_weight,
                user_weight_col=user_weight_col,
                cluster_col=cluster_col,
            )
            horizon_metric_rows.extend(rows)
            one_step_metric_rows.extend(one_step_rows)
    else:
        # In unknown-future mode, use the explicit test split as the untouched
        # single-origin holdout.
        history_for_test_df = pd.concat(
            [train_df, validation_df],
            ignore_index=True,
        )

        for cluster, test_group in test_df.groupby(cluster_col, sort=False):
            train_group = sort_time(
                history_for_test_df[
                    history_for_test_df[cluster_col] == cluster
                ]
            )
            test_group = sort_time(test_group).head(n_weeks)

            if len(train_group) < min_train_weeks or test_group.empty:
                continue

            rows, one_step_rows = _metric_rows_for_forecast_window(
                cluster=cluster,
                train_group=train_group,
                test_group=test_group,
                model_name=model_name,
                model_config=resolved_model_config,
                emb_prefix=emb_prefix,
                pca=pca,
                scaler=scaler,
                eval_split="test_holdout",
                backtest_window=1,
                cutoff=len(train_group),
                lag=lag,
                degree=degree,
                q_pos=q_pos,
                q_vel=q_vel,
                r=r,
                velocity_decay=velocity_decay,
                anchor_n_weeks=anchor_n_weeks,
                anchor_weight=anchor_weight,
                user_weight_col=user_weight_col,
                cluster_col=cluster_col,
            )
            horizon_metric_rows.extend(rows)
            one_step_metric_rows.extend(one_step_rows)

    # --------------------------------------------------
    # 5. Test rolling backtest metrics.
    # --------------------------------------------------
    history_for_test_df = pd.concat([train_df, validation_df], ignore_index=True)

    for cluster, test_group_all in test_df.groupby(cluster_col, sort=False):
        base_history_group = sort_time(
            history_for_test_df[history_for_test_df[cluster_col] == cluster]
        )
        test_group_all = sort_time(test_group_all)

        if len(base_history_group) < min_train_weeks or test_group_all.empty:
            continue

        possible_origins = list(range(len(test_group_all)))
        selected_origins = possible_origins[:max_backtest_windows]

        for backtest_window, origin in enumerate(selected_origins, start=1):
            train_group = pd.concat(
                [base_history_group, test_group_all.iloc[:origin]],
                ignore_index=True,
            )
            test_group = test_group_all.iloc[origin:origin + n_weeks]

            rows, one_step_rows = _metric_rows_for_forecast_window(
                cluster=cluster,
                train_group=train_group,
                test_group=test_group,
                model_name=model_name,
                model_config=resolved_model_config,
                emb_prefix=emb_prefix,
                pca=pca,
                scaler=scaler,
                eval_split="test_backtest",
                backtest_window=backtest_window,
                cutoff=len(train_group),
                lag=lag,
                degree=degree,
                q_pos=q_pos,
                q_vel=q_vel,
                r=r,
                velocity_decay=velocity_decay,
                anchor_n_weeks=anchor_n_weeks,
                anchor_weight=anchor_weight,
                user_weight_col=user_weight_col,
                cluster_col=cluster_col
            )
            horizon_metric_rows.extend(rows)
            one_step_metric_rows.extend(one_step_rows)

    prediction_df = pd.DataFrame(prediction_rows)

    horizon_metrics_df = _average_metric_rows(
        horizon_metric_rows,
        group_cols=["eval_split", cluster_col, "model", "horizon"],
        weight_col=user_weight_col
    )

    one_step_metrics_df = _average_metric_rows(
        one_step_metric_rows,
        group_cols=["eval_split", cluster_col, "model"],
        weight_col=user_weight_col
    )

    historical_displacement_threshold = median_historical_displacement(train_df, emb_prefix=emb_prefix, cluster_col=cluster_col)

    reliable_horizon_summary_df = summarize_reliable_horizon(
        horizon_metrics_df,
        median_historical_displacement=historical_displacement_threshold,
        min_cosine=min_cosine,
        min_rmse_improvement=min_rmse_improvement,
        min_displacement_cosine=min_displacement_cosine,
        weight_col=user_weight_col,
        cluster_col=cluster_col
    )

    if PROJECT_NAME is not None and not prediction_df.empty:
        save_prediction_and_metrics(
            prediction_df,
            horizon_metrics_df,
            reliable_horizon_summary_df,
            one_step_metrics_df,
            forecast_n_known_weeks if forecast_n_known_weeks is not None else n_weeks,
            PROJECT_NAME,
            cluster_col=cluster_col,
            user_weight_col=user_weight_col
        )

    return prediction_df, horizon_metrics_df, reliable_horizon_summary_df, one_step_metrics_df
