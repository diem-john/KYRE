import pandas as pd
import numpy as np
import os


from src.behavioral_anomaly_detector.utils import rolling_zscore, build_detection_timelines

def add_decoded_behavior_change_signals(
    decoded_weekly_centroid_emb,
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    timeline_col="timeline_type",
    feature_col="feature_id",
    prob_col="pred_used_prob",
    elapsed_col="pred_elapsed",
    top_k=10,
    rolling_window=8,
    min_periods=4,
):
    """
    Adds the columns for detecting changes in centroid decoded behavior.
    """
    df = decoded_weekly_centroid_emb.copy()

    if timeline_col not in df.columns:
        df[timeline_col] = "actual"

    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0)
    df[elapsed_col] = pd.to_numeric(df[elapsed_col], errors="coerce").fillna(0)

    df["pred_weighted_usage"] = df[prob_col] * df[elapsed_col]

    group_cols = [timeline_col, cluster_col]
    time_keys = group_cols + [year_col, week_col]

    prob_wide = (
        df.pivot_table(
            index=time_keys,
            columns=feature_col,
            values=prob_col,
            aggfunc="mean",
            fill_value=0,
        )
        .sort_index()
    )

    elapsed_wide = (
        df.pivot_table(
            index=time_keys,
            columns=feature_col,
            values=elapsed_col,
            aggfunc="mean",
            fill_value=0,
        )
        .sort_index()
    )

    weighted_wide = (
        df.pivot_table(
            index=time_keys,
            columns=feature_col,
            values="pred_weighted_usage",
            aggfunc="mean",
            fill_value=0,
        )
        .sort_index()
    )

    result_rows = []

    for group_key, prob_group in prob_wide.groupby(level=group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_cols, group_key))

        elapsed_group = elapsed_wide.xs(group_key, level=group_cols)
        weighted_group = weighted_wide.xs(group_key, level=group_cols)
        prob_group = prob_group.droplevel(group_cols).sort_index()
        elapsed_group = elapsed_group.sort_index()
        weighted_group = weighted_group.sort_index()

        prob_prev = prob_group.shift(1)
        elapsed_prev = elapsed_group.shift(1)
        weighted_prev = weighted_group.shift(1)

        prob_l1_change = (prob_group - prob_prev).abs().sum(axis=1) # sum of change in feature usage probabilities in using a feature
        elapsed_l1_change = (elapsed_group - elapsed_prev).abs().sum(axis=1) # sum of change in elapsed time in using a feature
        weighted_l1_change = (weighted_group - weighted_prev).abs().sum(axis=1) # sum of change in weighted prob and elaspsed time (probability * elapsed)

        # computes the jaccard similarity of top-k feature used. checks if there is a change in features used
        top_k_jaccard = []

        for i in range(len(prob_group)):
            if i == 0:
                top_k_jaccard.append(np.nan)
                continue

            curr_top = set(
                prob_group.iloc[i]
                .sort_values(ascending=False)
                .head(top_k)
                .index
            )

            prev_top = set(
                prob_group.iloc[i - 1]
                .sort_values(ascending=False)
                .head(top_k)
                .index
            )

            union = curr_top | prev_top
            intersection = curr_top & prev_top

            top_k_jaccard.append(
                len(intersection) / len(union) if union else np.nan
            )

        out = pd.DataFrame({
            **group_values,
            year_col: [idx[0] for idx in prob_group.index],
            week_col: [idx[1] for idx in prob_group.index],
            "decoded_prob_l1_change": prob_l1_change.values,
            "decoded_elapsed_l1_change": elapsed_l1_change.values,
            "decoded_weighted_usage_l1_change": weighted_l1_change.values,
            f"decoded_top_{top_k}_jaccard": top_k_jaccard,
        })

        result_rows.append(out)

    decoded_change_df = pd.concat(result_rows, ignore_index=True)

    z_cols = [
        "decoded_prob_l1_change",
        "decoded_elapsed_l1_change",
        "decoded_weighted_usage_l1_change",
    ]

    # A static threshold for detecting anomalous weeks as we create a robust rolling z_score for detecting is the change is unusual relative to the respective clusters.
    # z-score of 2.5 is considered anomalous. higher z score, more chance that the current week deviates from recent values
    for col in z_cols:
        decoded_change_df[f"{col}_z"] = (
            decoded_change_df
            .groupby(group_cols)[col]
            .transform(
                lambda s: rolling_zscore(
                    s,
                    window=rolling_window,
                    min_periods=min_periods,
                )
            )
        )

    return decoded_change_df


def add_centroid_movement_signals(
    concat_weekly_centroid_emb,
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    timeline_col="timeline_type",
    emb_cols=None,
    emb_prefix="emb_",
    drift_windows=(4, 8),
    baseline_window=8,
    rolling_window=8,
    min_periods=4,
):
    """
    Adds the columns for detecting changes in centroid movement.
    """
    df = concat_weekly_centroid_emb.copy()

    if timeline_col not in df.columns:
        df[timeline_col] = "actual"

    if emb_cols is None:
        emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]

    if not emb_cols:
        raise ValueError(
            "No embedding columns found. Provide emb_cols or set the correct emb_prefix."
        )

    group_cols = [timeline_col, cluster_col]
    df = df.sort_values(group_cols + [year_col, week_col]).reset_index(drop=True)

    output_parts = []

    for group_key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values([year_col, week_col]).copy()
        X = g[emb_cols].to_numpy(dtype=float)

        # checks the 1-week displacement.
        prev_X = np.roll(X, shift=1, axis=0)
        disp_1w = np.linalg.norm(X - prev_X, axis=1)
        disp_1w[0] = np.nan

        g["centroid_displacement_1w"] = disp_1w

        # Checks if there is a net drift in centroid movement for weeks 4 and 8 (drift_windows).
        # Calculates the net centroid movement relative to the current centroid position on longer weeks (drift_windows).
        for w in drift_windows:
            lag_X = np.roll(X, shift=w, axis=0)
            net_disp = np.linalg.norm(X - lag_X, axis=1)
            net_disp[:w] = np.nan

            g[f"net_displacement_{w}w"] = net_disp

        # Distance from recent rolling centroid baseline.
        rolling_baseline_disp = []

        # checks how far off the current centroid position with the recent average of previous centroid positions (mean(n_weeks_drift_windows)).
        for i in range(len(g)):
            start = max(0, i - baseline_window)
            past_window = X[start:i]

            if len(past_window) < min_periods:
                rolling_baseline_disp.append(np.nan)
                continue

            baseline = past_window.mean(axis=0)
            dist = np.linalg.norm(X[i] - baseline)

            rolling_baseline_disp.append(dist)

        g[f"rolling_baseline_displacement_{baseline_window}w"] = rolling_baseline_disp

        # Directional efficiency: Checks how much of the centroid path distance encompasses the net distance = net distance / path length.
        # Higher values means the distance traveled is in one direction.
        for w in drift_windows:
            efficiencies = []

            for i in range(len(g)):
                if i < w:
                    efficiencies.append(np.nan)
                    continue

                window_X = X[i - w:i + 1]

                net_distance = np.linalg.norm(window_X[-1] - window_X[0])

                step_distances = np.linalg.norm(
                    window_X[1:] - window_X[:-1],
                    axis=1,
                )

                path_length = step_distances.sum()

                efficiencies.append(
                    net_distance / path_length if path_length > 0 else np.nan
                )

            g[f"directional_efficiency_{w}w"] = efficiencies

        output_parts.append(g)

    movement_df = pd.concat(output_parts, ignore_index=True)

    movement_metric_cols = ["centroid_displacement_1w"]

    for w in drift_windows:
        movement_metric_cols.append(f"net_displacement_{w}w")

    movement_metric_cols.append(
        f"rolling_baseline_displacement_{baseline_window}w"
    )

    for col in movement_metric_cols:
        movement_df[f"{col}_z"] = (
            movement_df
            .groupby(group_cols)[col]
            .transform(
                lambda s: rolling_zscore(
                    s,
                    window=rolling_window,
                    min_periods=min_periods,
                )
            )
        )

    return movement_df

def detect_centroid_behavioral_anomaly(
    decoded_weekly_centroid_emb,
    concat_weekly_centroid_emb,
    cluster_col="final_cluster_label",
    year_col="year",
    week_col="week",
    is_prediction_col="is_prediction",
    timeline_col="timeline_type",
    feature_col="feature_id",
    prob_col="pred_used_prob",
    elapsed_col="pred_elapsed",
    emb_cols=None,
    emb_prefix="emb_",
    top_k=10,
    rolling_window=8,
    min_periods=4,
    decoded_z_threshold=2.5,
    movement_z_threshold=2.5,
    top_k_jaccard_threshold=0.6,
    directional_efficiency_threshold=0.4,
    save_dir=None,
):
    """
    Detects behavior-change weeks using separate actual and prediction timelines.

    Baselines are calculated as:
    - timeline_type == "actual": actual historical + actual historical
    - timeline_type == "prediction": actual historical context + predicted continuation

    This is important when actual and predicted rows overlap for the same
    cluster-week.

    Final anomaly logic:
    - Primary behavioral signal uses only:
        1. decoded_elapsed_change_flag
        2. decoded_top_feature_shift_flag

    - Secondary movement support uses only:
        1. movement_drift_flag

    Diagnostic-only signals are still computed and retained:
    - decoded_prob_change_flag
    - decoded_weighted_usage_change_flag
    - movement_1w_spike_flag
    - movement_directional_flag
    - movement_directional_drift_flag
    """

    # Create clean actual and prediction timelines.
    decoded_detection_df = build_detection_timelines(
        decoded_weekly_centroid_emb,
        cluster_col=cluster_col,
        year_col=year_col,
        week_col=week_col,
        is_prediction_col=is_prediction_col,
        timeline_col=timeline_col,
    )

    centroid_detection_df = build_detection_timelines(
        concat_weekly_centroid_emb,
        cluster_col=cluster_col,
        year_col=year_col,
        week_col=week_col,
        is_prediction_col=is_prediction_col,
        timeline_col=timeline_col,
    )

    # Add decoded behavior-change metrics.
    decoded_signals = add_decoded_behavior_change_signals(
        decoded_weekly_centroid_emb=decoded_detection_df,
        cluster_col=cluster_col,
        year_col=year_col,
        week_col=week_col,
        timeline_col=timeline_col,
        feature_col=feature_col,
        prob_col=prob_col,
        elapsed_col=elapsed_col,
        top_k=top_k,
        rolling_window=rolling_window,
        min_periods=min_periods,
    )

    # Add centroid movement-change metrics.
    movement_signals = add_centroid_movement_signals(
        concat_weekly_centroid_emb=centroid_detection_df,
        cluster_col=cluster_col,
        year_col=year_col,
        week_col=week_col,
        timeline_col=timeline_col,
        emb_cols=emb_cols,
        emb_prefix=emb_prefix,
        drift_windows=(4, 8),
        baseline_window=8,
        rolling_window=rolling_window,
        min_periods=min_periods,
    )

    # Merge decoded and movement signals.
    merge_keys = [timeline_col, cluster_col, year_col, week_col]

    out = movement_signals.merge(
        decoded_signals,
        on=merge_keys,
        how="left",
    )

    top_k_jaccard_col = f"decoded_top_{top_k}_jaccard"

    # ------------------------------------------------------------------
    # Primary decoded behavior flags
    # ------------------------------------------------------------------
    # These diagnostic flags are still computed, but not used directly
    # in decoded_usage_change_flag.
    out["decoded_prob_change_flag"] = (
        out["decoded_prob_l1_change_z"] >= decoded_z_threshold
    )

    out["decoded_elapsed_change_flag"] = (
        out["decoded_elapsed_l1_change_z"] >= decoded_z_threshold
    )

    out["decoded_weighted_usage_change_flag"] = (
        out["decoded_weighted_usage_l1_change_z"] >= decoded_z_threshold
    )

    out["decoded_top_feature_shift_flag"] = (
        out[top_k_jaccard_col] <= top_k_jaccard_threshold
    )

    # Final decoded behavior signal:
    # only elapsed-time change and top-feature composition shift are used.
    out["decoded_usage_change_flag"] = (
        out["decoded_elapsed_change_flag"]
        | out["decoded_top_feature_shift_flag"]
    )

    # ------------------------------------------------------------------
    # Behavioral ranking details
    # ------------------------------------------------------------------

    behavior_priority_conditions = [
        (
            out["decoded_elapsed_change_flag"]
            & out["decoded_top_feature_shift_flag"]
        ),
        out["decoded_top_feature_shift_flag"],
        out["decoded_elapsed_change_flag"],
    ]

    behavior_priority_values = [
        3,
        2,
        1,
    ]

    out["behavior_priority_rank"] = np.select(
        behavior_priority_conditions,
        behavior_priority_values,
        default=0,
    ).astype(int)

    behavior_priority_labels = {
        3: "elapsed_and_top_feature_change",
        2: "top_feature_change",
        1: "elapsed_time_change",
        0: "no_behavioral_change",
    }

    out["behavior_priority_type"] = (
        out["behavior_priority_rank"]
        .map(behavior_priority_labels)
    )

    # ------------------------------------------------------------------
    # Secondary movement flags
    # ------------------------------------------------------------------
    # 1-week spike is kept only as diagnostic.
    out["movement_1w_spike_flag"] = (
        out["centroid_displacement_1w_z"] >= movement_z_threshold
    )

    out["movement_4w_drift_flag"] = (
        out["net_displacement_4w_z"] >= movement_z_threshold
    )

    out["movement_8w_drift_flag"] = (
        out["net_displacement_8w_z"] >= movement_z_threshold
    )

    out["movement_far_from_recent_baseline_flag"] = (
        out["rolling_baseline_displacement_8w_z"] >= movement_z_threshold
    )

    # Directional efficiency is kept only as diagnostic.
    out["movement_directional_flag"] = (
        out["directional_efficiency_8w"] >= directional_efficiency_threshold
    )

    # Final movement support signal:
    # only longer-window drift and distance from recent baseline are used.
    out["movement_drift_flag"] = (
        out["movement_4w_drift_flag"]
        | out["movement_8w_drift_flag"]
        | out["movement_far_from_recent_baseline_flag"]
    )

    # ------------------------------------------------------------------
    # Movement ranking details
    # ------------------------------------------------------------------

    # Count how many of the main centroid movement signals were triggered.
    movement_signal_cols = [
        "movement_4w_drift_flag",
        "movement_8w_drift_flag",
        "movement_far_from_recent_baseline_flag",
    ]

    out["movement_signal_count"] = (
        out[movement_signal_cols]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    # Rank the type of centroid movement.
    #
    # Priority:
    # 4 = multiple movement signals
    # 3 = far from recent baseline
    # 2 = 4-week drift
    # 1 = 8-week drift
    # 0 = no main centroid movement
    movement_priority_conditions = [
        out["movement_signal_count"] >= 2,
        out["movement_far_from_recent_baseline_flag"],
        out["movement_4w_drift_flag"],
        out["movement_8w_drift_flag"],
    ]

    movement_priority_values = [
        4,
        3,
        2,
        1,
    ]

    out["movement_priority_rank"] = np.select(
        movement_priority_conditions,
        movement_priority_values,
        default=0,
    ).astype(int)

    movement_priority_labels = {
        4: "multiple_movement_signals",
        3: "far_from_recent_baseline",
        2: "4_week_centroid_drift",
        1: "8_week_centroid_drift",
        0: "no_main_movement",
    }

    out["movement_priority_type"] = (
        out["movement_priority_rank"]
        .map(movement_priority_labels)
    )

    # Diagnostic-only, not used in final scoring for anomaly.
    out["movement_directional_drift_flag"] = (
        out["movement_drift_flag"]
        & out["movement_directional_flag"]
    )

    # ------------------------------------------------------------------
    # Final anomaly flags
    # ------------------------------------------------------------------
    # Main behavior-change flag:
    # True if decoded elapsed behavior or top-feature composition changed.
    out["behavior_change_flag"] = out["decoded_usage_change_flag"]

    # Strong behavior-change flag:
    # True if decoded behavior changed and movement drift supports it.
    out["strong_behavior_change_flag"] = (
        out["decoded_usage_change_flag"]
        & out["movement_drift_flag"]
    )

    # Retained for diagnostics only.
    # This is no longer used in behavior_change_type or score.
    out["directional_behavior_change_flag"] = (
        out["decoded_usage_change_flag"]
        & out["movement_directional_drift_flag"]
    )
    
    # ------------------------------------------------------------------
    # Creates a more readable triggered-main-flags summary to know which individual anomaly criteria are triggered.
    # ------------------------------------------------------------------
    # These are the flags used in the final anomaly logic.
    main_trigger_flag_labels = {
        "decoded_elapsed_change_flag": "elapsed time changed",
        "decoded_top_feature_shift_flag": "top features shifted",
        "movement_4w_drift_flag": "4-week centroid drift",
        "movement_8w_drift_flag": "8-week centroid drift",
        "movement_far_from_recent_baseline_flag": "far from recent centroid baseline (8 week mean position)",
    }

    def summarize_triggered_flags(row):
        triggered = [
            label
            for flag_col, label in main_trigger_flag_labels.items()
            if bool(row.get(flag_col, False))
        ]

        return "; ".join(triggered) if triggered else "none"

    out["triggered_main_flags"] = out.apply(
        summarize_triggered_flags,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Behavior-change type and ordinal score
    # ------------------------------------------------------------------
    # Directional drift is no longer part of the score.
    # Movement support is only movement_drift_flag.
    conditions = [
        out["decoded_usage_change_flag"] & out["movement_drift_flag"], # decoded_change_with_movement_drift
        out["decoded_usage_change_flag"] & ~out["movement_drift_flag"], # decoded_change_only
        ~out["decoded_usage_change_flag"] & out["movement_drift_flag"], # movement_only_inspect
        ~out["decoded_usage_change_flag"] & out["movement_1w_spike_flag"], # short_term_jitter_only
    ]

    choices = [
        "decoded_change_with_movement_drift",
        "decoded_change_only",
        "movement_only_inspect",
        "short_term_jitter_only",
    ]

    out["behavior_change_type"] = np.select(
        conditions,
        choices,
        default="stable",
    )

    behavior_change_score_map = {
        "stable": 0,
        "short_term_jitter_only": 1,
        "movement_only_inspect": 2,
        "decoded_change_only": 3,
        "decoded_change_with_movement_drift": 4,
    }

    out["behavior_change_score"] = (
        out["behavior_change_type"]
        .map(behavior_change_score_map)
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Detailed anomaly priority ranking
    # ------------------------------------------------------------------
    #
    # Ranking order:
    # 1. Existing behavior_change_score:
    #       score 4 ranks above score 3
    #
    # 2. Behavior-change score:
    #       movement-supported decoded change > decoded change alone
    #
    # 3. Movement priority:
    #       multiple signals > far from baseline > 4-week drift > 8-week drift
    #
    out["anomaly_priority_rank"] = (
        out["behavior_priority_rank"] * 100
        + out["behavior_change_score"] * 10
        + out["movement_priority_rank"]
    ).astype(int)

    out["anomaly_priority_label"] = (
        out["behavior_priority_type"].astype(str)
        + " | "
        + out["movement_priority_type"].astype(str)
    )

    anomaly_priority_level_conditions = [
        # Level 5:
        # Both behavioral signals, with or without movement support.
        (
            out["decoded_elapsed_change_flag"]
            & out["decoded_top_feature_shift_flag"]
        ),

        # Level 4:
        # Top-feature composition changed and movement also supports it.
        (
            out["decoded_top_feature_shift_flag"]
            & ~out["decoded_elapsed_change_flag"]
            & out["movement_drift_flag"]
        ),

        # Level 3:
        # Top-feature composition changed without movement support.
        (
            out["decoded_top_feature_shift_flag"]
            & ~out["decoded_elapsed_change_flag"]
            & ~out["movement_drift_flag"]
        ),

        # Level 2:
        # Elapsed time changed and movement supports it.
        (
            out["decoded_elapsed_change_flag"]
            & ~out["decoded_top_feature_shift_flag"]
            & out["movement_drift_flag"]
        ),

        # Level 1:
        # Elapsed time changed without movement support.
        (
            out["decoded_elapsed_change_flag"]
            & ~out["decoded_top_feature_shift_flag"]
            & ~out["movement_drift_flag"]
        ),
    ]

    anomaly_priority_level_values = [
        5,
        4,
        3,
        2,
        1,
    ]

    out["anomaly_priority_level"] = np.select(
        anomaly_priority_level_conditions,
        anomaly_priority_level_values,
        default=0,
    ).astype(int)

    behavior_change_weeks_df = out.sort_values(
        [timeline_col, cluster_col, year_col, week_col]
    ).reset_index(drop=True)

    # Separate actual historical centroid positions from predicted centroid positions
    # for visualization.
    historical_behavior_change_df = (
        behavior_change_weeks_df
        .loc[
            (behavior_change_weeks_df[timeline_col] == "actual")
            & (~behavior_change_weeks_df[is_prediction_col].astype(bool))
        ]
        .copy()
        .reset_index(drop=True)
    )

    predicted_behavior_change_df = (
        behavior_change_weeks_df
        .loc[
            (behavior_change_weeks_df[timeline_col] == "prediction")
            & (behavior_change_weeks_df[is_prediction_col].astype(bool))
        ]
        .copy()
        .reset_index(drop=True)
    )

    # saves the output of the anomaly detection pipeline (historical and predicted)
    os.makedirs(save_dir, exist_ok=True)
    historical_behavior_change_df.to_csv(f'{save_dir}/historical_behavior_change.csv', index=False)
    predicted_behavior_change_df.to_csv(f'{save_dir}/predicted_behavior_change.csv', index=False)
    
    return (
        behavior_change_weeks_df,
        historical_behavior_change_df,
        predicted_behavior_change_df,
    )