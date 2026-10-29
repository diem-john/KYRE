import pandas as pd
from pathlib import Path

def load_centroid_weekly_embeddings(
    centroid_directory
):
    # Loads the centroid embedding parquet files with user behavior aggregations
    files = sorted(centroid_directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {centroid_directory}")

    centroid_df = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)

    # excludes cluster labels that are considered as outlier clusters
    outlier_mask = centroid_df["final_cluster_label"].astype(str).str.startswith("-1_")
    centroid_df = centroid_df[~outlier_mask]
    
    return centroid_df

def embedding_cols(df, prefix=None):
    """
    Return embedding columns in numeric suffix order when possible.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(
        cols,
        key=lambda c: int(c.replace(prefix, "")) if c.replace(prefix, "").isdigit() else c,
    )


def iso_week_start(year, week):
    """
    Convert ISO year/week columns to week-start dates. ISO weeks start on Monday.
    """
    year = year.astype(int).astype(str)
    week = week.astype(int).astype(str).str.zfill(2)
    return pd.to_datetime(year + "-W" + week + "-1", format="%G-W%V-%u")


def sort_time(df):
    """
    Sort by chronological week.
    """
    return df.sort_values(["year", "week"]).reset_index(drop=True)


def fill_weekly_gaps(group,
                     emb_cols,
                     cluster_col,
                     user_weight_col,
                     max_gap=None):
    """
    Reindex one cluster to a complete weekly grid and fill only short internal gaps.

    This does not fill missing tail weeks after the cluster's latest observed week.
    Tail extrapolation is avoided because there is no later observation to anchor it.
    """
    g = sort_time(group).copy()
    g["week_date"] = iso_week_start(g["year"], g["week"])

    full_weeks = pd.date_range(
        g["week_date"].min(),
        g["week_date"].max(),
        freq="W-MON",
    )

    g = g.set_index("week_date").reindex(full_weeks)
    g.index.name = "week_date"

    g[cluster_col] = group[cluster_col].iloc[0]

    iso = g.index.isocalendar()
    g["year"] = iso.year.astype(int)
    g["week"] = iso.week.astype(int)

    g["was_gap_filled"] = g[emb_cols[0]].isna()

    g[emb_cols] = g[emb_cols].interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both",
        limit_area="inside",
    )

    if user_weight_col in group.columns:
        g[user_weight_col] = pd.to_numeric(g[user_weight_col], errors="coerce")
        g[user_weight_col] = (
            g[user_weight_col]
            .interpolate(method="linear", limit=max_gap, limit_area="inside")
            .ffill()
            .bfill()
        )

    return g.copy().reset_index()


def latest_complete_streak(group, required_cols):
    """
    Keep the latest consecutive streak with complete embedding values.

    This replaces the old longest-streak logic. For current forecasting, a recent
    shorter streak is preferable to an older longer streak that stopped weeks ago.
    """
    g = sort_time(group).copy()
    valid = g[required_cols].notna().all(axis=1)

    if not valid.any():
        return g.iloc[0:0].copy()

    streak_id = (valid != valid.shift(fill_value=False)).cumsum()
    valid_streaks = [part for _, part in g.groupby(streak_id) if valid.loc[part.index].iloc[0]]

    latest_streak = max(valid_streaks, key=lambda part: part["week_date"].max())
    return latest_streak.reset_index(drop=True)


def preprocess_centroids(
    df,
    min_weeks,
    max_gap,
    max_staleness_weeks,
    emb_prefix,
    cluster_col,
    user_weight_col,
):
    """
    Clean every cluster into one recent usable weekly series.

    Rules:
    1. Fill short internal gaps only.
    2. Use the latest complete streak, not the longest historical streak.
    3. Skip clusters whose latest observed week is too stale relative to the
       global latest week.
    """
    emb_cols = embedding_cols(df, emb_prefix)
    work = df.copy()
    work["week_date"] = iso_week_start(work["year"], work["week"])
    global_latest_week = work["week_date"].max()

    kept, summary = [], []

    for cluster, group in work.groupby(cluster_col, sort=False):
        group = sort_time(group).copy()
        cluster_latest_week = pd.to_datetime(group["week_date"]).max()
        staleness_weeks = int((global_latest_week - cluster_latest_week).days // 7)

        filled = fill_weekly_gaps(group, emb_cols, max_gap=max_gap, cluster_col=cluster_col, user_weight_col=user_weight_col)
        streak = latest_complete_streak(filled, emb_cols)

        keep = (
            len(streak) >= min_weeks
            and staleness_weeks <= max_staleness_weeks
        )

        summary.append({
            cluster_col: cluster,
            "original_rows": len(group),
            "rows_after_week_grid": len(filled),
            "latest_complete_streak": len(streak),
            "latest_streak_start": streak["week_date"].min() if len(streak) else pd.NaT,
            "latest_streak_end": streak["week_date"].max() if len(streak) else pd.NaT,
            "cluster_latest_week": cluster_latest_week,
            "global_latest_week": global_latest_week,
            "staleness_weeks": staleness_weeks,
            "n_internal_gaps_filled_in_latest_streak": int(streak.get("was_gap_filled", pd.Series(dtype=bool)).sum()) if len(streak) else 0,
            "kept": keep,
        })

        if keep:
            kept.append(streak)

    clean_df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=df.columns)

    summary_df = pd.DataFrame(summary).sort_values(
        ["kept", "staleness_weeks", "latest_complete_streak"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    return clean_df, summary_df


def split_train_validation_test_sequences(
    df,
    validation_weeks,
    test_weeks,
    min_train_weeks,
    cluster_col,
    project_name,
    is_save_split_csv=False
):
    """
    Chronologically split each cluster's latest streak into train, validation, and test.

    Train is used for rolling backtest metrics and model/config selection.
    Validation is kept as a separate middle holdout sequence.
    Test is the latest sequence used for final holdout metrics.
    """
    train_parts, validation_parts, test_parts, summary = [], [], [], []

    for cluster, group in df.groupby(cluster_col, sort=False):
        group = sort_time(group).copy()
        n = len(group)
        required = min_train_weeks + validation_weeks + test_weeks
        keep = n >= required

        summary.append({
            cluster_col: cluster,
            "n_weeks": n,
            "min_required_weeks": required,
            "n_train": max(n - validation_weeks - test_weeks, 0) if keep else 0,
            "n_validation": validation_weeks if keep else 0,
            "n_test": test_weeks if keep else 0,
            "kept": keep,
        })

        if not keep:
            continue

        train_end = n - validation_weeks - test_weeks
        validation_end = n - test_weeks

        train_parts.append(group.iloc[:train_end].assign(sequence_split="train"))
        validation_parts.append(group.iloc[train_end:validation_end].assign(sequence_split="validation"))
        test_parts.append(group.iloc[validation_end:].assign(sequence_split="test"))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    validation_df = pd.concat(validation_parts, ignore_index=True) if validation_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()
    split_summary_df = pd.DataFrame(summary).sort_values(["kept", "n_weeks"], ascending=[False, False])

    if project_name is not None:
        output_dir = Path("projects") / project_name / "centroid_trajectory_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        split_summary_df.to_csv(output_dir / "sequence_split_summary.csv", index=False)
        if is_save_split_csv:
            train_df.to_csv(output_dir / "train_df.csv", index=False)
            validation_df.to_csv(output_dir / "validation_df.csv", index=False)
            test_df.to_csv(output_dir / "test_df.csv", index=False)

    return train_df, validation_df, test_df, split_summary_df
