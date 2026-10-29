import pandas as pd
import streamlit as st
import numpy as np

import ast
import json

import plotly.graph_objects as go

from src.trajectory_dashboard.config import ANOMALY_SCORE_THRESHOLD

def filter_out_overlapping_historical_records(
    df
):
    """
    The function handles overlapping "Historical" and "Predicted" weeks by retaining only the "Predicted" week records for these overlaps.
    """
    
    duplicate_col_check = ["final_cluster_label", "year", "week"]

    return (
        df.assign(_priority=df["source"].ne("Predicted"))
          .sort_values("_priority")
          .drop_duplicates(subset=duplicate_col_check, keep="first")
          .drop(columns="_priority")
          .reset_index(drop=True)
    )

def parse_selected_anomalous_cluster_week(
    selected_cluster_week_anomaly,
    prediction_anomaly_table
):
    """
    Parses the output of the `st.dataframe` to extract the selected cluster name, year-week, and source. 
    """
    
    selected_row = selected_cluster_week_anomaly.selection.rows[0]
    
    cluster_name = prediction_anomaly_table.iloc[selected_row]['Cluster']
    year_week = prediction_anomaly_table.iloc[selected_row]['Week']
    source = prediction_anomaly_table.iloc[selected_row]['Source']
    
    return cluster_name, year_week, source

def get_decoded_behavior(
    df: pd.DataFrame,
    cluster,
    year_week: str,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    1. Query records matching the exact cluster, year, week, and source.
    2. All matching records from the latest available previous week.

    Previous-record source priority:
    - Query source is "Prediction":
      prefer Prediction, then Historical.
    - Otherwise:
      prefer Historical, then Prediction.

    Both returned values are DataFrames with the same columns as `df`.
    """
    required_cols = {"final_cluster_label", "year", "week", "source"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    try:
        target_year, target_week = year_week.upper().split("-W")
        target_year = int(target_year)
        target_week = int(target_week)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            'year_week must use the format "YYYY-Www", such as "2026-W01".'
        ) from exc

    cluster_df = df[df["final_cluster_label"] == cluster].copy()

    requested_source = str(source).casefold()
    normalized_sources = cluster_df["source"].astype(str).str.casefold()

    # Exact query records.
    query_records = cluster_df[
        (cluster_df["year"] == target_year)
        & (cluster_df["week"] == target_week)
        & (normalized_sources == requested_source)
    ].copy()

    # Empty DataFrame with the same schema.
    empty_previous_records = df.iloc[0:0].copy()

    # Find records before the requested year-week.
    previous_records = cluster_df[
        (cluster_df["year"] < target_year)
        | (
            (cluster_df["year"] == target_year)
            & (cluster_df["week"] < target_week)
        )
    ].copy()

    if previous_records.empty:
        return query_records, empty_previous_records

    # Find the latest available previous year-week.
    latest_previous_week = (
        previous_records[["year", "week"]]
        .drop_duplicates()
        .sort_values(["year", "week"])
        .iloc[-1]
    )

    previous_week_records = previous_records[
        (previous_records["year"] == latest_previous_week["year"])
        & (previous_records["week"] == latest_previous_week["week"])
    ].copy()

    # Source priority applies only to previous-week records.
    if requested_source == "prediction":
        source_priority = ["prediction", "historical"]
    else:
        source_priority = ["historical", "prediction"]

    normalized_previous_sources = (
        previous_week_records["source"].astype(str).str.casefold()
    )

    for preferred_source in source_priority:
        matching_records = previous_week_records[
            normalized_previous_sources == preferred_source
        ].copy()

        if not matching_records.empty:
            return query_records, matching_records

    # If neither expected source exists, return all records from that week.
    return query_records, previous_week_records

def display_elapsed_time_changes(
    elapsed_changes,
    current_df,
    added,
    cards_per_row,
):
    """
    A reusable function that displays the changes in feature usage elapsed time as cards.
    The card includes information about the raw elapsed time difference and percent deltas per feature name.
    """
    
    changes = elapsed_changes.copy()

    if not changes.empty:
        changes["percent_change"] = np.where(
            changes["pred_elapsed_df1"] != 0,
            changes["elapsed_change"]
            / changes["pred_elapsed_df1"]
            * 100,
            np.nan,
        )

    added_changes = (
        current_df.loc[
            current_df["feature"].isin(added),
            ["feature", "pred_elapsed"],
        ]
        .rename(columns={"pred_elapsed": "pred_elapsed_df2"})
        .copy()
    )

    if changes.empty and added_changes.empty:
        st.info("No elapsed-time changes.")
        return

    st.markdown("#### Elapsed Time Changes")

    cards = []

    for _, record in changes.iterrows():
        percent_change = record["percent_change"]
        elapsed_change = record["elapsed_change"]

        delta = f"{elapsed_change:+.2f}"

        if not pd.isna(percent_change):
            delta += f" · {percent_change:+.1f}%"

        cards.append(
            {
                "feature": record["feature"],
                "value": record["pred_elapsed_df2"],
                "delta": delta,
                "help": f"Previous: {record['pred_elapsed_df1']:.2f}",
                "sort_value": abs(elapsed_change),
            }
        )

    for _, record in added_changes.iterrows():
        cards.append(
            {
                "feature": record["feature"],
                "value": record["pred_elapsed_df2"],
                "delta": None,
                "help": "Newly added feature",
                "sort_value": abs(record["pred_elapsed_df2"]),
            }
        )

    cards.sort(
        key=lambda card: card["sort_value"],
        reverse=True,
    )

    for start in range(0, len(cards), cards_per_row):
        row = cards[start : start + cards_per_row]
        columns = st.columns(len(row))

        for column, card in zip(columns, row):
            with column:
                st.metric(
                    label=str(card["feature"]),
                    value=f"{card['value']:.2f}",
                    delta=card["delta"],
                    delta_color="inverse",
                    border=True,
                    help=card["help"],
                )
def display_feature_usage_changes(
    added,
    removed
):
    """
    A reusable function that displays the changes in feature usage composition as cards.
    The card includes a list of added and removed features comparing two usage weeks.
    """
    st.markdown("#### Feature Usage Changes")
    if added and removed:
        added_col, removed_col = st.columns(2)

        with added_col:
            with st.container(border=True):
                st.markdown("##### ✅ Added Features")

                for feature in added:
                    st.markdown(f":green-background[{feature}]")

        with removed_col:
            with st.container(border=True):
                st.markdown("##### ❌ Removed Features")

                for feature in removed:
                    st.markdown(f":red-background[{feature}]")

    elif added:
        with st.container(border=True):
            st.markdown("##### ✅ Added Features")

            for feature in added:
                st.markdown(f":green-background[{feature}]")

    elif removed:
        with st.container(border=True):
            st.markdown("##### ❌ Removed Features")

            for feature in removed:
                st.markdown(f":red-background[{feature}]")
                
def compare_behavioral_changes(
    df1,
    df2,
):
    """
    Compare the decoded behavioral changes of two decoded behavior week dfs.

    Returns:
        added_features:
            Feature names present only in df2.

        removed_features:
            Feature names present only in df1.

        elapsed_change_dict:
            Dictionary mapping each shared feature to its elapsed-time change.
            Change is calculated as df2 - df1.

        elapsed_changes:
            DataFrame containing the original and changed elapsed times.
    """
    required_cols = {"feature", "pred_elapsed"}

    for name, frame in (("df1", df1), ("df2", df2)):
        missing_cols = required_cols - set(frame.columns)

        if missing_cols:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing_cols)}"
            )

        if frame["feature"].duplicated().any():
            duplicated = (
                frame.loc[frame["feature"].duplicated(), "feature"]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"{name} contains duplicate features: {duplicated}"
            )

    left = df1[["feature", "pred_elapsed"]].rename(
        columns={"pred_elapsed": "pred_elapsed_df1"}
    )

    right = df2[["feature", "pred_elapsed"]].rename(
        columns={"pred_elapsed": "pred_elapsed_df2"}
    )

    comparison = left.merge(
        right,
        on="feature",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    added_features = comparison.loc[
        comparison["_merge"] == "right_only",
        "feature",
    ].tolist()

    removed_features = comparison.loc[
        comparison["_merge"] == "left_only",
        "feature",
    ].tolist()

    elapsed_changes = comparison.loc[
        comparison["_merge"] == "both",
        ["feature", "pred_elapsed_df1", "pred_elapsed_df2"],
    ].copy()

    elapsed_changes["elapsed_change"] = (
        elapsed_changes["pred_elapsed_df2"]
        - elapsed_changes["pred_elapsed_df1"]
    )

    elapsed_changes = elapsed_changes.reset_index(drop=True)

    elapsed_change_dict = elapsed_changes.set_index(
        "feature"
    )["elapsed_change"].to_dict()

    return (
        added_features,
        removed_features,
        elapsed_change_dict,
        elapsed_changes,
    )
    
def _parse_decoded_behavior(value):
    """
    Convert decoded_behavior into a list of dictionaries.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []

        if isinstance(parsed, dict):
            return [parsed]

        if isinstance(parsed, list):
            return parsed

    return []

def prepare_decoded_behavior_long_df(
    df,
    decoded_behavior_col="decoded_behavior",
):
    """
    Convert the centroid-level dataframe into one row per
    cluster-week-feature.
    """
    required_cols = {
        "year",
        "week",
        "final_cluster_label",
        "behavior_change_score",
        "is_prediction",
        decoded_behavior_col,
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    working_df = df.copy()

    working_df["week_start"] = pd.to_datetime(
        working_df["year"].astype(str)
        + "-W"
        + working_df["week"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    working_df["year_week"] = (
        working_df["year"].astype(str)
        + "-W"
        + working_df["week"].astype(str).str.zfill(2)
    )

    working_df["decoded_behavior_parsed"] = working_df[
        decoded_behavior_col
    ].apply(_parse_decoded_behavior)

    long_df = working_df.explode(
        "decoded_behavior_parsed",
        ignore_index=True,
    )

    long_df = long_df[
        long_df["decoded_behavior_parsed"].apply(
            lambda value: isinstance(value, dict)
        )
    ].copy()

    long_df["feature"] = long_df[
        "decoded_behavior_parsed"
    ].apply(
        lambda value: value.get("feature")
    )

    long_df["used_prob"] = pd.to_numeric(
        long_df["decoded_behavior_parsed"].apply(
            lambda value: value.get(
                "used_prob",
                value.get("usage_probability"),
            )
        ),
        errors="coerce",
    )

    long_df["elapsed"] = pd.to_numeric(
        long_df["decoded_behavior_parsed"].apply(
            lambda value: value.get(
                "elapsed",
                value.get(
                    "elapsed_time",
                    value.get("elapsed_seconds"),
                ),
            )
        ),
        errors="coerce",
    )

    long_df = long_df.dropna(
        subset=["feature", "week_start"]
    ).copy()

    long_df["elapsed"] = long_df["elapsed"].fillna(0)
    long_df["used_prob"] = long_df["used_prob"].fillna(0)

    long_df["source"] = np.where(
        long_df["is_prediction"].astype(bool),
        "Predicted",
        "Historical",
    )

    long_df["is_anomaly"] = (
        long_df["behavior_change_score"]
        >= ANOMALY_SCORE_THRESHOLD
    )

    return long_df.sort_values(
        [
            "final_cluster_label",
            "week_start",
            "feature",
        ]
    ).reset_index(drop=True)
    
def plot_cluster_decoded_behavior_stacked_bar(
    decoded_long_df,
    cluster_label,
    year,
    week,
    anomaly_score_threshold=3,
    elapsed_unit="hours",
    top_n_features=None,
    latest_n_weeks=52,
):
    """
    Plot the historical and predicted decoded feature composition for one
    cluster, centered on a selected year-week.

    Historical weeks use historical decoded behavior. Weeks that contain
    prediction rows use predicted decoded behavior instead. Prediction
    periods are shaded gray.

    Every displayed week whose behavior_change_score is greater than or
    equal to ``anomaly_score_threshold`` is outlined as anomalous. Non-focused
    anomalous weeks use a muted red border without an annotation. The selected
    year-week uses a bright red border and an arrow annotation.

    Parameters
    ----------
    decoded_long_df : pandas.DataFrame
        Long-form decoded behavior data with one row per
        cluster-week-feature-source.

    cluster_label : str or int
        Cluster to display.

    year : int
        ISO year of the week to focus on.

    week : int or str
        ISO week to focus on. Accepts values such as 1, "1", "W01", or
        "2026-W01". When a full year-week string is supplied, its year is
        used and must match ``year``.

    anomaly_score_threshold : float, default=3
        Minimum behavior_change_score required for a displayed week to be
        marked as anomalous. Only the focused week receives the bright red
        arrow annotation.

    elapsed_unit : {"seconds", "minutes", "hours"}, default="hours"
        Unit used for elapsed-time values in the chart.

    top_n_features : int or None, default=None
        Keep only the top N features by elapsed time. Remaining features are
        grouped into "Other".

    latest_n_weeks : int, default=52
        Total number of unique weeks to display. The selected year-week is
        kept as close to the center as possible. At the start or end of the
        available timeline, the window shifts while retaining the selected
        week.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Stacked bar chart of weekly decoded feature elapsed time.

    visualized_weeks : list[str]
        Ordered year-week labels included in the displayed timeline.
    """

    required_cols = {
        "final_cluster_label",
        "week_start",
        "year_week",
        "feature",
        "elapsed",
        "used_prob",
        "is_prediction",
        "behavior_change_score",
        "behavior_change_type",
        "triggered_main_flags",
    }

    missing_cols = required_cols - set(decoded_long_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    try:
        latest_n_weeks = int(latest_n_weeks)
    except (TypeError, ValueError) as exc:
        raise ValueError("latest_n_weeks must be an integer.") from exc

    if latest_n_weeks < 1:
        raise ValueError("latest_n_weeks must be at least 1.")

    try:
        focus_year = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError("year must be an integer-like value.") from exc

    week_text = str(week).strip().upper()

    if "-W" in week_text:
        supplied_year_text, supplied_week_text = week_text.split("-W", 1)
        try:
            supplied_year = int(supplied_year_text)
            focus_week = int(supplied_week_text)
        except ValueError as exc:
            raise ValueError(
                "week must be an integer-like value, 'W01', or '2026-W01'."
            ) from exc

        if supplied_year != focus_year:
            raise ValueError(
                f"The year in week={week!r} does not match year={focus_year}."
            )
    else:
        try:
            focus_week = int(week_text.replace("W", ""))
        except ValueError as exc:
            raise ValueError(
                "week must be an integer-like value, 'W01', or '2026-W01'."
            ) from exc

    if not 1 <= focus_week <= 53:
        raise ValueError("week must be between 1 and 53.")

    elapsed_unit = str(elapsed_unit).lower()
    unit_divisors = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
    }
    if elapsed_unit not in unit_divisors:
        raise ValueError(
            "elapsed_unit must be 'seconds', 'minutes', or 'hours'."
        )

    if top_n_features is not None:
        try:
            top_n_features = int(top_n_features)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_n_features must be an integer or None.") from exc

        if top_n_features < 1:
            raise ValueError("top_n_features must be at least 1 or None.")

    # --------------------------------------------------------
    # 1. Prepare the selected cluster.
    # --------------------------------------------------------
    cluster_all_df = decoded_long_df.loc[
        decoded_long_df["final_cluster_label"] == cluster_label
    ].copy()

    if cluster_all_df.empty:
        raise ValueError(
            f"No decoded behavior data found for cluster {cluster_label!r}."
        )

    cluster_all_df["week_start"] = pd.to_datetime(
        cluster_all_df["week_start"],
        errors="coerce",
    )
    cluster_all_df["is_prediction"] = (
        cluster_all_df["is_prediction"].fillna(False).astype(bool)
    )
    cluster_all_df["year_week"] = cluster_all_df["year_week"].astype(str)

    cluster_all_df = cluster_all_df.dropna(
        subset=["week_start", "year_week", "feature"]
    ).copy()

    if cluster_all_df.empty:
        raise ValueError(
            f"No valid weekly decoded behavior remains for cluster "
            f"{cluster_label!r}."
        )

    # --------------------------------------------------------
    # 2. Build the complete weekly timeline and center the view
    #    on the requested ISO year-week.
    # --------------------------------------------------------
    timeline_df = (
        cluster_all_df.groupby(
            ["week_start", "year_week"],
            as_index=False,
        )
        .agg(has_prediction=("is_prediction", "max"))
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    iso_values = timeline_df["week_start"].dt.isocalendar()
    focus_mask = (
        iso_values["year"].astype(int).eq(focus_year)
        & iso_values["week"].astype(int).eq(focus_week)
    )

    if not focus_mask.any():
        raise ValueError(
            f"Week {focus_year}-W{focus_week:02d} was not found for "
            f"cluster {cluster_label!r}."
        )

    focus_index = int(np.flatnonzero(focus_mask.to_numpy())[0])
    focus_year_week = timeline_df.loc[focus_index, "year_week"]

    weeks_before = latest_n_weeks // 2
    start_index = focus_index - weeks_before
    end_index = start_index + latest_n_weeks

    if start_index < 0:
        end_index -= start_index
        start_index = 0

    if end_index > len(timeline_df):
        overflow = end_index - len(timeline_df)
        start_index = max(0, start_index - overflow)
        end_index = len(timeline_df)

    visible_timeline_df = timeline_df.iloc[start_index:end_index].copy()
    visible_timeline_df = visible_timeline_df.reset_index(drop=True)

    if visible_timeline_df.empty:
        raise ValueError("No weeks are available in the requested view.")

    week_order = visible_timeline_df["year_week"].tolist()
    focus_plot_index = week_order.index(focus_year_week)

    cluster_all_df = cluster_all_df.merge(
        visible_timeline_df[["week_start", "year_week"]],
        on=["week_start", "year_week"],
        how="inner",
    )

    # --------------------------------------------------------
    # 3. Choose one source per week.
    #
    #    - Prediction exists -> predicted rows
    #    - No prediction      -> historical rows
    # --------------------------------------------------------
    source_by_week_df = (
        cluster_all_df.groupby(
            ["week_start", "year_week"],
            as_index=False,
        )
        .agg(has_prediction=("is_prediction", "max"))
    )

    cluster_df = cluster_all_df.merge(
        source_by_week_df,
        on=["week_start", "year_week"],
        how="left",
    )

    cluster_df = cluster_df.loc[
        (
            cluster_df["has_prediction"]
            & cluster_df["is_prediction"]
        )
        |
        (
            ~cluster_df["has_prediction"]
            & ~cluster_df["is_prediction"]
        )
    ].copy()

    if cluster_df.empty:
        raise ValueError(
            "No historical or predicted decoded behavior is available in "
            "the selected timeline."
        )

    cluster_df["source"] = np.where(
        cluster_df["is_prediction"],
        "Predicted",
        "Historical",
    )

    # --------------------------------------------------------
    # 4. Clean and convert values.
    # --------------------------------------------------------
    cluster_df["elapsed_display"] = (
        pd.to_numeric(cluster_df["elapsed"], errors="coerce").fillna(0)
        / unit_divisors[elapsed_unit]
    )
    cluster_df["used_prob"] = pd.to_numeric(
        cluster_df["used_prob"], errors="coerce"
    ).fillna(0)
    cluster_df["behavior_change_score"] = pd.to_numeric(
        cluster_df["behavior_change_score"], errors="coerce"
    )
    cluster_df["behavior_change_type"] = (
        cluster_df["behavior_change_type"].fillna("None").astype(str)
    )
    cluster_df["triggered_main_flags"] = (
        cluster_df["triggered_main_flags"].fillna("None").astype(str)
    )

    # --------------------------------------------------------
    # 5. Optionally group lower-volume features into Other.
    # --------------------------------------------------------
    if top_n_features is None:
        cluster_df["feature_display"] = cluster_df["feature"].astype(str)
    else:
        feature_totals = (
            cluster_df.groupby("feature")["elapsed_display"]
            .sum()
            .sort_values(ascending=False)
        )
        top_features = set(feature_totals.head(top_n_features).index)

        cluster_df["feature_display"] = np.where(
            cluster_df["feature"].isin(top_features),
            cluster_df["feature"].astype(str),
            "Other",
        )

    # --------------------------------------------------------
    # 6. Aggregate the feature bars and weekly anomaly details.
    # --------------------------------------------------------
    def combine_nonempty(values):
        unique_values = dict.fromkeys(
            value
            for value in values.astype(str)
            if value and value != "None" and value.lower() != "nan"
        )
        return ", ".join(unique_values) or "None"

    plot_df = (
        cluster_df.groupby(
            [
                "week_start",
                "year_week",
                "feature_display",
                "source",
                "is_prediction",
            ],
            as_index=False,
        )
        .agg(
            elapsed_display=("elapsed_display", "sum"),
            used_prob=("used_prob", "sum"),
            behavior_change_score=("behavior_change_score", "max"),
            behavior_change_type=("behavior_change_type", combine_nonempty),
            triggered_main_flags=("triggered_main_flags", combine_nonempty),
        )
    )

    week_summary_df = (
        cluster_df.groupby(
            ["week_start", "year_week"],
            as_index=False,
        )
        .agg(
            is_prediction=("is_prediction", "max"),
            source=("source", "first"),
            behavior_change_score=("behavior_change_score", "max"),
            behavior_change_type=("behavior_change_type", combine_nonempty),
            triggered_main_flags=("triggered_main_flags", combine_nonempty),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    plot_df["year_week"] = pd.Categorical(
        plot_df["year_week"],
        categories=week_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["year_week", "feature_display"])

    # --------------------------------------------------------
    # 7. Create stacked feature-composition bars.
    # --------------------------------------------------------
    fig = go.Figure()

    features = (
        plot_df.groupby("feature_display", observed=True)["elapsed_display"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for feature in features:
        feature_df = plot_df.loc[
            plot_df["feature_display"] == feature
        ].copy()

        customdata = np.column_stack(
            [
                feature_df["source"],
                feature_df["used_prob"],
                feature_df["behavior_change_score"],
                feature_df["behavior_change_type"],
                feature_df["triggered_main_flags"],
            ]
        )

        fig.add_trace(
            go.Bar(
                x=feature_df["year_week"],
                y=feature_df["elapsed_display"],
                name=str(feature),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x} · " + str(feature) + "</b><br>"
                    f"Elapsed: %{{y:,.2f}} {elapsed_unit}<br>"
                    "Probability: %{customdata[1]:.3f}<br>"
                    "Source: %{customdata[0]}<br>"
                    "Score: %{customdata[2]:.2f}<br>"
                    # "Type: %{customdata[3]}<br>"
                    "Flags: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # 8. Shade every contiguous prediction period shown.
    # --------------------------------------------------------
    prediction_flags = visible_timeline_df["has_prediction"].astype(bool).tolist()
    prediction_runs = []
    run_start = None

    for index, is_prediction_week in enumerate(prediction_flags):
        if is_prediction_week and run_start is None:
            run_start = index

        is_last_index = index == len(prediction_flags) - 1
        run_ends_here = run_start is not None and (
            not is_prediction_week or is_last_index
        )

        if run_ends_here:
            if is_prediction_week and is_last_index:
                run_end = index
            else:
                run_end = index - 1

            prediction_runs.append((run_start, run_end))
            run_start = None

    for run_number, (run_start, run_end) in enumerate(prediction_runs):
        fig.add_vrect(
            x0=run_start - 0.5,
            x1=run_end + 0.5,
            fillcolor="gray",
            opacity=0.12,
            line_width=0,
            layer="below",
        )

        fig.add_vline(
            x=run_start - 0.5,
            line_color="gray",
            line_dash="dash",
            line_width=1.5,
        )

        if run_number == 0:
            fig.add_annotation(
                x=(run_start + run_end) / 2,
                y=1.06,
                xref="x",
                yref="paper",
                text="<b>Prediction period</b>",
                showarrow=False,
            )

    # --------------------------------------------------------
    # 9. Mark all anomalous weeks.
    #
    #    - Other anomalous weeks: muted red border only
    #    - Focused anomalous week: bright red border + annotation
    # --------------------------------------------------------
    focused_summary_df = week_summary_df.loc[
        week_summary_df["year_week"] == focus_year_week
    ]

    if focused_summary_df.empty:
        raise ValueError(
            f"No displayed behavior was found for focused week "
            f"{focus_year}-W{focus_week:02d}."
        )

    focused_row = focused_summary_df.iloc[0]
    focused_score = focused_row["behavior_change_score"]
    focused_is_anomaly = (
        pd.notna(focused_score)
        and float(focused_score) >= anomaly_score_threshold
    )

    weekly_totals = (
        plot_df.groupby("year_week", observed=True)["elapsed_display"]
        .sum()
        .to_dict()
    )

    anomalous_weeks_df = week_summary_df.loc[
        pd.to_numeric(
            week_summary_df["behavior_change_score"],
            errors="coerce",
        ) >= anomaly_score_threshold
    ].copy()

    muted_anomaly_color = "rgba(200, 0, 0, 0.80)"

    for _, anomaly_row in anomalous_weeks_df.iterrows():
        anomaly_week = anomaly_row["year_week"]

        if anomaly_week not in week_order:
            continue

        anomaly_total = float(weekly_totals.get(anomaly_week, 0))
        if anomaly_total <= 0:
            continue

        anomaly_index = week_order.index(anomaly_week)
        is_focused_week = anomaly_week == focus_year_week

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=anomaly_index - 0.43,
            x1=anomaly_index + 0.43,
            y0=0,
            y1=anomaly_total,
            line=dict(
                color="red" if is_focused_week else muted_anomaly_color,
                width=3 if is_focused_week else 2,
            ),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    if focused_is_anomaly:
        focused_total = float(weekly_totals.get(focus_year_week, 0))

        if focused_total > 0:
            fig.add_annotation(
                x=focus_year_week,
                y=focused_total,
                text=f"<b>Anomaly</b> {float(focused_score):.2f}",
                showarrow=True,
                arrowcolor="red",
                arrowwidth=1.5,
                arrowhead=2,
                ax=0,
                ay=-35,
                font=dict(color="red"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="red",
                borderwidth=1,
                hovertext=(
                    f"Source: {focused_row['source']}<br>"
                    # f"Type: {focused_row['behavior_change_type']}<br>"
                    f"Flags: {focused_row['triggered_main_flags']}"
                ),
            )

    other_anomalies_exist = anomalous_weeks_df["year_week"].ne(
        focus_year_week
    ).any()

    if other_anomalies_exist:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=(
                    "Other anomalies "
                    f"(score >= {anomaly_score_threshold})"
                ),
                marker=dict(
                    symbol="square-open",
                    size=13,
                    color=muted_anomaly_color,
                    line=dict(color=muted_anomaly_color, width=2),
                ),
                hoverinfo="skip",
            )
        )

    if focused_is_anomaly:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Focused anomaly",
                marker=dict(
                    symbol="square-open",
                    size=14,
                    color="red",
                    line=dict(color="red", width=3),
                ),
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # 10. Final layout.
    # --------------------------------------------------------
    focused_source = str(focused_row["source"])

    fig.update_layout(
        title=(
            "Historical and Predicted Feature Usage by Week"
            f"<br><sup>Cluster: {cluster_label} · "
            f"Focused week: {focus_year}-W{focus_week:02d} "
            f"({focused_source}) · "
            f"{len(week_order)} weeks displayed</sup>"
        ),
        xaxis_title="Week",
        yaxis_title=f"Decoded elapsed time ({elapsed_unit})",
        barmode="stack",
        hovermode="closest",
        height=650,
        bargap=0.15,
        legend_title_text="Feature",
        margin=dict(t=115, r=40, b=90, l=80),
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=week_order,
        tickangle=-45,
    )
    fig.update_yaxes(rangemode="tozero")

    return fig, week_order