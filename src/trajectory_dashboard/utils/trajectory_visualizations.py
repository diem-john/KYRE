import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def build_trajectory_3d(
    trajectory_data: pd.DataFrame,
    show_predicted_anomalies: bool = True,
    show_historical_anomalies: bool = False,
    anomaly_score_threshold: int = 3,
) -> go.Figure:
    """
    Build a 3D historical and predicted centroid trajectory.

    Historical trajectory:
        Filled circular markers with solid connecting lines.

    Predicted trajectory:
        Open diamond markers with dashed connecting lines.

    Historical and predicted rows belonging to the same cluster use
    the same color.
    """

    required_columns = [
        "source",
        "final_cluster_label",
        "year_week",
        "week_start",
        "pca_0",
        "pca_1",
        "pca_2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in trajectory_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Trajectory visualization is missing columns: "
            f"{missing_columns}"
        )

    plot_data = trajectory_data.copy()

    plot_data = plot_data.dropna(
        subset=[
            "week_start",
            "pca_0",
            "pca_1",
            "pca_2",
        ]
    )

    plot_data = plot_data.sort_values(
        [
            "final_cluster_label",
            "source",
            "week_start",
        ]
    )

    figure = go.Figure()

    cluster_values = sorted(
        plot_data["final_cluster_label"]
        .dropna()
        .unique()
        .tolist(),
        key=str,
    )

    color_palette = px.colors.qualitative.Dark24

    cluster_color_map = {
        cluster: color_palette[
            index % len(color_palette)
        ]
        for index, cluster in enumerate(cluster_values)
    }

    for cluster in cluster_values:
        cluster_color = cluster_color_map[cluster]

        cluster_data = plot_data.loc[
            plot_data["final_cluster_label"] == cluster
        ].copy()

        historical_data = (
            cluster_data.loc[
                cluster_data["source"] == "Historical"
            ]
            .sort_values("week_start")
            .copy()
        )

        predicted_data = (
            cluster_data.loc[
                cluster_data["source"] == "Predicted"
            ]
            .sort_values("week_start")
            .copy()
        )

        historical_non_overlap = historical_data.copy()
        historical_overlap = historical_data.iloc[0:0].copy()

        if not predicted_data.empty:
            predicted_start_week = predicted_data["week_start"].min()
            predicted_end_week = predicted_data["week_start"].max()

            historical_overlap = historical_data.loc[
                historical_data["week_start"].between(
                    predicted_start_week,
                    predicted_end_week,
                    inclusive="both",
                )
            ].copy()

            historical_non_overlap = historical_data.loc[
                ~historical_data["week_start"].between(
                    predicted_start_week,
                    predicted_end_week,
                    inclusive="both",
                )
            ].copy()

        # Add the final non-overlapping historical point as an anchor
        # so the faded overlap path remains connected.
        if (
            not historical_non_overlap.empty
            and not historical_overlap.empty
        ):
            first_overlap_week = (
                historical_overlap["week_start"].min()
            )

            overlap_anchor = historical_non_overlap.loc[
                historical_non_overlap["week_start"]
                < first_overlap_week
            ].sort_values("week_start").tail(1)

            if not overlap_anchor.empty:
                historical_overlap = pd.concat(
                    [
                        overlap_anchor,
                        historical_overlap,
                    ],
                    ignore_index=True,
                )

        # Historical trajectory outside the prediction period
        if not historical_non_overlap.empty:
            figure.add_trace(
                go.Scatter3d(
                    x=historical_non_overlap["pca_0"],
                    y=historical_non_overlap["pca_1"],
                    z=historical_non_overlap["pca_2"],
                    mode="lines+markers",
                    name=f"{cluster} - Historical",
                    legendgroup=str(cluster),
                    line={
                        "color": cluster_color,
                        "width": 4,
                    },
                    marker={
                        "color": cluster_color,
                        "size": 5,
                        "symbol": "circle",
                    },
                    opacity=0.9,
                    customdata=historical_non_overlap[
                        [
                            "final_cluster_label",
                            "year_week",
                            "source",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>Historical centroid</b><br>"
                        "Cluster: %{customdata[0]}<br>"
                        "Week: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )


        # Historical trajectory during the prediction period
        if not historical_overlap.empty:
            figure.add_trace(
                go.Scatter3d(
                    x=historical_overlap["pca_0"],
                    y=historical_overlap["pca_1"],
                    z=historical_overlap["pca_2"],
                    mode="lines+markers",
                    name=f"{cluster} - Historical overlap",
                    legendgroup=str(cluster),
                    showlegend=False,
                    line={
                        "color": cluster_color,
                        "width": 4,
                    },
                    marker={
                        "color": cluster_color,
                        "size": 5,
                        "symbol": "circle",
                    },
                    opacity=0.5,
                    customdata=historical_overlap[
                        [
                            "final_cluster_label",
                            "year_week",
                            "source",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>Historical centroid during prediction period</b><br>"
                        "Cluster: %{customdata[0]}<br>"
                        "Week: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )

        # Connect the final historical point before the prediction
        # period to the first predicted point.
        if (
            not historical_data.empty
            and not predicted_data.empty
        ):
            first_predicted = predicted_data.head(1)

            first_predicted_week = (
                first_predicted["week_start"].iloc[0]
            )

            historical_before_prediction = historical_data.loc[
                historical_data["week_start"]
                < first_predicted_week
            ].copy()

            if not historical_before_prediction.empty:
                final_historical = (
                    historical_before_prediction
                    .sort_values("week_start")
                    .tail(1)
                )

                connection_data = pd.concat(
                    [
                        final_historical,
                        first_predicted,
                    ],
                    ignore_index=True,
                )

                figure.add_trace(
                    go.Scatter3d(
                        x=connection_data["pca_0"],
                        y=connection_data["pca_1"],
                        z=connection_data["pca_2"],
                        mode="lines",
                        name=f"{cluster} - Forecast connection",
                        legendgroup=str(cluster),
                        showlegend=False,
                        line={
                            "color": cluster_color,
                            "width": 3,
                            "dash": "dot",
                        },
                        hoverinfo="skip",
                    )
                )

        # Predicted trajectory line
        if not predicted_data.empty:
            figure.add_trace(
                go.Scatter3d(
                    x=predicted_data["pca_0"],
                    y=predicted_data["pca_1"],
                    z=predicted_data["pca_2"],
                    mode="lines",
                    name=f"{cluster} - Predicted path",
                    legendgroup=str(cluster),
                    showlegend=False,
                    line={
                        "color": cluster_color,
                        "width": 4,
                        "dash": "dash",
                    },
                    hoverinfo="skip",
                )
            )

            # Predicted trajectory markers
            figure.add_trace(
                go.Scatter3d(
                    x=predicted_data["pca_0"],
                    y=predicted_data["pca_1"],
                    z=predicted_data["pca_2"],
                    mode="markers",
                    name=f"{cluster} - Predicted",
                    legendgroup=str(cluster),
                    marker={
                        "color": cluster_color,
                        "size": 7,
                        "symbol": "diamond-open",
                        "line": {
                            "color": cluster_color,
                            "width": 3,
                        },
                        "opacity": 1.0,
                    },
                    customdata=predicted_data[
                        [
                            "final_cluster_label",
                            "year_week",
                            "source",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>Predicted centroid</b><br>"
                        "Cluster: %{customdata[0]}<br>"
                        "Week: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )

    # =========================================================
    # Optional anomaly overlay
    # =========================================================

    anomaly_score_values = pd.to_numeric(
        plot_data["behavior_change_score"],
        errors="coerce",
    )

    base_anomaly_data = plot_data.loc[
        anomaly_score_values
        >= anomaly_score_threshold
    ].copy()


    def add_anomaly_trace(
        anomaly_data: pd.DataFrame,
        trace_name: str,
        legend_group: str,
    ) -> None:
        if anomaly_data.empty:
            return

        anomaly_custom_columns = [
            "final_cluster_label",
            "year_week",
            "source",
            "anomaly_priority_level",
            "behavior_change_type",
            "triggered_main_flags",
        ]

        available_anomaly_columns = [
            column
            for column in anomaly_custom_columns
            if column in anomaly_data.columns
        ]

        figure.add_trace(
            go.Scatter3d(
                x=anomaly_data["pca_0"],
                y=anomaly_data["pca_1"],
                z=anomaly_data["pca_2"],
                mode="markers",
                name=trace_name,
                legendgroup=legend_group,
                marker={
                    "size": 11,
                    "symbol": "circle-open",
                    "color": "red",
                    "line": {
                        "color": "red",
                        "width": 4,
                    },
                    "opacity": 0.8,
                },
                customdata=anomaly_data[
                    available_anomaly_columns
                ].to_numpy(),
                hovertemplate=(
                    "<b>Anomaly summary</b><br>"
                    "Cluster: %{customdata[0]}<br>"
                    "Week: %{customdata[1]}<br>"
                    "Source: %{customdata[2]}<br>"
                    "Priority level: %{customdata[3]}<br>"
                    "Behavior change: %{customdata[4]}<br>"
                    "Triggered main flags: %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )

    if show_predicted_anomalies:
        predicted_anomaly_data = (
            base_anomaly_data.loc[
                base_anomaly_data["source"]
                == "Predicted"
            ].copy()
        )

        add_anomaly_trace(
            anomaly_data=predicted_anomaly_data,
            trace_name="Predicted anomaly",
            legend_group="predicted_anomalies",
        )

    if show_historical_anomalies:
        historical_anomaly_data = (
            base_anomaly_data.loc[
                base_anomaly_data["source"]
                == "Historical"
            ].copy()
        )

        add_anomaly_trace(
            anomaly_data=historical_anomaly_data,
            trace_name="Historical anomaly",
            legend_group="historical_anomalies",
        )

    figure.update_layout(
        height=650,
        margin={
            "l": 0,
            "r": 220,
            "t": 30,
            "b": 0,
        },
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "title": {
                "text": "Trajectory",
            },
        },
        scene={
            "xaxis_title": "PCA 1",
            "yaxis_title": "PCA 2",
            "zaxis_title": "PCA 3",
            "aspectmode": "data",
        },
    )

    return figure


def build_decoded_stacked_bar(
    decoded_data: pd.DataFrame,
    anomaly_data: pd.DataFrame,
    cluster,
    prediction_period_source: str,
    feature_order: list[str],
    feature_color_map: dict,
    y_axis_max: float | None = None,
    anomaly_score_threshold: int = 3,
) -> go.Figure:
    """
    Build one continuous decoded-behavior timeline.

    Before the prediction boundary:
        Historical decoded values.

    After the prediction boundary:
        Either Historical or Predicted decoded values, depending on
        prediction_period_source.
    """

    required_columns = [
        "source",
        "final_cluster_label",
        "week_start",
        "year_week",
        "feature",
        "pred_elapsed",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in decoded_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Decoded stacked bar is missing columns: "
            f"{missing_columns}"
        )

    if prediction_period_source not in {
        "Historical",
        "Predicted",
    }:
        raise ValueError(
            "prediction_period_source must be either "
            "'Historical' or 'Predicted'."
        )

    cluster_data = decoded_data.loc[
        decoded_data["final_cluster_label"] == cluster
    ].copy()

    cluster_data["pred_elapsed"] = pd.to_numeric(
        cluster_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)

    cluster_data = cluster_data.dropna(
        subset=[
            "week_start",
            "feature",
        ]
    )

    historical_data = cluster_data.loc[
        cluster_data["source"] == "Historical"
    ].copy()

    predicted_data = cluster_data.loc[
        cluster_data["source"] == "Predicted"
    ].copy()

    figure = go.Figure()

    if historical_data.empty and predicted_data.empty:
        figure.update_layout(
            title="No decoded behavior available",
            height=600,
            annotations=[
                {
                    "text": (
                        "No decoded behavior is available for the "
                        "selected cluster and week range."
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return figure

    # The first predicted week is the boundary between
    # historical context and the prediction period.
    prediction_start = (
        predicted_data["week_start"].min()
        if not predicted_data.empty
        else pd.NaT
    )

    if pd.isna(prediction_start):
        # No prediction rows exist, so show historical values only.
        chart_data = historical_data.copy()

    elif prediction_period_source == "Historical":
        # Actual timeline:
        # historical values before and during the prediction period.
        chart_data = historical_data.copy()

    else:
        # Prediction timeline:
        # historical context before prediction start
        # + predicted values from prediction start onward.
        historical_context = historical_data.loc[
            historical_data["week_start"]
            < prediction_start
        ].copy()

        predicted_continuation = predicted_data.loc[
            predicted_data["week_start"]
            >= prediction_start
        ].copy()

        chart_data = pd.concat(
            [
                historical_context,
                predicted_continuation,
            ],
            ignore_index=True,
        )

    chart_data = chart_data.loc[
        chart_data["feature"].isin(feature_order)
    ].copy()

    weekly_feature_data = (
        chart_data.groupby(
            [
                "week_start",
                "year_week",
                "feature",
            ],
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            [
                "week_start",
                "feature",
            ]
        )
    )

    for feature in feature_order:
        feature_data = weekly_feature_data.loc[
            weekly_feature_data["feature"] == feature
        ].copy()

        if feature_data.empty:
            continue

        figure.add_trace(
            go.Bar(
                x=feature_data["week_start"],
                y=feature_data["pred_elapsed"],
                name=str(feature),
                marker={
                    "color": feature_color_map.get(feature),
                },
                customdata=feature_data[
                    [
                        "year_week",
                        "feature",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Week: %{customdata[0]}<br>"
                    "Elapsed time: %{y:,.2f} seconds"
                    "<extra></extra>"
                ),
            )
        )

    # =====================================================
    # Prediction-period boundary and shading
    # =====================================================

    if pd.notna(prediction_start):
        available_end_candidates = [
            weekly_feature_data["week_start"].max()
            if not weekly_feature_data.empty
            else pd.NaT,
            historical_data["week_start"].max()
            if not historical_data.empty
            else pd.NaT,
            predicted_data["week_start"].max()
            if not predicted_data.empty
            else pd.NaT,
        ]

        available_end_candidates = [
            value
            for value in available_end_candidates
            if pd.notna(value)
        ]

        chart_end = (
            max(available_end_candidates)
            if available_end_candidates
            else prediction_start
        )

        figure.add_vrect(
            x0=prediction_start,
            x1=chart_end + pd.Timedelta(days=4),
            fillcolor="gray",
            opacity=0.12,
            layer="below",
            line_width=0,
        )

        figure.add_vline(
            x=prediction_start,
            line_dash="dash",
            line_width=2,
            line_color="gray",
        )

        if not historical_data.empty:
            figure.add_annotation(
                x=prediction_start - pd.Timedelta(days=4),
                y=1.04,
                xref="x",
                yref="paper",
                text="<b>Historical period</b>",
                showarrow=False,
                xanchor="right",
            )

        prediction_label = (
            "Prediction period · historical values"
            if prediction_period_source == "Historical"
            else "Prediction period · predicted values"
        )

        figure.add_annotation(
            x=prediction_start + pd.Timedelta(days=4),
            y=1.04,
            xref="x",
            yref="paper",
            text=f"<b>{prediction_label}</b>",
            showarrow=False,
            xanchor="left",
        )


    # =====================================================
    # Anomaly outlines
    # =====================================================

    if (
        not anomaly_data.empty
        and "behavior_change_score" in anomaly_data.columns
    ):
        if prediction_period_source == "Historical":
            relevant_anomalies = anomaly_data.loc[
                (
                    anomaly_data["final_cluster_label"]
                    == cluster
                )
                & (
                    anomaly_data["source"]
                    == "Historical"
                )
            ].copy()

        elif pd.notna(prediction_start):
            historical_anomalies = anomaly_data.loc[
                (
                    anomaly_data["final_cluster_label"] == cluster
                )
                & (
                    anomaly_data["source"] == "Historical"
                )
                & (
                    anomaly_data["week_start"] < prediction_start
                )
            ].copy()

            predicted_anomalies = anomaly_data.loc[
                (
                    anomaly_data["final_cluster_label"] == cluster
                )
                & (
                    anomaly_data["source"] == "Predicted"
                )
                & (
                    anomaly_data["week_start"] >= prediction_start
                )
            ].copy()

            relevant_anomalies = pd.concat(
                [
                    historical_anomalies,
                    predicted_anomalies,
                ],
                ignore_index=True,
            )

        else:
            relevant_anomalies = anomaly_data.iloc[0:0].copy()

        relevant_anomalies[
            "behavior_change_score"
        ] = pd.to_numeric(
            relevant_anomalies[
                "behavior_change_score"
            ],
            errors="coerce",
        )

        relevant_anomalies = relevant_anomalies.loc[
            relevant_anomalies[
                "behavior_change_score"
            ]
            >= anomaly_score_threshold
        ]

        weekly_totals = (
            weekly_feature_data.groupby(
                "week_start"
            )["pred_elapsed"]
            .sum()
        )

        for _, anomaly_row in relevant_anomalies.iterrows():
            anomaly_week = anomaly_row["week_start"]

            if anomaly_week not in weekly_totals.index:
                continue

            total_elapsed = weekly_totals.loc[
                anomaly_week
            ]

            figure.add_shape(
                type="rect",
                x0=anomaly_week - pd.Timedelta(days=3),
                x1=anomaly_week + pd.Timedelta(days=3),
                y0=0,
                y1=total_elapsed,
                line={
                    "color": "red",
                    "width": 3,
                },
                fillcolor="rgba(0,0,0,0)",
            )

            anomaly_score = anomaly_row[
                "behavior_change_score"
            ]

            figure.add_annotation(
                x=anomaly_week,
                y=total_elapsed,
                text=f"Anomaly {anomaly_score:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font={
                    "color": "red",
                    "size": 11,
                },
                bgcolor="white",
                bordercolor="red",
                borderwidth=1,
                ay=-35,
            )

    if predicted_data.empty:
        chart_title = (
            "Historical Decoded Feature Elapsed Time by Week"
        )

        behavior_label = (
            "Historical decoded behavior only"
        )

    elif prediction_period_source == "Historical":
        chart_title = (
            "Historical Feature Elapsed Time by Week"
        )

        behavior_label = (
            "Observed historical values during the "
            "prediction period"
        )

    else:
        chart_title = (
            "Predicted Feature Elapsed Time by Week"
        )

        behavior_label = (
            "Forecast values during the prediction period"
        )

    figure.update_layout(
        title={
            "text": (
                f"{chart_title}"
                f"<br><sup>Cluster: {cluster} · "
                f"{behavior_label}</sup>"
            ),
            "x": 0.01,
        },
        barmode="stack",
        height=650,
        xaxis_title="Week",
        yaxis_title="Decoded elapsed time (seconds)",
        legend={
            "title": {
                "text": "Feature",
            },
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.01,
        },
        margin={
            "l": 60,
            "r": 220,
            "t": 110,
            "b": 100,
        },
        hovermode="x unified",
    )

    week_ticks = (
        cluster_data[
            [
                "week_start",
                "year_week",
            ]
        ]
        .drop_duplicates(subset=["week_start"])
        .sort_values("week_start")
    )

    figure.update_xaxes(
        tickmode="array",
        tickvals=week_ticks["week_start"],
        ticktext=week_ticks["year_week"],
        tickangle=-45,
    )

    if not cluster_data.empty:
        timeline_start = cluster_data["week_start"].min()
        timeline_end = cluster_data["week_start"].max()

        if (
            pd.notna(timeline_start)
            and pd.notna(timeline_end)
        ):
            figure.update_xaxes(
                range=[
                    timeline_start - pd.Timedelta(days=4),
                    timeline_end + pd.Timedelta(days=4),
                ]
            )

    if y_axis_max is not None:
        figure.update_yaxes(
            range=[
                0,
                y_axis_max,
            ]
        )

    return figure


def build_decoded_feature_trends(
    decoded_data: pd.DataFrame,
    anomaly_data: pd.DataFrame,
    cluster,
    top_n_features: int = 5,
    anomaly_score_threshold: int = 3,
) -> go.Figure:
    """
    Compare historical and predicted decoded elapsed time by feature.

    Historical:
        Solid lines with filled markers.

    Predicted:
        Dashed lines with open markers.
    """

    required_columns = [
        "source",
        "final_cluster_label",
        "week_start",
        "year_week",
        "feature",
        "pred_elapsed",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in decoded_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Decoded trend chart is missing columns: "
            f"{missing_columns}"
        )

    cluster_data = decoded_data.loc[
        decoded_data["final_cluster_label"] == cluster
    ].copy()

    cluster_data["pred_elapsed"] = pd.to_numeric(
        cluster_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)

    cluster_data = cluster_data.dropna(
        subset=[
            "week_start",
            "feature",
        ]
    )

    if cluster_data.empty:
        figure = go.Figure()
        figure.update_layout(
            title="No decoded feature data available"
        )
        return figure

    top_features = (
        cluster_data.groupby(
            "feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .nlargest(
            top_n_features,
            "pred_elapsed",
        )["feature"]
        .tolist()
    )

    trend_data = cluster_data.loc[
        cluster_data["feature"].isin(
            top_features
        )
    ].copy()

    trend_data = (
        trend_data.groupby(
            [
                "source",
                "week_start",
                "year_week",
                "feature",
            ],
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            [
                "feature",
                "source",
                "week_start",
            ]
        )
    )

    color_palette = (
        px.colors.qualitative.Plotly
    )

    feature_color_map = {
        feature: color_palette[
            index % len(color_palette)
        ]
        for index, feature
        in enumerate(top_features)
    }

    figure = go.Figure()

    for feature in top_features:
        feature_color = feature_color_map[
            feature
        ]

        historical_feature = trend_data.loc[
            (
                trend_data["feature"]
                == feature
            )
            & (
                trend_data["source"]
                == "Historical"
            )
        ].copy()

        predicted_feature = trend_data.loc[
            (
                trend_data["feature"]
                == feature
            )
            & (
                trend_data["source"]
                == "Predicted"
            )
        ].copy()

        if not historical_feature.empty:
            figure.add_trace(
                go.Scatter(
                    x=historical_feature[
                        "week_start"
                    ],
                    y=historical_feature[
                        "pred_elapsed"
                    ],
                    mode="lines+markers",
                    name=str(feature),
                    legendgroup=str(feature),
                    line={
                        "color": feature_color,
                        "width": 3,
                    },
                    marker={
                        "color": feature_color,
                        "size": 7,
                        "symbol": "circle",
                    },
                    customdata=historical_feature[
                        [
                            "year_week",
                            "feature",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "Historical<br>"
                        "Week: %{customdata[0]}<br>"
                        "Elapsed time: %{y:,.2f} seconds"
                        "<extra></extra>"
                    ),
                )
            )

        if not predicted_feature.empty:
            figure.add_trace(
                go.Scatter(
                    x=predicted_feature[
                        "week_start"
                    ],
                    y=predicted_feature[
                        "pred_elapsed"
                    ],
                    mode="lines+markers",
                    name=f"{feature} · Predicted",
                    legendgroup=str(feature),
                    showlegend=False,
                    line={
                        "color": feature_color,
                        "width": 3,
                        "dash": "dash",
                    },
                    marker={
                        "color": feature_color,
                        "size": 8,
                        "symbol": "circle-open",
                        "line": {
                            "color": feature_color,
                            "width": 2,
                        },
                    },
                    customdata=predicted_feature[
                        [
                            "year_week",
                            "feature",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "Predicted<br>"
                        "Week: %{customdata[0]}<br>"
                        "Elapsed time: %{y:,.2f} seconds"
                        "<extra></extra>"
                    ),
                )
            )

    predicted_rows = trend_data.loc[
        trend_data["source"] == "Predicted"
    ]

    prediction_start = (
        predicted_rows["week_start"].min()
        if not predicted_rows.empty
        else pd.NaT
    )

    if pd.notna(prediction_start):
        chart_end = trend_data[
            "week_start"
        ].max()

        figure.add_vrect(
            x0=prediction_start,
            x1=chart_end
            + pd.Timedelta(days=4),
            fillcolor="gray",
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text="Prediction period",
            annotation_position="top left",
        )

        figure.add_vline(
            x=prediction_start,
            line_dash="dash",
            line_width=2,
            line_color="gray",
        )

    # Highlight anomaly weeks using vertical red rectangles.
    if (
        not anomaly_data.empty
        and "behavior_change_score"
        in anomaly_data.columns
    ):
        cluster_anomalies = anomaly_data.loc[
            anomaly_data[
                "final_cluster_label"
            ]
            == cluster
        ].copy()

        cluster_anomalies[
            "behavior_change_score"
        ] = pd.to_numeric(
            cluster_anomalies[
                "behavior_change_score"
            ],
            errors="coerce",
        )

        cluster_anomalies = cluster_anomalies.loc[
            cluster_anomalies[
                "behavior_change_score"
            ]
            >= anomaly_score_threshold
        ]

        for _, anomaly_row in (
            cluster_anomalies.iterrows()
        ):
            anomaly_week = anomaly_row[
                "week_start"
            ]

            figure.add_vrect(
                x0=anomaly_week
                - pd.Timedelta(days=3),
                x1=anomaly_week
                + pd.Timedelta(days=3),
                fillcolor="rgba(255,0,0,0.04)",
                line_color="red",
                line_width=2,
                layer="below",
            )

    figure.update_layout(
        title={
            "text": (
                "Weekly Decoded Feature "
                "Elapsed-Time Trends"
                f"<br><sup>Cluster: {cluster}</sup>"
            ),
            "x": 0.01,
        },
        height=600,
        xaxis_title="Week",
        yaxis_title="Decoded elapsed time (seconds)",
        legend={
            "title": {
                "text": "Feature",
            },
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.01,
        },
        margin={
            "l": 60,
            "r": 220,
            "t": 80,
            "b": 100,
        },
        hovermode="x unified",
    )

    week_ticks = (
        trend_data[
            [
                "week_start",
                "year_week",
            ]
        ]
        .drop_duplicates(subset=["week_start"])
        .sort_values("week_start")
    )

    figure.update_xaxes(
        tickmode="array",
        tickvals=week_ticks["week_start"],
        ticktext=week_ticks["year_week"],
        tickangle=-45,
    )

    return figure


def prepare_decoded_composition_data(
    decoded_data: pd.DataFrame,
    cluster,
    top_n_features: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare historical and predicted feature-composition data.

    The function:
        1. filters to one cluster;
        2. calculates the combined top features across both sources;
        3. groups all remaining features into Other;
        4. calculates each feature's weekly elapsed-time share.
    """

    required_columns = [
        "source",
        "final_cluster_label",
        "week_start",
        "year_week",
        "feature",
        "pred_elapsed",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in decoded_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Decoded composition data is missing columns: "
            f"{missing_columns}"
        )

    cluster_data = decoded_data.loc[
        decoded_data["final_cluster_label"] == cluster
    ].copy()

    cluster_data["pred_elapsed"] = pd.to_numeric(
        cluster_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)

    cluster_data = cluster_data.dropna(
        subset=[
            "week_start",
            "year_week",
            "feature",
        ]
    )

    if cluster_data.empty:
        return pd.DataFrame(), []

    feature_totals = (
        cluster_data.groupby(
            "feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            "pred_elapsed",
            ascending=False,
        )
    )

    top_features = (
        feature_totals
        .head(top_n_features)["feature"]
        .tolist()
    )

    cluster_data["composition_feature"] = np.where(
        cluster_data["feature"].isin(top_features),
        cluster_data["feature"],
        "Other",
    )

    composition_data = (
        cluster_data.groupby(
            [
                "source",
                "week_start",
                "year_week",
                "composition_feature",
            ],
            as_index=False,
        )["pred_elapsed"]
        .sum()
    )

    weekly_totals = (
        composition_data.groupby(
            [
                "source",
                "week_start",
            ]
        )["pred_elapsed"]
        .transform("sum")
    )

    composition_data["elapsed_share"] = np.where(
        weekly_totals > 0,
        composition_data["pred_elapsed"]
        / weekly_totals
        * 100.0,
        0.0,
    )

    feature_order = top_features.copy()

    if (
        composition_data["composition_feature"]
        .eq("Other")
        .any()
    ):
        feature_order.append("Other")

    composition_data[
        "composition_feature"
    ] = pd.Categorical(
        composition_data["composition_feature"],
        categories=feature_order,
        ordered=True,
    )

    composition_data = composition_data.sort_values(
        [
            "source",
            "week_start",
            "composition_feature",
        ]
    )

    return composition_data, feature_order


def build_decoded_composition_donut(
    composition_data: pd.DataFrame,
    cluster,
    source: str,
    selected_week_start,
    selected_year_week: str,
    feature_order: list[str],
    feature_color_map: dict,
) -> go.Figure:
    """
    Build one donut chart for a selected cluster, source, and week.
    """

    week_data = composition_data.loc[
        (
            composition_data["source"] == source
        )
        & (
            composition_data["week_start"]
            == selected_week_start
        )
    ].copy()

    figure = go.Figure()

    if week_data.empty:
        figure.update_layout(
            title={
                "text": (
                    f"{source} Feature Composition"
                    f"<br><sup>Cluster: {cluster} · "
                    f"{selected_year_week}</sup>"
                ),
                "x": 0.01,
            },
            height=450,
            annotations=[
                {
                    "text": (
                        f"{source} decoded behavior is not "
                        f"available for {selected_year_week}."
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "align": "center",
                }
            ],
        )
        return figure

    week_data[
        "composition_feature"
    ] = week_data[
        "composition_feature"
    ].astype(str)

    week_data = (
        week_data.set_index(
            "composition_feature"
        )
        .reindex(feature_order)
        .fillna(
            {
                "pred_elapsed": 0.0,
                "elapsed_share": 0.0,
            }
        )
        .reset_index()
    )

    total_elapsed = week_data[
        "pred_elapsed"
    ].sum()

    donut_colors = [
        feature_color_map.get(feature)
        for feature in week_data[
            "composition_feature"
        ]
    ]

    figure.add_trace(
        go.Pie(
            labels=week_data["composition_feature"],
            values=week_data["pred_elapsed"],
            hole=0.50,
            sort=False,
            marker={
                "colors": donut_colors,
            },
            customdata=week_data[
                [
                    "elapsed_share",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"Source: {source}<br>"
                f"Week: {selected_year_week}<br>"
                "Elapsed time: %{value:,.2f} seconds<br>"
                "Share: %{customdata[0]:.2f}%"
                "<extra></extra>"
            ),
            textinfo="none",
        )
    )

    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=(
            "<b>Total elapsed</b>"
            f"<br>{total_elapsed:,.0f} sec"
        ),
        showarrow=False,
        align="center",
        font={
            "size": 13,
        },
    )

    figure.update_layout(
        title={
            "text": (
                f"{source} Feature Composition"
                f"<br><sup>Cluster: {cluster} · "
                f"{selected_year_week}</sup>"
            ),
            "x": 0.01,
        },
        height=450,
        legend={
            "title": {
                "text": "Feature",
            },
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={
            "l": 20,
            "r": 180,
            "t": 80,
            "b": 30,
        },
    )

    return figure