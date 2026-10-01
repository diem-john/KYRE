import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def shorten_text(
    value: str,
    max_length: int = 22,
) -> str:
    value = str(value)

    if len(value) <= max_length:
        return value

    return value[: max_length - 3] + "..."

def build_behavior_timeline(
    cluster_data: pd.DataFrame,
    selected_target_week=None,
    feature_color_map: dict[str, str] | None = None,
) -> go.Figure:
    plot_data = cluster_data.copy()

    plot_data["pred_elapsed"] = pd.to_numeric(
        plot_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)

    plot_data = plot_data.dropna(
        subset=[
            "source",
            "week_start",
            "year_week",
            "feature",
        ]
    )

    historical_data = plot_data.loc[
        plot_data["source"] == "Historical"
    ].copy()

    predicted_data = plot_data.loc[
        plot_data["source"] == "Predicted"
    ].copy()

    prediction_start = (
        predicted_data["week_start"].min()
        if not predicted_data.empty
        else pd.NaT
    )

    if pd.notna(prediction_start):
        historical_context = historical_data.loc[
            historical_data["week_start"]
            < prediction_start
        ].copy()

        predicted_timeline = predicted_data.loc[
            predicted_data["week_start"]
            >= prediction_start
        ].copy()

        timeline_data = pd.concat(
            [
                historical_context,
                predicted_timeline,
            ],
            ignore_index=True,
        )

    else:
        timeline_data = historical_data.copy()

    if timeline_data.empty:
        figure = go.Figure()

        figure.update_layout(
            title="No decoded behavior timeline available",
            height=600,
        )

        return figure

    timeline_data["display_feature"] = (
        timeline_data["feature"].astype(str)
    )

    weekly_feature_data = (
        timeline_data.groupby(
            [
                "week_start",
                "year_week",
                "source",
                "display_feature",
            ],
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            [
                "week_start",
                "display_feature",
            ]
        )
    )

    feature_order = (
        weekly_feature_data.groupby(
            "display_feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            "pred_elapsed",
            ascending=False,
        )["display_feature"]
        .tolist()
    )

    if feature_color_map is None:
        feature_color_map = (
            build_feature_color_map(
                feature_order
            )
        )

    figure = go.Figure()

    for feature in feature_order:
        feature_data = weekly_feature_data.loc[
            weekly_feature_data["display_feature"]
            == feature
        ].copy()

        if feature_data.empty:
            continue

        figure.add_trace(
            go.Bar(
                x=feature_data["week_start"],
                y=feature_data["pred_elapsed"],
                name=str(feature),
                marker={
                    "color": feature_color_map.get(
                        str(feature),
                        "#B0B0B0",
                    ),
                },
                customdata=feature_data[
                    [
                        "year_week",
                        "source",
                        "display_feature",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "Week: %{customdata[0]}<br>"
                    "Source: %{customdata[1]}<br>"
                    "Elapsed time: %{y:,.2f} seconds"
                    "<extra></extra>"
                ),
            )
        )

    if pd.notna(prediction_start):
        chart_end = weekly_feature_data[
            "week_start"
        ].max()

        figure.add_vrect(
            x0=prediction_start,
            x1=chart_end + pd.Timedelta(days=4),
            fillcolor="gray",
            opacity=0.10,
            layer="below",
            line_width=0,
        )

        figure.add_vline(
            x=prediction_start,
            line_dash="dash",
            line_width=2,
            line_color="gray",
        )

        figure.add_annotation(
            x=prediction_start - pd.Timedelta(days=4),
            y=1.04,
            xref="x",
            yref="paper",
            text="<b>Historical</b>",
            showarrow=False,
            xanchor="right",
        )

        figure.add_annotation(
            x=prediction_start + pd.Timedelta(days=4),
            y=1.04,
            xref="x",
            yref="paper",
            text="<b>Forecast</b>",
            showarrow=False,
            xanchor="left",
        )

    if selected_target_week in set(
        weekly_feature_data["week_start"]
    ):
        figure.add_vrect(
            x0=selected_target_week
            - pd.Timedelta(days=3),
            x1=selected_target_week
            + pd.Timedelta(days=3),
            fillcolor="rgba(255, 165, 0, 0.08)",
            line={
                "color": "orange",
                "width": 3,
            },
            layer="above",
        )

    week_ticks = (
        weekly_feature_data[
            [
                "week_start",
                "year_week",
            ]
        ]
        .drop_duplicates(
            subset=["week_start"]
        )
        .sort_values("week_start")
    )

    figure.update_layout(
        title={
            "text": (
                "Historical and Forecast Feature Elapsed Time"
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

    figure.update_xaxes(
        tickmode="array",
        tickvals=week_ticks["week_start"],
        ticktext=week_ticks["year_week"],
        tickangle=-45,
    )

    return figure


def build_feature_comparison_chart(
    feature_comparison: pd.DataFrame,
    latest_historical_label: str,
    selected_target_label: str,
    top_n_features: int = 10,
    feature_color_map: dict[str, str] | None = None,
) -> go.Figure:
    comparison_data = (
        feature_comparison.copy()
    )

    comparison_data["feature"] = (
        comparison_data["feature"]
        .astype(str)
    )

    comparison_data[
        "maximum_elapsed"
    ] = comparison_data[
        [
            "historical_elapsed",
            "forecast_elapsed",
        ]
    ].max(axis=1)

    comparison_data = (
        comparison_data.sort_values(
            "maximum_elapsed",
            ascending=False,
        )
        .head(top_n_features)
        .copy()
    )

    feature_order = (
        comparison_data[
            "feature"
        ].tolist()
    )

    if feature_color_map is None:
        feature_color_map = (
            build_feature_color_map(
                feature_order
            )
        )

    feature_colors = [
        feature_color_map.get(
            feature,
            "#B0B0B0",
        )
        for feature in feature_order
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            name=(
                "Historical · "
                f"{latest_historical_label}"
            ),
            x=feature_order,
            y=comparison_data[
                "historical_elapsed"
            ],
            marker={
                "color": feature_colors,
                "opacity": 0.45,
                "line": {
                    "color": "#666666",
                    "width": 1.5,
                },
                "pattern": {
                    "shape": "/",
                    "solidity": 0.2,
                },
            },
            customdata=comparison_data[
                [
                    "feature",
                    "historical_elapsed",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Period: Historical<br>"
                "Elapsed time: "
                "%{customdata[1]:,.0f} sec"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            name=(
                "Forecast · "
                f"{selected_target_label}"
            ),
            x=feature_order,
            y=comparison_data[
                "forecast_elapsed"
            ],
            marker={
                "color": feature_colors,
                "opacity": 1.0,
                "line": {
                    "color": "#333333",
                    "width": 1.5,
                },
                "pattern": {
                    "shape": "",
                },
            },
            customdata=comparison_data[
                [
                    "feature",
                    "forecast_elapsed",
                ]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Period: Forecast<br>"
                "Elapsed time: "
                "%{customdata[1]:,.0f} sec"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        barmode="group",
        xaxis={
            "title": "Feature",
            "categoryorder": "array",
            "categoryarray": feature_order,
            "tickangle": -35,
        },
        yaxis={
            "title": "Decoded elapsed time (seconds)",
            "rangemode": "tozero",
        },
        legend={
            "title": {
                "text": "Period",
            },
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 50,
            "r": 20,
            "t": 70,
            "b": 130,
        },
        height=550,
        hovermode="x unified",
    )

    return figure

def build_feature_color_map(
    features,
) -> dict:
    feature_list = sorted(
        {
            str(feature)
            for feature in features
        }
    )

    color_palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Set3
    )

    return {
        feature: color_palette[
            index % len(color_palette)
        ]
        for index, feature in enumerate(
            feature_list
        )
    }

def build_feature_composition_donut(
    feature_comparison: pd.DataFrame,
    value_column: str,
    chart_title: str,
    feature_color_map: dict,
) -> go.Figure:
    composition_data = feature_comparison[
        [
            "feature",
            value_column,
            "status",
        ]
    ].copy()

    composition_data[value_column] = pd.to_numeric(
        composition_data[value_column],
        errors="coerce",
    ).fillna(0.0)

    composition_data = composition_data.loc[
        composition_data[value_column] > 0
    ].copy()

    composition_data = composition_data.sort_values(
        value_column,
        ascending=False,
    )

    total_elapsed = composition_data[
        value_column
    ].sum()

    figure = go.Figure()

    if composition_data.empty or total_elapsed <= 0:
        figure.update_layout(
            title={
                "text": chart_title,
                "x": 0.5,
            },
            height=520,
            annotations=[
                {
                    "text": "No feature usage available",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {
                        "size": 16,
                    },
                }
            ],
        )

        return figure

    composition_data["composition_share"] = (
        composition_data[value_column]
        / total_elapsed
        * 100.0
    )

    slice_colors = [
        feature_color_map.get(
            str(feature),
            "#B0B0B0",
        )
        for feature in composition_data["feature"]
    ]

    composition_data["status"] = (
        composition_data["status"]
        .fillna("Unknown")
        .astype(str)
    )

    figure.add_trace(
        go.Pie(
            labels=composition_data["feature"],
            values=composition_data[value_column],
            hole=0.55,
            sort=False,
            marker={
                "colors": slice_colors,
            },
            customdata=composition_data[
                ["status"]
            ].to_numpy(),
            textinfo="percent",
            textposition="inside",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Elapsed time: %{value:,.2f} seconds<br>"
                "Composition share: %{percent:.2%}<br>"
                "Status: %{customdata[0]}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": chart_title,
            "x": 0.5,
        },
        height=520,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 150,
        },
        annotations=[
            {
                "text": (
                    f"Total<br>"
                    f"{total_elapsed:,.0f} sec"
                ),
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {
                    "size": 15,
                },
            }
        ],
    )

    return figure

def build_feature_trend_chart(
    cluster_data: pd.DataFrame,
    top_n_features: int = 8,
    feature_color_map: dict[str, str] | None = None,
) -> go.Figure:
    trend_source_data = cluster_data.copy()

    trend_source_data["pred_elapsed"] = pd.to_numeric(
        trend_source_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)

    trend_source_data = trend_source_data.dropna(
        subset=[
            "week_start",
            "year_week",
            "feature",
            "source",
        ]
    )

    if trend_source_data.empty:
        figure = go.Figure()

        figure.update_layout(
            title="No feature trend data available",
            height=500,
        )

        return figure


    # =====================================================
    # Select the major features across the full 52 weeks
    # =====================================================

    feature_totals = (
        trend_source_data.groupby(
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
        feature_totals["feature"]
        .head(top_n_features)
        .tolist()
    )


    # =====================================================
    # Prepare weekly feature trend data
    # =====================================================

    trend_data = trend_source_data.loc[
        trend_source_data["feature"].isin(
            top_features
        )
    ].copy()

    trend_data = (
        trend_data.groupby(
            [
                "week_start",
                "year_week",
                "feature",
                "source",
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


    # =====================================================
    # Build the line chart
    # =====================================================

    figure = px.line(
        trend_data,
        x="week_start",
        y="pred_elapsed",
        color="feature",
        markers=True,
        color_discrete_map=feature_color_map,
        category_orders={
            "feature": top_features,
        },
        custom_data=[
            "year_week",
            "feature",
            "source",
        ],
    )

    # =====================================================
    # Shade the predicted portion of the timeline
    # =====================================================

    predicted_weeks = (
        trend_data.loc[
            trend_data["source"] == "Predicted",
            "week_start",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if not predicted_weeks.empty:
        first_predicted_week = (
            predicted_weeks.iloc[0]
        )

        final_chart_week = (
            trend_data["week_start"].max()
        )

        # Extend by one week so the final predicted week is
        # completely covered by the shaded region.
        predicted_region_end = (
            final_chart_week
            + pd.Timedelta(days=7)
        )

        figure.add_vrect(
            x0=first_predicted_week,
            x1=predicted_region_end,
            fillcolor="gray",
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        figure.add_vline(
            x=first_predicted_week,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )

        # Label the historical and forecast regions
        figure.add_annotation(
            x=first_predicted_week,
            y=1.04,
            xref="x",
            yref="paper",
            text="<b>Historical</b>",
            showarrow=False,
            xanchor="right",
        )

        figure.add_annotation(
            x=first_predicted_week,
            y=1.04,
            xref="x",
            yref="paper",
            text="<b>Forecast</b>",
            showarrow=False,
            xanchor="left",
        )

    figure.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Week: %{customdata[0]}<br>"
            "Source: %{customdata[2]}<br>"
            "Elapsed time: %{y:,.2f} seconds"
            "<extra></extra>"
        ),
        line={
            "width": 2,
        },
        marker={
            "size": 7,
        },
    )


    # =====================================================
    # Prepare readable week labels
    # =====================================================

    week_ticks = (
        trend_data[
            [
                "week_start",
                "year_week",
            ]
        ]
        .drop_duplicates(
            subset=["week_start"]
        )
        .sort_values("week_start")
    )


    # =====================================================
    # Final chart layout
    # =====================================================

    figure.update_layout(
        title={
            "text": "Historical and Predicted Feature Trends",
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
            "b": 90,
        },
        hovermode="x unified",
    )

    figure.update_xaxes(
        tickmode="array",
        tickvals=week_ticks[
            "week_start"
        ].tolist(),
        ticktext=week_ticks[
            "year_week"
        ].tolist(),
        tickangle=-45,
    )

    return figure

def classify_feature_change(
    historical_elapsed: float,
    forecast_elapsed: float,
) -> str:
    if (
        historical_elapsed <= 0
        and forecast_elapsed > 0
    ):
        return "Appearing"

    if (
        historical_elapsed > 0
        and forecast_elapsed <= 0
    ):
        return "Disappearing"

    if forecast_elapsed > historical_elapsed:
        return "Increasing"

    if forecast_elapsed < historical_elapsed:
        return "Decreasing"

    return "Stable"

def format_feature_change_list(
    change_data: pd.DataFrame,
) -> str:
    if change_data.empty:
        return "No features in this category."

    lines = []

    for position, row in enumerate(
        change_data.itertuples(),
        start=1,
    ):
        lines.append(
            f"{position}. `{row.feature}`: "
            f"**{row.elapsed_change:+,.0f} seconds**"
        )

    return "\n\n".join(lines)