from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# Make the repository root importable
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.trajectory_dashboard.config import (
    PROJECT_NAME,
    MODEL_NAME,
)

from src.trajectory_dashboard.data_loader import (
    get_dashboard_paths,
    validate_dashboard_paths,
    load_dashboard_data,
    validate_dashboard_columns,
    prepare_anomaly_data,
    prepare_decoded_data,
    add_dashboard_pca_coordinates,
)

from src.trajectory_dashboard.utils.trajectory_visualizations import (
    build_trajectory_3d,
    build_decoded_stacked_bar,
    prepare_decoded_composition_data,
    build_decoded_composition_donut,
)

# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Trajectory Overview",
    layout="wide",
)

st.title("Trajectory Prediction Overview")

st.write(
    "Review the historical centroid trajectories and the "
    "trajectory predictions produced by the project."
)


# =========================================================
# Load project results
# =========================================================

@st.cache_data(
    show_spinner="Loading project trajectory results..."
)
def load_trajectory_results(
    project_name: str,
    model_name: str,
):
    dashboard_paths = get_dashboard_paths(
        project_name=project_name,
        model_name=model_name,
    )

    all_files_exist, missing_files = validate_dashboard_paths(
        dashboard_paths
    )

    if not all_files_exist:
        return {
            "trajectory_data": None,
            "decoded_data": None,
            "paths": dashboard_paths,
            "missing_files": missing_files,
            "missing_columns": {},
        }

    dashboard_data = load_dashboard_data(
        dashboard_paths
    )

    missing_columns = validate_dashboard_columns(
        dashboard_data
    )

    if missing_columns:
        return {
            "trajectory_data": None,
            "decoded_data": None,
            "paths": dashboard_paths,
            "missing_files": [],
            "missing_columns": missing_columns,
        }

    trajectory_data = prepare_anomaly_data(
        dashboard_data
    )

    trajectory_data = add_dashboard_pca_coordinates(
        trajectory_data
    )

    decoded_data = prepare_decoded_data(
        dashboard_data
    )

    return {
        "trajectory_data": trajectory_data,
        "decoded_data": decoded_data,
        "paths": dashboard_paths,
        "missing_files": [],
        "missing_columns": {},
    }


loaded_results = load_trajectory_results(
    project_name=PROJECT_NAME,
    model_name=MODEL_NAME,
)

trajectory_data = loaded_results["trajectory_data"]
decoded_data = loaded_results["decoded_data"]
dashboard_paths = loaded_results["paths"]
missing_files = loaded_results["missing_files"]
missing_columns = loaded_results["missing_columns"]


# =========================================================
# Validate loaded results
# =========================================================

if missing_files:
    st.error(
        "Some required project result files could not be found."
    )

    for missing_file in missing_files:
        st.write(f"- `{missing_file}`")

    st.stop()


if missing_columns:
    st.error(
        "Some required columns are missing from the project "
        "result files."
    )

    for data_name, columns in missing_columns.items():
        st.write(f"**{data_name}**")
        st.write(", ".join(columns))

    st.stop()


if trajectory_data is None or trajectory_data.empty:
    st.error(
        "No trajectory results were available."
    )
    st.stop()

decoded_data_available = (
    decoded_data is not None
    and not decoded_data.empty
)


# =========================================================
# Overview filters
# =========================================================

st.sidebar.header("Overview Filters")

# Source filter
source_options = sorted(
    trajectory_data["source"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_sources = st.sidebar.multiselect(
    "Source",
    options=source_options,
    default=source_options,
)

# Cluster filter
cluster_options = sorted(
    trajectory_data["final_cluster_label"]
    .dropna()
    .unique()
    .tolist(),
    key=str,
)

selected_clusters = st.sidebar.multiselect(
    "Cluster",
    options=cluster_options,
    default=cluster_options,
)

# Available dataset weeks
available_weeks = (
    trajectory_data[
        [
            "week_start",
            "year_week",
        ]
    ]
    .dropna(subset=["week_start"])
    .drop_duplicates(subset=["week_start"])
    .sort_values("week_start")
    .reset_index(drop=True)
)

week_options = available_weeks["week_start"].tolist()

if not week_options:
    st.error(
        "No valid weekly dates were found in the trajectory results."
    )
    st.stop()

week_label_map = dict(
    zip(
        available_weeks["week_start"],
        available_weeks["year_week"],
    )
)

# Default to the latest 52 available weeks.
default_week_count = 54

default_start_index = max(
    0,
    len(week_options) - default_week_count,
)

default_end_index = len(week_options) - 1

selected_start_week, selected_end_week = (
    st.sidebar.select_slider(
        "Week range",
        options=week_options,
        value=(
            week_options[default_start_index],
            week_options[default_end_index],
        ),
        format_func=lambda value: week_label_map[value],
    )
)

show_predicted_anomalies = st.sidebar.toggle(
    "Show predicted anomaly markers",
    value=True,
)

show_historical_anomalies = st.sidebar.toggle(
    "Show historical anomaly markers",
    value=False,
)


# =========================================================
# Apply overview filters
# =========================================================

filtered_trajectory = trajectory_data.loc[
    trajectory_data["source"].isin(selected_sources)
    & trajectory_data["final_cluster_label"].isin(
        selected_clusters
    )
    & trajectory_data["week_start"].between(
        selected_start_week,
        selected_end_week,
        inclusive="both",
    )
].copy()

if filtered_trajectory.empty:
    st.warning(
        "No trajectory rows match the selected filters."
    )
    st.stop()

filtered_historical = filtered_trajectory.loc[
    filtered_trajectory["source"] == "Historical"
].copy()

filtered_predicted = filtered_trajectory.loc[
    filtered_trajectory["source"] == "Predicted"
].copy()


# =========================================================
# Filter decoded behavior using the main week range
# =========================================================

if decoded_data_available:
    filtered_decoded_data = decoded_data.loc[
        decoded_data["source"].isin(
            selected_sources
        )
        & decoded_data["week_start"].between(
            selected_start_week,
            selected_end_week,
            inclusive="both",
        )
        & decoded_data["final_cluster_label"].isin(
            selected_clusters
        )
    ].copy()

else:
    filtered_decoded_data = pd.DataFrame()


# =========================================================
# Basic summary values
# =========================================================

historical_week_count = (
    filtered_historical["week_start"]
    .dropna()
    .nunique()
)

predicted_week_count = (
    filtered_predicted["week_start"]
    .dropna()
    .nunique()
)

cluster_count = (
    filtered_trajectory["final_cluster_label"]
    .dropna()
    .nunique()
)


# =========================================================
# Dashboard overview header
# =========================================================

st.caption(
    f"Project: {PROJECT_NAME}  •  Model: {MODEL_NAME}"
)


# =========================================================
# Summary cards
# =========================================================

summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)

summary_col1.metric(
    "Historical weeks",
    f"{historical_week_count:,}",
)

summary_col2.metric(
    "Predicted weeks",
    f"{predicted_week_count:,}",
)

summary_col3.metric(
    "Clusters",
    f"{cluster_count:,}",
)


# =========================================================
# Compact trajectory period summary
# =========================================================

historical_start = (
    filtered_historical["week_start"].min()
)

historical_end = (
    filtered_historical["week_start"].max()
)

predicted_start = (
    filtered_predicted["week_start"].min()
)

predicted_end = (
    filtered_predicted["week_start"].max()
)


historical_period_text = (
    f"{historical_start.strftime('%G-W%V')} "
    f"to {historical_end.strftime('%G-W%V')}"
    if (
        pd.notna(historical_start)
        and pd.notna(historical_end)
    )
    else "No historical period"
)

predicted_period_text = (
    f"{predicted_start.strftime('%G-W%V')} "
    f"to {predicted_end.strftime('%G-W%V')}"
    if (
        pd.notna(predicted_start)
        and pd.notna(predicted_end)
    )
    else "No predicted period"
)


period_col1, period_col2 = st.columns(2)

with period_col1:
    st.markdown(
        f"""
        **Historical period**  
        {historical_period_text}  
        """,
        unsafe_allow_html=True,
    )

with period_col2:
    st.markdown(
        f"""
        **Prediction period**  
        {predicted_period_text}  
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# Main trajectory visualization
# =========================================================

st.subheader("Centroid Trajectory")

st.caption(
    "Filled circles and solid lines represent historical centroids. "
    "Historical points that overlap the prediction period use lower "
    "opacity. Open diamonds and dashed lines represent predicted "
    "centroids. Red rings highlight detected anomalies."
)


# =========================================================
# Prepare playback weeks
# =========================================================

visualization_weeks = (
    filtered_trajectory[
        [
            "week_start",
            "year_week",
        ]
    ]
    .dropna(subset=["week_start"])
    .drop_duplicates(subset=["week_start"])
    .sort_values("week_start")
    .reset_index(drop=True)
)

visualization_week_options = (
    visualization_weeks["week_start"].tolist()
)

if not visualization_week_options:
    st.warning(
        "No weeks are available for the trajectory visualization."
    )
    st.stop()

visualization_week_label_map = dict(
    zip(
        visualization_weeks["week_start"],
        visualization_weeks["year_week"],
    )
)


# =========================================================
# Initialize playback week
# =========================================================

playback_state_key = "trajectory_playback_week"

if (
    playback_state_key not in st.session_state
    or st.session_state[playback_state_key]
    not in visualization_week_options
):
    st.session_state[playback_state_key] = (
        visualization_week_options[-1]
    )


selected_playback_week = (
    st.session_state[playback_state_key]
)


# =========================================================
# Limit the chart to the selected playback week
# =========================================================

visualization_trajectory = filtered_trajectory.loc[
    filtered_trajectory["week_start"]
    <= selected_playback_week
].copy()


# =========================================================
# Build and display the 3D trajectory
# =========================================================

trajectory_figure = build_trajectory_3d(
    trajectory_data=visualization_trajectory,
    show_predicted_anomalies=(
        show_predicted_anomalies
    ),
    show_historical_anomalies=(
        show_historical_anomalies
    ),
    anomaly_score_threshold=3,
)

st.plotly_chart(
    trajectory_figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToAdd": [
            "resetCameraLastSave3d",
        ],
    },
)


# =========================================================
# Playback slider below the visualization
# =========================================================

st.select_slider(
    "Trajectory playback week",
    options=visualization_week_options,
    format_func=lambda value: (
        visualization_week_label_map[value]
    ),
    key=playback_state_key,
)

st.caption(
    "Move the slider to progressively display the trajectory "
    "up to a selected week. This does not change the sidebar "
    "week range."
)


# =========================================================
# Decoded behavior overview
# =========================================================

st.subheader("Decoded Behavior")

if filtered_decoded_data.empty:
    st.info(
        "No decoded behavior is available for the selected "
        "clusters and week range."
    )

else:
    decoded_cluster_options = sorted(
        filtered_decoded_data[
            "final_cluster_label"
        ]
        .dropna()
        .unique()
        .tolist(),
        key=str,
    )

    selected_behavior_cluster = st.selectbox(
        "Cluster for decoded behavior",
        options=decoded_cluster_options,
        format_func=str,
        key="decoded_behavior_cluster",
    )


    # =====================================================
    # Data for the selected behavior cluster
    # =====================================================

    cluster_decoded_data = (
        filtered_decoded_data.loc[
            filtered_decoded_data[
                "final_cluster_label"
            ]
            == selected_behavior_cluster
        ].copy()
    )

    # Use the complete main week range for anomaly overlays,
    # independent of the trajectory playback position.
    cluster_anomaly_data = trajectory_data.loc[
        trajectory_data["source"].isin(
            selected_sources
        )
        & (
            trajectory_data["final_cluster_label"]
            == selected_behavior_cluster
        )
        & trajectory_data["week_start"].between(
            selected_start_week,
            selected_end_week,
            inclusive="both",
        )
    ].copy()


    # =====================================================
    # Apply anomaly marker toggle settings
    # =====================================================

    enabled_anomaly_sources = []

    if show_historical_anomalies:
        enabled_anomaly_sources.append(
            "Historical"
        )

    if show_predicted_anomalies:
        enabled_anomaly_sources.append(
            "Predicted"
        )


    if enabled_anomaly_sources:
        cluster_anomaly_data = (
            cluster_anomaly_data.loc[
                cluster_anomaly_data["source"].isin(
                    enabled_anomaly_sources
                )
            ]
            .copy()
        )

    else:
        cluster_anomaly_data = (
            cluster_anomaly_data.iloc[0:0].copy()
        )


    # =====================================================
    # Prepare elapsed-time values
    # =====================================================

    cluster_decoded_data["pred_elapsed"] = pd.to_numeric(
        cluster_decoded_data["pred_elapsed"],
        errors="coerce",
    ).fillna(0.0)


    # =====================================================
    # Shared top features
    # =====================================================

    feature_totals = (
        cluster_decoded_data.groupby(
            "feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
        .sort_values(
            "pred_elapsed",
            ascending=False,
        )
    )

    feature_order = (
        feature_totals["feature"]
        .head(8)
        .tolist()
    )


    if not feature_order:
        st.info(
            "No decoded features are available for the "
            "selected cluster."
        )

    else:
        # ================================================
        # Shared feature colors
        # ================================================

        feature_palette = (
            px.colors.qualitative.Plotly
        )

        feature_color_map = {
            feature: feature_palette[
                index % len(feature_palette)
            ]
            for index, feature
            in enumerate(feature_order)
        }


        # ================================================
        # Shared y-axis maximum
        # ================================================

        shared_bar_data = (
            cluster_decoded_data.loc[
                cluster_decoded_data[
                    "feature"
                ].isin(feature_order)
            ]
            .groupby(
                [
                    "source",
                    "week_start",
                ],
                as_index=False,
            )["pred_elapsed"]
            .sum()
        )

        maximum_weekly_total = (
            shared_bar_data["pred_elapsed"].max()
            if not shared_bar_data.empty
            else 0.0
        )

        shared_y_axis_max = (
            maximum_weekly_total * 1.20
            if maximum_weekly_total > 0
            else None
        )


        # ================================================
        # Check source availability and same-week overlap
        # ================================================

        historical_decoded_weeks = set(
            cluster_decoded_data.loc[
                cluster_decoded_data["source"]
                == "Historical",
                "week_start",
            ]
            .dropna()
            .tolist()
        )

        predicted_decoded_weeks = set(
            cluster_decoded_data.loc[
                cluster_decoded_data["source"]
                == "Predicted",
                "week_start",
            ]
            .dropna()
            .tolist()
        )

        overlapping_decoded_weeks = (
            historical_decoded_weeks
            & predicted_decoded_weeks
        )

        has_historical_decoded = bool(
            historical_decoded_weeks
        )

        has_predicted_decoded = bool(
            predicted_decoded_weeks
        )

        has_decoded_overlap = bool(
            overlapping_decoded_weeks
        )


        # ================================================
        # Decoded behavior visualization selector
        # ================================================

        behavior_header_col1, behavior_header_col2 = (
            st.columns(
                [4, 2],
                vertical_alignment="bottom",
            )
        )

        with behavior_header_col1:
            st.markdown(
                "#### Decoded Feature Composition by Week"
            )

        with behavior_header_col2:
            selected_behavior_view = st.radio(
                "Decoded behavior view",
                options=[
                    "Bar",
                    "Donut",
                ],
                horizontal=True,
                label_visibility="collapsed",
                key="overview_decoded_behavior_view",
            )

        if selected_behavior_view == "Bar":
            if (
                has_historical_decoded
                and has_predicted_decoded
            ):
                st.caption(
                    "Compare historical and predicted decoded "
                    "elapsed-time composition by week. Both charts "
                    "use the same features, colors, and y-axis scale."
                )

            elif has_historical_decoded:
                st.caption(
                    "View historical decoded feature elapsed-time "
                    "composition across the selected week range."
                )

            elif has_predicted_decoded:
                st.caption(
                    "View predicted decoded feature elapsed-time "
                    "composition across the selected week range."
                )

        else:
            if (
                has_historical_decoded
                and has_predicted_decoded
            ):
                st.caption(
                    "Compare historical and predicted feature "
                    "composition across three consecutive weeks."
                )

            elif has_historical_decoded:
                st.caption(
                    "View historical feature composition across "
                    "three consecutive weeks."
                )

            elif has_predicted_decoded:
                st.caption(
                    "View predicted feature composition across "
                    "three consecutive weeks."
                )


        # ================================================
        # Bar view
        # ================================================

        if selected_behavior_view == "Bar":
            if (
                has_historical_decoded
                and has_predicted_decoded
            ):
                historical_bar_figure = (
                    build_decoded_stacked_bar(
                        decoded_data=cluster_decoded_data,
                        anomaly_data=cluster_anomaly_data,
                        cluster=selected_behavior_cluster,
                        prediction_period_source="Historical",
                        feature_order=feature_order,
                        feature_color_map=feature_color_map,
                        y_axis_max=shared_y_axis_max,
                        anomaly_score_threshold=3,
                    )
                )

                predicted_bar_figure = (
                    build_decoded_stacked_bar(
                        decoded_data=cluster_decoded_data,
                        anomaly_data=cluster_anomaly_data,
                        cluster=selected_behavior_cluster,
                        prediction_period_source="Predicted",
                        feature_order=feature_order,
                        feature_color_map=feature_color_map,
                        y_axis_max=shared_y_axis_max,
                        anomaly_score_threshold=3,
                    )
                )

                st.plotly_chart(
                    historical_bar_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "displayModeBar": True,
                        "responsive": True,
                    },
                    key="historical_decoded_stacked_bar",
                )

                st.plotly_chart(
                    predicted_bar_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "displayModeBar": True,
                        "responsive": True,
                    },
                    key="predicted_decoded_stacked_bar",
                )

            elif has_historical_decoded:
                historical_bar_figure = (
                    build_decoded_stacked_bar(
                        decoded_data=cluster_decoded_data,
                        anomaly_data=cluster_anomaly_data,
                        cluster=selected_behavior_cluster,
                        prediction_period_source="Historical",
                        feature_order=feature_order,
                        feature_color_map=feature_color_map,
                        y_axis_max=shared_y_axis_max,
                        anomaly_score_threshold=3,
                    )
                )

                st.plotly_chart(
                    historical_bar_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "displayModeBar": True,
                        "responsive": True,
                    },
                    key="historical_decoded_stacked_bar",
                )

            elif has_predicted_decoded:
                predicted_bar_figure = (
                    build_decoded_stacked_bar(
                        decoded_data=cluster_decoded_data,
                        anomaly_data=cluster_anomaly_data,
                        cluster=selected_behavior_cluster,
                        prediction_period_source="Predicted",
                        feature_order=feature_order,
                        feature_color_map=feature_color_map,
                        y_axis_max=shared_y_axis_max,
                        anomaly_score_threshold=3,
                    )
                )

                st.plotly_chart(
                    predicted_bar_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "displayModeBar": True,
                        "responsive": True,
                    },
                    key="predicted_decoded_stacked_bar",
                )

            else:
                st.info(
                    "No historical or predicted decoded behavior "
                    "is available for the selected cluster and "
                    "week range."
                )

        if selected_behavior_view == "Donut":
            # ================================================
            # Prepare historical and predicted composition
            # ================================================

            (
                composition_data,
                composition_feature_order,
            ) = prepare_decoded_composition_data(
                decoded_data=cluster_decoded_data,
                cluster=selected_behavior_cluster,
                top_n_features=10,
            )

            if (
                composition_data.empty
                or not composition_feature_order
            ):
                st.info(
                    "No decoded feature composition is available for "
                    "the selected cluster."
                )

            else:
                # ============================================
                # Reuse the same colors as the bar chart
                # ============================================

                composition_feature_color_map = {
                    feature: feature_color_map.get(
                        feature,
                        "#B0B0B0",
                    )
                    for feature in composition_feature_order
                }

                if "Other" in composition_feature_order:
                    composition_feature_color_map[
                        "Other"
                    ] = "#B0B0B0"


                # ============================================
                # Prepare historical and predicted weeks
                # ============================================

                historical_composition_weeks = (
                    composition_data.loc[
                        composition_data["source"]
                        == "Historical",
                        [
                            "week_start",
                            "year_week",
                        ],
                    ]
                    .drop_duplicates(
                        subset=["week_start"]
                    )
                    .sort_values("week_start")
                    .reset_index(drop=True)
                )

                predicted_composition_weeks = (
                    composition_data.loc[
                        composition_data["source"]
                        == "Predicted",
                        [
                            "week_start",
                            "year_week",
                        ],
                    ]
                    .drop_duplicates(
                        subset=["week_start"]
                    )
                    .sort_values("week_start")
                    .reset_index(drop=True)
                )

                historical_composition_week_set = set(
                    historical_composition_weeks[
                        "week_start"
                    ].tolist()
                )

                predicted_composition_week_set = set(
                    predicted_composition_weeks[
                        "week_start"
                    ].tolist()
                )


                # ============================================
                # Build one shared week timeline
                # ============================================

                composition_navigation_weeks = (
                    pd.concat(
                        [
                            historical_composition_weeks,
                            predicted_composition_weeks,
                        ],
                        ignore_index=True,
                    )
                    .drop_duplicates(
                        subset=["week_start"]
                    )
                    .sort_values("week_start")
                    .reset_index(drop=True)
                )

                composition_week_options = (
                    composition_navigation_weeks[
                        "week_start"
                    ].tolist()
                )

                composition_week_label_map = dict(
                    zip(
                        composition_navigation_weeks[
                            "week_start"
                        ],
                        composition_navigation_weeks[
                            "year_week"
                        ],
                    )
                )


                # ============================================
                # Prepare valid center weeks
                # ============================================

                if len(composition_week_options) >= 3:
                    composition_slider_options = (
                        composition_week_options[1:-1]
                    )
                else:
                    composition_slider_options = []


                if not composition_slider_options:
                    st.info(
                        "At least three weeks are required to display "
                        "the feature composition comparison."
                    )

                else:
                    composition_state_key = (
                        "overview_composition_center_week"
                    )

                    valid_predicted_slider_weeks = [
                        week
                        for week in predicted_composition_weeks[
                            "week_start"
                        ].tolist()
                        if week in composition_slider_options
                    ]

                    default_composition_week = (
                        valid_predicted_slider_weeks[0]
                        if valid_predicted_slider_weeks
                        else composition_slider_options[-1]
                    )

                    if (
                        composition_state_key
                        not in st.session_state
                        or st.session_state[
                            composition_state_key
                        ] not in composition_slider_options
                    ):
                        st.session_state[
                            composition_state_key
                        ] = default_composition_week


                    # ========================================
                    # Previous and next callbacks
                    # ========================================

                    def move_overview_composition_week(
                        step: int,
                    ) -> None:
                        current_week = st.session_state[
                            composition_state_key
                        ]

                        current_index = (
                            composition_slider_options.index(
                                current_week
                            )
                        )

                        new_index = min(
                            max(
                                current_index + step,
                                0,
                            ),
                            len(composition_slider_options) - 1,
                        )

                        st.session_state[
                            composition_state_key
                        ] = composition_slider_options[
                            new_index
                        ]


                    selected_center_week = (
                        st.session_state[
                            composition_state_key
                        ]
                    )

                    selected_slider_index = (
                        composition_slider_options.index(
                            selected_center_week
                        )
                    )

                    previous_disabled = (
                        selected_slider_index == 0
                    )

                    next_disabled = (
                        selected_slider_index
                        == len(composition_slider_options) - 1
                    )


                    (
                        previous_col,
                        slider_col,
                        next_col,
                    ) = st.columns(
                        [
                            1.2,
                            6,
                            1.2,
                        ]
                    )

                    with previous_col:
                        st.button(
                            "Previous",
                            key=(
                                "overview_composition_previous"
                            ),
                            disabled=previous_disabled,
                            on_click=(
                                move_overview_composition_week
                            ),
                            args=(-1,),
                            use_container_width=True,
                        )

                    with slider_col:
                        st.select_slider(
                            "Composition comparison week",
                            options=composition_slider_options,
                            format_func=lambda value: (
                                composition_week_label_map[
                                    value
                                ]
                            ),
                            key=composition_state_key,
                        )

                    with next_col:
                        st.button(
                            "Next",
                            key=(
                                "overview_composition_next"
                            ),
                            disabled=next_disabled,
                            on_click=(
                                move_overview_composition_week
                            ),
                            args=(1,),
                            use_container_width=True,
                        )


                    # ========================================
                    # Select previous, center, and next weeks
                    # ========================================

                    selected_center_week = (
                        st.session_state[
                            composition_state_key
                        ]
                    )

                    selected_center_index = (
                        composition_week_options.index(
                            selected_center_week
                        )
                    )

                    visible_composition_weeks = (
                        composition_week_options[
                            selected_center_index - 1:
                            selected_center_index + 2
                        ]
                    )


                    # ========================================
                    # Week heading row
                    # ========================================

                    composition_column_widths = [
                        0.8,
                        *(
                            [3]
                            * len(visible_composition_weeks)
                        ),
                    ]

                    week_header_columns = st.columns(
                        composition_column_widths
                    )

                    with week_header_columns[0]:
                        st.markdown("**Source**")

                    for column_index, week_start in enumerate(
                        visible_composition_weeks,
                        start=1,
                    ):
                        week_label = (
                            composition_week_label_map[
                                week_start
                            ]
                        )

                        with week_header_columns[
                            column_index
                        ]:
                            st.markdown(
                                (
                                    "<div style='text-align:center;'>"
                                    f"<b>{week_label}</b>"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )

                    # ========================================
                    # Determine visible source availability
                    # ========================================

                    visible_composition_week_set = set(
                        visible_composition_weeks
                    )

                    visible_historical_weeks = (
                        visible_composition_week_set
                        & historical_composition_week_set
                    )

                    visible_predicted_weeks = (
                        visible_composition_week_set
                        & predicted_composition_week_set
                    )

                    show_historical_donuts = bool(
                        visible_historical_weeks
                    )

                    show_predicted_donuts = bool(
                        visible_predicted_weeks
                    )


                    # ========================================
                    # Historical donut row
                    # ========================================

                    if show_historical_donuts:
                        historical_donut_columns = st.columns(
                            composition_column_widths
                        )

                        with historical_donut_columns[0]:
                            st.markdown("**Historical**")

                        for column_index, week_start in enumerate(
                            visible_composition_weeks,
                            start=1,
                        ):
                            week_label = (
                                composition_week_label_map[
                                    week_start
                                ]
                            )

                            with historical_donut_columns[
                                column_index
                            ]:
                                historical_week_available = (
                                    week_start
                                    in historical_composition_week_set
                                )

                                if not historical_week_available:
                                    st.markdown(
                                        """
                                        <div style="
                                            height: 220px;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            text-align: center;
                                            color: gray;
                                        ">
                                            No historical data<br>
                                            for this week
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                else:
                                    historical_donut_figure = (
                                        build_decoded_composition_donut(
                                            composition_data=(
                                                composition_data
                                            ),
                                            cluster=(
                                                selected_behavior_cluster
                                            ),
                                            source="Historical",
                                            selected_week_start=(
                                                week_start
                                            ),
                                            selected_year_week=(
                                                week_label
                                            ),
                                            feature_order=(
                                                composition_feature_order
                                            ),
                                            feature_color_map=(
                                                composition_feature_color_map
                                            ),
                                        )
                                    )

                                    historical_donut_figure.update_layout(
                                        title=None,
                                        height=220,
                                        showlegend=False,
                                        margin={
                                            "l": 0,
                                            "r": 0,
                                            "t": 0,
                                            "b": 0,
                                        },
                                    )

                                    historical_donut_figure.update_traces(
                                        textinfo="none",
                                        domain={
                                            "x": [0.12, 0.88],
                                            "y": [0.08, 0.92],
                                        },
                                    )

                                    st.plotly_chart(
                                        historical_donut_figure,
                                        use_container_width=True,
                                        config={
                                            "displaylogo": False,
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        key=(
                                            "overview_historical_composition_"
                                            f"{selected_behavior_cluster}_"
                                            f"{week_start}"
                                        ),
                                    )


                    # ========================================
                    # Predicted donut row
                    # ========================================

                    if show_predicted_donuts:
                        predicted_donut_columns = st.columns(
                            composition_column_widths
                        )

                        with predicted_donut_columns[0]:
                            st.markdown("**Predicted**")

                        for column_index, week_start in enumerate(
                            visible_composition_weeks,
                            start=1,
                        ):
                            week_label = (
                                composition_week_label_map[
                                    week_start
                                ]
                            )

                            with predicted_donut_columns[
                                column_index
                            ]:
                                predicted_week_available = (
                                    week_start
                                    in predicted_composition_week_set
                                )

                                if not predicted_week_available:
                                    st.markdown(
                                        """
                                        <div style="
                                            height: 220px;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            text-align: center;
                                            color: gray;
                                        ">
                                            No predicted data<br>
                                            for this week
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                else:
                                    predicted_donut_figure = (
                                        build_decoded_composition_donut(
                                            composition_data=(
                                                composition_data
                                            ),
                                            cluster=(
                                                selected_behavior_cluster
                                            ),
                                            source="Predicted",
                                            selected_week_start=(
                                                week_start
                                            ),
                                            selected_year_week=(
                                                week_label
                                            ),
                                            feature_order=(
                                                composition_feature_order
                                            ),
                                            feature_color_map=(
                                                composition_feature_color_map
                                            ),
                                        )
                                    )

                                    predicted_donut_figure.update_layout(
                                        title=None,
                                        height=220,
                                        showlegend=False,
                                        margin={
                                            "l": 0,
                                            "r": 0,
                                            "t": 0,
                                            "b": 0,
                                        },
                                    )

                                    predicted_donut_figure.update_traces(
                                        textinfo="none",
                                        domain={
                                            "x": [0.12, 0.88],
                                            "y": [0.08, 0.92],
                                        },
                                    )

                                    st.plotly_chart(
                                        predicted_donut_figure,
                                        use_container_width=True,
                                        config={
                                            "displaylogo": False,
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        key=(
                                            "overview_predicted_composition_"
                                            f"{selected_behavior_cluster}_"
                                            f"{week_start}"
                                        ),
                                    )

                    if (
                        show_historical_donuts
                        and show_predicted_donuts
                    ):
                        st.caption(
                            "Each column compares historical and predicted "
                            "feature composition for the same week. The ten "
                            "largest features are shown separately and the "
                            "remaining features are grouped as Other."
                        )

                    elif show_historical_donuts:
                        st.caption(
                            "Only historical feature composition is available "
                            "for the visible weeks."
                        )

                    elif show_predicted_donuts:
                        st.caption(
                            "Only predicted feature composition is available "
                            "for the visible weeks."
                        )


# =========================================================
# Compact anomaly overview
# =========================================================

col_1, col_2 = st.columns([4, 1])
with col_1:
    st.markdown("## Behavioral Anomaly Overview")
with col_2:
    if st.button("View Detailed Anomaly Page", type="primary"):
        st.switch_page("pages/2_Anomaly_Analysis.py")

behavior_scores = pd.to_numeric(
    filtered_trajectory.get(
        "behavior_change_score",
        pd.Series(
            index=filtered_trajectory.index,
            dtype=float,
        ),
    ),
    errors="coerce",
)

filtered_anomalies = filtered_trajectory.loc[
    behavior_scores >= 3
].copy()


# ============================
# Anomaly Count Overview
# ============================

# Anomaly Count
total_anomaly_count = len(filtered_anomalies)
historical_anomaly_count = len(
    filtered_anomalies.loc[
        filtered_anomalies["source"] == "Historical"
    ]
)
predicted_anomaly_count = len(
    filtered_anomalies.loc[
        filtered_anomalies["source"] == "Predicted"
    ]
)

# Affected Cluster Count
affected_cluster_count = (
    filtered_anomalies["final_cluster_label"]
    .dropna()
    .nunique()
)
affected_historical_cluster_df = filtered_anomalies.loc[
    filtered_anomalies["source"] == "Historical"
]
affected_prediction_cluster_df = filtered_anomalies.loc[
    filtered_anomalies["source"] == "Predicted"
]
affected_historical_cluster_count = affected_historical_cluster_df["final_cluster_label"].dropna().nunique()
affected_prediction_cluster_count = affected_prediction_cluster_df["final_cluster_label"].dropna().nunique()


anomaly_col1, anomaly_col2 = (
    st.columns(2)
)

with anomaly_col1:
    st.metric(
        "Total Detected Behavioral Anomalies",
        f"{total_anomaly_count:,} Anomalies",
    )
    col1_sub1, col1_sub2 = st.columns(2)
    with col1_sub1:
        st.markdown(f"Anomalies on Predicted Weeks: **{predicted_anomaly_count:,} anomalies.**")
    with col1_sub2:
        st.markdown(f"Anomalies on Historical Weeks: **{historical_anomaly_count:,} anomalies.**")
        


with anomaly_col2:
    st.metric(
        "Total Affected Clusters",
        f"{affected_cluster_count:,} Clusters",
    )
    col2_sub1, col2_sub2 = st.columns(2)
    with col2_sub1:
        st.markdown(f"Affected Clusters in Predicted Weeks: **{affected_prediction_cluster_count:,} clusters**.")
    with col2_sub2:
        st.markdown(f"Affected Clusters in Historical Weeks: **{affected_historical_cluster_count:,} clusters**.")


# =========================================================
# Highest-priority anomalies
# =========================================================

if filtered_anomalies.empty:
    st.info(
        "No anomalies were found for the selected filters."
    )

else:
    
    st.markdown("#### High Priority Predicted Week Anomalies")
    
    anomaly_table = filtered_anomalies.copy()

    sort_columns = []
    sort_ascending = []

    # Ensures that the anomaly scores are in numeric datatype before sorting
    if "anomaly_priority_rank" in anomaly_table.columns:
        anomaly_table["anomaly_priority_rank"] = (
            pd.to_numeric(
                anomaly_table["anomaly_priority_rank"],
                errors="coerce",
            )
        )

        sort_columns.append("anomaly_priority_rank")
        sort_ascending.append(False)
        
    if "behavior_change_score" in anomaly_table.columns:
        anomaly_table["behavior_change_score"] = (
            pd.to_numeric(
                anomaly_table["behavior_change_score"],
                errors="coerce",
            )
        )

        sort_columns.append("behavior_change_score")
        sort_ascending.append(False)

    sort_columns.append("week_start")
    sort_ascending.append(False)
    
    # Sorts, filters and renames the columns of the "filtered" anomaly records
    anomaly_table = anomaly_table.sort_values(
        sort_columns,
        ascending=sort_ascending,
        na_position="last",
    )

    # Defines the essential column names for displaying the anomalous week selection DF
    preferred_columns = [
        "final_cluster_label",
        "year_week",
        "source",
        "anomaly_priority_level",
        "anomaly_priority_rank",
        "triggered_main_flags",
    ]

    # Ensures that essential column names are present
    available_anomaly_columns = [
        column
        for column in preferred_columns
        if column in anomaly_table.columns
    ]

    # Filters, creates a copy, and renames the defined essential columns
    selection_of_detected_anomalies = anomaly_table[
        available_anomaly_columns
    ].copy()

    selection_of_detected_anomalies = selection_of_detected_anomalies.rename(
        columns={
            "final_cluster_label": "Cluster",
            "year_week": "Week",
            "source": "Source",
            "anomaly_priority_level": "Priority Level",
            "anomaly_priority_rank": "Priority Code",
            "triggered_main_flags": "Triggered Main Flags",
        }
    )

    selection_of_detected_anomalies = selection_of_detected_anomalies.loc[
        selection_of_detected_anomalies["Source"] == "Predicted"
    ]
        
    st.dataframe(
        selection_of_detected_anomalies,
        use_container_width=True,
        hide_index=True,
        height=245,
    )


