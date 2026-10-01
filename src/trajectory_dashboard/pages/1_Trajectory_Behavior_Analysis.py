from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

from src.trajectory_dashboard.utils.trajectory_analysis import (
    shorten_text,
    build_behavior_timeline,
    build_feature_comparison_chart,
    build_feature_trend_chart,
    build_feature_color_map,
    build_feature_composition_donut,
    classify_feature_change,
    format_feature_change_list,
)

from src.trajectory_dashboard.utils.trajectory_visualizations import (
    prepare_decoded_composition_data,
    build_decoded_composition_donut,
)

from src.trajectory_dashboard.utils.anomaly_dashboard_utils import (
    display_feature_usage_changes,
)


# =========================================================
# Make the repository root importable
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    prepare_decoded_data,
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Trajectory Behavior Analysis",
    layout="wide",
)


# =========================================================
# Page header
# =========================================================

st.title(
    "Trajectory Behavior Analysis"
)

st.write(
    "Analyze how decoded feature usage and elapsed time "
    "change from historical behavior into the forecast period."
)


# =========================================================
# Load decoded behavior results
# =========================================================

@st.cache_data(
    show_spinner="Loading decoded behavior results..."
)
def load_behavior_results(
    project_name: str,
    model_name: str,
):
    dashboard_paths = get_dashboard_paths(
        project_name=project_name,
        model_name=model_name,
    )

    all_files_exist, missing_files = (
        validate_dashboard_paths(
            dashboard_paths
        )
    )

    if not all_files_exist:
        return {
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
            "decoded_data": None,
            "paths": dashboard_paths,
            "missing_files": [],
            "missing_columns": missing_columns,
        }

    decoded_data = prepare_decoded_data(
        dashboard_data
    )

    return {
        "decoded_data": decoded_data,
        "paths": dashboard_paths,
        "missing_files": [],
        "missing_columns": {},
    }

loaded_results = load_behavior_results(
    project_name=PROJECT_NAME,
    model_name=MODEL_NAME,
)

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


if decoded_data is None or decoded_data.empty:
    st.error(
        "No decoded behavior results are available."
    )
    st.stop()


# =========================================================
# Prepare decoded behavior columns
# =========================================================

decoded_data = decoded_data.copy()

decoded_data["pred_elapsed"] = pd.to_numeric(
    decoded_data["pred_elapsed"],
    errors="coerce",
).fillna(0.0)

decoded_data = decoded_data.dropna(
    subset=[
        "final_cluster_label",
        "source",
        "week_start",
        "year_week",
        "feature",
    ]
)

decoded_data = decoded_data.sort_values(
    [
        "final_cluster_label",
        "week_start",
        "source",
        "feature",
    ]
).reset_index(drop=True)


# =========================================================
# Analysis controls
# =========================================================

cluster_options = sorted(
    decoded_data["final_cluster_label"]
    .dropna()
    .unique()
    .tolist(),
    key=str,
)

default_cluster = st.session_state.get(
    "analysis_cluster"
)

if default_cluster not in cluster_options:
    default_cluster = cluster_options[0]

selected_cluster = st.selectbox(
    "Cluster",
    options=cluster_options,
    index=cluster_options.index(
        default_cluster
    ),
    format_func=str,
    key="trajectory_behavior_cluster",
)

st.session_state["analysis_cluster"] = (
    selected_cluster
)


cluster_decoded_data = decoded_data.loc[
    decoded_data["final_cluster_label"]
    == selected_cluster
].copy()

# =========================================================
# Prepare the latest 52 weeks for composition charts
# =========================================================

composition_available_weeks = (
    cluster_decoded_data[
        [
            "week_start",
            "year_week",
        ]
    ]
    .dropna(
        subset=[
            "week_start",
        ]
    )
    .drop_duplicates(
        subset=[
            "week_start",
        ]
    )
    .sort_values(
        "week_start"
    )
    .reset_index(
        drop=True
    )
)

last_52_composition_weeks = (
    composition_available_weeks
    .tail(52)
    .copy()
)

last_52_week_starts = (
    last_52_composition_weeks[
        "week_start"
    ].tolist()
)

last_52_weeks_decoded_data = (
    cluster_decoded_data.loc[
        cluster_decoded_data[
            "week_start"
        ].isin(
            last_52_week_starts
        )
    ]
    .copy()
)

# =========================================================
# Prepare a unified 52-week trend dataset
# Predicted replaces Historical during overlapping weeks
# =========================================================

predicted_overlap_weeks = set(
    last_52_weeks_decoded_data.loc[
        last_52_weeks_decoded_data["source"]
        == "Predicted",
        "week_start",
    ]
    .dropna()
    .tolist()
)

latest_52_unified_trend_data = (
    last_52_weeks_decoded_data.loc[
        ~(
            (
                last_52_weeks_decoded_data["source"]
                == "Historical"
            )
            & (
                last_52_weeks_decoded_data[
                    "week_start"
                ].isin(
                    predicted_overlap_weeks
                )
            )
        )
    ]
    .sort_values(
        [
            "week_start",
            "source",
            "feature",
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# Shared feature order and colors for all trend views
# =========================================================

trend_feature_totals = (
    latest_52_unified_trend_data.groupby(
        "feature",
        as_index=False,
    )["pred_elapsed"]
    .sum()
    .sort_values(
        "pred_elapsed",
        ascending=False,
    )
)

trend_feature_order = (
    trend_feature_totals[
        "feature"
    ]
    .astype(str)
    .tolist()
)

trend_feature_color_map = (
    build_feature_color_map(
        trend_feature_order
    )
)


# =========================================================
# Latest 52-week behavior trend
# =========================================================
st.divider()

trend_header_col1, trend_header_col2 = st.columns(
    [4, 2],
    vertical_alignment="bottom",
)

with trend_header_col1:
    st.subheader(
        "Latest 52-Week Behavior Trend"
    )

with trend_header_col2:
    selected_trend_view = st.radio(
        "Trend view",
        options=[
            "Bar",
            "Line",
            "Donut",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="behavior_analysis_trend_view",
    )

if selected_trend_view == "Bar":
    st.caption(
        "View decoded feature elapsed-time composition across "
        "the latest 52 weeks using stacked bars."
    )

elif selected_trend_view == "Line":
    st.caption(
        "View the major decoded feature trends across the latest "
        "52 weeks."
    )

else:
    st.caption(
        "View weekly decoded feature composition across the "
        "latest 52 weeks."
    )

if selected_trend_view == "Bar":
    behavior_timeline_figure = build_behavior_timeline(
        cluster_data=latest_52_unified_trend_data,
        selected_target_week=None,
        feature_color_map=trend_feature_color_map,
    )

    st.plotly_chart(
        behavior_timeline_figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "responsive": True,
        },
        key="trajectory_behavior_timeline",
    )


if selected_trend_view == "Line":
    feature_trend_figure = (
        build_feature_trend_chart(
            cluster_data=latest_52_unified_trend_data,
            top_n_features=8,
            feature_color_map=trend_feature_color_map,
        )
    )

    st.plotly_chart(
        feature_trend_figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "responsive": True,
        },
        key="historical_predicted_feature_trend",
    )


# =========================================================
# Feature composition trend
# =========================================================

if selected_trend_view == "Donut":
    # =========================================================
    # Prepare weekly composition data
    # =========================================================

    (
        weekly_composition_data,
        weekly_composition_feature_order,
    ) = prepare_decoded_composition_data(
        decoded_data=latest_52_unified_trend_data,
        cluster=selected_cluster,
        top_n_features=5,
    )

    if (
        weekly_composition_data.empty
        or not weekly_composition_feature_order
    ):
        st.info(
            "No decoded feature composition is available for "
            "the selected cluster within the latest 52 weeks."
        )

    else:
        # =====================================================
        # Shared feature colors across all composition donuts
        # =====================================================

        weekly_composition_color_map = {
            feature: trend_feature_color_map.get(
                str(feature),
                "#B0B0B0",
            )
            for feature in weekly_composition_feature_order
        }

        if "Other" in weekly_composition_feature_order:
            weekly_composition_color_map["Other"] = "#B0B0B0"

        # =====================================================
        # Prepare unified composition weeks
        # =====================================================

        composition_navigation_weeks = (
            weekly_composition_data[
                [
                    "week_start",
                    "year_week",
                    "source",
                ]
            ]
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

        composition_week_source_map = dict(
            zip(
                composition_navigation_weeks[
                    "week_start"
                ],
                composition_navigation_weeks[
                    "source"
                ],
            )
        )


        # =====================================================
        # Prepare center-week slider options
        # =====================================================

        if len(composition_week_options) >= 3:
            composition_slider_options = (
                composition_week_options[1:-1]
            )
        else:
            composition_slider_options = []


        if not composition_slider_options:
            st.info(
                "At least three weeks are required to display "
                "the three-week feature composition trend."
            )

        else:
            # =================================================
            # Initialize selected center week
            # =================================================

            composition_state_key = (
                "behavior_analysis_composition_center_week"
            )

            default_composition_week = (
                composition_slider_options[-1]
            )

            if (
                composition_state_key not in st.session_state
                or st.session_state[
                    composition_state_key
                ] not in composition_slider_options
            ):
                st.session_state[
                    composition_state_key
                ] = default_composition_week


            # =================================================
            # Previous and next week callback
            # =================================================

            def move_composition_week(
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


            selected_center_week = st.session_state[
                composition_state_key
            ]

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
                        "behavior_analysis_composition_previous"
                    ),
                    disabled=previous_disabled,
                    on_click=move_composition_week,
                    args=(-1,),
                    use_container_width=True,
                )

            with slider_col:
                st.select_slider(
                    "Composition trend week",
                    options=composition_slider_options,
                    format_func=lambda value: (
                        composition_week_label_map[value]
                    ),
                    key=composition_state_key,
                )

            with next_col:
                st.button(
                    "Next",
                    key=(
                        "behavior_analysis_composition_next"
                    ),
                    disabled=next_disabled,
                    on_click=move_composition_week,
                    args=(1,),
                    use_container_width=True,
                )


            selected_center_week = st.session_state[
                composition_state_key
            ]

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


            # =================================================
            # Week heading row
            # =================================================

            composition_column_widths = [
                0.9,
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

                with week_header_columns[column_index]:
                    st.markdown(
                        (
                            "<div style='text-align:center;'>"
                            f"<b>{week_label}</b>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )


            # =================================================
            # Unified donut row
            # =================================================

            unified_donut_columns = st.columns(
                composition_column_widths
            )

            with unified_donut_columns[0]:
                st.markdown("**Trend**")

            for column_index, week_start in enumerate(
                visible_composition_weeks,
                start=1,
            ):
                week_label = (
                    composition_week_label_map[
                        week_start
                    ]
                )

                week_source = (
                    composition_week_source_map[
                        week_start
                    ]
                )

                with unified_donut_columns[column_index]:
                    st.caption(
                        week_source
                    )

                    unified_donut_figure = (
                        build_decoded_composition_donut(
                            composition_data=(
                                weekly_composition_data
                            ),
                            cluster=selected_cluster,
                            source=week_source,
                            selected_week_start=week_start,
                            selected_year_week=week_label,
                            feature_order=(
                                weekly_composition_feature_order
                            ),
                            feature_color_map=(
                                weekly_composition_color_map
                            ),
                        )
                    )

                    unified_donut_figure.update_layout(
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

                    unified_donut_figure.update_traces(
                        textinfo="none",
                        domain={
                            "x": [0.12, 0.88],
                            "y": [0.08, 0.92],
                        },
                    )

                    st.plotly_chart(
                        unified_donut_figure,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "displayModeBar": False,
                            "responsive": True,
                        },
                        key=(
                            "behavior_unified_composition_"
                            f"{selected_cluster}_"
                            f"{week_start}_"
                            f"{week_source}"
                        ),
                    )

predicted_week_options_data = (
    cluster_decoded_data.loc[
        cluster_decoded_data["source"]
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

predicted_week_options = (
    predicted_week_options_data[
        "week_start"
    ].tolist()
)

predicted_week_label_map = dict(
    zip(
        predicted_week_options_data[
            "week_start"
        ],
        predicted_week_options_data[
            "year_week"
        ],
    )
)


# =========================================================
# Prepare historical baseline week options
# =========================================================

first_predicted_week = min(
    predicted_week_options
)

historical_week_options_data = (
    cluster_decoded_data.loc[
        (
            cluster_decoded_data["source"]
            == "Historical"
        )
        & (
            cluster_decoded_data["week_start"]
            < first_predicted_week
        ),
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

historical_week_options = (
    historical_week_options_data[
        "week_start"
    ].tolist()
)

historical_week_label_map = dict(
    zip(
        historical_week_options_data[
            "week_start"
        ],
        historical_week_options_data[
            "year_week"
        ],
    )
)

if not predicted_week_options:
    st.warning(
        "No predicted decoded weeks are available for the "
        "selected cluster."
    )
    st.stop()

if not historical_week_options:
    st.warning(
        "No historical decoded weeks before the forecast "
        "period are available for the selected cluster."
    )
    st.stop()


default_target_week = st.session_state.get(
    "analysis_week"
)

if default_target_week not in predicted_week_options:
    default_target_week = predicted_week_options[0]

default_baseline_week = st.session_state.get(
    "analysis_baseline_week"
)

if default_baseline_week not in historical_week_options:
    default_baseline_week = historical_week_options[-1]

st.divider()

st.subheader(
    "Historical vs Forecast Comparison"
)

baseline_col, target_col = st.columns(2)

with baseline_col:
    selected_baseline_week = st.selectbox(
        "Historical baseline week",
        options=historical_week_options,
        index=historical_week_options.index(
            default_baseline_week
        ),
        format_func=lambda value: (
            historical_week_label_map[value]
        ),
        key="trajectory_behavior_baseline_week",
    )

with target_col:
    selected_target_week = st.selectbox(
        "Target forecast week",
        options=predicted_week_options,
        index=predicted_week_options.index(
            default_target_week
        ),
        format_func=lambda value: (
            predicted_week_label_map[value]
        ),
        key="trajectory_behavior_target_week",
    )

st.session_state["analysis_baseline_week"] = (
    selected_baseline_week
)

st.session_state["analysis_week"] = (
    selected_target_week
)

selected_baseline_label = (
    historical_week_label_map[
        selected_baseline_week
    ]
)

selected_target_label = (
    predicted_week_label_map[
        selected_target_week
    ]
)

st.caption(
    f"Cluster {selected_cluster} · "
    f"Historical baseline: "
    f"{selected_baseline_label} · "
    f"Target forecast: {selected_target_label}"
)


# =========================================================
# Prepare target forecast behavior
# =========================================================

target_forecast_data = (
    cluster_decoded_data.loc[
        (
            cluster_decoded_data["source"]
            == "Predicted"
        )
        & (
            cluster_decoded_data["week_start"]
            == selected_target_week
        )
    ]
    .groupby(
        "feature",
        as_index=False,
    )["pred_elapsed"]
    .sum()
)

selected_baseline_data = (
    cluster_decoded_data.loc[
        (
            cluster_decoded_data["source"]
            == "Historical"
        )
        & (
            cluster_decoded_data["week_start"]
            == selected_baseline_week
        )
    ]
    .groupby(
        "feature",
        as_index=False,
    )["pred_elapsed"]
    .sum()
)

if not target_forecast_data.empty:
    top_predicted_row = (
        target_forecast_data.sort_values(
            "pred_elapsed",
            ascending=False,
        )
        .iloc[0]
    )

    top_predicted_feature = str(
        top_predicted_row["feature"]
    )

else:
    top_predicted_feature = "No feature"

predicted_total_elapsed = (
    target_forecast_data["pred_elapsed"].sum()
)

feature_comparison = pd.merge(
    selected_baseline_data.rename(
        columns={
            "pred_elapsed": "historical_elapsed",
        }
    ),
    target_forecast_data.rename(
        columns={
            "pred_elapsed": "forecast_elapsed",
        }
    ),
    on="feature",
    how="outer",
).fillna(
    {
        "historical_elapsed": 0.0,
        "forecast_elapsed": 0.0,
    }
)

feature_comparison["elapsed_change"] = (
    feature_comparison["forecast_elapsed"]
    - feature_comparison["historical_elapsed"]
)


# =========================================================
# Prepare comparison summary values
# =========================================================

historical_total_elapsed = (
    feature_comparison[
        "historical_elapsed"
    ].sum()
)

forecast_total_elapsed = (
    feature_comparison[
        "forecast_elapsed"
    ].sum()
)

total_elapsed_change = (
    forecast_total_elapsed
    - historical_total_elapsed
)

if historical_total_elapsed > 0:
    total_elapsed_percent_change = (
        total_elapsed_change
        / historical_total_elapsed
        * 100.0
    )
else:
    total_elapsed_percent_change = None


feature_comparison[
    "status"
] = feature_comparison.apply(
    lambda row: classify_feature_change(
        historical_elapsed=float(
            row["historical_elapsed"]
        ),
        forecast_elapsed=float(
            row["forecast_elapsed"]
        ),
    ),
    axis=1,
)

status_counts = (
    feature_comparison[
        "status"
    ]
    .value_counts()
    .to_dict()
)

increasing_count = status_counts.get(
    "Increasing",
    0,
)

decreasing_count = status_counts.get(
    "Decreasing",
    0,
)

appearing_count = status_counts.get(
    "Appearing",
    0,
)

disappearing_count = status_counts.get(
    "Disappearing",
    0,
)

stable_count = status_counts.get(
    "Stable",
    0,
)

total_feature_count = len(
    feature_comparison
)

changed_feature_count = (
    total_feature_count
    - stable_count
)

if total_elapsed_percent_change is not None:
    total_elapsed_percent_text = (
        f"{total_elapsed_percent_change:+.2f}%"
    )
else:
    total_elapsed_percent_text = (
        "No historical baseline"
    )

if not feature_comparison.empty:
    largest_increase_row = (
        feature_comparison.sort_values(
            "elapsed_change",
            ascending=False,
        )
        .iloc[0]
    )

    largest_increase_feature = str(
        largest_increase_row["feature"]
    )

    largest_increase_value = float(
        largest_increase_row["elapsed_change"]
    )


    largest_decrease_row = (
        feature_comparison.sort_values(
            "elapsed_change",
            ascending=True,
        )
        .iloc[0]
    )

    largest_decrease_feature = str(
        largest_decrease_row["feature"]
    )

    largest_decrease_value = float(
        largest_decrease_row["elapsed_change"]
    )

else:
    largest_increase_feature = "No feature"
    largest_increase_value = 0.0

    largest_decrease_feature = "No feature"
    largest_decrease_value = 0.0

if largest_increase_value <= 0:
    largest_increase_feature = "No increase"
    largest_increase_value = 0.0

if largest_decrease_value >= 0:
    largest_decrease_feature = "No decrease"
    largest_decrease_value = 0.0


# =========================================================
# Behavior summary cards
# =========================================================

st.subheader("Behavior Summary")

st.caption(
    "Summary of the selected target forecast week compared "
    "with the selected historical baseline week."
)


# =========================================================
# First summary row
# =========================================================

(
    summary_col1,
    summary_col2,
    summary_col3,
    summary_col4,
) = st.columns(4)

summary_col1.metric(
    "Historical baseline",
    selected_baseline_label,
)

summary_col2.metric(
    "Target forecast",
    selected_target_label,
)

summary_col3.metric(
    "Total elapsed change",
    f"{total_elapsed_change:+,.0f} sec",
    delta=total_elapsed_percent_text,
)

summary_col4.metric(
    "Features changed",
    (
        f"{changed_feature_count} "
        f"of {total_feature_count}"
    ),
)


# =========================================================
# Second summary row
# =========================================================

(
    summary_col5,
    summary_col6,
    summary_col7,
    summary_col8,
) = st.columns(4)

summary_col5.metric(
    "Largest increase",
    shorten_text(
        largest_increase_feature,
        max_length=24,
    ),
    delta=(
        f"{largest_increase_value:+,.0f} sec"
    ),
    help=largest_increase_feature,
)

summary_col6.metric(
    "Largest decrease",
    shorten_text(
        largest_decrease_feature,
        max_length=24,
    ),
    delta=(
        f"{largest_decrease_value:+,.0f} sec"
    ),
    help=largest_decrease_feature,
)

summary_col7.metric(
    "Appearing features",
    f"{appearing_count:,}",
)

summary_col8.metric(
    "Disappearing features",
    f"{disappearing_count:,}",
)
    

# =========================================================
# Latest historical vs selected forecast
# =========================================================

st.divider()

st.subheader(
    "Historical vs Forecast Feature Comparison"
)

st.caption(
    "Compare the selected target forecast week with "
    "the selected target forecast week."
)

if (
    selected_baseline_week
    not in historical_week_options
    or selected_target_week
    not in predicted_week_options
):
    st.info(
        "A pre-forecast historical week and a selected "
        "forecast week are required for this comparison."
    )

else:
    selected_historical_comparison_week = (
        selected_baseline_week
    )

    selected_forecast_comparison_week = (
        selected_target_week
    )

    selected_historical_comparison_label = (
        selected_baseline_label
    )

    selected_forecast_comparison_label = (
        selected_target_label
    )

    selected_historical_comparison_data = (
        cluster_decoded_data.loc[
            (
                cluster_decoded_data["source"]
                == "Historical"
            )
            & (
                cluster_decoded_data["week_start"]
                == selected_historical_comparison_week
            )
        ]
        .groupby(
            "feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
    )

    selected_forecast_comparison_data = (
        cluster_decoded_data.loc[
            (
                cluster_decoded_data["source"]
                == "Predicted"
            )
            & (
                cluster_decoded_data["week_start"]
                == selected_forecast_comparison_week
            )
        ]
        .groupby(
            "feature",
            as_index=False,
        )["pred_elapsed"]
        .sum()
    )

    selected_feature_comparison = pd.merge(
        selected_historical_comparison_data.rename(
            columns={
                "pred_elapsed": "historical_elapsed",
            }
        ),
        selected_forecast_comparison_data.rename(
            columns={
                "pred_elapsed": "forecast_elapsed",
            }
        ),
        on="feature",
        how="outer",
    ).fillna(
        {
            "historical_elapsed": 0.0,
            "forecast_elapsed": 0.0,
        }
    )

    selected_feature_comparison[
        "elapsed_change"
    ] = (
        selected_feature_comparison[
            "forecast_elapsed"
        ]
        - selected_feature_comparison[
            "historical_elapsed"
        ]
    )

    comparison_features = (
        selected_feature_comparison[
            "feature"
        ]
        .astype(str)
        .tolist()
    )

    comparison_feature_color_map = (
        trend_feature_color_map.copy()
    )

    missing_comparison_features = [
        feature
        for feature in comparison_features
        if feature not in comparison_feature_color_map
    ]

    if missing_comparison_features:
        fallback_color_map = build_feature_color_map(
            missing_comparison_features
        )

        comparison_feature_color_map.update(
            fallback_color_map
        )

    feature_comparison_figure = (
        build_feature_comparison_chart(
            feature_comparison=(
                selected_feature_comparison
            ),
            latest_historical_label=(
                selected_historical_comparison_label
            ),
            selected_target_label=(
                selected_forecast_comparison_label
            ),
            top_n_features=10,
            feature_color_map=(
                comparison_feature_color_map
            ),
        )
    )

    st.plotly_chart(
        feature_comparison_figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "responsive": True,
        },
        key=(
            "historical_forecast_"
            "feature_comparison"
        ),
    )

    historical_comparison_total = (
        selected_feature_comparison[
            "historical_elapsed"
        ].sum()
    )

    forecast_comparison_total = (
        selected_feature_comparison[
            "forecast_elapsed"
        ].sum()
    )

    selected_feature_comparison[
        "historical_share"
    ] = 0.0

    selected_feature_comparison[
        "forecast_share"
    ] = 0.0


    if historical_comparison_total > 0:
        selected_feature_comparison[
            "historical_share"
        ] = (
            selected_feature_comparison[
                "historical_elapsed"
            ]
            / historical_comparison_total
            * 100.0
        )


    if forecast_comparison_total > 0:
        selected_feature_comparison[
            "forecast_share"
        ] = (
            selected_feature_comparison[
                "forecast_elapsed"
            ]
            / forecast_comparison_total
            * 100.0
        )

    selected_feature_comparison[
        "share_change"
    ] = (
        selected_feature_comparison[
            "forecast_share"
        ]
        - selected_feature_comparison[
            "historical_share"
        ]
    )

    selected_feature_comparison[
        "percent_change"
    ] = pd.NA

    historical_positive_mask = (
        selected_feature_comparison[
            "historical_elapsed"
        ]
        > 0
    )

    selected_feature_comparison.loc[
        historical_positive_mask,
        "percent_change",
    ] = (
        selected_feature_comparison.loc[
            historical_positive_mask,
            "elapsed_change",
        ]
        / selected_feature_comparison.loc[
            historical_positive_mask,
            "historical_elapsed",
        ]
        * 100.0
    )

    selected_feature_comparison[
        "status"
    ] = selected_feature_comparison.apply(
        lambda row: classify_feature_change(
            historical_elapsed=float(
                row["historical_elapsed"]
            ),
            forecast_elapsed=float(
                row["forecast_elapsed"]
            ),
        ),
        axis=1,
    )

    selected_feature_comparison[
        "absolute_change"
    ] = (
        selected_feature_comparison[
            "elapsed_change"
        ].abs()
    )


    # =========================================================
    # Feature composition comparison
    # =========================================================

    st.divider()

    st.subheader(
        "Feature Composition Comparison"
    )

    st.caption(
        "Compare how each feature contributes to total decoded "
        "elapsed time in the historical baseline and selected "
        "forecast week."
    )

    feature_color_map = (
        comparison_feature_color_map
    )

    historical_composition_figure = (
        build_feature_composition_donut(
            feature_comparison=(
                selected_feature_comparison
            ),
            value_column="historical_elapsed",
            chart_title=(
                "Historical baseline · "
                f"{selected_historical_comparison_label}"
            ),
            feature_color_map=feature_color_map,
        )
    )

    forecast_composition_figure = (
        build_feature_composition_donut(
            feature_comparison=(
                selected_feature_comparison
            ),
            value_column="forecast_elapsed",
            chart_title=(
                "Target forecast · "
                f"{selected_forecast_comparison_label}"
            ),
            feature_color_map=feature_color_map,
        )
    )

    composition_col1, composition_col2 = (
        st.columns(2)
    )

    with composition_col1:
        st.plotly_chart(
            historical_composition_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "responsive": True,
            },
            key="historical_feature_composition",
        )

    with composition_col2:
        st.plotly_chart(
            forecast_composition_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "responsive": True,
            },
            key="forecast_feature_composition",
        )

    feature_change_table = (
        selected_feature_comparison.sort_values(
            "absolute_change",
            ascending=False,
        )
        .copy()
    )

    feature_change_table = feature_change_table[
        [
            "feature",
            "historical_elapsed",
            "forecast_elapsed",
            "elapsed_change",
            "percent_change",
            "historical_share",
            "forecast_share",
            "share_change",
            "status",
        ]
    ].copy()

    feature_change_table = feature_change_table.rename(
        columns={
            "feature": "Feature",
            "historical_elapsed": "Historical Elapsed",
            "forecast_elapsed": "Forecast Elapsed",
            "elapsed_change": "Elapsed Change",
            "percent_change": "Percent Change",
            "historical_share": "Historical Share",
            "forecast_share": "Forecast Share",
            "share_change": "Share Change",
            "status": "Status",
        }
    )

    st.subheader(
        "Feature Change Breakdown"
    )

    st.caption(
        "The table shows how each decoded feature changes "
        "between the selected historical and forecast weeks. "
        "Features are sorted by the largest absolute elapsed-time "
        "change."
    )

    feature_change_row_count = len(
        feature_change_table
    )

    feature_change_table_height = min(
        250,
        38 + feature_change_row_count * 35,
    )

    st.dataframe(
        feature_change_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn(
                "Feature",
                help="Decoded feature name.",
            ),
            "Historical Elapsed": (
                st.column_config.NumberColumn(
                    "Historical Elapsed",
                    format="%.0f sec",
                )
            ),
            "Forecast Elapsed": (
                st.column_config.NumberColumn(
                    "Forecast Elapsed",
                    format="%.0f sec",
                )
            ),
            "Elapsed Change": (
                st.column_config.NumberColumn(
                    "Elapsed Change",
                    format="%+.0f sec",
                )
            ),
            "Percent Change": (
                st.column_config.NumberColumn(
                    "Percent Change",
                    format="%+.2f%%",
                )
            ),
            "Historical Share": (
                st.column_config.NumberColumn(
                    "Historical Share",
                    format="%.2f%%",
                )
            ),
            "Forecast Share": (
                st.column_config.NumberColumn(
                    "Forecast Share",
                    format="%.2f%%",
                )
            ),
            "Share Change": (
                st.column_config.NumberColumn(
                    "Share Change",
                    format="%+.2f pts",
                )
            ),
            "Status": st.column_config.TextColumn(
                "Status",
            ),
        },
        height=feature_change_table_height,
    )

    historical_total_elapsed = (
        selected_feature_comparison[
            "historical_elapsed"
        ].sum()
    )

    forecast_total_elapsed = (
        selected_feature_comparison[
            "forecast_elapsed"
        ].sum()
    )

    total_elapsed_change = (
        forecast_total_elapsed
        - historical_total_elapsed
    )

    if historical_total_elapsed > 0:
        total_elapsed_percent_change = (
            total_elapsed_change
            / historical_total_elapsed
            * 100.0
        )
    else:
        total_elapsed_percent_change = None

    status_counts = (
        selected_feature_comparison[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    increasing_count = status_counts.get(
        "Increasing",
        0,
    )

    decreasing_count = status_counts.get(
        "Decreasing",
        0,
    )

    appearing_count = status_counts.get(
        "Appearing",
        0,
    )

    disappearing_count = status_counts.get(
        "Disappearing",
        0,
    )

    stable_count = status_counts.get(
        "Stable",
        0,
    )

    increase_candidates = (
        selected_feature_comparison.loc[
            selected_feature_comparison[
                "elapsed_change"
            ]
            > 0
        ]
    )

    decrease_candidates = (
        selected_feature_comparison.loc[
            selected_feature_comparison[
                "elapsed_change"
            ]
            < 0
        ]
    )


    # =========================================================
    # Prepare important feature movements
    # =========================================================

    top_increases = (
        increase_candidates.sort_values(
            "elapsed_change",
            ascending=False,
        )
        .head(3)
        .copy()
    )

    top_decreases = (
        decrease_candidates.sort_values(
            "elapsed_change",
            ascending=True,
        )
        .head(3)
        .copy()
    )

    appearing_features = (
        selected_feature_comparison.loc[
            selected_feature_comparison["status"]
            == "Appearing",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    disappearing_features = (
        selected_feature_comparison.loc[
            selected_feature_comparison["status"]
            == "Disappearing",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    top_increases_text = format_feature_change_list(
        top_increases
    )

    top_decreases_text = format_feature_change_list(
        top_decreases
    )

    if not increase_candidates.empty:
        largest_comparison_increase_row = (
            increase_candidates.loc[
                increase_candidates[
                    "elapsed_change"
                ].idxmax()
            ]
        )

        largest_comparison_increase_feature = str(
            largest_comparison_increase_row[
                "feature"
            ]
        )

        largest_comparison_increase_value = float(
            largest_comparison_increase_row[
                "elapsed_change"
            ]
        )

    else:
        largest_comparison_increase_feature = (
            "None"
        )

        largest_comparison_increase_value = 0.0

    if not decrease_candidates.empty:
        largest_comparison_decrease_row = (
            decrease_candidates.loc[
                decrease_candidates[
                    "elapsed_change"
                ].idxmin()
            ]
        )

        largest_comparison_decrease_feature = str(
            largest_comparison_decrease_row[
                "feature"
            ]
        )

        largest_comparison_decrease_value = float(
            largest_comparison_decrease_row[
                "elapsed_change"
            ]
        )

    else:
        largest_comparison_decrease_feature = (
            "None"
        )

        largest_comparison_decrease_value = 0.0

    if total_elapsed_change > 0:
        overall_direction = "increased"
    elif total_elapsed_change < 0:
        overall_direction = "decreased"
    else:
        overall_direction = "remained unchanged"

    if total_elapsed_percent_change is not None:
        overall_change_text = (
            f"Total decoded elapsed time "
            f"{overall_direction} by "
            f"{abs(total_elapsed_change):,.0f} seconds "
            f"({abs(total_elapsed_percent_change):.2f}%)."
        )
    else:
        overall_change_text = (
            f"Total decoded elapsed time "
            f"{overall_direction} by "
            f"{abs(total_elapsed_change):,.0f} seconds."
        )

    if total_elapsed_percent_change is not None:
        interpretation_percent_text = (
            f"({abs(total_elapsed_percent_change):.2f}%)."
        )
    else:
        interpretation_percent_text = "."

    st.divider()

    st.subheader(
        "Interpretation"
    )

    st.caption(
        "A summary of the main behavioral differences between "
        "the historical baseline and selected forecast week."
    )

    st.markdown(
        f"""
    Compared with historical week
    `{selected_historical_comparison_label}`, total decoded elapsed
    time in forecast week `{selected_forecast_comparison_label}`
    **{overall_direction} by {abs(total_elapsed_change):,.0f} seconds**
    {interpretation_percent_text}

    **{changed_feature_count} of {total_feature_count} features**
    changed between the two weeks.
    """
    )


    # =========================================================
    # Largest feature movements
    # =========================================================

    interpretation_col1, interpretation_col2 = (
        st.columns(2)
    )

    with interpretation_col1:
        st.markdown("#### Largest increases")
        st.markdown(top_increases_text)

    with interpretation_col2:
        st.markdown("#### Largest decreases")
        st.markdown(top_decreases_text)


    # =========================================================
    # Appearing and disappearing features
    # =========================================================
    
    display_feature_usage_changes(
        added=appearing_features,
        removed=disappearing_features,
    )