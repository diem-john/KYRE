from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import numpy as np

import ast
import json

import plotly.graph_objects as go
from src.behavioral_anomaly_detector.utils import build_decoded_behavior_column
from src.trajectory_prediction.centroid_anomaly_visualization_forecast import plot_behavior_change_3d
from src.trajectory_dashboard.config import ANOMALY_FLAG_MAPPING, ANOMALY_SCORE_THRESHOLD
from src.trajectory_dashboard.utils.anomaly_dashboard_utils import (
    filter_out_overlapping_historical_records,
    parse_selected_anomalous_cluster_week,
    get_decoded_behavior,
    compare_behavioral_changes,
    display_elapsed_time_changes,
    display_feature_usage_changes,
    prepare_decoded_behavior_long_df,
    plot_cluster_decoded_behavior_stacked_bar)


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


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Anomaly Analysis",
    layout="wide",
)

st.title("Behavioral Anomaly Analysis")


st.write(
    "This page is dedicated for the analysis of the detected behavioral anomalies."
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

trajectory_data_df = loaded_results["trajectory_data"]
decoded_data_df = loaded_results["decoded_data"]
dashboard_paths = loaded_results["paths"]
missing_files = loaded_results["missing_files"]
missing_columns = loaded_results["missing_columns"]

trajectory_data_df = filter_out_overlapping_historical_records(trajectory_data_df)

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


if trajectory_data_df is None or decoded_data_df.empty:
    st.error(
        "No trajectory results were available."
    )
    st.stop()
    
#======================
# Display Page for Behavioral Anomaly Overview
#======================

st.divider()

st.markdown("### Behavioral Anomaly Overview")

# Filter trajectory data records with >=3 anomaly scores
behavior_scores = pd.to_numeric(
    trajectory_data_df.get(
        "behavior_change_score",
        pd.Series(
            index=trajectory_data_df.index,
            dtype=float,
        ),
    ),
    errors="coerce",
)

filtered_anomalies = trajectory_data_df.loc[
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
        "Anomalies on Predicted Weeks",
        f"{predicted_anomaly_count:,} Anomalies",
    )
    st.caption(f"Anomalies on Historical Weeks: {historical_anomaly_count} anomalies.")
    st.caption(f"Total Detected Anomalies: {total_anomaly_count} anomalies.")
        

with anomaly_col2:
    st.metric(
        "Affected Clusters in Predicted Weeks",
        f"{affected_prediction_cluster_count:,} Clusters",
    )
    st.caption(f"Affected Clusters in Historical Weeks: {affected_historical_cluster_count} clusters.")
    st.caption(f"Total Affected Clusters: {affected_cluster_count} clusters.")

st.divider()

# =========================================================
# Displays the Highest-priority anomalies DF selection
# =========================================================

if filtered_anomalies.empty:
    st.info(
        "No anomalies were found for the selected filters."
    )

else:
    col1, col2 = st.columns([4,1])
    
    with col1:
        st.markdown("#### High Priority Behavioral Anomalies")
    with col2:
        is_show_historical_anomalies = st.checkbox('Include Historical Anomalies',help='Enable to view the anomalies on the historical weeks.')
        
    st.caption("""
        The list below shows detected behavioral anomalies by priority (only anomalies on the predicted weeks are shown by default).
        Changes in feature usage and usage time are prioritized, while centroid movement is used only as a diagnostic signal.
        Select an anomalous week using the checkbox on the left to view detailed information about each detected anomaly.
    """)

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

    # Shows anomalous week selection DF of the historical and predicted anomalies with toggle to remove the historical anomalies
    prediction_anomaly_table = selection_of_detected_anomalies.copy()
    
    if is_show_historical_anomalies:
        selected_cluster_week_anomaly = st.dataframe(
            selection_of_detected_anomalies,
            use_container_width=True,
            hide_index=True,
            height=245,
            on_select="rerun",
            selection_mode="single-row",
        )
        
    else: # Shows predicted anomalies only
        # Filters the anomaly table to retain only those behaviorally anomalous predicted weeks
        prediction_anomaly_table = prediction_anomaly_table.loc[
                prediction_anomaly_table["Source"] == "Predicted"
            ]
        
        selected_cluster_week_anomaly = st.dataframe(
            prediction_anomaly_table,
            use_container_width=True,
            hide_index=True,
            height=245,
            on_select="rerun",
            selection_mode="single-row",
        )
        
    st.divider()
    
    if len(selected_cluster_week_anomaly.selection.rows): # Checks if there is a selected week for anomaly analysis
        
        # Extracts the selected week's cluster name, year-week, and source (historical or predicted week)
        selected_cluster_name, selected_year_week, selected_source = parse_selected_anomalous_cluster_week(selected_cluster_week_anomaly, prediction_anomaly_table)
        
        st.markdown(f'### Cluster Behavior Anomaly Analysis')

        st.caption(f"Cluster: {selected_cluster_name} | Selected Year-Week: {selected_year_week}")
    
        # Filters the trajectory data to extract only the historical and predicted data of only the selected cluster
        selected_cluster_week_historical_data_df = trajectory_data_df.copy()
        selected_cluster_week_historical_data_df = selected_cluster_week_historical_data_df[selected_cluster_week_historical_data_df['final_cluster_label'] == selected_cluster_name]
        
        # Retrieves the specific year and week of the selected cluster anomaly record
        selected_cluster_week_anomaly_data_df = selected_cluster_week_historical_data_df[
            (selected_cluster_week_historical_data_df['year_week'] == selected_year_week) &
            (selected_cluster_week_historical_data_df['source'] == selected_source)
        ]
        
        # Parses a list of triggered anomaly flags from the retrived anomaly record
        triggered_flags = str(selected_cluster_week_anomaly_data_df['anomaly_priority_label'].item()).split(' | ')
        
        # Retrieves the decoded behavior of both the selected week and the week preceeding it for comparison
        query_record, previous_record = get_decoded_behavior(
            decoded_data_df,
            selected_cluster_name,
            selected_year_week,
            selected_source
        )
        
        # Compares and generates the behavioral changes of the selected anomalous week and week before it
        added, removed, elapsed_changes_df, elapsed_change_dict = compare_behavioral_changes(previous_record, query_record)
        
        def print_anomaly_interpretation(
            selected_cluster_name,
            selected_year_week,
            triggered_flags,
        ):
            has_elapsed = has_features = False
        
            for flags in triggered_flags:
                anomaly = ANOMALY_FLAG_MAPPING[flags].split()
                if bool("Elapsed" in anomaly):
                    has_elapsed = True
                if bool("Features" in anomaly):
                    has_features = True
            
            change = None
            if has_elapsed and has_features:
                change = 'change in feature usage composition and feature usage elapsed time'
            elif has_elapsed and not has_features:
                change = 'change in feature usage elapsed time'
            elif has_features and not has_elapsed:
                change = 'change in feature usage composition'
            else:
                change = 'UNDEFINED'
            
            # selected_cluster_name, selected_year_week, selected_source
            st.markdown(f"""
                **Cluster** ***{selected_cluster_name}***'s ***{selected_year_week}*** **week** has been marked as anomalous due to
                a ***{change}***.
            """)
            
        print_anomaly_interpretation(
            selected_cluster_name,
            selected_year_week,
            triggered_flags
        )
        
        # Displays the behavioral changes of the selected anomalous week and week preceeding it
        for flags in triggered_flags:
            
            anomaly = ANOMALY_FLAG_MAPPING[flags].split()

            has_elapsed = "Elapsed" in anomaly
            has_features = "Features" in anomaly

            # Displays detected behavior change
            if has_elapsed and has_features: # Displays behavioral changes in two columns if changes in features and elapsed time are present
                
                col1, col2 = st.columns(2)
                with col1:
                    display_elapsed_time_changes(elapsed_change_dict, query_record, added, cards_per_row=3)

                with col2:
                    display_feature_usage_changes(added, removed)

            else:
                if has_elapsed:
                    display_elapsed_time_changes(elapsed_change_dict, query_record, added, cards_per_row=3)

                if has_features:
                    display_feature_usage_changes(added, removed)

            # A diagnostic indicator for detected movement anomalies.
            if "Movement" in anomaly and not has_elapsed and "No" not in anomaly:
                st.caption(f"Diagnostics: There's a detected movement change ({ANOMALY_FLAG_MAPPING[flags]})")
            
            
        # ======================================================
        # Visualization of Feature Usages (Stacked Bar Chart)
        # ======================================================
        with st.spinner("Visualizing Cluster's Feature Usage Composition..."):
            
            # Seperates the historical and predicted weeks before visualization
            historical_behavior_change_df = selected_cluster_week_historical_data_df[selected_cluster_week_historical_data_df['source']=="Historical"]
            predicted_behavior_change_df = selected_cluster_week_historical_data_df[selected_cluster_week_historical_data_df['source']=="Predicted"]
            
            # Builds and decoded behavior column for faster feature usage and elapsed time visualization
            historical_behavior_change_df, predicted_behavior_change_df = build_decoded_behavior_column(
                historical_behavior_change_df,
                predicted_behavior_change_df,
                decoded_data_df
            )

            # Formats the decoded behavior data into a `centroid-week-feature` usage row format.
            concat_behavior_change_df = pd.concat([historical_behavior_change_df, predicted_behavior_change_df], ignore_index=True)
            decoded_behavior_long_df = prepare_decoded_behavior_long_df(
                concat_behavior_change_df
            )
        
            # Plots the feature usage of a cluster with anomaly highlight only on the selected year-week
            fig, week_order = plot_cluster_decoded_behavior_stacked_bar(
                decoded_long_df=decoded_behavior_long_df,
                cluster_label=selected_cluster_name,
                anomaly_score_threshold=ANOMALY_SCORE_THRESHOLD,
                elapsed_unit="seconds",
                top_n_features=10,
                latest_n_weeks=52,
                year=selected_year_week.split("-")[0],
                week=selected_year_week.split("W")[1]
            )
        
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "displayModeBar": True,
                    "responsive": True,
                },
            )
        
        #==================================================================================
        # Visualization of Centroid Movement (Centroid Trajectory) with Decoded Behavior
        #==================================================================================
        
        with st.spinner("Visualizing Cluster's Movement Trajectory..."):
                        
            # Filters the selected cluster's trajectory data to only include weeks visualized in the last visualization
            cluster_filtered_visualization = selected_cluster_week_historical_data_df[selected_cluster_week_historical_data_df['year_week'].isin(week_order)]

            # Separates the historical and predicted week rows then adds the decoded behavior column
            historical_behavior_change_df = cluster_filtered_visualization[cluster_filtered_visualization['source']=="Historical"]
            predicted_behavior_change_df = cluster_filtered_visualization[cluster_filtered_visualization['source']=="Predicted"]
            
            historical_behavior_change_df, predicted_behavior_change_df = build_decoded_behavior_column(
                historical_behavior_change_df,
                predicted_behavior_change_df,
                decoded_data_df
            )

            # Plots the centroid trajector of a cluster with anomaly highlight only on the selected year-week
            trajectory_figure = plot_behavior_change_3d(
                historical_behavior_change_df=historical_behavior_change_df,
                predicted_behavior_change_df=predicted_behavior_change_df,
                anomaly_score_threshold=3,
                title=f"PCA Centroid Behavior Change 3D",
                animate_by_week=True,
                year=selected_year_week.split("-")[0],
                week=selected_year_week.split("W")[1]
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
        
##############################
    
# st.divider()
# st.markdown("## *DEVELOPER DF REFERENCE ONLY*")

# st.caption("the selected week anomaly data")
# st.dataframe(
#     selected_cluster_week_anomaly_data_df
# )

# st.caption("previous week decoded behavior")
# st.dataframe(
#     previous_record 
# )
# st.caption("query decoded behavior")
# st.dataframe(
#     query_record 
# )

# st.caption("selected_cluster_week_historical_data_df")
# st.dataframe(
#     selected_cluster_week_historical_data_df,
#     use_container_width=True,
#     hide_index=True,
#     height=245,
# )

# decoded_data_available = (
#     decoded_data_df is not None
#     and not decoded_data_df.empty
# )

# st.caption("trajectory_data")
# st.dataframe(
#     trajectory_data_df,
#     use_container_width=True,
#     hide_index=True,
# )

# st.caption("decoded_data")
# st.dataframe(
#     decoded_data_df,
#     use_container_width=True,
#     hide_index=True,
# )