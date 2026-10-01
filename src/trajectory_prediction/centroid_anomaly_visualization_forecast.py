import ast
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# =========================================================
# Data preparation helpers
# =========================================================

REQUIRED_COLS = [
    "year",
    "week",
    "final_cluster_label",
    "pca_0",
    "pca_1",
    "decoded_behavior",
    "behavior_change_score",
    "behavior_change_type",
    "triggered_main_flags", 
    "anomaly_priority_level",
]


def _normalize_week_start(value):
    """Return a timezone-naive, normalized pandas Timestamp."""
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not interpret {value!r} as a weekly timestamp."
        ) from exc

    if pd.isna(timestamp):
        raise ValueError("Weekly timestamp cannot be missing.")

    # Preserve the represented calendar date while removing timezone metadata.
    timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _to_week_start(year, week):
    return _normalize_week_start(
        pd.Timestamp.fromisocalendar(int(year), int(week), 1)
    )


def _week_label(year, week):
    return f"{int(year)}_{int(week):02d}"


def _resolve_focus_week_start(year=None, week=None):
    """
    Convert optional year/week inputs into the Monday of the ISO week.

    Both values must be supplied together. ``week`` accepts values such as
    1, "1", "W01", or "2026-W01".
    """
    if year is None and week is None:
        return None

    if year is None or week is None:
        raise ValueError("year and week must either both be provided or both be None.")

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
            focus_week = int(week_text.removeprefix("W"))
        except ValueError as exc:
            raise ValueError(
                "week must be an integer-like value, 'W01', or '2026-W01'."
            ) from exc

    if not 1 <= focus_week <= 53:
        raise ValueError("week must be between 1 and 53.")

    try:
        return _to_week_start(focus_year, focus_week)
    except ValueError as exc:
        raise ValueError(
            f"{focus_year}-W{focus_week:02d} is not a valid ISO year-week."
        ) from exc


def _make_color_map(groups):
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Alphabet
    )
    return {group: palette[i % len(palette)] for i, group in enumerate(groups)}


def _get_z_col(df_history, df_predicted):
    """
    Prefer pca_3 when supplied, as requested. Fall back to pca_2 because the
    stated input schema contains pca_0, pca_1, and pca_2.
    """
    common_cols = set(df_history.columns).intersection(df_predicted.columns)
    if "pca_3" in common_cols:
        return "pca_3"
    if "pca_2" in common_cols:
        return "pca_2"
    raise ValueError(
        "Both dataframes must contain either 'pca_3' or 'pca_2' for 3D plotting."
    )


def _parse_decoded_behavior(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue

    return []


def _format_decoded_behavior(value):
    items = _parse_decoded_behavior(value)
    if not items:
        raw = html.escape(str(value)) if value is not None else ""
        return raw or "No decoded behavior"

    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue

        feature = html.escape(str(item.get("feature", "unknown")))
        used_prob = item.get("used_prob", item.get("pred_used_prob"))
        elapsed = item.get("elapsed", item.get("pred_elapsed"))

        used_prob_text = (
            f"{float(used_prob):.3f}" if pd.notna(used_prob) else "N/A"
        )
        elapsed_text = f"{float(elapsed):.3f}" if pd.notna(elapsed) else "N/A"

        lines.append(
            f"<b>{feature}</b> | used_prob={used_prob_text} | elapsed={elapsed_text}"
        )

    return "<br>".join(lines) if lines else "No decoded behavior"


def _prepare_behavior_df(df, source, clusters=None, max_weeks_per_cluster=None):
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{source} dataframe is missing columns: {missing}")

    keep_cols = REQUIRED_COLS.copy()
    if "pca_2" in df.columns:
        keep_cols.append("pca_2")
    if "pca_3" in df.columns:
        keep_cols.append("pca_3")

    prepared = df.loc[:, list(dict.fromkeys(keep_cols))].copy()

    if clusters is not None:
        prepared = prepared[
            prepared["final_cluster_label"].isin(clusters)
        ].copy()

    prepared["week_start"] = pd.to_datetime(
        [
            _to_week_start(year, week)
            for year, week in zip(prepared["year"], prepared["week"])
        ],
        errors="raise",
    )
    prepared["week_label"] = [
        _week_label(year, week)
        for year, week in zip(prepared["year"], prepared["week"])
    ]
    prepared["source"] = source
    prepared["decoded_behavior_hover"] = prepared["decoded_behavior"].map(
        _format_decoded_behavior
    )

    prepared = prepared.sort_values(["final_cluster_label", "week_start"])

    if max_weeks_per_cluster is not None:
        prepared = (
            prepared.groupby("final_cluster_label", group_keys=False)
            .tail(max_weeks_per_cluster)
        )

    return prepared.reset_index(drop=True)


def _get_slider_weeks(df_history, df_predicted):
    weeks = pd.concat(
        [
            df_history[["week_start", "week_label"]],
            df_predicted[["week_start", "week_label"]],
        ],
        ignore_index=True,
    )

    if weeks.empty:
        return [], {}

    weeks["week_start"] = pd.to_datetime(
        weeks["week_start"],
        errors="raise",
    ).map(_normalize_week_start)

    weeks = (
        weeks.drop_duplicates("week_start")
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    # Explicit conversion avoids numpy.datetime64 / datetime.date values from
    # reaching the trace builders on different pandas versions.
    values = [
        _normalize_week_start(value)
        for value in weeks["week_start"]
    ]
    labels = {
        _normalize_week_start(value): label
        for value, label in zip(
            weeks["week_start"],
            weeks["week_label"],
        )
    }
    return values, labels


def _mark_history_prediction_overlap(df_history, df_predicted):
    """
    A historical row overlaps if there is a prediction for the same:
    final_cluster_label + week_start.

    Overlapping historical rows are still shown, but with lower opacity.
    """
    pred_keys = set(
        zip(
            df_predicted["final_cluster_label"].astype(str),
            df_predicted["week_start"],
        )
    )

    df = df_history.copy()

    df["overlaps_prediction"] = [
        (str(cluster), week_start) in pred_keys
        for cluster, week_start in zip(
            df["final_cluster_label"],
            df["week_start"],
        )
    ]

    return df


def _split_path_by_overlap_2d(sub):
    """
    Splits historical path into normal and overlapping segments.

    A segment is considered overlapping if either endpoint overlaps a prediction week.
    """
    normal_x, normal_y = [], []
    overlap_x, overlap_y = [], []

    if len(sub) < 2:
        return normal_x, normal_y, overlap_x, overlap_y

    rows = list(sub.itertuples(index=False))

    for a, b in zip(rows[:-1], rows[1:]):
        segment_overlaps = bool(a.overlaps_prediction) or bool(b.overlaps_prediction)

        x_pair = [a.pca_0, b.pca_0, None]
        y_pair = [a.pca_1, b.pca_1, None]

        if segment_overlaps:
            overlap_x.extend(x_pair)
            overlap_y.extend(y_pair)
        else:
            normal_x.extend(x_pair)
            normal_y.extend(y_pair)

    return normal_x, normal_y, overlap_x, overlap_y


def _split_path_by_overlap_3d(sub, z_col):
    normal_x, normal_y, normal_z = [], [], []
    overlap_x, overlap_y, overlap_z = [], [], []

    if len(sub) < 2:
        return normal_x, normal_y, normal_z, overlap_x, overlap_y, overlap_z

    rows = list(sub.itertuples(index=False))

    for a, b in zip(rows[:-1], rows[1:]):
        segment_overlaps = bool(a.overlaps_prediction) or bool(b.overlaps_prediction)

        x_pair = [a.pca_0, b.pca_0, None]
        y_pair = [a.pca_1, b.pca_1, None]
        z_pair = [getattr(a, z_col), getattr(b, z_col), None]

        if segment_overlaps:
            overlap_x.extend(x_pair)
            overlap_y.extend(y_pair)
            overlap_z.extend(z_pair)
        else:
            normal_x.extend(x_pair)
            normal_y.extend(y_pair)
            normal_z.extend(z_pair)

    return normal_x, normal_y, normal_z, overlap_x, overlap_y, overlap_z


def _last_history_before_first_forecast(df_history, cluster, first_forecast_week):
    """
    Returns the last historical centroid before the first predicted week.
    This anchors the predicted path to the last real historical point.
    """
    hist = df_history[
        (df_history["final_cluster_label"] == cluster)
        & (df_history["week_start"] < first_forecast_week)
    ].sort_values("week_start")

    if len(hist) == 0:
        return None

    return hist.iloc[-1]


def _customdata(df):
    if df.empty:
        return np.empty((0, 7), dtype=object)
    return np.column_stack(
        [
            df["week_label"].astype(str).to_numpy(),
            df["final_cluster_label"].astype(str).to_numpy(),
            df["behavior_change_type"].astype(str).to_numpy(),
            df["triggered_main_flags"].astype(str).to_numpy(),
            df["decoded_behavior_hover"].astype(str).to_numpy(),
            df["source"].astype(str).to_numpy(),
            df["anomaly_priority_level"].to_numpy(),
        ]
    )


def _select_anomalous_rows(
    combined_df,
    anomaly_score_threshold,
    focus_week_start=None,
):
    """
    Return anomalous rows, optionally restricted to one focused ISO week.

    When ``focus_week_start`` is None, all anomalous weeks are returned,
    preserving the original visualization behavior.
    """
    scores = pd.to_numeric(
        combined_df["behavior_change_score"],
        errors="coerce",
    )
    anomaly_mask = scores >= anomaly_score_threshold

    if focus_week_start is not None:
        focus_week_start = _normalize_week_start(focus_week_start)
        normalized_weeks = pd.to_datetime(
            combined_df["week_start"],
            errors="coerce",
        ).map(
            lambda value: (
                _normalize_week_start(value)
                if pd.notna(value)
                else pd.NaT
            )
        )
        anomaly_mask &= normalized_weeks.eq(focus_week_start)

    return combined_df.loc[anomaly_mask].copy()


# =========================================================
# Trace builders
# =========================================================


def _build_2d_traces(
    df_history,
    df_predicted,
    clusters,
    color_map,
    anomaly_score_threshold,
    upto_week_start=None,
    focus_week_start=None,
):
    traces = []
    upto_week_start = _normalize_week_start(upto_week_start)
    focus_week_start = _normalize_week_start(focus_week_start)

    for cluster in clusters:
        # -------------------------
        # Historical traces
        # -------------------------
        hist_sub = (
            df_history[df_history["final_cluster_label"] == cluster]
            .sort_values("week_start")
        )

        if upto_week_start is not None:
            hist_sub = hist_sub.loc[pd.to_datetime(hist_sub["week_start"]) <= upto_week_start]

        normal_x, normal_y, overlap_x, overlap_y = _split_path_by_overlap_2d(hist_sub)

        traces.append(
            go.Scatter(
                x=normal_x,
                y=normal_y,
                mode="lines",
                name=f"{cluster} - historical path",
                legendgroup=f"{cluster}-Historical",
                line=dict(color=color_map[cluster], width=2, dash="solid"),
                opacity=1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        traces.append(
            go.Scatter(
                x=overlap_x,
                y=overlap_y,
                mode="lines",
                name=f"{cluster} - historical path overlap",
                legendgroup=f"{cluster}-Historical",
                line=dict(color=color_map[cluster], width=2, dash="solid"),
                opacity=0.5,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        normal_hist = hist_sub[~hist_sub["overlaps_prediction"]] if len(hist_sub) else hist_sub
        overlap_hist = hist_sub[hist_sub["overlaps_prediction"]] if len(hist_sub) else hist_sub

        for sub, opacity, showlegend in [
            (normal_hist, 1.0, True),
            (overlap_hist, 0.5, False),
        ]:
            traces.append(
                go.Scatter(
                    x=sub["pca_0"],
                    y=sub["pca_1"],
                    mode="markers",
                    name=f"{cluster} - Historical centroid",
                    legendgroup=f"{cluster}-Historical",
                    marker=dict(
                        color=color_map[cluster],
                        size=8,
                        symbol="circle",
                        opacity=opacity,
                        line=dict(color=color_map[cluster], width=2),
                    ),
                    customdata=_customdata(sub),
                    hovertemplate=(
                    "<b>%{customdata[5]} centroid</b><br>"
                    "Cluster: %{customdata[1]}<br>"
                    "Week: %{customdata[0]}<br><br>"

                    "<b>Anomaly summary</b><br>"
                    "Priority level: %{customdata[6]}<br>"
                    "Triggered main flags: %{customdata[3]}<br><br>"

                    "pca_0: %{x:.5f}<br>"
                    "pca_1: %{y:.5f}<br><br>"

                    "<b>Decoded behavior</b><br>"
                    "%{customdata[4]}"

                    "<extra></extra>"
                ),
                    showlegend=showlegend,
                )
            )

        # -------------------------
        # Predicted traces
        # -------------------------
        pred_sub = (
            df_predicted[df_predicted["final_cluster_label"] == cluster]
            .sort_values("week_start")
        )

        if upto_week_start is not None:
            pred_sub = pred_sub.loc[pd.to_datetime(pred_sub["week_start"]) <= upto_week_start]

        path_x, path_y = [], []

        if len(pred_sub) > 0:
            first_forecast_week = pred_sub["week_start"].min()
            anchor = _last_history_before_first_forecast(
                df_history,
                cluster,
                first_forecast_week,
            )

            if anchor is not None:
                path_x.append(anchor["pca_0"])
                path_y.append(anchor["pca_1"])

            path_x.extend(pred_sub["pca_0"].tolist())
            path_y.extend(pred_sub["pca_1"].tolist())

        traces.append(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines",
                name=f"{cluster} - predicted path",
                legendgroup=f"{cluster}-Predicted",
                line=dict(color=color_map[cluster], width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        traces.append(
            go.Scatter(
                x=pred_sub["pca_0"],
                y=pred_sub["pca_1"],
                mode="markers",
                name=f"{cluster} - Predicted centroid",
                legendgroup=f"{cluster}-Predicted",
                marker=dict(
                    color=color_map[cluster],
                    size=10,
                    symbol="diamond-open",
                    line=dict(color=color_map[cluster], width=2),
                ),
                customdata=_customdata(pred_sub),
                hovertemplate=(
                "<b>%{customdata[5]} centroid</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Week: %{customdata[0]}<br><br>"

                "<b>Anomaly summary</b><br>"
                "Priority level: %{customdata[6]}<br>"
                "Triggered main flags: %{customdata[3]}<br><br>"

                "pca_0: %{x:.5f}<br>"
                "pca_1: %{y:.5f}<br><br>"

                "<b>Decoded behavior</b><br>"
                "%{customdata[4]}"

                "<extra></extra>"
            ),
                showlegend=True,
            )
        )

        # -------------------------
        # Anomaly rings
        # -------------------------
        combined_sub = pd.concat([hist_sub, pred_sub], ignore_index=True)
        anomalous = _select_anomalous_rows(
            combined_sub,
            anomaly_score_threshold,
            focus_week_start=focus_week_start,
        )

        traces.append(
            go.Scatter(
                x=anomalous["pca_0"],
                y=anomalous["pca_1"],
                mode="markers",
                name="Anomalous centroid-week",
                legendgroup="anomaly",
                opacity=0.5,
                marker=dict(
                    size=18,
                    symbol="circle-open",
                    color="red",
                    line=dict(color="red", width=4),
                ),
                customdata=_customdata(anomalous),
                hovertemplate=(
                    "<b>ANOMALOUS CENTROID-WEEK</b><br>"
                    "Source: %{customdata[5]}<br>"
                    "Cluster: %{customdata[1]}<br>"
                    "Week: %{customdata[0]}<br><br>"

                    "<b>Anomaly summary</b><br>"
                    "Priority level: %{customdata[6]}<br>"
                    "Triggered main flags: %{customdata[3]}<br><br>"

                    "<b>Decoded behavior</b><br>"
                    "%{customdata[4]}"

                    "<extra></extra>"
                ),
                showlegend=(cluster == clusters[0]),
            )
        )

    return traces


def _build_3d_traces(
    df_history,
    df_predicted,
    clusters,
    color_map,
    z_col,
    anomaly_score_threshold,
    upto_week_start=None,
    focus_week_start=None,
):
    traces = []
    upto_week_start = _normalize_week_start(upto_week_start)
    focus_week_start = _normalize_week_start(focus_week_start)

    for cluster in clusters:
        # -------------------------
        # Historical traces
        # -------------------------
        hist_sub = (
            df_history[df_history["final_cluster_label"] == cluster]
            .sort_values("week_start")
        )

        if upto_week_start is not None:
            hist_sub = hist_sub.loc[pd.to_datetime(hist_sub["week_start"]) <= upto_week_start]

        (
            normal_x,
            normal_y,
            normal_z,
            overlap_x,
            overlap_y,
            overlap_z,
        ) = _split_path_by_overlap_3d(hist_sub, z_col)

        traces.append(
            go.Scatter3d(
                x=normal_x,
                y=normal_y,
                z=normal_z,
                mode="lines",
                name=f"{cluster} - historical path",
                legendgroup=f"{cluster}-Historical",
                line=dict(color=color_map[cluster], width=4, dash="solid"),
                opacity=1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        traces.append(
            go.Scatter3d(
                x=overlap_x,
                y=overlap_y,
                z=overlap_z,
                mode="lines",
                name=f"{cluster} - historical path overlap",
                legendgroup=f"{cluster}-Historical",
                line=dict(color=color_map[cluster], width=4, dash="solid"),
                opacity=0.5,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        normal_hist = hist_sub[~hist_sub["overlaps_prediction"]] if len(hist_sub) else hist_sub
        overlap_hist = hist_sub[hist_sub["overlaps_prediction"]] if len(hist_sub) else hist_sub

        for sub, opacity, showlegend in [
            (normal_hist, 1.0, True),
            (overlap_hist, 0.5, False),
        ]:
            traces.append(
                go.Scatter3d(
                    x=sub["pca_0"],
                    y=sub["pca_1"],
                    z=sub[z_col],
                    mode="markers",
                    name=f"{cluster} - Historical centroid",
                    legendgroup=f"{cluster}-Historical",
                    marker=dict(
                        color=color_map[cluster],
                        size=5,
                        symbol="circle",
                        opacity=opacity,
                        line=dict(color=color_map[cluster], width=2),
                    ),
                    customdata=_customdata(sub),
                    hovertemplate=(
                    "<b>%{customdata[5]} centroid</b><br>"
                    "Cluster: %{customdata[1]}<br>"
                    "Week: %{customdata[0]}<br><br>"

                    "<b>Anomaly summary</b><br>"
                    "Priority level: %{customdata[6]}<br>"
                    "Triggered main flags: %{customdata[3]}<br><br>"

                    "pca_0: %{x:.5f}<br>"
                    "pca_1: %{y:.5f}<br>"
                    f"{z_col}: %{{z:.5f}}<br><br>"

                    "<b>Decoded behavior</b><br>"
                    "%{customdata[4]}"

                    "<extra></extra>"
                ),
                    showlegend=showlegend,
                )
            )

        # -------------------------
        # Predicted traces
        # -------------------------
        pred_sub = (
            df_predicted[df_predicted["final_cluster_label"] == cluster]
            .sort_values("week_start")
        )

        if upto_week_start is not None:
            pred_sub = pred_sub.loc[pd.to_datetime(pred_sub["week_start"]) <= upto_week_start]

        path_x, path_y, path_z = [], [], []

        if len(pred_sub) > 0:
            first_forecast_week = pred_sub["week_start"].min()
            anchor = _last_history_before_first_forecast(
                df_history,
                cluster,
                first_forecast_week,
            )

            if anchor is not None:
                path_x.append(anchor["pca_0"])
                path_y.append(anchor["pca_1"])
                path_z.append(anchor[z_col])

            path_x.extend(pred_sub["pca_0"].tolist())
            path_y.extend(pred_sub["pca_1"].tolist())
            path_z.extend(pred_sub[z_col].tolist())

        traces.append(
            go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="lines",
                name=f"{cluster} - predicted path",
                legendgroup=f"{cluster}-Predicted",
                line=dict(color=color_map[cluster], width=4, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        traces.append(
            go.Scatter3d(
                x=pred_sub["pca_0"],
                y=pred_sub["pca_1"],
                z=pred_sub[z_col],
                mode="markers",
                name=f"{cluster} - Predicted centroid",
                legendgroup=f"{cluster}-Predicted",
                marker=dict(
                    color=color_map[cluster],
                    size=7,
                    symbol="diamond-open",
                    line=dict(color=color_map[cluster], width=2),
                ),
                customdata=_customdata(pred_sub),
                hovertemplate=(
                "<b>%{customdata[5]} centroid</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Week: %{customdata[0]}<br><br>"

                "<b>Anomaly summary</b><br>"
                "Priority level: %{customdata[6]}<br>"
                "Triggered main flags: %{customdata[3]}<br><br>"

                "pca_0: %{x:.5f}<br>"
                "pca_1: %{y:.5f}<br>"
                f"{z_col}: %{{z:.5f}}<br><br>"

                "<b>Decoded behavior</b><br>"
                "%{customdata[4]}"

                "<extra></extra>"
            ),
                showlegend=True,
            )
        )

        # -------------------------
        # Anomaly rings
        # -------------------------
        combined_sub = pd.concat([hist_sub, pred_sub], ignore_index=True)
        anomalous = _select_anomalous_rows(
            combined_sub,
            anomaly_score_threshold,
            focus_week_start=focus_week_start,
        )

        traces.append(
            go.Scatter3d(
                x=anomalous["pca_0"],
                y=anomalous["pca_1"],
                z=anomalous[z_col],
                mode="markers",
                name="Anomalous centroid-week",
                legendgroup="anomaly",
                opacity=0.5,
                marker=dict(
                    size=11,
                    symbol="circle-open",
                    color="red",
                    line=dict(color="red", width=5),
                ),
                customdata=_customdata(anomalous),
                hovertemplate=(
                "<b>ANOMALOUS CENTROID-WEEK</b><br>"
                "Source: %{customdata[5]}<br>"
                "Cluster: %{customdata[1]}<br>"
                "Week: %{customdata[0]}<br><br>"

                "<b>Anomaly summary</b><br>"
                "Priority level: %{customdata[6]}<br>"
                "Triggered main flags: %{customdata[3]}<br><br>"

                "<b>Decoded behavior</b><br>"
                "%{customdata[4]}"

                "<extra></extra>"
            ),
                showlegend=(cluster == clusters[0]),
            )
        )

    return traces


# =========================================================
# Main plot functions
# =========================================================


def plot_behavior_change_2d(
    historical_behavior_change_df,
    predicted_behavior_change_df,
    clusters=None,
    anomaly_score_threshold=3,
    max_history_weeks_per_cluster=None,
    title="Centroid behavior-change trajectory (2D)",
    animate_by_week=True,
    year=None,
    week=None,
):
    """
    Plot the 2D historical and predicted centroid trajectories.

    When ``year`` and ``week`` are supplied, only that week is eligible for
    an anomaly ring, and the animation initially opens on that week. When
    both are None, all anomalous weeks are marked as in the original version.
    """
    history = _prepare_behavior_df(
        historical_behavior_change_df,
        source="Historical",
        clusters=clusters,
        max_weeks_per_cluster=max_history_weeks_per_cluster,
    )
    predicted = _prepare_behavior_df(
        predicted_behavior_change_df,
        source="Predicted",
        clusters=clusters,
    )

    history = _mark_history_prediction_overlap(history, predicted)

    cluster_groups = sorted(
        set(history["final_cluster_label"].dropna()).union(
            predicted["final_cluster_label"].dropna()
        ),
        key=str,
    )
    if not cluster_groups:
        raise ValueError("No rows are available after filtering.")

    color_map = _make_color_map(cluster_groups)
    slider_weeks, slider_labels = _get_slider_weeks(history, predicted)
    focus_week_start = _resolve_focus_week_start(year=year, week=week)

    if focus_week_start is not None and focus_week_start not in slider_weeks:
        raise ValueError(
            f"Focused week {focus_week_start.isocalendar().year}-W"
            f"{focus_week_start.isocalendar().week:02d} was not found after "
            "applying the cluster and history filters."
        )

    if focus_week_start is not None:
        initial_week = focus_week_start
    else:
        initial_week = slider_weeks[0] if animate_by_week else slider_weeks[-1]

    initial_slider_index = slider_weeks.index(initial_week)

    def frame_data(current_week):
        return _build_2d_traces(
            history,
            predicted,
            cluster_groups,
            color_map,
            anomaly_score_threshold,
            upto_week_start=current_week,
            focus_week_start=focus_week_start,
        )

    fig = go.Figure(data=frame_data(initial_week))

    if animate_by_week:
        fig.frames = [
            go.Frame(
                data=frame_data(week),
                name=slider_labels[week],
                layout=go.Layout(
                    title_text=f"{title} | Week {slider_labels[week]}"
                ),
            )
            for week in slider_weeks
        ]
        fig.update_layout(
            sliders=[{
                "active": initial_slider_index,
                "currentvalue": {"prefix": "Week: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "method": "animate",
                        "label": slider_labels[week],
                        "args": [
                            [slider_labels[week]],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for week in slider_weeks
                ],
            }]
        )

    fig.update_layout(
        title=f"{title} | Week {slider_labels[initial_week]}",
        xaxis_title="pca_0",
        yaxis_title="pca_1",
        showlegend=True,
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_behavior_change_3d(
    historical_behavior_change_df,
    predicted_behavior_change_df,
    clusters=None,
    anomaly_score_threshold=3,
    max_history_weeks_per_cluster=None,
    title="Centroid behavior-change trajectory (3D)",
    animate_by_week=True,
    year=None,
    week=None,
):
    """
    Plot the 3D historical and predicted centroid trajectories.

    When ``year`` and ``week`` are supplied, only that week is eligible for
    an anomaly ring, and the animation initially opens on that week. When
    both are None, all anomalous weeks are marked as in the original version.
    """
    z_col = _get_z_col(
        historical_behavior_change_df,
        predicted_behavior_change_df,
    )
    history = _prepare_behavior_df(
        historical_behavior_change_df,
        source="Historical",
        clusters=clusters,
        max_weeks_per_cluster=max_history_weeks_per_cluster,
    )
    predicted = _prepare_behavior_df(
        predicted_behavior_change_df,
        source="Predicted",
        clusters=clusters,
    )

    history = _mark_history_prediction_overlap(history, predicted)

    cluster_groups = sorted(
        set(history["final_cluster_label"].dropna()).union(
            predicted["final_cluster_label"].dropna()
        ),
        key=str,
    )
    if not cluster_groups:
        raise ValueError("No rows are available after filtering.")

    color_map = _make_color_map(cluster_groups)
    slider_weeks, slider_labels = _get_slider_weeks(history, predicted)
    focus_week_start = _resolve_focus_week_start(year=year, week=week)

    if focus_week_start is not None and focus_week_start not in slider_weeks:
        raise ValueError(
            f"Focused week {focus_week_start.isocalendar().year}-W"
            f"{focus_week_start.isocalendar().week:02d} was not found after "
            "applying the cluster and history filters."
        )

    if focus_week_start is not None:
        initial_week = focus_week_start
    else:
        initial_week = slider_weeks[0] if animate_by_week else slider_weeks[-1]

    initial_slider_index = slider_weeks.index(initial_week)

    def frame_data(current_week):
        return _build_3d_traces(
            history,
            predicted,
            cluster_groups,
            color_map,
            z_col,
            anomaly_score_threshold,
            upto_week_start=current_week,
            focus_week_start=focus_week_start,
        )

    fig = go.Figure(data=frame_data(initial_week))

    if animate_by_week:
        fig.frames = [
            go.Frame(
                data=frame_data(week),
                name=slider_labels[week],
                layout=go.Layout(
                    title_text=f"{title} | Week {slider_labels[week]}"
                ),
            )
            for week in slider_weeks
        ]
        fig.update_layout(
            sliders=[{
                "active": initial_slider_index,
                "currentvalue": {"prefix": "Week: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "method": "animate",
                        "label": slider_labels[week],
                        "args": [
                            [slider_labels[week]],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for week in slider_weeks
                ],
            }]
        )

    fig.update_layout(
        title=f"{title} | Week {slider_labels[initial_week]}",
        showlegend=True,
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis_title="pca_0",
            yaxis_title="pca_1",
            zaxis_title=z_col,
        ),
    )
    return fig


# =========================================================
# End-to-end wrapper
# =========================================================


def visualize_centroid_behavior_changes(
    historical_behavior_change_df,
    predicted_behavior_change_df,
    saving_dir=None,
    clusters=None,
    anomaly_score_threshold=3,
    max_history_weeks_per_cluster=None,
    is_save_vis_as_file=True,
    animate_by_week=True,
    file_prefix="Centroid Behavior Change",
    model=None,
    year=None,
    week=None,
):
    """
    Visualize historical and predicted centroid trajectories.

    A centroid-week is flagged with a thick red hollow ring when:
        behavior_change_score >= anomaly_score_threshold

    When year and week are provided, only that ISO week is marked if it is
    anomalous. When both are None, all anomalous weeks are marked.

    Only the columns needed by the visualization are retained internally.
    No horizon or z_n_model_space_pred columns are required.
    """
    fig2d = plot_behavior_change_2d(
        historical_behavior_change_df=historical_behavior_change_df,
        predicted_behavior_change_df=predicted_behavior_change_df,
        clusters=clusters,
        anomaly_score_threshold=anomaly_score_threshold,
        max_history_weeks_per_cluster=max_history_weeks_per_cluster,
        title=f"PCA Centroid Behavior Change 2D | {model.capitalize()} Approach",
        animate_by_week=animate_by_week,
        year=year,
        week=week,
    )

    fig3d = plot_behavior_change_3d(
        historical_behavior_change_df=historical_behavior_change_df,
        predicted_behavior_change_df=predicted_behavior_change_df,
        clusters=clusters,
        anomaly_score_threshold=anomaly_score_threshold,
        max_history_weeks_per_cluster=max_history_weeks_per_cluster,
        title=f"PCA Centroid Behavior Change 3D | {model.capitalize()} Approach",
        animate_by_week=animate_by_week,
        year=year,
        week=week,
    )

    if saving_dir is not None and is_save_vis_as_file:
        output_dir = Path(saving_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pio.write_html(
            fig2d,
            file=str(output_dir / f"PCA {file_prefix} 2D {model.capitalize()} Approach.html"),
            auto_open=False,
            full_html=True,
            config={"responsive": True},
        )
        pio.write_html(
            fig3d,
            file=str(output_dir / f"PCA {file_prefix} 3D {model.capitalize()} Approach.html"),
            auto_open=False,
            full_html=True,
            config={"responsive": True},
        )

    return fig2d, fig3d


# Backward-compatible alias for existing calling code.
def visualize_centroid_forecasts(
    df_centroid_history,
    df_centroid_predicted,
    saving_dir=None,
    clusters=None,
    max_history_weeks_per_cluster=None,
    is_save_vis_as_file=True,
    animate_by_week=True,
    anomaly_score_threshold=3,
    model=None,
    year=None,
    week=None,
    **_,
):
    return visualize_centroid_behavior_changes(
        historical_behavior_change_df=df_centroid_history,
        predicted_behavior_change_df=df_centroid_predicted,
        saving_dir=saving_dir,
        clusters=clusters,
        anomaly_score_threshold=anomaly_score_threshold,
        max_history_weeks_per_cluster=max_history_weeks_per_cluster,
        is_save_vis_as_file=is_save_vis_as_file,
        animate_by_week=animate_by_week,
        model=model,
        year=year,
        week=week,
    )