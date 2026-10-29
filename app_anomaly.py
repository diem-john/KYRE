from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / "projects"
VALIDATOR = ROOT / "scripts" / "validate_anomaly_real_data.py"

# Page configuration is managed by app_unified.py.


def get_projects() -> list[str]:
    if not PROJECTS.is_dir():
        return []
    return sorted(
        p.name for p in PROJECTS.iterdir()
        if p.is_dir() and (p / "cluster_embeddings" / "centroids_with_usage_labels").is_dir()
    )


def paths_for(name: str) -> dict[str, Path]:
    root = PROJECTS / name
    return {
        "root": root,
        "data": root / "cluster_embeddings" / "centroids_with_usage_labels",
        "models": root / "neural_models",
        "output": root / "anomaly_validation",
        "report": root / "anomaly_validation" / "validation_report.json",
    }


@st.cache_data(show_spinner=False)
def load_report(path_str: str, mtime: float) -> dict[str, Any] | None:
    del mtime
    try:
        with Path(path_str).open(encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {"report": obj}
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner="Reading Parquet preview…")
def load_parquet_preview(files: tuple[str, ...], rows_per_file: int) -> pd.DataFrame:
    frames = []
    for filename in files:
        try:
            frame = pd.read_parquet(filename).head(rows_per_file).copy()
            frame["_source_file"] = Path(filename).name
            frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_table(path_str: str, mtime: float) -> pd.DataFrame:
    del mtime
    try:
        path = Path(path_str)
        return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def walk_dicts(obj: Any, path: tuple[str, ...] = ()):
    if isinstance(obj, dict):
        yield path, obj
        for key, value in obj.items():
            yield from walk_dicts(value, path + (str(key),))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk_dicts(value, path + (str(i),))


def split_summaries(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    wanted = ("rows", "mean_anomaly_score", "median_anomaly_score",
              "warning_or_high_count", "warning_or_high_rate", "label_counts")
    for path, node in walk_dicts(report):
        if not path or path[-1].lower() not in {"train", "validation", "test"}:
            continue
        if isinstance(node, dict) and any(k in node for k in wanted):
            split = path[-1].lower()
            quality = sum(k in node for k in wanted)
            old = result.get(split)
            if old is None or quality > sum(k in old for k in wanted):
                result[split] = node
    return result


def metric_row(report: dict[str, Any]) -> None:
    keys = [
        ("raw_rows", "Raw rows"),
        ("clean_rows", "Clean rows"),
        ("embedding_dimensions", "Embedding dimensions"),
        ("pca_dimensions", "PCA dimensions"),
        ("cluster_count_kept", "Clusters retained"),
    ]
    values = [(label, report[key]) for key, label in keys if key in report]
    if values:
        cols = st.columns(len(values))
        for col, (label, value) in zip(cols, values):
            col.metric(label, f"{value:,}" if isinstance(value, int) else str(value))


def result_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.name != "validation_report.json"
        and p.suffix.lower() in {".csv", ".parquet", ".pq"}
    )


st.sidebar.title("🔎 KYRE Anomaly Detection")
st.sidebar.caption("Trajectory monitoring workspace")
projects = get_projects()
default_name = "trajectory_prediction_data_hitachi"
if projects:
    chosen = st.sidebar.selectbox(
        "Project dataset", projects,
        index=projects.index(default_name) if default_name in projects else 0,
    )
else:
    chosen = st.sidebar.text_input("Project folder", default_name)
P = paths_for(chosen)
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Explorer", "Run Detection", "Investigate Results", "Validation Report"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline signals**\n\n"
    "- Robust motion score\n"
    "- Isolation Forest score\n"
    "- Optional neural prediction residual\n"
    "- Maximum-rank fusion"
)
st.sidebar.caption("Anomaly scores are rankings, not probabilities.")

report = None
if P["report"].is_file():
    report = load_report(str(P["report"]), P["report"].stat().st_mtime)


if page == "Overview":
    st.title("Anomaly Detection Overview")
    st.write("Review the latest validation run and navigate to data, detection controls, and flagged results.")
    if report:
        st.success(f"Loaded `{P['report'].relative_to(ROOT)}`")
        metric_row(report)
        summaries = split_summaries(report)
        if summaries:
            st.subheader("Split-level summary")
            rows = []
            for split in ("train", "validation", "test"):
                d = summaries.get(split)
                if d:
                    rate = d.get("warning_or_high_rate")
                    rows.append({
                        "Split": split.title(),
                        "Rows": d.get("rows"),
                        "Mean score": d.get("mean_anomaly_score"),
                        "Median score": d.get("median_anomaly_score"),
                        "Warning / High": d.get("warning_or_high_count"),
                        "Flagged rate": f"{rate:.1%}" if isinstance(rate, (int, float)) else "—",
                    })
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            if PLOTLY_AVAILABLE and not summary_df.empty and "Flagged rate" in summary_df:
                plot_df = summary_df.copy()
                plot_df["Flagged rate (%)"] = pd.to_numeric(
                    plot_df["Flagged rate"].astype(str).str.rstrip("%"), errors="coerce"
                )
                plot_df = plot_df.dropna(subset=["Flagged rate (%)"])
                if not plot_df.empty:
                    st.plotly_chart(
                        px.bar(plot_df, x="Split", y="Flagged rate (%)",
                               title="Flagged rate by data split", text_auto=".1f",
                               labels={"Flagged rate (%)": "Flagged rate (%)"}),
                        use_container_width=True,
                    )
            label_frames = {
                split: d["label_counts"] for split, d in summaries.items()
                if isinstance(d.get("label_counts"), dict) and d["label_counts"]
            }
            if label_frames:
                st.subheader("Label distribution")
                labels_df = pd.DataFrame(label_frames).fillna(0).rename_axis("Label").reset_index()
                labels_long = labels_df.melt(id_vars="Label", var_name="Split", value_name="Count")
                if PLOTLY_AVAILABLE:
                    st.plotly_chart(
                        px.bar(labels_long, x="Label", y="Count", color="Split",
                               barmode="group", title="Label counts across splits", text_auto=True),
                        use_container_width=True,
                    )
                else:
                    st.bar_chart(pd.DataFrame(label_frames).fillna(0))
        neural = report.get("neural_signal")
        if isinstance(neural, dict):
            st.subheader("Neural prediction signal")
            a, b, c = st.columns(3)
            a.metric("Enabled", "Yes" if neural.get("enabled") else "No")
            b.metric("Device", str(neural.get("device", "—")))
            c.metric("Lookback", f"{neural.get('lookback_weeks', '—')} weeks")
    else:
        st.info("No validation report found. Use **Run Detection** to execute the existing validation workflow.")
    a, b, c = st.columns(3)
    a.metric("Centroid data", "Found" if P["data"].is_dir() else "Missing")
    b.metric("Validation script", "Found" if VALIDATOR.is_file() else "Missing")
    c.metric("Neural model folder", "Found" if P["models"].is_dir() else "Missing")
    st.warning(
        "The pipeline-generated labels are not independent ground truth. Do not interpret them as "
        "verified anomalies or use them to claim real-world precision/recall."
    )


elif page == "Data Explorer":
    st.title("Data Explorer")
    st.write("Preview centroid embedding files used by the real-data validation workflow.")
    files = sorted(P["data"].glob("*.parquet")) if P["data"].is_dir() else []
    if not files:
        st.error(f"No Parquet files found in `{P['data']}`.")
    else:
        st.caption(f"{len(files)} Parquet files available.")
        max_files = st.slider("Default preview file count", 1, min(30, len(files)), min(5, len(files)))
        rows_per_file = st.slider("Maximum rows per selected file", 10, 1000, 200, step=10)
        choices = st.multiselect(
            "Parquet files",
            [str(p) for p in files],
            default=[str(p) for p in files[:max_files]],
            format_func=lambda s: Path(s).name,
        )
        if choices:
            df = load_parquet_preview(tuple(choices), rows_per_file)
            if df.empty:
                st.error("Could not read the selected Parquet files. Check the installed Parquet engine.")
            else:
                st.metric("Preview rows", f"{len(df):,}")
                st.dataframe(df.head(500), use_container_width=True, hide_index=True)
                zcols = [
                    c for c in df.columns
                    if str(c).startswith(("z_", "pca_", "embedding_", "emb_"))
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                cluster_col = next((c for c in ("cluster", "cluster_id", "cluster_label") if c in df), None)
                c1, c2, c3 = st.columns(3)
                c1.metric("Columns", len(df.columns))
                c2.metric("Numeric columns", len(df.select_dtypes(include="number").columns))
                c3.metric("Potential embedding columns", len(zcols))
                if cluster_col:
                    st.subheader("Cluster counts in selected preview")
                    cluster_counts = df[cluster_col].astype(str).value_counts().rename_axis("Cluster").reset_index(name="Rows")
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(
                            px.bar(cluster_counts, x="Cluster", y="Rows",
                                   title="Rows by cluster in preview", text_auto=True),
                            use_container_width=True,
                        )
                    else:
                        st.bar_chart(cluster_counts.set_index("Cluster"))
                if zcols:
                    feature = st.selectbox("Embedding / PCA feature", zcols)
                    plot_df = df[[feature]].copy()
                    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
                    plot_df = plot_df.dropna().reset_index(drop=True)
                    plot_df["Preview row"] = range(1, len(plot_df) + 1)
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(
                            px.line(plot_df, x="Preview row", y=feature,
                                    title=f"{feature} across preview rows"),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            px.histogram(plot_df, x=feature, nbins=40,
                                         title=f"Distribution of {feature}"),
                            use_container_width=True,
                        )
                    else:
                        st.line_chart(plot_df.set_index("Preview row")[feature])
                    st.dataframe(plot_df[feature].describe().to_frame("value"))


elif page == "Run Detection":
    st.title("Run Anomaly Detection")
    st.write("This runs `scripts/validate_anomaly_real_data.py`; the app does not duplicate scoring logic.")
    if not VALIDATOR.is_file() or not P["data"].is_dir():
        st.error("The validation script or centroid data directory is missing.")
    else:
        st.info(
            "Models are fitted on the chronological training split. When neural scoring is enabled, "
            "the held-out validation split is used for calibration. The test split must not be used "
            "to tune thresholds."
        )
        with st.form("run_detection_form"):
            st.subheader("Preprocessing and splits")
            a, b, c = st.columns(3)
            pca_dims = a.number_input("PCA dimensions", 2, 192, 32)
            min_weeks = a.number_input("Minimum weeks", 2, 100, 8)
            max_gap = a.number_input("Maximum gap to fill", 0, 20, 2)
            validation_weeks = b.number_input("Validation weeks", 1, 104, 12)
            test_weeks = b.number_input("Test weeks", 1, 260, 24)
            max_staleness = b.number_input("Maximum staleness (weeks)", 0, 52, 3)
            window_size = c.number_input("Window size", 2, 104, 12)
            window_stride = c.number_input("Window stride", 1, 52, 1)
            estimators = c.number_input("Isolation Forest estimators", 10, 2000, 200, step=10)

            st.subheader("Neural residual signal")
            default_checkpoint = P["models"] / "hybrid_cnn_lstm_attention_best.pth"
            use_neural = st.checkbox("Enable neural prediction residuals", value=default_checkpoint.is_file())
            checkpoints = sorted(P["models"].glob("*.pth")) if P["models"].is_dir() else []
            checkpoint_values = [str(p.relative_to(ROOT)) for p in checkpoints]
            default_idx = checkpoint_values.index(str(default_checkpoint.relative_to(ROOT))) if default_checkpoint.is_file() and str(default_checkpoint.relative_to(ROOT)) in checkpoint_values else 0
            if checkpoint_values:
                checkpoint = st.selectbox("Checkpoint", checkpoint_values, index=default_idx, disabled=not use_neural)
            else:
                checkpoint = st.text_input("Checkpoint path relative to project root", disabled=not use_neural)
            device = st.selectbox("Inference device", ["cpu", "cuda"], disabled=not use_neural)

            st.subheader("Synthetic sensitivity check")
            d, e = st.columns(2)
            spike = d.number_input("Spike magnitude", 1.0, 50.0, 8.0, step=1.0)
            injection_count = e.number_input("Trajectories to inject", 0, 100, 5)
            confirm = st.checkbox("I understand this reruns validation and may overwrite validation_report.json.")
            run = st.form_submit_button("Run validation", type="primary", use_container_width=True)

        if run:
            if not confirm:
                st.error("Confirm the report overwrite warning before running.")
            elif use_neural and not checkpoint:
                st.error("Select a checkpoint or disable neural scoring.")
            else:
                cmd = [
                    sys.executable, str(VALIDATOR),
                    "--project-name", chosen,
                    "--projects-root", str(PROJECTS),
                    "--pca-dims", str(int(pca_dims)),
                    "--min-weeks", str(int(min_weeks)),
                    "--max-gap", str(int(max_gap)),
                    "--validation-weeks", str(int(validation_weeks)),
                    "--test-weeks", str(int(test_weeks)),
                    "--max-staleness-weeks", str(int(max_staleness)),
                    "--window-size", str(int(window_size)),
                    "--window-stride", str(int(window_stride)),
                    "--estimators", str(int(estimators)),
                    "--spike-magnitude", str(float(spike)),
                    "--max-injection-trajectories", str(int(injection_count)),
                ]
                if use_neural:
                    cp = Path(checkpoint)
                    if not cp.is_absolute():
                        cp = ROOT / cp
                    cmd.extend(["--neural-checkpoint", str(cp), "--neural-device", device])
                st.subheader("Live run progress")
                progress_bar = st.progress(0, text="Starting validation process…")
                progress_status = st.empty()
                elapsed_status = st.empty()
                log_box = st.empty()
                started = time.time()
                previous_duration = st.session_state.get("last_validation_duration_seconds")
                log_lines = []
                result_returncode = None
                launch_error = None
                # Approximate phases inferred from logs; this is not a formal progress API.
                phase_markers = [
                    (("load", "read data", "raw rows", "dataset"), 0.10, "Loading and validating data"),
                    (("clean", "preprocess", "pca", "embedding"), 0.25, "Cleaning data and preparing embeddings"),
                    (("split", "train", "validation", "test"), 0.40, "Building chronological splits"),
                    (("isolation", "fit", "motion score", "statistical"), 0.58, "Fitting statistical detectors"),
                    (("neural", "checkpoint", "residual", "inference"), 0.72, "Computing neural residuals"),
                    (("window", "fusion", "anomaly score", "threshold", "calibrat"), 0.84, "Scoring windows and calibrating labels"),
                    (("synthetic", "spike", "report", "saved", "complete", "finished"), 0.95, "Running sensitivity check and writing report"),
                ]
                try:
                    import os
                    import queue
                    import threading
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.Popen(
                        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=env,
                    )
                    line_queue = queue.Queue()

                    def read_output(pipe, output_queue):
                        try:
                            for output_line in iter(pipe.readline, ""):
                                output_queue.put(output_line.rstrip())
                        finally:
                            output_queue.put(None)
                            pipe.close()

                    reader = threading.Thread(target=read_output, args=(proc.stdout, line_queue), daemon=True)
                    reader.start()
                    stream_finished = False
                    displayed_progress = 0.03
                    while proc.poll() is None or not stream_finished:
                        try:
                            line = line_queue.get(timeout=0.35)
                            if line is None:
                                stream_finished = True
                            elif line:
                                log_lines.append(line)
                                low = line.lower()
                                for markers, amount, label in phase_markers:
                                    if any(marker in low for marker in markers):
                                        displayed_progress = max(displayed_progress, amount)
                                        progress_status.info(f"Current stage: {label}")
                                log_box.code("\n".join(log_lines[-12:]) or "Waiting for validator output…")
                        except queue.Empty:
                            pass
                        elapsed = time.time() - started
                        if previous_duration and previous_duration > 0:
                            estimated_progress = min(0.94, elapsed / previous_duration * 0.90)
                            displayed_progress = max(displayed_progress, estimated_progress)
                            remaining = max(0.0, previous_duration - elapsed)
                            elapsed_status.caption(
                                f"Elapsed: {elapsed:.0f}s · Estimated remaining: ~{remaining:.0f}s "
                                "(based on the previous run; actual time may vary)."
                            )
                        else:
                            elapsed_status.caption(
                                f"Elapsed: {elapsed:.0f}s · ETA unavailable on the first run; "
                                "the progress indicator uses detected log stages when available."
                            )
                        progress_bar.progress(min(0.97, displayed_progress), text="Approximate progress")
                    result_returncode = proc.returncode
                except Exception as exc:
                    launch_error = exc
                elapsed = time.time() - started
                st.session_state["validation_stdout"] = "\n".join(log_lines)
                st.session_state["validation_stderr"] = ""
                if launch_error is not None:
                    progress_bar.progress(1.0, text="Could not start validation")
                    st.exception(launch_error)
                elif result_returncode == 0:
                    progress_bar.progress(1.0, text="Validation completed")
                    st.session_state["last_validation_duration_seconds"] = elapsed
                    st.success(f"Validation completed in {elapsed:.1f} seconds.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    progress_bar.progress(1.0, text="Validation failed")
                    st.session_state["last_validation_duration_seconds"] = elapsed
                    st.error(f"Validation failed with exit code {result_returncode} after {elapsed:.1f} seconds.")
                    st.code("\n".join(log_lines) or "(No standard output)")
        if st.session_state.get("validation_stdout"):
            with st.expander("Latest validation output", expanded=True):
                st.text(st.session_state["validation_stdout"])
                if st.session_state.get("validation_stderr"):
                    st.text_area("Standard error", st.session_state["validation_stderr"], height=180)


elif page == "Investigate Results":
    st.title("Investigate Results")
    st.write("Explore saved row-level CSV/Parquet score tables, if the validation workflow has produced them.")
    tables = result_files(P["output"])
    if not tables:
        st.info(
            "No row-level CSV/Parquet tables were found in the validation output folder. "
            "This version can display the JSON summary, but interactive per-trajectory investigation "
            "requires the validation script to save scored timestep/window tables."
        )
    else:
        default_table = next(
            (i for i, path in enumerate(tables) if path.name == "timestep_scores.csv"),
            0,
        )
        selected = st.selectbox(
            "Result table", tables, index=default_table,
            format_func=lambda p: p.name,
        )
        df = load_table(str(selected), selected.stat().st_mtime)
        if df.empty:
            st.warning("The selected table could not be read or is empty.")
        else:
            score_col = next((c for c in ("anomaly_score", "score", "fused_score") if c in df), None)
            label_col = next((c for c in ("anomaly_label", "label", "anomaly_status") if c in df), None)
            cluster_col = next((c for c in ("cluster", "cluster_id", "trajectory_id") if c in df), None)
            filtered = df.copy()
            f1, f2 = st.columns(2)
            if label_col:
                labels = sorted(filtered[label_col].dropna().astype(str).unique())
                chosen_labels = f1.multiselect("Labels", labels, default=labels)
                filtered = filtered[filtered[label_col].astype(str).isin(chosen_labels)]
            if cluster_col:
                clusters = sorted(filtered[cluster_col].dropna().astype(str).unique())
                chosen_clusters = f2.multiselect("Cluster / trajectory", clusters)
                if chosen_clusters:
                    filtered = filtered[filtered[cluster_col].astype(str).isin(chosen_clusters)]
            # Time-series forecast and anomaly overlay for scored timestep results.
            if selected.name == "timestep_scores.csv":
                st.divider()
                st.subheader("Time-series prediction and flagged anomalies")
                forecast_dims = sorted(
                    [
                        col[len("actual_"):]
                        for col in filtered.columns
                        if col.startswith("actual_")
                        and f"predicted_{col[len('actual_'):]}" in filtered.columns
                    ],
                    key=lambda name: (
                        0, int(name.split("_")[-1])
                    ) if name.split("_")[-1].isdigit() else (1, name),
                )
                chart_df = filtered.copy()
                if cluster_col and not chart_df.empty:
                    chart_clusters = sorted(chart_df[cluster_col].dropna().astype(str).unique())
                    if chart_clusters:
                        preferred = (
                            chosen_clusters[0] if "chosen_clusters" in locals() and len(chosen_clusters) == 1
                            else chart_clusters[0]
                        )
                        chart_cluster = st.selectbox(
                            "Trajectory / cluster to visualize",
                            chart_clusters,
                            index=chart_clusters.index(preferred) if preferred in chart_clusters else 0,
                            key="forecast_chart_cluster",
                        )
                        chart_df = chart_df[chart_df[cluster_col].astype(str) == chart_cluster].copy()
                if "sequence_split" in chart_df.columns and not chart_df.empty:
                    split_options = [
                        split for split in ("train", "validation", "test")
                        if split in chart_df["sequence_split"].astype(str).unique()
                    ]
                    if split_options:
                        selected_split = st.selectbox(
                            "Time period", split_options,
                            index=split_options.index("test") if "test" in split_options else 0,
                            key="forecast_chart_split",
                        )
                        chart_df = chart_df[
                            chart_df["sequence_split"].astype(str) == selected_split
                        ].copy()

                if "year" in chart_df.columns and "week" in chart_df.columns:
                    chart_df["timestamp"] = pd.to_datetime(
                        chart_df["year"].astype("Int64").astype(str)
                        + "-W"
                        + chart_df["week"].astype("Int64").astype(str).str.zfill(2)
                        + "-1",
                        format="%G-W%V-%u",
                        errors="coerce",
                    )
                else:
                    chart_df["timestamp"] = pd.NaT
                chart_df = chart_df.dropna(subset=["timestamp"]).sort_values("timestamp")

                if chart_df.empty:
                    st.info("No valid year/week timestamps are available for this selection.")
                elif PLOTLY_AVAILABLE:
                    if forecast_dims:
                        dim = st.selectbox(
                            "PCA coordinate to plot", forecast_dims,
                            format_func=lambda name: name.replace("z_", "PCA component "),
                            key="forecast_chart_dimension",
                        )
                        actual_col = f"actual_{dim}"
                        predicted_col = f"predicted_{dim}"
                        chart_df[actual_col] = pd.to_numeric(chart_df[actual_col], errors="coerce")
                        chart_df[predicted_col] = pd.to_numeric(chart_df[predicted_col], errors="coerce")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=chart_df["timestamp"], y=chart_df[actual_col],
                            mode="lines+markers", name="Observed",
                            connectgaps=False,
                        ))
                        fig.add_trace(go.Scatter(
                            x=chart_df["timestamp"], y=chart_df[predicted_col],
                            mode="lines+markers", name="One-step-ahead prediction",
                            connectgaps=False,
                        ))
                        if label_col:
                            for anomaly_label, symbol in (("Warning", "diamond"), ("High anomaly", "x")):
                                flagged = chart_df[
                                    chart_df[label_col].astype(str) == anomaly_label
                                ].copy()
                                if not flagged.empty:
                                    fig.add_trace(go.Scatter(
                                        x=flagged["timestamp"], y=flagged[actual_col],
                                        mode="markers", name=f"{anomaly_label} (observed)",
                                        marker={"symbol": symbol, "size": 11},
                                        customdata=flagged[[score_col]].to_numpy() if score_col else None,
                                        hovertemplate=(
                                            "%{x|%Y-%m-%d}<br>Observed: %{y:.4f}"
                                            + (f"<br>Anomaly score: %{{customdata[0]:.4f}}" if score_col else "")
                                            + "<extra>%{fullData.name}</extra>"
                                        ),
                                    ))
                        fig.update_layout(
                            title="Observed vs one-step-ahead prediction",
                            xaxis_title="Week",
                            yaxis_title=f"{dim} (PCA-space coordinate)",
                            hovermode="x unified",
                            legend_title="Series",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Predictions are rolling one-step-ahead estimates using only prior observations. "
                            "The chart is in PCA space, not physical units. Initial points may have no prediction "
                            "because the model requires sufficient lookback history."
                        )
                    else:
                        st.info(
                            "This run has no saved neural forecasts. Enable **Neural prediction residuals** "
                            "and rerun validation to overlay observed vs predicted PCA coordinates. "
                            "Showing the selected PCA coordinate with flagged anomalies instead."
                        )
                        position_cols = sorted(
                            [col for col in chart_df.columns if col.startswith("position_")],
                            key=lambda name: int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else 9999,
                        )
                        if position_cols:
                            position_col = st.selectbox(
                                "PCA coordinate", position_cols, key="fallback_position_dimension"
                            )
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=chart_df["timestamp"], y=chart_df[position_col],
                                mode="lines+markers", name="Observed PCA coordinate",
                            ))
                            if label_col:
                                flagged = chart_df[chart_df[label_col].astype(str) != "Normal"]
                                if not flagged.empty:
                                    fig.add_trace(go.Scatter(
                                        x=flagged["timestamp"], y=flagged[position_col],
                                        mode="markers", name="Flagged anomaly",
                                        marker={"symbol": "x", "size": 11},
                                    ))
                            fig.update_layout(
                                title="Observed trajectory coordinate with flagged anomalies",
                                xaxis_title="Week", yaxis_title="PCA-space coordinate",
                                hovermode="x unified",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    if score_col:
                        score_plot = chart_df.copy()
                        score_plot[score_col] = pd.to_numeric(score_plot[score_col], errors="coerce")
                        score_plot = score_plot.dropna(subset=[score_col])
                        if not score_plot.empty:
                            score_fig = px.line(
                                score_plot, x="timestamp", y=score_col,
                                color=label_col if label_col else None,
                                markers=True, title="Anomaly score over time",
                            )
                            st.plotly_chart(score_fig, use_container_width=True)
                else:
                    st.warning("Install Plotly to enable the interactive forecast/anomaly overlay.")

            if score_col:
                scores = pd.to_numeric(filtered[score_col], errors="coerce").dropna()
                if not scores.empty:
                    st.subheader("Score distribution")
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(
                            px.histogram(filtered, x=score_col, nbins=40,
                                         color=label_col if label_col else None,
                                         marginal="box", title="Anomaly score distribution"),
                            use_container_width=True,
                        )
                    else:
                        st.bar_chart(scores.value_counts(bins=20, sort=False).sort_index())
                    if label_col:
                        st.subheader("Score summary by label")
                        summary = filtered.groupby(label_col)[score_col].agg(
                            ["count", "mean", "median", "max"]
                        ).reset_index()
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                        if PLOTLY_AVAILABLE:
                            st.plotly_chart(
                                px.box(filtered, x=label_col, y=score_col, points="outliers",
                                       title="Score spread by anomaly label"),
                                use_container_width=True,
                            )
                    datetime_col = next(
                        (c for c in filtered.columns if any(
                            k in str(c).lower() for k in ("timestamp", "datetime", "date", "time")
                        )), None
                    )
                    if datetime_col and score_col and PLOTLY_AVAILABLE:
                        time_df = filtered.copy()
                        time_df[datetime_col] = pd.to_datetime(time_df[datetime_col], errors="coerce")
                        time_df[score_col] = pd.to_numeric(time_df[score_col], errors="coerce")
                        time_df = time_df.dropna(subset=[datetime_col, score_col]).sort_values(datetime_col)
                        if not time_df.empty:
                            st.subheader("Anomaly score over time")
                            if len(time_df) > 5000:
                                time_df = time_df.iloc[::max(1, len(time_df) // 5000)]
                                st.caption("The time-series chart is downsampled for responsiveness; the table retains all filtered rows.")
                            st.plotly_chart(
                                px.line(time_df, x=datetime_col, y=score_col,
                                        color=label_col if label_col else None,
                                        title="Anomaly score over time"),
                                use_container_width=True,
                            )
            st.dataframe(filtered.head(2000), use_container_width=True, hide_index=True)
            st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), f"{selected.stem}_filtered.csv", "text/csv")
    if report:
        with st.expander("Validation report JSON"):
            st.json(report)


elif page == "Validation Report":
    st.title("Validation Report")
    if not report:
        st.info("No readable validation_report.json exists for this project yet.")
    else:
        metric_row(report)
        if isinstance(report.get("neural_signal"), dict):
            st.subheader("Neural signal")
            st.json(report["neural_signal"])
        limitations = report.get("limitations")
        if limitations:
            st.subheader("Interpretation and limitations")
            if isinstance(limitations, list):
                for item in limitations:
                    st.markdown(f"- {item}")
            else:
                st.write(limitations)
        with st.expander("Full JSON report", expanded=True):
            st.json(report)
        st.download_button(
            "Download JSON report",
            json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            "validation_report.json",
            "application/json",
        )

st.sidebar.divider()
if not PLOTLY_AVAILABLE:
    st.sidebar.warning("Plotly is not installed. Install it with: pip install plotly")
st.sidebar.caption("KYRE Neural Prediction Lab")