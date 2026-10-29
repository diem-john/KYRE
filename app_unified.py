"""Unified KYRE entry point: trajectory inference followed by anomaly detection.

Run with:
    streamlit run app_unified.py

Anomaly detection is integrated into the trajectory inference workflow; it is
not exposed as a separate sidebar page.
"""
from pathlib import Path
import runpy

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="KYRE | Trajectory Inference & Anomaly Detection",
    page_icon="🧭",
    layout="wide",
)

# Execute the existing trajectory application as the single user-facing workflow.
# Its Inference tab now runs anomaly scoring immediately after the model's
# one-step-ahead trajectory inference and shows combined results in place.
runpy.run_path(
    str(ROOT / "src" / "neural_prediction" / "streamlit_app.py"),
    run_name="__main__",
)
