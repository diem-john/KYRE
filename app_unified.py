"""Unified KYRE Streamlit entry point.

Run with:
    streamlit run app_unified.py

The prediction and anomaly workflows remain separate pages so their existing
UI and model pipelines can be preserved while sharing one application shell.
"""

import streamlit as st

# Page configuration belongs to this entry point. Child pages intentionally
# do not call st.set_page_config().
st.set_page_config(
    page_title="KYRE | Neural Prediction & Anomaly Intelligence",
    page_icon="🧭",
    layout="wide",
)

pages = {
    "Trajectory Prediction": [
        st.Page(
            "src/neural_prediction/streamlit_app.py",
            title="Trajectory Prediction Lab",
            icon="📈",
            default=True,
        ),
    ],
    "Anomaly Intelligence": [
        st.Page(
            "app_anomaly.py",
            title="Anomaly Detection",
            icon="🔎",
        ),
    ],
}

selected_page = st.navigation(pages, position="sidebar", expanded=True)
selected_page.run()
