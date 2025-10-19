# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

# --- Default Settings ---
DEFAULTS = {
    "scrap_threshold": 6.5,
    "prior_shift": True,
    "prior_shift_guard": 20,
    "use_quick_hook": False,
    "manual_s": 1.0,
    "manual_gamma": 0.5,
    "rolling_validation": True,
    "include_rate_features": True
}

# --- Initialize Session State ---
def initialize_session_state():
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Settings")

# Reset Toggle
if st.sidebar.button("Reset to Recommended Defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.experimental_rerun()  # Force UI refresh

# Scrap Threshold
st.sidebar.slider(
    "Scrap % Threshold", 1.0, 15.0, key="scrap_threshold", step=0.1
)

# Prior Shift Guard
st.sidebar.slider(
    "Prior Shift Guard", 0, 50, key="prior_shift_guard"
)

# Include *_rate Features
st.session_state["include_rate_features"] = st.sidebar.checkbox(
    "Include *_rate Features", value=st.session_state["include_rate_features"]
)

# Prior Shift
st.session_state["prior_shift"] = st.sidebar.checkbox(
    "Enable Prior Shift", value=st.session_state["prior_shift"]
)

# Quick-Hook Toggle
st.session_state["use_quick_hook"] = st.sidebar.checkbox(
    "Use Manual Quick-Hook", value=st.session_state["use_quick_hook"]
)

# Manual Quick-Hook Sliders
if st.session_state["use_quick_hook"]:
    st.sidebar.slider(
        "Manual s", 0.1, 2.0, key="manual_s", step=0.1
    )
    st.sidebar.slider(
        "Manual γ", 0.1, 1.0, key="manual_gamma", step=0.05
    )

# Rolling Validation
st.session_state["rolling_validation"] = st.sidebar.checkbox(
    "Run 6-2-1 Rolling Validation", value=st.session_state["rolling_validation"]
)

# --- Use settings in the pipeline ---
settings = {key: st.session_state[key] for key in DEFAULTS}
