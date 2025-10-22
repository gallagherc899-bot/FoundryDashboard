# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
# --- REQUIRED IMPORT FOR THE FIX ---
from pandas.util import hash_pandas_object
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

# -----------------------------
# Page / constants
# -----------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)
# TOP_K_PARETO is no longer used for dynamic 80% cutoff, but kept as a fallback.
TOP_K_PARETO = 8 
# Set rate cols to always be true for Pareto output (manager requirement)
USE_RATE_COLS_PERMANENT = True 


# -----------------------------
# Helpers - MODIFIED for 80% Pareto Rule
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.lower()
    ) # ... rest of the function definition truncated for brevity
    return df
# ... end of load_and_clean (original content continues below)


# --- FUNCTION WITH CACHING FIX (Corresponds to the call on line 563) ---
# The fix uses 'hash_funcs' to instruct Streamlit how to safely hash the
# complex input types (DataFrame and Model) to resolve UnhashableParamError.
@st.cache_data(
    show_spinner=False,
    # Explicitly define how to hash complex types.
    hash_funcs={
        # 1. Pandas DataFrame (for df_ml): Hash by content and index.
        pd.DataFrame: lambda df: hash_pandas_object(df, index=True).sum(),
        # 2. RandomForestClassifier (for model): Hash by its parameters.
        RandomForestClassifier: lambda model: str(model.get_params())
    }
)
def predict_part_risk(model: RandomForestClassifier, part_id: str, df_ml: pd.DataFrame, feature_cols: list[str]):
    # Placeholder implementation based on context.
    # RETAIN YOUR ORIGINAL FUNCTION LOGIC HERE. The fix is only in the decorator.

    # 1. Filter data for the selected part
    part_data = df_ml[df_ml['part_id'] == part_id]
    
    if part_data.empty:
        # Return default values if no data
        return 0.0, pd.DataFrame() 

    # 2. Prepare features (X)
    X = part_data[feature_cols]

    # 3. Get prediction probability
    # model.predict_proba returns [[prob_of_0, prob_of_1]]
    prob_scrap = model.predict_proba(X)[:, 1].mean()

    # 4. Determine key feature drivers (part_drivers)
    # This is highly simplified and assumes some form of feature importance/SHAP analysis
    # which is not provided. Returning placeholder for structure.
    part_drivers = pd.DataFrame({
        'feature': ['placeholder_feature_1', 'placeholder_feature_2'],
        'delta_prob_raw': [0.15, 0.08],
        'share_%': [65, 35],
        'cumulative_%': [65, 100]
    }).sort_values('delta_prob_raw', ascending=False)
    
    return prob_scrap, part_drivers

# ... original file content continues below (lines that were cut off in the snippet)
# Diagnostics
st.subheader("Model Diagnostics")
test_brier = np.nan
try:
    test_brier = brier_score_loss(y_test, p_test) if len(X_test) and len(p_test) else np.nan
except Exception:
    pass
st.write(f"Calibration: **{calib_method}** | Test Brier: {test_brier:.4f}")
st.caption("Adjusted risk = calibrated prob × s × (part_scale^γ). part_scale is the per-part exceedance prevalence relative to train global prevalence.")
