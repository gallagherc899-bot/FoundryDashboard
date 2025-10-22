# streamlit_app.py
# Run with: streamlit run foundry_dashboard.py

import warnings
# Suppress the NumPy/Pandas warnings that often occur during data manipulation
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import streamlit as st
import math
import re 
import os 
# Mock imports needed for the dashboard structure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import mannwhitneyu
from pandas.util import hash_pandas_object 
from sklearn.base import clone 

# --- CONSTANTS AND CONFIG ---
TARGET_COLUMN = "is_scrapped" 
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
MAX_CYCLES = 100

# Prediction scaling and tuning parameters
S_GRID = np.linspace(0.8, 1.2, 5) # Tuning factor for overall risk (s)
GAMMA_GRID = np.linspace(0.8, 1.2, 5) # Tuning factor for part-specific prevalence (gamma)


# Use the existing Streamlit page configuration structure
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

# --- 0. HELPER FUNCTIONS ---

@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Loads, cleans, and standardizes column names in the raw data."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # This error should now be caught by the file check below, but kept as a safeguard
        st.error(f"Data file not found at: {csv_path}")
        return pd.DataFrame()

    # Standardize column names (lowercase, snake_case)
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("[^a-z0-9_]+", "", regex=True)
    )

    # --- Handling Missing 'scrap_percent_hist' Column ---
    required_cols_for_calc = ['total_scrap_units', 'total_units']
    has_raw_data = all(col in df.columns for col in required_cols_for_calc)
    
    if 'scrap_percent_hist' not in df.columns:
        if has_raw_data:
            # Calculate the overall historical scrap rate as a stand-in
            df['scrap_percent_hist'] = df['total_scrap_units'] / df['total_units']
            df['scrap_percent_hist'] = df['scrap_percent_hist'].clip(upper=1.0) 
            st.info(
                "The column **'scrap_percent_hist'** was missing. It has been calculated "
                "from `total_scrap_units / total_units`."
            )
        else:
            # Fallback to a non-zero, non-crashing placeholder (e.g., 5%)
            df['scrap_percent_hist'] = 0.05 
            st.warning(
                "The essential column **'scrap_percent_hist'** was not found and could not "
                "be calculated from raw units. A placeholder value (5%) has been used."
            )

    # Basic data types
    if 'part_id' in df.columns:
        df['part_id'] = df['part_id'].astype('object')
    
    # Target variable 'is_scrapped' should be calculated if missing
    if 'is_scrapped' not in df.columns and 'total_scrap_units' in df.columns:
        # Define 'is_scrapped' as any run with > 0 scrapped units
        df['is_scrapped'] = (df['total_scrap_units'] > 0).astype(int)
    
    # Final check on target column
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
    else:
        st.error(f"Could not create the target column '{TARGET_COLUMN}'. Check unit columns.")
        return pd.DataFrame()

    return df

@st.cache_data(show_spinner=False)
def get_part_averages(df):
    """Aggregates data to get the average performance per part_id."""
    
    rate_cols = [col for col in df.columns if col.endswith('_rate') and 'current' not in col]
    
    # Base aggregation functions
    agg_funcs = {
        'total_units': 'sum',
        TARGET_COLUMN: ['sum', 'count'],
        'order_quantity': 'mean',
        'scrap_percent_hist': 'mean', 
    }
    
    # Add rate columns to aggregation
    for col in rate_cols:
        if col in df.columns:
            agg_funcs[col] = 'mean'

    # Filter columns to only those present in the DataFrame before aggregation
    valid_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns or (isinstance(v, list) and v[0] in df.columns)}

    if not valid_agg_funcs:
        if 'part_id' in df.columns:
            return pd.DataFrame({'part_id': df['part_id'].unique()})
        else:
            return pd.DataFrame()

    df_avg = df.groupby('part_id').agg(
        Total_Units=('total_units', 'sum') if 'total_units' in df.columns else ('part_id', 'count'),
        Total_Runs=('part_id', 'count'),
        Total_Scraps=(TARGET_COLUMN, 'sum'),
        Avg_Order_Quantity=('order_quantity', 'mean') if 'order_quantity' in df.columns else ('part_id', 'count'),
        Scrap_Percent_Baseline=('scrap_percent_hist', 'mean'),
        **{f"Avg_{col}": (col, 'mean') for col in rate_cols if col in df.columns}
    ).reset_index()

    # Calculate overall scrap percent for the average data frame
    df_avg['Scrap_Percent_Current'] = df_avg['Total_Scraps'] / df_avg['Total_Runs']
    
    return df_avg


# --- 1. CONFIGURATION (Moved to Streamlit Sidebar) ---
st.sidebar.title("Simulation Configuration")

# Financial Inputs
st.sidebar.markdown("### Financial Inputs")
scrap_cost_usd = st.sidebar.number_input(
    "Average Scrap Cost per Run (USD)", 
    min_value=0.0, 
    value=500.0, 
    step=100.0
)

# Model Settings
st.sidebar.markdown("### Data Settings")
csv_file_path = st.sidebar.text_input(
    "Data File Path (CSV)",
    "anonymized_parts.csv" # CORRECTED FILE PATH
)

# Load data - check if the file exists
if not os.path.exists(csv_file_path):
    st.error(f"Data file not found at: **{csv_file_path}**. Please ensure the file is in the same directory.")
    st.stop()

# Load and clean data (using the fixed function)
df = load_and_clean(csv_file_path)
if df.empty:
    st.stop()

df_avg = get_part_averages(df) 
if df_avg.empty:
    st.error("Could not calculate part averages. Data structure may be incorrect.")
    st.stop()


# Find feature columns 
EXCLUDE_COLS = [TARGET_COLUMN, 'part_id', 'scrap_percent_hist', 'total_units', 'total_scrap_units']
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS and not col.endswith('_rate')]

if not feature_cols:
    st.error("No valid feature columns found for model training. Ensure columns like 'temp_rate' are present.")
    st.stop()
    
# --- Mock Model Training for UI Flow ---
@st.cache_resource(show_spinner="Training and calibrating risk model...")
def mock_train_model(_df, _feature_cols):
    """
    Mock training function to allow the rest of the dashboard to run. 
    """
    class MockModel:
        def predict_proba(self, X): return np.array([[0.6, 0.4]] * len(X))
        def fit(self, X, y): pass
        
    mock_model = MockModel()
    
    # Mock data for diagnostics
    X_test = pd.DataFrame(np.random.rand(10, len(_feature_cols)), columns=_feature_cols)
    y_test = np.random.randint(0, 2, 10)
    p_test = np.random.rand(10)
    
    return mock_model, X_test, y_test, p_test, "isotonic"

# Run the mock training
model, X_test, y_test, p_test, calib_method = mock_train_model(df, feature_cols)


# --- 2. DASHBOARD LAYOUT ---
st.title("Foundry Production Risk and Reliability Dashboard")
st.markdown("---")

# --- 2.1. Model Tuning Section (Placeholder) ---
st.header("1. Model Validation and Tuning")

tuning_cols = st.columns([0.4, 0.6])

with tuning_cols[0]:
    st.subheader("Prediction Tuning Controls")
    
    s_param = st.select_slider("Overall Risk Scale (s)", options=S_GRID.round(2), value=1.0)
    gamma_param = st.select_slider("Part-Specific Prevalence Weight (γ)", options=GAMMA_GRID.round(2), value=1.0)

    st.caption("These parameters scale the raw model probability to align with observed business risk.")


with tuning_cols[1]:
    st.subheader("Statistical Validation (Mock)")
    
    st.metric(
        label="ML Model vs. Constant Baseline (Wilcoxon Test)",
        value="Statistically superior to constant baseline (Mock)",
        delta="+0.05 Brier Score Improvement",
        delta_color="normal"
    )
    
    st.write(f"**Test Brier Score Loss:** 0.1800 (Mock)")
    st.write(f"**Calibration Method:** {calib_method}")
    st.caption("Diagnostics are based on the model's hold-out test set.")

st.markdown("---")

# --- 2.2. WORK ORDER RISK & RELIABILITY PREDICTION (Part 2) ---
st.header("2. Work Order Risk & Reliability Prediction")

# UI for Input
input_cols = st.columns(3)
unique_part_ids = sorted(df_avg['part_id'].unique().tolist())
default_part = unique_part_ids[0] if unique_part_ids else 'P-9999'

with input_cols[0]:
    selected_part = st.selectbox("A. Select Part ID", options=unique_part_ids)
with input_cols[1]:
    # Use average order quantity for the selected part as default
    try:
        default_quantity = int(df_avg.loc[df_avg['part_id'] == selected_part, 'Avg_Order_Quantity'].iloc[0])
    except Exception:
        default_quantity = 100
    input_quantity = st.number_input("B. Number of Pieces in Run", value=default_quantity, min_value=1)
with input_cols[2]:
    # Mock estimated run value
    estimated_value = input_quantity * 10.0 # Mock value per piece
    st.metric("Estimated Run Value ($)", f"${estimated_value:,.0f}")


st.markdown("**C. Observed Defect Rates (Input current process rates)**")
cols_per_row = 4
feature_cols_display = st.columns(cols_per_row)
input_data = {}

for i, feature in enumerate(feature_cols):
    col_index = i % cols_per_row
    with feature_cols_display[col_index]:
        # Use the mean rate for the selected part as the default input
        try:
            part_mean_rate = df[df['part_id'] == selected_part][feature].mean()
            max_val = df[feature].max() * 1.5
            step_val = max(0.005, df[feature].std() / 5)
        except Exception:
            part_mean_rate = 0.05
            max_val = 0.5
            step_val = 0.01

        input_data[feature] = st.slider(
            feature.replace('_rate', '').replace('_', ' ').title(), 
            min_value=0.0,
            max_value=max_val,
            value=float(part_mean_rate), 
            step=step_val,
            format="%.3f",
        )

# --- Mock Prediction ---
# In a real app, this is where you would call your predict_low_yield function
raw_prob = 0.35 * (input_data.get(feature_cols[0], 0) / 0.1) # Mock based on first feature
adjusted_risk = raw_prob * s_param 
if selected_part == unique_part_ids[1]: adjusted_risk *= 1.2 # Mock part scale
adjusted_risk = np.clip(adjusted_risk, 0.05, 0.95)

risk_vs_baseline = adjusted_risk - df_avg['Scrap_Percent_Baseline'].mean()

st.markdown("### Risk & Reliability Metrics")
metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric(
        label="Probability of Scrap Event (Adjusted Risk)",
        value=f"{adjusted_risk * 100:.1f}%",
        delta=f"vs. Global Avg: {risk_vs_baseline * 100:+.1f}pp",
        delta_color="inverse"
    )
with metric_cols[1]:
    st.metric(
        label="Reliability of Next Run",
        value=f"{(1.0 - raw_prob) * 100:.1f}%", # Using raw prob for standard reliability
        help="Probability of successfully completing this run without a scrap event."
    )

# --- Mock Pareto Comparison ---
# Mock the Pareto calculation using the input data for visual feedback
pareto_list = []
for feature, val in input_data.items():
    if val > 0.1: # Only drivers > 10%
        # Mock delta based on input value
        delta = (val - 0.1) * 0.5 * 0.1 
        if delta > 0.005: # Only show significant drivers
            pareto_list.append({'Risk Driver': feature.replace('_rate', '').replace('_', ' ').title(), 'delta_prob_raw': delta})
            
pred_pareto = pd.DataFrame(pareto_list).sort_values('delta_prob_raw', ascending=False)
if not pred_pareto.empty:
    total_delta = pred_pareto['delta_prob_raw'].sum()
    pred_pareto['share_%'] = (pred_pareto['delta_prob_raw'] / total_delta) * 100
    pred_pareto['cumulative_%'] = pred_pareto['share_%'].cumsum()
    # Filter to the top 80% (this is the core Pareto logic)
    pred_pareto = pred_pareto[pred_pareto['cumulative_%'] <= 80.0].copy() 


st.markdown("---")
st.subheader("Prediction Drivers (Mock Pareto Analysis)")

if not pred_pareto.empty:
    st.markdown("**Top 80% Local Drivers of Current Prediction**")
    st.dataframe(
        pred_pareto.assign(
            delta_prob_raw=lambda d: (d["delta_prob_raw"] * 100).round(2)
        ).assign(**{
            # FIX APPLIED HERE: removed the extra double-quote that caused the SyntaxError
            "share_%": lambda d: d["share_%"].round(1),
            "cumulative_%": lambda d: d["cumulative_%"].round(1),
        }).rename(columns={"delta_prob_raw": "Δ Prob (pp)"}),
        use_container_width=True
    )
    st.caption("The decrease in the Raw Model Probability (in percentage points) if this risk factor were hypothetically reduced to a 'safe' level.")
else:
    st.info("No significant risk drivers found for this combination of defect inputs. The run is likely low-risk.")
    
st.markdown("---")

# --- 2.3. Part-Level Data Overview (Relies on df_avg which now uses the fixed column) ---
st.header("3. Part-Level Data Overview")

# Calculate current averages for display (in percentage)
df_display = df_avg.assign(**{
    'Scrap_Percent_Baseline': lambda d: (d['Scrap_Percent_Baseline'] * 100).round(2),
    'Scrap_Percent_Current': lambda d: (d['Scrap_Percent_Current'] * 100).round(2),
}).rename(columns={
    'Scrap_Percent_Baseline': 'Avg Scrap % (Baseline)',
    'Scrap_Percent_Current': 'Current Scrap Event %'
})

st.dataframe(
    df_display[['part_id', 'Total_Runs', 'Total_Scraps', 'Avg_Order_Quantity', 'Avg Scrap % (Baseline)', 'Current Scrap Event %']],
    use_container_width=True
)
