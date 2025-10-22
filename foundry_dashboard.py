# streamlit_app.py
# Run this file with: streamlit run foundry_dashboard.py

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import streamlit as st
import re 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import wilcoxon 
from pandas.util import hash_pandas_object
from sklearn.base import clone 

# --- CONSTANTS AND CONFIG ---
TARGET_COLUMN = "is_low_yield" # Target: Did the run exceed the scrap threshold?
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

# Prediction scaling and tuning parameters
S_GRID = np.linspace(0.8, 1.2, 9) # Tuning factor for overall risk (s)
GAMMA_GRID = np.linspace(0.8, 1.2, 9) # Tuning factor for part-specific prevalence (gamma)

# Use the existing Streamlit page configuration structure
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

# --- 2. DATA LOADING AND PREPARATION ---

def clean_col_name(col_raw):
    """Aggressively cleans column names."""
    col = str(col_raw).strip().lower()
    col = re.sub(r'[^\w_]', '', col.replace(' ', '_').replace('/', ''))
    while '__' in col:
        col = col.replace('__', '_')
    return col.strip('_')

@st.cache_data(
    show_spinner="Loading and preparing data...",
    hash_funcs={pd.DataFrame: lambda df: hash_pandas_object(df, index=True).sum()}
)
def load_and_prepare_data(filepath: str = 'anonymized_parts.csv'):
    """
    Loads, cleans, and prepares data, setting up the target variable and features.
    """
    try:
        df_historical = pd.read_csv(filepath)
        
        # Apply cleaning to ALL columns
        df_historical.columns = [clean_col_name(c) for c in df_historical.columns]
        
        # Standardize crucial column names
        final_rename_map = {}
        for col in df_historical.columns:
            if 'part_id' in col: final_rename_map[col] = 'part_id'
            elif 'scrap_percent' in col: final_rename_map[col] = 'scrap_percent_hist'
            elif 'order_quantity' in col: final_rename_map[col] = 'order_quantity'
            elif 'weight_lbs' in col: final_rename_map[col] = 'piece_weight_lbs'
            
        df_historical.rename(columns=final_rename_map, inplace=True)
        
        # Ensure scrap percent is a rate (0 to 1)
        if 'scrap_percent_hist' in df_historical.columns:
            df_historical['scrap_percent_hist'] = df_historical['scrap_percent_hist'].clip(upper=100) / 100.0
        
        # --- ML Data Preparation ---
        df_ml = df_historical.copy()
        
        # 1. Define Target (Low Yield Exceedance: greater than median scrap rate)
        scrap_threshold = df_ml['scrap_percent_hist'].median()
        if scrap_threshold == 0:
            scrap_threshold = df_ml['scrap_percent_hist'].mean()
        
        df_ml[TARGET_COLUMN] = (df_ml['scrap_percent_hist'] > scrap_threshold).astype(int)
        
        if df_ml[TARGET_COLUMN].nunique() < 2:
            st.error("Target variable preparation failed: Need at least two classes (low/high yield).")
            return pd.DataFrame(), pd.DataFrame(), [], 0.0, {}

        # 2. Identify Features
        feature_cols = [col for col in df_ml.columns if col.endswith('_rate')]
        
        # --- Prevalences (for Risk Tuning) ---
        global_prevalence = df_ml[TARGET_COLUMN].mean()
        
        low_yield_runs_per_part = df_ml.groupby('part_id')[TARGET_COLUMN].sum()
        total_runs_per_part = df_ml.groupby('part_id').size()
        part_prevalence = low_yield_runs_per_part / total_runs_per_part
        
        # Scale: Part Prevalence / Global Prevalence
        part_prevalence_scale = (part_prevalence / global_prevalence).replace([np.inf, -np.inf], 0).fillna(0).to_dict()
        
        # --- Summary Data for Display ---
        df_avg = df_historical.groupby('part_id').agg(
            Scrap_Percent_Baseline=('scrap_percent_hist', 'max'),
            Avg_Order_Quantity=('order_quantity', 'mean'),
            Avg_Piece_Weight=('piece_weight_lbs', 'mean'),
            Total_Runs=('scrap_percent_hist', 'count') 
        ).reset_index()

        return df_ml, df_avg, feature_cols, global_prevalence, part_prevalence_scale

    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please ensure the file is correctly named and available.")
        return pd.DataFrame(), pd.DataFrame(), [], 0.0, {}
    except Exception as e:
        st.error(f"A severe error occurred during data processing: {e}")
        return pd.DataFrame(), pd.DataFrame(), [], 0.0, {}


# Load data and global context
df_ml, df_avg, feature_cols, global_prevalence, part_prevalence_scale = load_and_prepare_data()

if df_ml.empty or not feature_cols:
    st.stop()


# --- 3. MACHINE LEARNING MODEL (Training) ---

@st.cache_resource(
    show_spinner="Training and calibrating risk model...",
    hash_funcs={
        pd.DataFrame: lambda df: hash_pandas_object(df, index=True).sum(),
        CalibratedClassifierCV: lambda model: str(model.get_params()),
    }
)
def train_random_forest_model(df_ml: pd.DataFrame, feature_cols: list):
    """
    Trains a Random Forest and calibrates its probability outputs.
    """
    X = df_ml[feature_cols].fillna(0) 
    y = df_ml[TARGET_COLUMN].astype(int)

    # Train-test split (70% Train/Calib, 30% Test)
    X_tc, X_test, y_tc, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    # Further split for base training and calibration (approx 49% Train, 21% Calib)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_tc, y_tc, test_size=0.3, random_state=RANDOM_STATE, stratify=y_tc
    )

    base_model = RandomForestClassifier(
        n_estimators=DEFAULT_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    base_model.fit(X_train, y_train)

    # Calibrate the pre-fitted base model
    calib_method = "isotonic" 
    model = CalibratedClassifierCV(
        estimator=base_model,
        method=calib_method,
        cv="prefit"
    )
    model.fit(X_calib, y_calib)

    p_test = model.predict_proba(X_test)[:, 1] 

    return model, X_test, y_test, p_test, calib_method

# Train the model and retrieve diagnostics data
model, X_test, y_test, p_test, calib_method = train_random_forest_model(df_ml, feature_cols)

# --- 4. PREDICTION AND RELIABILITY UTILITIES ---

def predict_low_yield(model, X_single, part_id, part_prevalence_scale, s_param, gamma_param, global_prevalence):
    """
    Generates prediction and applies the tuning formula:
    Adjusted Risk = Raw Prob * s * (Part Scale ^ gamma)
    """
    # 1. Raw Probability from Calibrated Model
    raw_prob = model.predict_proba(X_single)[0, 1]

    # 2. Part-Specific Prevalence Scale (P_scale)
    # The part scale is the Part's Historical Low-Yield Prevalence / Global Low-Yield Prevalence
    part_scale_raw = part_prevalence_scale.get(part_id, 1.0)
    
    # Apply gamma tuning and cap scale to prevent unstable results
    p_scale = (part_scale_raw ** gamma_param) 
    
    # 3. Apply Global (s) and Part-Specific (gamma-tuned) scaling
    adjusted_risk = raw_prob * s_param * p_scale
    
    # Ensure risk is capped at 100% and floored at 0%
    adjusted_risk = np.clip(adjusted_risk, 0.0, 1.0)
    
    # Calculate risk versus the global average for the delta metric
    risk_vs_baseline = adjusted_risk - global_prevalence

    return raw_prob, adjusted_risk, p_scale, risk_vs_baseline

def calculate_mttf_reliability(part_id, df_avg, raw_prob):
    """
    Calculates MTTF (Mean Runs To Failure) and Reliability of the Next Run.
    Failure rate (lambda) is based on the raw model prediction (raw_prob).
    """
    # Use the raw model probability (P(Low Yield)) as the predicted failure rate (lambda)
    lambda_pred = raw_prob 
    
    # Historical baseline (for context)
    lambda_hist = df_avg.loc[df_avg['part_id'] == part_id, 'Scrap_Percent_Baseline'].iloc[0]
    
    # Calculate MTTF (Mean Runs to Failure) = 1 / Lambda
    mttf_hist = 1.0 / lambda_hist if lambda_hist > 1e-6 else np.inf
    mttf_pred = 1.0 / lambda_pred if lambda_pred > 1e-6 else np.inf

    # Reliability of Next Run (Simple definition: 1 - P(Failure))
    reliability_next_run_simple = 1.0 - lambda_pred

    return mttf_hist, mttf_pred, reliability_next_run_simple

def calculate_predicted_pareto(model, X_single, feature_cols, p_base):
    """
    Calculates the local feature importance (Predicted Pareto) for a single prediction.
    Measures the drop in probability when an active risk factor is hypothetically removed (set to 0).
    """
    X_base = X_single.copy()
    pareto_list = []
    
    # Use the 5th percentile from the full dataset as the "safe" value
    percentiles_5 = df_ml[feature_cols].quantile(0.05).to_dict()
    
    for feature in feature_cols:
        
        # Only analyze factors that are currently high (non-zero or above a minimal threshold)
        current_val = X_single[feature].iloc[0]
        if current_val > percentiles_5[feature] * 1.5: 
            
            # Create a counterfactual where this specific risk factor is reduced to the 5th percentile ("safe")
            X_temp = X_base.copy()
            X_temp[feature] = percentiles_5[feature]
            
            # Predict probability when feature is "removed"
            p_counterfactual = model.predict_proba(X_temp)[0, 1]
            
            # The delta is the decrease in risk from "removing" this factor
            delta_prob = p_base - p_counterfactual
            
            if delta_prob > 0.001: # Filter out negligible drivers
                pareto_list.append({
                    'Risk Driver': feature.replace('_rate', '').replace('_', ' ').title(),
                    'delta_prob_raw': delta_prob
                })
                
    if not pareto_list:
        return pd.DataFrame()

    pred_pareto = pd.DataFrame(pareto_list).sort_values(
        'delta_prob_raw', ascending=False
    ).reset_index(drop=True)
    
    # Calculate Pareto contribution
    total_delta = pred_pareto['delta_prob_raw'].sum()
    
    if total_delta > 1e-6:
        pred_pareto['share_%'] = (pred_pareto['delta_prob_raw'] / total_delta) * 100
        pred_pareto['cumulative_%'] = pred_pareto['share_%'].cumsum()
        
        # Filter for the top 80% contribution
        pred_pareto = pred_pareto[pred_pareto['cumulative_%'] <= 80.0].copy()
    
    return pred_pareto


# --- 5. DASHBOARD LAYOUT & EXECUTION ---

# --- 5.1. CONFIGURATION (Sidebar) ---
st.sidebar.title("Simulation Configuration")

st.sidebar.subheader("Financial Metrics (USD)")
MATERIAL_COST_PER_LB = st.sidebar.number_input(
    "Material Cost/Value per lb ($)", value=2.50, step=0.10, format="%.2f", key='sidebar_material_cost'
)
LABOR_OVERHEAD_COST_PER_LB = st.sidebar.number_input(
    "Labor/Overhead Cost per lb ($)", value=0.50, step=0.10, format="%.2f", key='sidebar_labor_cost'
)
AVG_NON_MATERIAL_COST_PER_FAILURE = st.sidebar.number_input(
    "Non-Material Cost per Failure ($)", value=150.00, step=10.00, key='sidebar_failure_cost'
)


# --- 5.2. MODEL VALIDATION AND TUNING (Part 1) ---
st.title("Foundry Production Risk and Reliability Dashboard")
st.markdown("---")

st.header("1. Model Validation and Tuning")
st.markdown("Adjust the scaling factors to align the model's predictions with observed scrap rates.")

tuning_cols = st.columns([0.4, 0.6])

with tuning_cols[0]:
    st.subheader("Prediction Tuning Controls")
    
    s_param = st.select_slider("Overall Risk Scale (s)", options=S_GRID.round(2), value=1.0, key='s_tuning')
    gamma_param = st.select_slider("Part-Specific Prevalence Weight (γ)", options=GAMMA_GRID.round(2), value=1.0, key='gamma_tuning')

    st.caption(f"Adjusted Risk = Raw Prob × **{s_param:.2f}** (s) × (Part Scale $^\gamma$ **{gamma_param:.2f}**)")


with tuning_cols[1]:
    st.subheader("Statistical Validation")
    
    # Wilcoxon Signed-Rank Test Implementation
    ml_residuals = p_test - y_test
    global_mean = y_test.mean()
    baseline_residuals = global_mean - y_test
    
    wilcoxon_result = "N/A"
    wilcoxon_delta = "N/A"
    p_wilcoxon = 1.0
    
    try:
        # Test if ML residuals are significantly smaller than baseline residuals
        stat, p_wilcoxon = wilcoxon(np.abs(ml_residuals), np.abs(baseline_residuals), alternative='less')
        
        if p_wilcoxon < 0.05:
            wilcoxon_result = "Statistically superior to constant baseline."
            wilcoxon_delta = f"Model is more effective (p={p_wilcoxon:.4f})"
        else:
            wilcoxon_result = "No statistical improvement over baseline."
            wilcoxon_delta = f"No significant improvement (p={p_wilcoxon:.4f})"
            
    except ValueError:
        wilcoxon_result = "Cannot run Wilcoxon test (data constraint)."
        wilcoxon_delta = "N/A"

    st.metric(
        label="ML Model vs. Constant Baseline (Wilcoxon Test)",
        value=wilcoxon_result,
        delta=wilcoxon_delta,
        delta_color="normal" if p_wilcoxon < 0.05 else "off",
        help="Compares the absolute errors of the ML model against a simple model that always predicts the global mean. A significant result (p < 0.05) means the ML model is better."
    )
    
    # Model Diagnostics (Brier Score)
    test_brier = brier_score_loss(y_test, p_test) if len(X_test) else np.nan
    st.write(f"**Test Brier Score Loss:** {test_brier:.4f} (Lower is better)")
    st.write(f"**Calibration Method:** {calib_method}")
    st.caption("Diagnostics are based on the model's hold-out test set.")

st.markdown("---")

# --- 5.3. WORK ORDER RISK & RELIABILITY PREDICTION (Part 2) ---
st.header("2. Work Order Risk & Reliability Prediction")
st.markdown("Input the conditions for the next production run to estimate risk and immediate defect drivers.")

# UI for Input
input_cols = st.columns(4)
unique_part_ids = sorted(df_avg['part_id'].unique().tolist())
default_part = unique_part_ids[0] if unique_part_ids else 'None'
default_weight = df_avg.loc[df_avg['part_id'] == default_part, 'Avg_Piece_Weight'].iloc[0].round(2) if not df_avg.empty else 1.0
default_quantity = int(df_avg.loc[df_avg['part_id'] == default_part, 'Avg_Order_Quantity'].iloc[0]) if not df_avg.empty else 100

with input_cols[0]:
    selected_part_id = st.selectbox("A. Select Part ID", options=unique_part_ids, index=0, key='part_id_select')
with input_cols[1]:
    input_weight = st.number_input("B. Piece Weight (lbs)", value=default_weight, min_value=0.01, key='input_weight')
with input_cols[2]:
    input_quantity = st.number_input("C. Number of Pieces in Run", value=default_quantity, min_value=1, key='input_quantity')
with input_cols[3]:
    # Placeholder for financial output/cost estimate, using input_cost for context
    st.metric("Estimated Run Value ($)", f"${input_weight * input_quantity * (MATERIAL_COST_PER_LB + LABOR_OVERHEAD_COST_PER_LB):,.0f}", help="Total material and labor/overhead cost of the run.")


st.markdown("**D. Observed Defect Rates (Input current process rates)**")
# Defect Rate Inputs (Sliders based on the feature_cols)
cols_per_row = 4
feature_cols_display = st.columns(cols_per_row)
input_data = {}
for i, feature in enumerate(feature_cols):
    col_index = i % cols_per_row
    with feature_cols_display[col_index]:
        # Determine step based on data magnitude
        max_val = df_ml[feature].max()
        step_val = max(0.01, max_val / 20)
        
        # Use the mean rate for the selected part as the default input
        part_mean_rate = df_ml[df_ml['part_id'] == selected_part_id][feature].mean()
        
        # Unique key for defect rate inputs
        input_data[feature] = st.slider(
            feature.replace('_rate', '').replace('_', ' ').title(), 
            min_value=0.0,
            max_value=max_val * 1.5, # Allow input above historical max for simulation
            value=float(part_mean_rate), 
            step=step_val,
            format="%.3f",
            key=f"pred_input_{feature}"
        )
        
# Prepare data for prediction
X_single = pd.DataFrame([input_data])[feature_cols]

# Make the prediction
raw_prob, adjusted_risk, p_scale, risk_vs_baseline = predict_low_yield(
    model, X_single, selected_part_id, part_prevalence_scale, s_param, gamma_param, global_prevalence
)

# Calculate Reliability Metrics
mttf_hist, mttf_pred, reliability_next_run = calculate_mttf_reliability(
    selected_part_id, df_avg, raw_prob
)

st.markdown("### Risk & Reliability Metrics")
metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric(
        label="Probability of Low Yield (Adjusted Risk)",
        value=f"{adjusted_risk * 100:.1f}%",
        delta=f"vs. Global Avg: {risk_vs_baseline * 100:+.1f}pp",
        delta_color="inverse"
    )
with metric_cols[1]:
    st.metric(
        label="Reliability of Next Run",
        value=f"{reliability_next_run * 100:.1f}%",
        help="Probability of successfully completing this run without a Low Yield event (1 - Raw Probability)."
    )
with metric_cols[2]:
    st.metric(
        label="MTTF Predicted (Runs to Failure)",
        value=f"{mttf_pred:,.0f}" if mttf_pred != np.inf else "Infinity",
        help="Predicted Mean Runs To Failure based on the current run's Raw Model Probability."
    )
with metric_cols[3]:
    st.metric(
        label="MTTF Historical (Runs to Failure)",
        value=f"{mttf_hist:,.0f}" if mttf_hist != np.inf else "Infinity",
        help="Mean Runs To Failure based on Part ID's max historical scrap rate."
    )

st.markdown("---")

# --- Pareto Comparison (Predicted) ---
st.subheader("Top Defect Drivers: Predicted Pareto Analysis")
st.markdown("This shows which currently high defect rates are contributing most to the predicted low-yield risk.")

# Calculate Paretos (Using a clone of the model for safe resource usage)
model_clone_for_pareto = clone(model)
pred_pareto = calculate_predicted_pareto(model_clone_for_pareto, X_single, feature_cols, raw_prob)

if not pred_pareto.empty:
    st.dataframe(
        pred_pareto.assign(
            delta_prob_raw=lambda d: (d["delta_prob_raw"] * 100).round(2)
        ).assign(**{
            "share_%": lambda d: d["share_%"].round(1),
            "cumulative_%": lambda d: d["cumulative_%"].round(1),
        }).rename(columns={"delta_prob_raw": "Δ Prob (pp)"}),
        use_container_width=True
    )
    st.caption("The decrease in the Raw Model Probability (in percentage points) if this risk factor were hypothetically reduced to a 'safe' level. Factors driving the top 80% of the risk delta are shown.")
else:
    st.info("No significant risk drivers found for this combination of defect inputs. The run is likely low-risk.")
    
st.markdown("---")

# --- 5.4. Part-Level Data Overview (Part 3) ---
st.header("3. Part-Level Data Overview")
st.markdown("A look at the baseline performance metrics across all parts.")

# Calculate current averages for display (in percentage)
df_display = df_avg.assign(**{
    'Scrap_Percent_Baseline': lambda d: (d['Scrap_Percent_Baseline'] * 100).round(2),
}).rename(columns={'Scrap_Percent_Baseline': 'Max Scrap % (Baseline)'})

st.dataframe(
    df_display[['part_id', 'Total_Runs', 'Avg_Order_Quantity', 'Avg_Piece_Weight', 'Max Scrap % (Baseline)']],
    use_container_width=True
)
