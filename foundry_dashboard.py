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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import mannwhitneyu
from pandas.util import hash_pandas_object # Required for Streamlit caching fix
from sklearn.base import clone # REQUIRED FOR FEATURE IMPORTANCE FIX

# --- CONSTANTS AND CONFIG ---
# Used for the single-sample filtering fix
TARGET_COLUMN = "is_scrapped" 
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
# Hardcoded max cycles for simulation stability
MAX_CYCLES = 100

# Prediction scaling and tuning parameters
S_GRID = np.linspace(0.8, 1.2, 5) # Tuning factor for overall risk (s)
GAMMA_GRID = np.linspace(0.8, 1.2, 5) # Tuning factor for part-specific prevalence (gamma)


# Use the existing Streamlit page configuration structure
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

# --- 1. CONFIGURATION (Moved to Streamlit Sidebar) ---
st.sidebar.title("Simulation Configuration")

# Financial Inputs
st.sidebar.subheader("Financial Metrics (USD)")
MATERIAL_COST_PER_LB = st.sidebar.number_input(
    "Material Cost/Value per lb ($)", value=2.50, step=0.10, format="%.2f"
)
LABOR_OVERHEAD_COST_PER_LB = st.sidebar.number_input(
    "Labor/Overhead Cost per lb ($)", value=0.50, step=0.10, format="%.2f"
)
AVG_NON_MATERIAL_COST_PER_FAILURE = st.sidebar.number_input(
    "Non-Material Cost per Failure ($)", value=150.00, step=10.00
)

# Simulation Target and Effort
st.sidebar.subheader("Target & Effort")
TARGET_SCRAP_PERCENT = st.sidebar.slider(
    "Ultimate Scrap Target (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1
)
REDUCTION_SCENARIOS = [
    0.30, 0.25, 0.20, 0.15, 0.10, 0.05
]

# Prediction Tuning Sliders
st.sidebar.subheader("Prediction Tuning (Expert Only)")
s_param = st.sidebar.select_slider("Overall Risk Scale (s)", options=S_GRID.round(2), value=1.0)
gamma_param = st.sidebar.select_slider("Part-Specific Prevalence Weight (γ)", options=GAMMA_GRID.round(2), value=1.0)


# --- 2. DATA LOADING AND PREPARATION ---

def clean_col_name(col_raw):
    """
    Hyper-aggressive cleaning function: converts to snake_case, removes symbols, 
    and handles common separators like # and %.
    """
    col = str(col_raw).strip().lower()
    # 1. Normalize common separators and symbols
    col = col.replace(' ', '_').replace('#', '_id').replace('%', '_percent')
    
    # 2. Aggressively remove all non-word characters (except underscore)
    col = re.sub(r'[^\w_]', '', col)
    
    # 3. Ensure no double underscores remain
    while '__' in col:
        col = col.replace('__', '_')
        
    return col.strip('_')

@st.cache_data
def load_and_prepare_data():
    """
    Loads, cleans, and prepares data, and calculates prevalence scales needed for prediction.
    """
    try:
        # NOTE: Assumes 'anonymized_parts.csv' is in the same directory.
        df_historical = pd.read_csv('anonymized_parts.csv')
        
        # 1. CRITICAL FIX: Ignore and drop the original Work Order ID column by its position (Index 0).
        df_historical['work_order_id'] = df_historical.index 
        df_historical.drop(df_historical.columns[0], axis=1, inplace=True)

        # 2. Apply Hyper-aggressive, universal cleaning to ALL remaining columns
        df_historical.columns = [clean_col_name(c) for c in df_historical.columns]
        
        # 3. Standardize the rest of the columns
        final_rename_map = {}
        for col in df_historical.columns:
            if col == 'work_order_id': continue
            elif 'part_id' in col: final_rename_map[col] = 'part_id'
            elif 'scrap_percent' in col: final_rename_map[col] = 'scrap_percent_hist'
            elif 'order_quantity' in col: final_rename_map[col] = 'order_quantity'
            elif 'pieces_scrapped' in col: final_rename_map[col] = 'pieces_scrapped'
            elif 'piece_weight_lbs' in col: final_rename_map[col] = 'piece_weight_lbs'
            
        df_historical.rename(columns=final_rename_map, inplace=True)
        
        # Convert historical scrap percentage to decimal
        if 'scrap_percent_hist' in df_historical.columns:
            df_historical['scrap_percent_hist'] = df_historical['scrap_percent_hist'] / 100.0
        
        # --- Data for Simulation (df_avg: Averages/Max rates per Part ID) ---
        df_avg = df_historical.groupby('part_id').agg(
            Scrap_Percent_Baseline=('scrap_percent_hist', 'max'),
            Avg_Order_Quantity=('order_quantity', 'mean'),
            Avg_Piece_Weight=('piece_weight_lbs', 'mean'),
            Total_Runs=('work_order_id', 'count') 
        ).reset_index()

        # FIX: Used np.minimum for vectorized operation
        df_avg['Est_Annual_Scrap_Weight_lbs'] = df_avg.apply(
            lambda row: row['Scrap_Percent_Baseline'] * row['Avg_Order_Quantity'] * row['Avg_Piece_Weight'] * np.minimum(52, row['Total_Runs']), 
            axis=1
        )
        
        # Calculate total historical scrap pieces for cost avoidance estimation
        df_historical['scrap_pieces'] = df_historical['pieces_scrapped'] 
        total_historical_scrap_pieces = df_historical['scrap_pieces'].sum()
        
        # --- Data for Machine Learning (df_ml: Work Orders with Binary Scrap Causes) ---
        df_ml = df_historical.copy()
        
        # Target variable: 1 if LOW YIELD (HIGH scrap, above median), 0 if HIGH YIELD (LOW scrap, at or below median)
        scrap_median = df_ml['scrap_percent_hist'].median()
        df_ml[TARGET_COLUMN] = (df_ml['scrap_percent_hist'] > scrap_median).astype(int)
        
        # Handle single-class data case (fallback to mean)
        if df_ml[TARGET_COLUMN].nunique() < 2:
            scrap_mean = df_ml['scrap_percent_hist'].mean()
            df_ml[TARGET_COLUMN] = (df_ml['scrap_percent_hist'] > scrap_mean).astype(int)
            st.sidebar.warning(f"Using mean ({scrap_mean:.4f}) to split scrap groups. Target 1 = LOW YIELD runs.")
        else:
            st.sidebar.info(f"Using median ({scrap_median:.4f}) to split runs into Low Yield (1) and High Yield (0) groups.")
            
        if df_ml[TARGET_COLUMN].nunique() < 2:
            st.error("Cannot create a binary target: all scrap percentages are identical. Model training is impossible.")
            return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame(), {}, 0.0

        # Calculate global prevalence for prediction scaling
        global_prevalence = df_ml[TARGET_COLUMN].mean()
        
        # Calculate part-specific prevalence scale
        low_yield_runs_per_part = df_ml.groupby('part_id')[TARGET_COLUMN].sum()
        total_runs_per_part = df_ml.groupby('part_id')['work_order_id'].count()
        part_prevalence = low_yield_runs_per_part / total_runs_per_part
        
        # Create a dictionary of part_id to its prevalence scale relative to the global mean
        # Handle cases where global_prevalence is near zero (though checked above)
        part_prevalence_scale = (part_prevalence / global_prevalence).replace([np.inf, -np.inf], 0).fillna(0).to_dict()

        # Features for ML: Scrap Cause columns (Rate columns)
        rate_cols = [col for col in df_ml.columns if col.endswith('_rate')]
        
        # Convert rate columns into binary features (1 if cause was present, 0 otherwise)
        for col in rate_cols:
            df_ml[col] = (df_ml[col] > 0).astype(int)

        return df_avg, df_ml, total_historical_scrap_pieces, df_historical, part_prevalence_scale, global_prevalence

    except FileNotFoundError:
        st.error("Error: 'anonymized_parts.csv' not found. Please ensure the file is correctly named and available.")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame(), {}, 0.0
    except Exception as e:
        st.error(f"A severe error occurred during data processing: {e}")
        st.info("Please verify the structure of your CSV, especially column headers.")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame(), {}, 0.0


# Load and prepare data (Updated call)
df_avg, df_ml, total_historical_scrap_pieces, df_historical, part_prevalence_scale, global_prevalence = load_and_prepare_data()

if df_avg.empty or df_ml.empty:
    st.stop() # Stop the script if data loading failed

# Extract feature column names for reuse
feature_cols = [col for col in df_ml.columns if col.endswith('_rate')]


# --- 3. MACHINE LEARNING MODEL (Training) ---

@st.cache_data(
    show_spinner="Training Model...",
    hash_funcs={
        pd.DataFrame: lambda df: hash_pandas_object(df, index=True).sum(),
        CalibratedClassifierCV: lambda model: str(model.get_params()),
        RandomForestClassifier: lambda model: str(model.get_params())
    }
)
def train_random_forest_model(df_ml: pd.DataFrame, feature_cols: list):
    """
    Trains and calibrates a Random Forest Classifier.
    Returns: model, X_train, X_test, y_train, y_test, calib_method, p_test
    """
    if df_ml.empty or df_ml[TARGET_COLUMN].nunique() < 2:
        return None, pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), "None", pd.Series()

    # 1. Filter out single-sample classes to prevent StratifiedKFold error
    class_counts = df_ml[TARGET_COLUMN].value_counts()
    single_sample_classes = class_counts[class_counts == 1].index.tolist()
    if single_sample_classes:
        df_ml = df_ml[~df_ml[TARGET_COLUMN].isin(single_sample_classes)].copy()
        
    if df_ml[TARGET_COLUMN].nunique() < 2:
        st.error("Model training failed: insufficient data after cleanup.")
        return None, pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), "None", pd.Series()

    # Define features (X) and target (y)
    X = df_ml[feature_cols].fillna(0) 
    y = df_ml[TARGET_COLUMN].astype(int)

    # Train-test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    base_model = RandomForestClassifier(
        n_estimators=DEFAULT_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    calib_method = "isotonic" 

    # Train and calibrate
    model = CalibratedClassifierCV(
        estimator=base_model,
        method=calib_method,
        cv=5
    )
    model.fit(X_train, y_train)

    # Predict probabilities
    p_test = model.predict_proba(X_test)[:, 1] # Probability of class 1

    return model, X_train, X_test, y_train, y_test, calib_method, p_test


# Train the model and retrieve all necessary data splits (Updated call)
model, X_train, X_test, y_train, y_test, calib_method, p_test = train_random_forest_model(df_ml, feature_cols)

if model is None:
    st.stop() # Stop the script if model training failed


# --- 4. SIMULATION & PREDICTION FUNCTIONS ---

def calculate_cycles_to_target(initial_scrap_rate, reduction_factor, target_scrap):
    """
    Calculates the number of cycles (e.g., weeks) required to reach the target 
    scrap rate, given an annual reduction factor applied to the excess.
    """
    target_rate = target_scrap / 100.0
    initial_rate = initial_scrap_rate 

    if initial_rate <= target_rate:
        return 0 
    
    current_rate = initial_rate
    cycles = 0
    
    while current_rate > target_rate and cycles < MAX_CYCLES: 
        reduction_amount = (current_rate - target_rate) * reduction_factor
        current_rate -= reduction_amount
        cycles += 1
    
    return cycles if cycles < MAX_CYCLES else MAX_CYCLES 

def calculate_feature_importances(model, feature_cols, X_train, y_train):
    """
    Calculates global feature importance by re-fitting the base Random Forest estimator 
    on the training data (X_train, y_train) to safely extract importances.
    """
    # FIX: model.estimator is the UN-FITTED base model. We must clone and fit it 
    # to the training data to get feature_importances_.
    try:
        base_rf = clone(model.estimator)
        base_rf.fit(X_train, y_train) # Fit on the data used for the main model's CV folds

        importances = pd.Series(
            base_rf.feature_importances_, 
            index=feature_cols
        ).sort_values(ascending=False)
    except Exception as e:
        st.error(f"Error during feature importance calculation: {e}. Check X_train/y_train contents.")
        return pd.DataFrame()

    # Calculate U-test p-values (Mann-Whitney U-test for feature significance)
    p_values = {}
    X_df = X_train.copy()
    X_df['target'] = y_train
    
    class_0 = X_df[X_df['target'] == 0]
    class_1 = X_df[X_df['target'] == 1]
    
    for feature in importances.index:
        if len(class_0) > 1 and len(class_1) > 1:
            try:
                # Compare the feature distributions in the two classes (0 and 1)
                # Since features are binary (0/1), this compares the prevalence of the feature in each target class
                stat, p = mannwhitneyu(class_0[feature].values, class_1[feature].values, alternative='two-sided')
                p_values[feature] = p
            except ValueError:
                p_values[feature] = 1.0 
        else:
            p_values[feature] = 1.0

    importance_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance (Gini)': importances.values,
        'P-Value (Mann-Whitney U)': [p_values.get(f, 1.0) for f in importances.index]
    })
    
    def get_star(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ''
    
    importance_df['Significance'] = importance_df['P-Value (Mann-Whitney U)'].apply(get_star)

    return importance_df

def run_simulation(df_avg, cost_material_per_lb, cost_labor_per_lb, cost_per_failure, target_scrap, reduction_scenarios):
    """
    Runs the cost avoidance simulation across different reduction scenarios.
    """
    cost_per_lb = cost_material_per_lb + cost_labor_per_lb
    
    # Calculate baseline costs (based on the annual estimated scrap weight)
    df_avg['Baseline_Cost'] = (
        df_avg['Est_Annual_Scrap_Weight_lbs'] * cost_per_lb 
        + df_avg['Total_Runs'] * cost_per_failure
    )
    total_baseline_cost = df_avg['Baseline_Cost'].sum()
    
    results = []
    
    for factor in reduction_scenarios:
        # Calculate new scrap rate after applying the factor to the excess over target
        df_avg[f'Reduced_Scrap_Rate_{factor}'] = df_avg.apply(
            lambda row: max(
                target_scrap/100.0, 
                row['Scrap_Percent_Baseline'] - (row['Scrap_Percent_Baseline'] - target_scrap/100.0) * factor
            ), 
            axis=1
        )
        
        # Calculate reduced scrap weight
        df_avg[f'Reduced_Scrap_Weight_{factor}'] = df_avg[f'Reduced_Scrap_Rate_{factor}'] * df_avg['Avg_Order_Quantity'] * df_avg['Avg_Piece_Weight'] * np.minimum(52, df_avg['Total_Runs'])
        
        # Calculate reduced cost
        df_avg[f'Reduced_Cost_{factor}'] = (
            df_avg[f'Reduced_Scrap_Weight_{factor}'] * cost_per_lb
            + df_avg['Total_Runs'] * cost_per_failure
        )
        
        total_reduced_cost = df_avg[f'Reduced_Cost_{factor}'].sum()
        cost_avoidance = total_baseline_cost - total_reduced_cost
        
        # Calculate cycles
        avg_baseline_scrap = df_avg['Scrap_Percent_Baseline'].mean()
        cycles = calculate_cycles_to_target(avg_baseline_scrap, factor, target_scrap)
        
        results.append({
            'Reduction_Effort': f"{int(factor * 100)}% Reduction on Excess",
            'Total_Cost_Avoidance': cost_avoidance,
            'Cycles_to_Target (Weeks)': cycles
        })
        
    df_results = pd.DataFrame(results)
    
    return df_results, total_baseline_cost

def predict_low_yield(model, X_single, part_id, part_prevalence_scale, s_param, gamma_param, global_prevalence):
    """
    Makes a prediction and applies the risk adjustment/scaling.
    Formula: Adjusted risk = calibrated prob × s × (part_scale^γ)
    """
    
    # 1. Calibrated Probability (Raw Prediction)
    raw_prob = model.predict_proba(X_single)[0, 1]
    
    # 2. Get Part Scaling Factor
    # Default to 1.0 if part_id is new/not in training data
    p_scale = part_prevalence_scale.get(part_id, 1.0)
    
    # 3. Apply the Adjusted Risk Formula
    adjusted_risk = raw_prob * s_param * (p_scale ** gamma_param)
    
    # Ensure the adjusted risk is a valid probability (between 0 and 1)
    adjusted_risk = np.clip(adjusted_risk, 0.0, 1.0)
    
    # Calculate the Risk vs Baseline (using global prevalence as a proxy for 'average baseline')
    risk_vs_baseline = adjusted_risk / global_prevalence if global_prevalence > 0 else 0
    
    return raw_prob, adjusted_risk, p_scale, risk_vs_baseline

def calculate_prediction_pareto(model, X_single, feature_cols, p_base):
    """
    Calculates the local feature importance (Pareto) for a single prediction.
    Measures the drop in probability when an active risk factor is hypothetically removed.
    """
    X_base = X_single.copy()
    pareto_list = []
    
    for feature in feature_cols:
        # Check if the risk factor is PRESENT (1) in the current prediction
        if X_single[feature].iloc[0] == 1:
            
            # Create a counterfactual where this specific active risk factor is removed (set to 0)
            X_temp = X_base.copy()
            X_temp[feature] = 0
            
            # Predict probability when feature is removed
            p_counterfactual = model.predict_proba(X_temp)[0, 1]
            
            # The delta is the decrease in risk from removing this factor
            delta_prob = p_base - p_counterfactual
            
            if delta_prob > 0.005: # Only consider meaningful drivers (0.5 percentage point change)
                pareto_list.append({
                    'feature': feature,
                    'delta_prob_raw': delta_prob
                })
                
    if not pareto_list:
        return pd.DataFrame()

    pred_pareto = pd.DataFrame(pareto_list).sort_values(
        'delta_prob_raw', ascending=False
    ).reset_index(drop=True)
    
    # Calculate Pareto contribution
    pred_pareto['cumulative_prob'] = pred_pareto['delta_prob_raw'].cumsum()
    total_delta = pred_pareto['delta_prob_raw'].sum()
    
    # Avoid division by zero if total_delta is extremely small
    if total_delta > 1e-6:
        pred_pareto['share_%'] = (pred_pareto['delta_prob_raw'] / total_delta) * 100
        pred_pareto['cumulative_%'] = (pred_pareto['cumulative_prob'] / total_delta) * 100
        
        # Filter for the top 80% contribution
        pred_pareto = pred_pareto[pred_pareto['cumulative_%'] <= 80.0].copy()
    else:
        pred_pareto['share_%'] = 0.0
        pred_pareto['cumulative_%'] = 0.0

    return pred_pareto.rename(columns={'feature': 'Risk Driver'})


# --- 5. DASHBOARD LAYOUT ---

st.title("Foundry Production Risk Dashboard")
st.markdown("---")

# --- 5.1. Causal Feature Analysis & Model Performance ---
st.header("1. Causal Feature Analysis & Model Performance")

# Updated call to use X_train and y_train
importance_df = calculate_feature_importances(model, feature_cols, X_train, y_train)

col_imp, col_diag = st.columns([0.6, 0.4])

with col_imp:
    st.subheader("Model Feature Importance (Top Drivers of Low Yield)")
    if not importance_df.empty:
        st.dataframe(
            importance_df.head(10).assign(**{
                'Importance (Gini)': lambda d: (d['Importance (Gini)'] * 100).round(2),
                'P-Value (Mann-Whitney U)': lambda d: d['P-Value (Mann-Whitney U)'].apply(lambda x: f"{x:.4f}")
            }).rename(columns={'Importance (Gini)': 'Importance (%)'}),
            use_container_width=True
        )
        st.caption("*Significance: *** < 0.001, ** < 0.01, * < 0.05. Importance is based on the Gini impurity reduction in the Random Forest model.")
    else:
        st.info("Feature Importance could not be calculated.")

with col_diag:
    st.subheader("Model Diagnostics")
    test_brier = np.nan
    try:
        test_brier = brier_score_loss(y_test, p_test) if len(X_test) and len(p_test) else np.nan
    except Exception:
        pass
    st.metric(
        label="Test Brier Score Loss",
        value=f"{test_brier:.4f}" if not np.isnan(test_brier) else "N/A",
        help="Lower is better. Measures the calibration and accuracy of probability predictions."
    )
    st.write(f"Calibration Method: **{calib_method}**")
    
st.markdown("---")

# --- 5.2. Cost Avoidance Simulation ---
st.header("2. Cost Avoidance Simulation")

simulation_results, total_baseline_cost = run_simulation(
    df_avg, 
    MATERIAL_COST_PER_LB, 
    LABOR_OVERHEAD_COST_PER_LB, 
    AVG_NON_MATERIAL_COST_PER_FAILURE, 
    TARGET_SCRAP_PERCENT, 
    REDUCTION_SCENARIOS
)

col_metric, col_table = st.columns([0.4, 0.6])

with col_metric:
    st.subheader("Baseline Metrics")
    st.metric(
        label="Total Historical Scrap Pieces Analyzed", 
        value=f"{total_historical_scrap_pieces:,.0f}"
    )
    st.metric(
        label="Estimated Annual Baseline Cost of Scrap/Failures",
        value=f"${total_baseline_cost:,.2f}",
        delta_color="off"
    )
    st.caption(f"Target Scrap Rate: {TARGET_SCRAP_PERCENT:.1f}%")

with col_table:
    st.subheader("Cost Avoidance Scenarios")
    
    # Format the table for display
    formatted_results = simulation_results.assign(**{
        'Total_Cost_Avoidance': lambda d: d['Total_Cost_Avoidance'].map('${:,.2f}'.format),
    }).rename(columns={
        'Total_Cost_Avoidance': 'Estimated Annual Cost Avoidance',
        'Cycles_to_Target (Weeks)': f'Cycles to Reach {TARGET_SCRAP_PERCENT:.1f}% Target'
    })
    
    st.dataframe(formatted_results, use_container_width=True)

st.markdown("---")


# --- 5.3. Work Order Risk Prediction ---
st.header("3. Work Order Risk Prediction")
st.markdown("Input the Part ID and any observed defect conditions to get a real-time Low Yield Risk prediction.")

# UI for Input
input_cols = st.columns([0.3, 0.7])

with input_cols[0]:
    # Part ID selection
    unique_part_ids = sorted(df_avg['part_id'].unique().tolist())
    selected_part_id = st.selectbox("Select Part ID", options=unique_part_ids, index=0)

with input_cols[1]:
    # Defect Rate Inputs (Only features ending in _rate)
    st.markdown("**Check all observed defect conditions (Input as 1/Present)**")
    
    # Create the input fields dynamically
    num_features = len(feature_cols)
    cols_per_row = 4
    
    # Initialize the input dictionary
    input_data = {'part_id': selected_part_id}
    
    # Create the input columns
    feature_cols_display = st.columns(cols_per_row)
    
    for i, feature in enumerate(feature_cols):
        col_index = i % cols_per_row
        with feature_cols_display[col_index]:
            # Use a checkbox to represent 1 (Present) or 0 (Absent)
            input_data[feature] = st.checkbox(feature.replace('_rate', ''), key=f"pred_input_{feature}")
            
# Convert boolean inputs to integers (1 or 0)
for feature in feature_cols:
    input_data[feature] = 1 if input_data[feature] else 0

# Prepare data for prediction
X_single = pd.DataFrame([input_data])[feature_cols]

# Make the prediction
raw_prob, adjusted_risk, p_scale, risk_vs_baseline = predict_low_yield(
    model, X_single, selected_part_id, part_prevalence_scale, s_param, gamma_param, global_prevalence
)

# Run Pareto analysis on the prediction
pred_pareto = calculate_prediction_pareto(model, X_single, feature_cols, raw_prob)

st.markdown("### Prediction Results")
result_cols = st.columns(3)

with result_cols[0]:
    st.metric(
        label="Adjusted Low Yield Risk",
        value=f"{adjusted_risk * 100:.1f}%",
        help="Calibrated probability adjusted for part-specific historical prevalence (tuned by S and γ)."
    )

with result_cols[1]:
    st.metric(
        label="Raw Model Probability",
        value=f"{raw_prob * 100:.1f}%",
        help="The direct output of the calibrated Random Forest model."
    )

with result_cols[2]:
    st.metric(
        label="Risk vs. Global Baseline",
        value=f"{risk_vs_baseline:.1f}x",
        help="How much more likely this run is to be Low Yield compared to the overall average run."
    )

st.markdown("---")
st.subheader("Prediction Drivers (Pareto Analysis)")

if not pred_pareto.empty:
    st.markdown("**Top 80% Local Drivers of Current Prediction**")
    st.dataframe(
        pred_pareto.assign(
            delta_prob_raw=lambda d: (d["delta_prob_raw"] * 100).round(2)
        ).assign(**{
            "share_%": lambda d: d["share_%"].round(1),
            "cumulative_%": lambda d: d["cumulative_%"].round(1),
        }).rename(columns={"delta_prob_raw": "Δ Prob (pp)"}),
        use_container_width=True
    )
    st.caption("Shows the decrease in the Raw Model Probability (in percentage points) if this risk factor were hypothetically removed. Factors driving the top 80% of the risk delta are shown.")
else:
    st.info("No significant risk drivers found for this combination of defect inputs.")
    
st.markdown("---")

# --- 5.4. Part-Level Data Overview ---
st.header("4. Part-Level Data Overview")

# Calculate current averages for display (in percentage)
df_display = df_avg.assign(**{
    'Scrap_Percent_Baseline': lambda d: (d['Scrap_Percent_Baseline'] * 100).round(2),
}).rename(columns={'Scrap_Percent_Baseline': 'Max Scrap % (Baseline)'})

st.dataframe(
    df_display[['part_id', 'Total_Runs', 'Avg_Order_Quantity', 'Avg_Piece_Weight', 'Max Scrap % (Baseline)']],
    use_container_width=True
)
st.caption("Baseline scrap represents the maximum historical scrap percentage observed for that Part ID.")

# End of File Generation
