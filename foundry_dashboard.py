# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
# Suppress the NumPy/Pandas warnings that often occur during data manipulation
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import streamlit as st
import math
import re 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import mannwhitneyu
from pandas.util import hash_pandas_object # Required for Streamlit caching fix

# --- CONSTANTS AND CONFIG ---
# Used for the single-sample filtering fix
TARGET_COLUMN = "is_scrapped" 
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
# (Note: Other constants from the full application have been preserved below)

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
# Hardcoded max cycles for simulation stability
MAX_CYCLES = 100

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
    Loads, cleans, and prepares data. It now explicitly drops the original Work Order ID column 
    (assumed to be index 0) and replaces it with the row index 
    to guarantee a valid unique identifier for ML and aggregation.
    
    CRITICAL FIX: Since all runs have scrap, the binary target is now based on 
    being above the median scrap percentage, creating two classes (high/low yield).
    """
    try:
        # NOTE: Assumes 'anonymized_parts.csv' is in the same directory.
        df_historical = pd.read_csv('anonymized_parts.csv')
        
        # 1. CRITICAL FIX: Ignore and drop the original Work Order ID column by its position (Index 0).
        
        # Create a guaranteed unique run ID based on the index before dropping the original column.
        df_historical['work_order_id'] = df_historical.index 
        
        # Drop the original first column, regardless of its header content.
        df_historical.drop(df_historical.columns[0], axis=1, inplace=True)


        # 2. Apply Hyper-aggressive, universal cleaning to ALL remaining columns
        df_historical.columns = [clean_col_name(c) for c in df_historical.columns]
        
        # 3. Standardize the rest of the columns
        final_rename_map = {}
        for col in df_historical.columns:
            # We must be careful not to rename 'work_order_id' which we just created
            if col == 'work_order_id':
                continue
            elif 'part_id' in col:
                final_rename_map[col] = 'part_id'
            elif 'scrap_percent' in col:
                final_rename_map[col] = 'scrap_percent_hist'
            elif 'order_quantity' in col:
                final_rename_map[col] = 'order_quantity'
            elif 'pieces_scrapped' in col:
                final_rename_map[col] = 'pieces_scrapped'
            elif 'piece_weight_lbs' in col:
                final_rename_map[col] = 'piece_weight_lbs'
            
        df_historical.rename(columns=final_rename_map, inplace=True)
        
        # Convert historical scrap percentage to decimal
        if 'scrap_percent_hist' in df_historical.columns:
            df_historical['scrap_percent_hist'] = df_historical['scrap_percent_hist'] / 100.0
        
        # --- Data for Simulation (df_avg: Averages/Max rates per Part ID) ---
        # We use 'work_order_id' for count, which is now the safe row index
        df_avg = df_historical.groupby('part_id').agg(
            Scrap_Percent_Baseline=('scrap_percent_hist', 'max'),
            Avg_Order_Quantity=('order_quantity', 'mean'),
            Avg_Piece_Weight=('piece_weight_lbs', 'mean'),
            Total_Runs=('work_order_id', 'count') 
        ).reset_index()

        df_avg['Est_Annual_Scrap_Weight_lbs'] = df_avg.apply(
            lambda row: row['Scrap_Percent_Baseline'] * row['Avg_Order_Quantity'] * row['Avg_Piece_Weight'] * min(52, row['Total_Runs']), 
            axis=1
        )
        
        # Calculate total historical scrap pieces for cost avoidance estimation
        df_historical['scrap_pieces'] = df_historical['pieces_scrapped'] 
        total_historical_scrap_pieces = df_historical['scrap_pieces'].sum()
        
        # --- Data for Machine Learning (df_ml: Work Orders with Binary Scrap Causes) ---
        df_ml = df_historical.copy()
        
        # Determine the threshold for 'Low Yield' based on the median
        scrap_median = df_ml['scrap_percent_hist'].median()
        
        # Target variable for ML: 1 if LOW YIELD (HIGH scrap, above median), 0 if HIGH YIELD (LOW scrap, at or below median)
        df_ml[TARGET_COLUMN] = (df_ml['scrap_percent_hist'] > scrap_median).astype(int)
        
        # Check if the median split resulted in two classes. If not (e.g., all values are identical), fall back to mean.
        if df_ml[TARGET_COLUMN].nunique() < 2:
            scrap_mean = df_ml['scrap_percent_hist'].mean()
            df_ml[TARGET_COLUMN] = (df_ml['scrap_percent_hist'] > scrap_mean).astype(int)
            st.sidebar.warning(f"Using mean ({scrap_mean:.4f}) to split scrap groups. Target 1 = LOW YIELD runs.")
        else:
            st.sidebar.info(f"Using median ({scrap_median:.4f}) to split runs into Low Yield (1) and High Yield (0) groups.")
            
        # Ensure we always have two classes before proceeding
        if df_ml[TARGET_COLUMN].nunique() < 2:
            # If even the mean split fails, something is fundamentally wrong with the data variance.
            st.error("Cannot create a binary target: all scrap percentages are identical. Model training is impossible.")
            return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()


        # Features for ML: Scrap Cause columns (Rate columns)
        # Find all columns ending in '_rate'
        rate_cols = [col for col in df_ml.columns if col.endswith('_rate')]
        
        # Convert rate columns into binary features (1 if cause was present, 0 otherwise)
        for col in rate_cols:
            df_ml[col] = (df_ml[col] > 0).astype(int)

        return df_avg, df_ml, total_historical_scrap_pieces, df_historical

    except FileNotFoundError:
        st.error("Error: 'anonymized_parts.csv' not found. Please ensure the file is correctly named and available.")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()
    except Exception as e:
        st.error(f"A severe error occurred during data processing: {e}")
        st.info("Please verify the structure of your CSV, especially column headers.")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()

# Load and prepare data
df_avg, df_ml, total_historical_scrap_pieces, df_historical = load_and_prepare_data()

if df_avg.empty or df_ml.empty:
    st.stop() # Stop the script if data loading failed

# --- 3. MACHINE LEARNING MODEL (with single-sample class filtering fix and DEBUG) ---

@st.cache_data(
    show_spinner="Training Model...",
    hash_funcs={
        pd.DataFrame: lambda df: hash_pandas_object(df, index=True).sum(),
        CalibratedClassifierCV: lambda model: str(model.get_params()),
        RandomForestClassifier: lambda model: str(model.get_params())
    }
)
def train_random_forest_model(df_ml: pd.DataFrame):
    """
    Trains and calibrates a Random Forest Classifier.
    Includes filtering for single-sample classes and diagnostic output.
    """
    if df_ml.empty:
        st.error("The ML DataFrame is empty. Cannot train model.")
        return None, pd.DataFrame(), pd.Series(dtype=int), [], "None", pd.Series(dtype=float)

    # =========================================================================
    # DEBUG START: Show initial class counts
    # =========================================================================
    initial_counts = df_ml[TARGET_COLUMN].value_counts()
    st.sidebar.info(
        f"Initial Target Class Counts:\n{initial_counts.to_string()}", 
        icon="🔍"
    )
    # =========================================================================
    
    # 1. Count the occurrences of each class in the target column
    class_counts = df_ml[TARGET_COLUMN].value_counts()
    
    # 2. Identify classes with only one sample (these cause the ValueError)
    # NOTE: Since the target is now 0 (High Yield) and 1 (Low Yield), this filter 
    #       is still necessary if one of the new classes only has one sample.
    single_sample_classes = class_counts[class_counts == 1].index.tolist()
    
    df_ml_original_len = len(df_ml)
    
    if single_sample_classes:
        # 3. Filter the DataFrame to exclude rows with single-sample classes
        df_filtered = df_ml[~df_ml[TARGET_COLUMN].isin(single_sample_classes)].copy()
        
        st.warning(
            f"Filtered out {df_ml_original_len - len(df_filtered)} row(s) because the target variable "
            f"'{TARGET_COLUMN}' had a class with only one sample (class label(s): {single_sample_classes}). "
            "This prevents the `ValueError` during stratified splitting."
        )
        df_ml = df_filtered
        
    # =========================================================================
    # DEBUG END: Show final class counts before the critical check
    # =========================================================================
    final_counts = df_ml[TARGET_COLUMN].value_counts()
    st.sidebar.info(
        f"Final Target Class Counts (after filter):\n{final_counts.to_string()}", 
        icon="✅"
    )
    # =========================================================================

    if df_ml.empty or df_ml[TARGET_COLUMN].nunique() < 2:
        st.error(
            "Final Error: After filtering, the data contains only one class or is empty. "
            "Model training cannot proceed without at least two classes (0 and 1)."
        )
        return None, pd.DataFrame(), pd.Series(dtype=int), [], "None", pd.Series(dtype=float)

    # Identify feature columns (ending with '_rate' from data loading)
    feature_cols = [col for col in df_ml.columns if col.endswith('_rate')] 
    
    # Define features (X) and target (y)
    X = df_ml[feature_cols].fillna(0) 
    y = df_ml[TARGET_COLUMN].astype(int)

    # Train-test split (now safe from single-sample classes, assuming final checks pass)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Model configuration
    base_model = RandomForestClassifier(
        n_estimators=DEFAULT_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    # Calibration selection
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

    # Rename the output for clarity on what the model is predicting
    st.caption("NOTE: The model now predicts the probability of a run being **Low Yield** (High Scrap, above the median) versus High Yield (Low Scrap, at or below the median).")


    return model, X_test, y_test, feature_cols, calib_method, p_test


# --- 4. SIMULATION & CALCULATION FUNCTIONS ---

def calculate_cycles_to_target(initial_scrap_rate, reduction_factor, target_scrap):
# ... (rest of the file remains unchanged from here) ...
