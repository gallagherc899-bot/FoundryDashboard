# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import streamlit as st

# Assuming these imports are already present
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# --- IMPORTANT: DEFINE YOUR TARGET COLUMN ---
# You must ensure this variable matches the name of your target column 
# (the column containing the class labels, e.g., 0/1 for binary classification)
TARGET_COLUMN = "is_scrapped" 

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
# ... (rest of constants) ...

@st.cache_data(show_spinner=False)
def train_random_forest_model(df_ml: pd.DataFrame):
    """
    Trains and calibrates a Random Forest Classifier.

    Includes a robust filter to remove single-sample classes from the target 
    variable to prevent ValueError during stratified train_test_split.
    """
    if df_ml.empty:
        st.error("The ML DataFrame is empty. Cannot train model.")
        return None, pd.DataFrame(), pd.Series(), [], "None", pd.Series()

    # =========================================================================
    #  << FIX: Filter out single-sample classes to prevent Stratification Error >>
    # =========================================================================
    
    # 1. Count the occurrences of each class in the target column
    class_counts = df_ml[TARGET_COLUMN].value_counts()
    
    # 2. Identify classes with only one sample (these cause the ValueError)
    single_sample_classes = class_counts[class_counts == 1].index.tolist()
    
    if single_sample_classes:
        # 3. Filter the DataFrame to exclude rows with single-sample classes
        df_filtered = df_ml[~df_ml[TARGET_COLUMN].isin(single_sample_classes)].copy()
        
        # Streamlit notification for the user
        st.warning(
            f"Filtered out {len(df_ml) - len(df_filtered)} row(s) because the target variable "
            f"'{TARGET_COLUMN}' had a class with only one sample (class label(s): {single_sample_classes}). "
            "This prevents the `ValueError` during stratified splitting."
        )
        df_ml = df_filtered
        
    if df_ml.empty or df_ml[TARGET_COLUMN].nunique() < 2:
        st.error("After filtering, there are insufficient data or classes (need >= 2) to train the model.")
        return None, pd.DataFrame(), pd.Series(), [], "None", pd.Series()
    # =========================================================================


    # Assuming feature_cols are derived from df_ml columns, excluding ID/Date/Target
    feature_cols = [col for col in df_ml.columns if col not in [TARGET_COLUMN, 'other_id_cols', 'date_col']] 
    
    # Define features (X) and target (y)
    # Assuming fillna(0) for missing features based on common ML preprocessing
    X = df_ml[feature_cols].fillna(0) 
    y = df_ml[TARGET_COLUMN].astype(int)

    # Train-test split (this is where the stratification previously failed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Model configuration
    model = RandomForestClassifier(
        n_estimators=DEFAULT_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    # Calibration selection (assuming method is pre-determined)
    calib_method = "isotonic" 

    # Train and calibrate
    model = CalibratedClassifierCV(
        estimator=model,
        method=calib_method,
        cv=5
    )
    model.fit(X_train, y_train)

    # Predict probabilities
    p_test = model.predict_proba(X_test)[:, 1] # Probability of class 1

    return model, X_test, y_test, feature_cols, calib_method, p_test
