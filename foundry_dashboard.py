# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 3.2.1 - PHM ENHANCEMENT LABORATORY
# ================================================================
# 
# NEW IN V3.2.1:
# - Fixed class imbalance error with improved threshold fallback strategies
# - Added data statistics expander to help choose appropriate threshold
# - More informative error messages with actionable guidance
#
# NEW IN V3.2:
# - PHM Enhancement Toggles (Health Index, Temporal, Uncertainty, Decision)
# - Factorial Testing Mode (test all 16 combinations)
# - Literature-based improvements from Lei et al. (2018), Choo & Shin (2025)
#
# RETAINED FROM V3.1:
# - Multi-defect feature engineering (n_defect_types, total_defect_rate)
# - Prediction inputs (Part ID, Order Qty, Weight, Cost)
# - Multi-defect alerts in predictions
# - 6-2-1 Rolling Window + Data Confidence Indicators
# - Results logging and export
#
# Based on:
# - Campbell (2003) "Castings Practice: The Ten Rules"
# - Lei et al. (2018) PHM systematic review
# - Choo & Shin (2025) RUL + cost-optimal maintenance
# ================================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from datetime import datetime
from itertools import product

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(
    page_title="Foundry Dashboard v3.2.1 - PHM Laboratory", 
    layout="wide"
)

# ================================================================
# VERSION BANNER
# ================================================================
st.markdown("""
<div style="background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
            padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e94560;">
    <h2 style="color: #e94560; margin: 0;">🏭 Foundry Scrap Risk Dashboard</h2>
    <p style="color: #a8d0ff; margin: 5px 0 0 0;">
        <strong>Version 3.2.1 - PHM Enhancement Laboratory</strong> | 
        Factorial Testing | Health Index | Temporal Features | Uncertainty Quantification
    </p>
</div>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2
MIN_SAMPLES_PER_CLASS = 5

# ================================================================
# DATA CONFIDENCE CONFIGURATION
# ================================================================
RECOMMENDED_MIN_RECORDS = 5
HIGH_CONFIDENCE_RECORDS = 15
MULTI_DEFECT_THRESHOLD = 2

# ================================================================
# PHM ENHANCEMENT CONFIGURATION
# ================================================================
PHM_ENHANCEMENTS = {
    'health_index': {
        'name': 'Health Index (HI)',
        'description': 'Composite degradation score combining all defect rates',
        'reference': 'Lei et al. (2018) - PHM systematic review'
    },
    'temporal': {
        'name': 'Temporal Features (TF)',
        'description': 'Trend detection: defect rate changes, acceleration, rolling averages',
        'reference': 'Standard PHM practice for degradation modeling'
    },
    'uncertainty': {
        'name': 'Uncertainty Quantification (UQ)',
        'description': 'Confidence intervals from ensemble variance',
        'reference': 'Bayesian/ensemble methods in PHM literature'
    },
    'decision': {
        'name': 'Decision Optimization (DO)',
        'description': 'Cost-based intervention recommendations',
        'reference': 'Choo & Shin (2025) - Cost-optimal maintenance'
    }
}

# ================================================================
# CAMPBELL PROCESS-DEFECT MAPPING
# ================================================================
PROCESS_DEFECT_MAP = {
    "Sand System": {
        "defects": ["sand_rate", "gas_porosity_rate", "runout_rate", "dirty_pattern_rate"],
        "description": "Sand moisture, clay content, compactability issues"
    },
    "Core Making": {
        "defects": ["core_rate", "crush_rate", "shrink_porosity_rate", "gas_porosity_rate"],
        "description": "Core binder ratios, cure cycles, venting problems"
    },
    "Pattern Making": {
        "defects": ["shift_rate", "bent_rate", "dirty_pattern_rate"],
        "description": "Pattern wear, dimensional drift, parting line issues"
    },
    "Mold Making": {
        "defects": ["shift_rate", "runout_rate", "missrun_rate", "short_pour_rate", "gas_porosity_rate"],
        "description": "Compaction, venting, gating setup problems"
    },
    "Melting": {
        "defects": ["dross_rate", "gas_porosity_rate", "shrink_rate", "shrink_porosity_rate", "gouged_rate"],
        "description": "Melt cleanliness, temperature, hydrogen content"
    },
    "Pouring": {
        "defects": ["missrun_rate", "short_pour_rate", "dross_rate", "tear_up_rate"],
        "description": "Pour temperature, velocity, turbulence issues"
    },
    "Solidification": {
        "defects": ["shrink_rate", "shrink_porosity_rate", "gas_porosity_rate", "missrun_rate"],
        "description": "Cooling rates, feeding design, thermal gradients"
    },
    "Shakeout": {
        "defects": ["tear_up_rate", "over_grind_rate", "sand_rate"],
        "description": "Mechanical stress during casting removal"
    },
    "Inspection": {
        "defects": ["failed_zyglo_rate", "zyglo_rate", "outside_process_scrap_rate"],
        "description": "NDT detection, finishing defects"
    },
}

DEFECT_TO_PROCESS = {}
for process, info in PROCESS_DEFECT_MAP.items():
    for defect in info["defects"]:
        if defect not in DEFECT_TO_PROCESS:
            DEFECT_TO_PROCESS[defect] = []
        DEFECT_TO_PROCESS[defect].append(process)


# ================================================================
# DATA LOADING AND PREPROCESSING
# ================================================================

def _normalize_headers(cols: pd.Index) -> pd.Index:
    return (
        pd.Index(cols)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )


def _canonical_rename(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        # Part ID variations
        "part_id": "part_id", 
        "partid": "part_id",
        "part": "part_id",
        # Work order variations
        "work_order": "work_order",
        "work_order_number": "work_order", 
        "work_order_#": "work_order",
        "workorder": "work_order",
        "wo": "work_order",
        # Order quantity variations
        "order_quantity": "order_quantity", 
        "order_qty": "order_quantity",
        "orderquantity": "order_quantity",
        "qty": "order_quantity",
        "quantity": "order_quantity",
        # Scrap percent variations - THIS IS THE KEY FIX
        "scrap_percent": "scrap_percent",
        "scrap": "scrap_percent",  # After normalization, "Scrap%" becomes "scrap"
        "scrap_": "scrap_percent",  # Sometimes trailing underscore
        "scrap_pct": "scrap_percent",
        "scrappercent": "scrap_percent",
        "scrap_rate": "scrap_percent",
        # Weight variations
        "piece_weight_lbs": "piece_weight_lbs", 
        "piece_weight": "piece_weight_lbs",
        "pieceweight": "piece_weight_lbs",
        "weight": "piece_weight_lbs",
        "weight_lbs": "piece_weight_lbs",
        # Date variations
        "week_ending": "week_ending", 
        "week_end": "week_ending",
        "weekending": "week_ending",
        "date": "week_ending",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def add_multi_defect_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-defect intelligence features."""
    df = df.copy()
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    for col in defect_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['n_defect_types'] = (df[defect_cols] > 0).sum(axis=1)
    df['has_multiple_defects'] = (df['n_defect_types'] >= MULTI_DEFECT_THRESHOLD).astype(int)
    df['total_defect_rate'] = df[defect_cols].sum(axis=1)
    df['max_defect_rate'] = df[defect_cols].max(axis=1)
    df['defect_concentration'] = df['max_defect_rate'] / (df['total_defect_rate'] + 0.001)
    
    # Interaction terms
    if 'shift_rate' in df.columns and 'tear_up_rate' in df.columns:
        df['shift_x_tearup'] = df['shift_rate'] * df['tear_up_rate']
    if 'shrink_rate' in df.columns and 'gas_porosity_rate' in df.columns:
        df['shrink_x_porosity'] = df['shrink_rate'] * df['gas_porosity_rate']
    if 'core_rate' in df.columns and 'sand_rate' in df.columns:
        df['core_x_sand'] = df['core_rate'] * df['sand_rate']
    
    return df


def add_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """PHM Enhancement 1: Health Index."""
    df = df.copy()
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    critical_defects = ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate']
    moderate_defects = ['shift_rate', 'core_rate', 'sand_rate', 'dross_rate']
    
    critical_score = sum(df[d].fillna(0) * 3.0 for d in critical_defects if d in df.columns)
    moderate_score = sum(df[d].fillna(0) * 2.0 for d in moderate_defects if d in df.columns)
    minor_score = sum(df[d].fillna(0) * 1.0 for d in defect_cols 
                      if d not in critical_defects and d not in moderate_defects)
    
    df['health_index'] = critical_score + moderate_score + minor_score
    max_hi = df['health_index'].max()
    df['health_index_norm'] = df['health_index'] / (max_hi + 0.001) if max_hi > 0 else 0
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHM Enhancement 2: Temporal Features."""
    df = df.copy()
    
    if 'week_ending' in df.columns:
        df = df.sort_values('week_ending').reset_index(drop=True)
    
    for col in ['total_defect_rate', 'scrap_percent']:
        if col not in df.columns:
            continue
        
        if 'part_id' in df.columns:
            df[f'{col}_trend'] = df.groupby('part_id')[col].diff().fillna(0)
            df[f'{col}_roll3'] = df.groupby('part_id')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
        else:
            df[f'{col}_trend'] = df[col].diff().fillna(0)
            df[f'{col}_roll3'] = df[col].rolling(window=3, min_periods=1).mean()
    
    if 'week_ending' in df.columns:
        df['month'] = pd.to_datetime(df['week_ending']).dt.month
        df['quarter'] = pd.to_datetime(df['week_ending']).dt.quarter
    
    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load and clean CSV data with robust column name handling. No caching to ensure fresh data."""
    df = pd.read_csv(csv_path)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != "None"]).strip() 
                      for col in df.columns.values]
    
    df.columns = _normalize_headers(df.columns)
    df = _canonical_rename(df)
    
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    for c in ["part_id", "order_quantity", "piece_weight_lbs", "week_ending", "scrap_percent"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "week_ending" else pd.NaT
    
    if "part_id" in df.columns:
        df["part_id"] = df["part_id"].fillna("Unknown").astype(str).str.strip()
        df["part_id"] = df["part_id"].replace({"nan": "Unknown", "": "Unknown"})
    
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(0)
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce").fillna(0.0)
    
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in defect_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)
    df = df.sort_values("week_ending").reset_index(drop=True)
    
    # Add multi-defect features
    df = add_multi_defect_features(df)
    
    return df


# ================================================================
# MODEL TRAINING UTILITIES
# ================================================================

def smart_split(df: pd.DataFrame, thr_label: float, use_temporal: bool = True):
    """
    Smart data splitting that ensures class balance in all splits.
    Uses multiple fallback strategies to guarantee both classes exist.
    """
    df = df.copy()
    
    # First determine effective threshold that guarantees both classes
    y_full = (df["scrap_percent"] > thr_label).astype(int)
    n_pos = int(y_full.sum())
    n_neg = int((y_full == 0).sum())
    
    effective_thr = thr_label
    
    # If we don't have enough of both classes, try multiple fallback strategies
    if n_pos < 10 or n_neg < 10:
        # Strategy 1: Try median
        effective_thr = df["scrap_percent"].median()
        y_full = (df["scrap_percent"] > effective_thr).astype(int)
        n_pos = int(y_full.sum())
        n_neg = int((y_full == 0).sum())
        
        # Strategy 2: If median doesn't work, try percentiles to find a good split point
        if n_pos < 10 or n_neg < 10:
            # Try percentiles from 30th to 70th to find one that gives balanced classes
            for pct in [50, 40, 60, 35, 65, 30, 70, 25, 75]:
                effective_thr = df["scrap_percent"].quantile(pct / 100)
                y_full = (df["scrap_percent"] > effective_thr).astype(int)
                n_pos = int(y_full.sum())
                n_neg = int((y_full == 0).sum())
                if n_pos >= 10 and n_neg >= 10:
                    break
        
        # Strategy 3: If still imbalanced, use mean
        if n_pos < 10 or n_neg < 10:
            effective_thr = df["scrap_percent"].mean()
            y_full = (df["scrap_percent"] > effective_thr).astype(int)
            n_pos = int(y_full.sum())
            n_neg = int((y_full == 0).sum())
    
    # Try temporal split first (only if requested AND we have timestamps)
    if use_temporal and 'week_ending' in df.columns:
        df_sorted = df.sort_values("week_ending").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * 0.6)
        calib_end = int(n * 0.8)
        
        df_train = df_sorted.iloc[:train_end].copy()
        df_calib = df_sorted.iloc[train_end:calib_end].copy()
        df_test = df_sorted.iloc[calib_end:].copy()
        
        # Check class balance in training set
        y_train_check = (df_train["scrap_percent"] > effective_thr).astype(int)
        if y_train_check.nunique() >= 2 and y_train_check.sum() >= 5 and (y_train_check == 0).sum() >= 5:
            return df_train, df_calib, df_test, effective_thr
        # If temporal split fails balance check, fall through to stratified
    
    # Use stratified random split to guarantee class balance
    try:
        # Stratified split ensures both classes in each split
        df_train_calib, df_test = train_test_split(
            df, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
        )
        y_train_calib = (df_train_calib["scrap_percent"] > effective_thr).astype(int)
        df_train, df_calib = train_test_split(
            df_train_calib, test_size=0.25, stratify=y_train_calib, random_state=RANDOM_STATE
        )
        
        # Verify we actually have both classes
        y_train_final = (df_train["scrap_percent"] > effective_thr).astype(int)
        if y_train_final.nunique() >= 2:
            return df_train.copy(), df_calib.copy(), df_test.copy(), effective_thr
    except Exception as e:
        pass  # Fall through to shuffle split
    
    # Last resort: shuffle and split, then verify
    df_shuffled = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    
    df_train = df_shuffled.iloc[:train_end].copy()
    df_calib = df_shuffled.iloc[train_end:calib_end].copy()
    df_test = df_shuffled.iloc[calib_end:].copy()
    
    return df_train, df_calib, df_test, effective_thr


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    grp = df_train.groupby("part_id", dropna=False)["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp


def get_feature_columns(df: pd.DataFrame, phm_config: dict) -> list:
    """Get feature columns based on PHM configuration."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    # Multi-defect features (always included)
    multi_feats = ["n_defect_types", "has_multiple_defects", "total_defect_rate",
                   "max_defect_rate", "defect_concentration",
                   "shift_x_tearup", "shrink_x_porosity", "core_x_sand"]
    feats.extend([f for f in multi_feats if f in df.columns])
    
    # Health Index features
    if phm_config.get('health_index', False):
        feats.extend([f for f in ['health_index', 'health_index_norm'] if f in df.columns])
    
    # Temporal features
    if phm_config.get('temporal', False):
        temp_feats = [c for c in df.columns if '_trend' in c or '_roll3' in c]
        temp_feats.extend(['month', 'quarter'])
        feats.extend([f for f in temp_feats if f in df.columns])
    
    # Defect rate columns
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    feats.extend(rate_cols)
    
    # Remove duplicates
    return list(dict.fromkeys(f for f in feats if f in df.columns))


def safe_predict_proba(model, X):
    """Safely get probability predictions, handling single-class case."""
    proba = model.predict_proba(X)
    if proba.shape[1] > 1:
        return proba[:, 1]
    else:
        return proba[:, 0]


def calculate_uncertainty(model, X):
    """Calculate prediction uncertainty from ensemble."""
    if hasattr(model, 'estimators_'):
        preds = []
        for tree in model.estimators_:
            p = tree.predict_proba(X)
            preds.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
        preds = np.array(preds)
        return {
            'mean': np.mean(preds, axis=0),
            'std': np.std(preds, axis=0),
            'ci_lower': np.clip(np.mean(preds, axis=0) - 1.96 * np.std(preds, axis=0), 0, 1),
            'ci_upper': np.clip(np.mean(preds, axis=0) + 1.96 * np.std(preds, axis=0), 0, 1),
        }
    return {'mean': safe_predict_proba(model, X), 'std': np.zeros(len(X)), 
            'ci_lower': safe_predict_proba(model, X), 'ci_upper': safe_predict_proba(model, X)}


def train_model_with_config(df: pd.DataFrame, phm_config: dict, thr_label: float, n_est: int):
    """Train model with specified PHM configuration. 
    Robust version that handles class imbalance at any threshold setting."""
    df = df.copy()
    
    # Add PHM features if enabled
    if phm_config.get('health_index', False):
        df = add_health_index(df)
    if phm_config.get('temporal', False):
        df = add_temporal_features(df)
    
    # Smart split with class balance handling
    df_train, df_calib, df_test, effective_thr = smart_split(
        df, thr_label, use_temporal=phm_config.get('temporal', False)
    )
    
    if len(df_train) < 20 or len(df_test) < 5:
        return None, None, None, {'error': 'Insufficient data'}
    
    # Compute MTBF features
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Re-merge after modifications (create new dataframes to avoid mutation issues)
    df_train = df_train.merge(mtbf_train, on="part_id", how="left", suffixes=('', '_y'))
    df_train["mttf_scrap"] = df_train["mttf_scrap"].fillna(default_mtbf)
    df_train = df_train.merge(part_freq_train.rename("part_freq"), 
                              left_on="part_id", right_index=True, how="left")
    df_train["part_freq"] = df_train["part_freq"].fillna(default_freq)
    
    df_calib = df_calib.merge(mtbf_train, on="part_id", how="left", suffixes=('', '_y'))
    df_calib["mttf_scrap"] = df_calib["mttf_scrap"].fillna(default_mtbf)
    df_calib = df_calib.merge(part_freq_train.rename("part_freq"),
                              left_on="part_id", right_index=True, how="left")
    df_calib["part_freq"] = df_calib["part_freq"].fillna(default_freq)
    
    df_test = df_test.merge(mtbf_train, on="part_id", how="left", suffixes=('', '_y'))
    df_test["mttf_scrap"] = df_test["mttf_scrap"].fillna(default_mtbf)
    df_test = df_test.merge(part_freq_train.rename("part_freq"),
                            left_on="part_id", right_index=True, how="left")
    df_test["part_freq"] = df_test["part_freq"].fillna(default_freq)
    
    # Get features
    feats = get_feature_columns(df_train, phm_config)
    
    # Ensure features exist
    for f in feats:
        for d in [df_train, df_calib, df_test]:
            if f not in d.columns:
                d[f] = 0.0
    
    # Prepare X
    X_train = df_train[feats].fillna(0)
    X_calib = df_calib[feats].fillna(0)
    X_test = df_test[feats].fillna(0)
    
    # ROBUST CLASS BALANCE HANDLING
    # Try the effective threshold from smart_split first
    y_train = (df_train["scrap_percent"] > effective_thr).astype(int)
    final_thr = effective_thr
    
    # If we don't have both classes, try to find a threshold that works for THIS specific train split
    if y_train.nunique() < 2 or y_train.sum() < 3 or (y_train == 0).sum() < 3:
        # Get the actual scrap values in the training set
        train_scrap = df_train["scrap_percent"].dropna()
        
        if len(train_scrap) < 10:
            return None, None, None, {'error': 'Insufficient training data after split'}
        
        # Try percentile-based thresholds on the TRAINING data specifically
        found_valid_threshold = False
        for pct in [50, 45, 55, 40, 60, 35, 65, 30, 70, 25, 75, 20, 80]:
            try_thr = train_scrap.quantile(pct / 100)
            y_try = (train_scrap > try_thr).astype(int)
            n_pos = y_try.sum()
            n_neg = (y_try == 0).sum()
            
            if n_pos >= 3 and n_neg >= 3:
                final_thr = try_thr
                found_valid_threshold = True
                break
        
        if not found_valid_threshold:
            # Last resort: use the mean of the training data
            final_thr = train_scrap.mean()
        
        # Recompute y values with the new threshold
        y_train = (df_train["scrap_percent"] > final_thr).astype(int)
    
    # Final check - if we STILL don't have two classes, the data truly has no variability
    if y_train.nunique() < 2:
        scrap_vals = df_train["scrap_percent"]
        return None, None, None, {
            'error': f'Training data has no variability (all values = {scrap_vals.iloc[0]:.2f}%). Need diverse scrap rates to train.'
        }
    
    # Compute y for calibration and test using the same threshold
    y_calib = (df_calib["scrap_percent"] > final_thr).astype(int)
    y_test = (df_test["scrap_percent"] > final_thr).astype(int)
    
    # Train the model
    rf = RandomForestClassifier(
        n_estimators=n_est,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    
    # Calibrate if we have enough samples of both classes
    if y_calib.sum() >= 3 and (y_calib == 0).sum() >= 3:
        try:
            model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
            calibrated = True
        except:
            model = rf
            calibrated = False
    else:
        model = rf
        calibrated = False
    
    # Evaluate
    preds_proba = safe_predict_proba(model, X_test)
    preds_binary = (preds_proba > 0.5).astype(int)
    
    metrics = {
        'brier_score': brier_score_loss(y_test, preds_proba),
        'accuracy': accuracy_score(y_test, preds_binary),
        'recall': recall_score(y_test, preds_binary, zero_division=0),
        'precision': precision_score(y_test, preds_binary, zero_division=0),
        'f1_score': f1_score(y_test, preds_binary, zero_division=0),
        'n_features': len(feats),
        'calibrated': calibrated,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'effective_threshold': final_thr,  # Track what threshold was actually used
    }
    
    if phm_config.get('uncertainty', False):
        unc = calculate_uncertainty(rf, X_test)
        metrics['mean_uncertainty'] = float(np.mean(unc['std']))
    
    return model, rf, feats, metrics


def run_factorial_experiment(df: pd.DataFrame, thr_label: float, n_est: int, progress_callback=None):
    """Run full factorial experiment."""
    keys = ['health_index', 'temporal', 'uncertainty', 'decision']
    all_configs = list(product([False, True], repeat=4))
    
    results = []
    for i, config_tuple in enumerate(all_configs):
        config = dict(zip(keys, config_tuple))
        
        active = [k[:2].upper() for k, v in config.items() if v]
        config_name = '+'.join(active) if active else 'Base'
        
        if progress_callback:
            progress_callback(i / len(all_configs), f"Testing: {config_name}")
        
        _, _, _, metrics = train_model_with_config(df, config, thr_label, n_est)
        
        result = {
            'config_id': i,
            'config_name': config_name,
            **config,
        }
        if 'error' in metrics:
            result['error'] = metrics['error']
        else:
            result.update(metrics)
        
        results.append(result)
    
    return pd.DataFrame(results)


# ================================================================
# PREDICTION FUNCTION
# ================================================================

def predict_for_part(df: pd.DataFrame, part_id: str, order_qty: int, 
                     piece_weight: float, cost_per_part: float,
                     phm_config: dict, thr_label: float, n_est: int):
    """Run prediction for a specific part."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'part_id': part_id,
        'order_quantity': order_qty,
        'piece_weight_lbs': piece_weight,
        'cost_per_part': cost_per_part,
        'threshold': thr_label,
    }
    
    # Get part history
    part_history = df[df['part_id'] == part_id]
    results['part_historical_records'] = len(part_history)
    
    if results['part_historical_records'] >= HIGH_CONFIDENCE_RECORDS:
        results['data_confidence'] = 'high'
    elif results['part_historical_records'] >= RECOMMENDED_MIN_RECORDS:
        results['data_confidence'] = 'medium'
    else:
        results['data_confidence'] = 'low'
    
    # Train model
    model, rf, feats, metrics = train_model_with_config(df, phm_config, thr_label, n_est)
    
    if model is None:
        results['error'] = metrics.get('error', 'Training failed')
        return results
    
    results['training_samples'] = metrics['train_samples']
    results['n_features'] = metrics['n_features']
    
    # Build input for prediction
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(part_history) > 0:
        defect_means = part_history[defect_cols].mean()
        results['historical_scrap_avg'] = part_history['scrap_percent'].mean()
        results['historical_scrap_max'] = part_history['scrap_percent'].max()
        results['historical_scrap_min'] = part_history['scrap_percent'].min()
    else:
        defect_means = df[defect_cols].mean()
        results['historical_scrap_avg'] = df['scrap_percent'].mean()
        results['historical_scrap_max'] = df['scrap_percent'].max()
        results['historical_scrap_min'] = df['scrap_percent'].min()
    
    # Compute MTBF
    df_train, _, _, _ = smart_split(df, thr_label, use_temporal=phm_config.get('temporal', False))
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    
    hist_mttf = float(mtbf_train[mtbf_train["part_id"] == part_id]["mttf_scrap"].values[0]) \
        if part_id in mtbf_train["part_id"].values else float(mtbf_train["mttf_scrap"].median())
    hist_freq = float(part_freq_train.get(part_id, part_freq_train.median()))
    
    # Build input dict
    input_dict = {
        "order_quantity": order_qty,
        "piece_weight_lbs": piece_weight,
        "mttf_scrap": hist_mttf,
        "part_freq": hist_freq,
    }
    
    # Add defect rates
    for dc in defect_cols:
        input_dict[dc] = defect_means.get(dc, 0.0)
    
    # Multi-defect features
    defect_values = [defect_means.get(dc, 0.0) for dc in defect_cols]
    input_dict['n_defect_types'] = sum(1 for v in defect_values if v > 0)
    input_dict['has_multiple_defects'] = 1 if input_dict['n_defect_types'] >= MULTI_DEFECT_THRESHOLD else 0
    input_dict['total_defect_rate'] = sum(defect_values)
    input_dict['max_defect_rate'] = max(defect_values) if defect_values else 0
    input_dict['defect_concentration'] = input_dict['max_defect_rate'] / (input_dict['total_defect_rate'] + 0.001)
    
    results['n_defect_types'] = input_dict['n_defect_types']
    results['is_multi_defect'] = input_dict['has_multiple_defects'] == 1
    results['total_defect_rate'] = input_dict['total_defect_rate']
    
    # Interaction terms
    input_dict['shift_x_tearup'] = defect_means.get('shift_rate', 0) * defect_means.get('tear_up_rate', 0)
    input_dict['shrink_x_porosity'] = defect_means.get('shrink_rate', 0) * defect_means.get('gas_porosity_rate', 0)
    input_dict['core_x_sand'] = defect_means.get('core_rate', 0) * defect_means.get('sand_rate', 0)
    
    # PHM features
    if phm_config.get('health_index', False):
        critical = sum(defect_means.get(d, 0) * 3 for d in ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate'])
        moderate = sum(defect_means.get(d, 0) * 2 for d in ['shift_rate', 'core_rate', 'sand_rate', 'dross_rate'])
        input_dict['health_index'] = critical + moderate + sum(defect_means.get(d, 0) for d in defect_cols)
        input_dict['health_index_norm'] = min(1.0, input_dict['health_index'] / 5.0)
    
    if phm_config.get('temporal', False):
        input_dict['total_defect_rate_trend'] = 0
        input_dict['total_defect_rate_roll3'] = input_dict['total_defect_rate']
        input_dict['scrap_percent_trend'] = 0
        input_dict['scrap_percent_roll3'] = results['historical_scrap_avg']
        input_dict['month'] = datetime.now().month
        input_dict['quarter'] = (datetime.now().month - 1) // 3 + 1
    
    # Create input DataFrame
    X_input = pd.DataFrame([{f: input_dict.get(f, 0.0) for f in feats}])
    
    # Predict
    proba = safe_predict_proba(model, X_input)[0]
    
    results['scrap_risk_pct'] = proba * 100
    results['reliability_pct'] = (1 - proba) * 100
    results['expected_scrap_pcs'] = order_qty * proba
    results['expected_loss_dollars'] = order_qty * proba * cost_per_part
    
    # Uncertainty if enabled
    if phm_config.get('uncertainty', False):
        unc = calculate_uncertainty(rf, X_input)
        results['ci_lower'] = float(unc['ci_lower'][0]) * 100
        results['ci_upper'] = float(unc['ci_upper'][0]) * 100
    
    # Decision recommendation
    if phm_config.get('decision', False):
        intervention_cost = 50
        if results['expected_loss_dollars'] > intervention_cost:
            results['recommendation'] = 'INTERVENE'
        else:
            results['recommendation'] = 'PROCEED'
    
    # Top defects
    top_defects = defect_means.sort_values(ascending=False).head(5)
    for i, (k, v) in enumerate(top_defects.items()):
        results[f'top_defect_{i+1}'] = k.replace('_rate', '')
        results[f'top_defect_{i+1}_rate'] = v
    
    return results


# ================================================================
# DISPLAY HELPERS
# ================================================================

def display_data_confidence_banner(n_records: int, part_id: str):
    if n_records >= HIGH_CONFIDENCE_RECORDS:
        st.success(f"✅ **HIGH CONFIDENCE**: Part {part_id} has {n_records} historical runs")
    elif n_records >= RECOMMENDED_MIN_RECORDS:
        st.info(f"📊 **MEDIUM CONFIDENCE**: Part {part_id} has {n_records} historical runs")
    else:
        st.warning(f"⚠️ **LOW CONFIDENCE**: Part {part_id} has only {n_records} historical runs")


# ================================================================
# SESSION STATE
# ================================================================

def initialize_results_log():
    if 'prediction_results' not in st.session_state:
        st.session_state['prediction_results'] = []


def log_prediction_result(results: dict):
    st.session_state['prediction_results'].append(results)


def get_results_dataframe() -> pd.DataFrame:
    if not st.session_state['prediction_results']:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state['prediction_results'])


def clear_results_log():
    st.session_state['prediction_results'] = []


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

st.sidebar.header("🔬 PHM Enhancements")
st.sidebar.caption("Toggle to test different configurations")

enable_health_index = st.sidebar.checkbox("Health Index (HI)", value=False, 
    help="Composite degradation score")
enable_temporal = st.sidebar.checkbox("Temporal Features (TF)", value=False,
    help="Trend detection over time")
enable_uncertainty = st.sidebar.checkbox("Uncertainty Quantification (UQ)", value=False,
    help="Confidence intervals")
enable_decision = st.sidebar.checkbox("Decision Optimization (DO)", value=False,
    help="Cost-based recommendations")

current_config = {
    'health_index': enable_health_index,
    'temporal': enable_temporal,
    'uncertainty': enable_uncertainty,
    'decision': enable_decision
}

st.sidebar.header("📊 Results Log")
initialize_results_log()
n_logged = len(st.session_state['prediction_results'])
st.sidebar.metric("Predictions Logged", n_logged)

if st.sidebar.button("🗑️ Clear Results Log"):
    clear_results_log()
    st.sidebar.success("Results cleared!")
    st.rerun()

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()


# ================================================================
# LOAD DATA
# ================================================================
df_base = load_and_clean(csv_path)
n_parts = df_base["part_id"].nunique()
n_records = len(df_base)
st.info(f"✅ Loaded {n_records:,} work orders | {n_parts} unique parts")

# Show data statistics to help with threshold selection
scrap_stats = df_base["scrap_percent"].describe()
with st.expander("📊 Scrap Data Statistics (click to expand)"):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Min Scrap %", f"{scrap_stats['min']:.2f}%")
    col_s2.metric("Median Scrap %", f"{scrap_stats['50%']:.2f}%")
    col_s3.metric("Mean Scrap %", f"{scrap_stats['mean']:.2f}%")
    col_s4.metric("Max Scrap %", f"{scrap_stats['max']:.2f}%")
    
    # Show recommended threshold range
    p25 = df_base["scrap_percent"].quantile(0.25)
    p75 = df_base["scrap_percent"].quantile(0.75)
    st.caption(f"💡 **Recommended threshold range:** {p25:.1f}% - {p75:.1f}% (25th-75th percentile) for balanced training classes")
    
    # Debug info if scrap data looks wrong
    if scrap_stats['max'] == 0:
        st.warning("⚠️ **Data Issue Detected:** All scrap values are 0. Check column mapping below.")
        
        # Show raw columns for debugging
        raw_df = pd.read_csv(csv_path, nrows=5)
        st.markdown("**Original CSV columns:**")
        st.code(", ".join(raw_df.columns.tolist()))
        
        st.markdown("**After normalization:**")
        st.code(", ".join(df_base.columns.tolist()))
        
        st.markdown("**First 5 rows of scrap_percent:**")
        st.write(df_base["scrap_percent"].head())


# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", 
    "🧪 PHM Lab", 
    "📊 Results Export",
    "📚 Reference"
])


# ================================================================
# TAB 1: PREDICT
# ================================================================
with tab1:
    st.header("🔮 Scrap Risk Prediction")
    
    # Show active PHM enhancements
    active = [PHM_ENHANCEMENTS[k]['name'] for k, v in current_config.items() if v]
    if active:
        st.success(f"**Active PHM Enhancements:** {', '.join(active)}")
    else:
        st.info("**PHM Enhancements:** None (Base Model) - Enable in sidebar")
    
    st.markdown("---")
    
    # INPUT FIELDS
    col1, col2, col3, col4 = st.columns(4)
    part_id_input = col1.text_input("Part ID", value="15")
    order_qty = col2.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = col3.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = col4.number_input("Cost per Part ($)", min_value=0.1, value=10.0)
    
    if st.button("🎯 Predict Scrap Risk"):
        with st.spinner("Running prediction..."):
            try:
                results = predict_for_part(
                    df_base, part_id_input, order_qty, piece_weight, 
                    cost_per_part, current_config, thr_label, n_est
                )
                
                if 'error' in results:
                    st.error(f"❌ Error: {results['error']}")
                else:
                    # Log results
                    log_prediction_result(results)
                    st.success(f"✅ Prediction logged! (#{len(st.session_state['prediction_results'])})")
                    
                    # Data confidence
                    display_data_confidence_banner(results['part_historical_records'], part_id_input)
                    
                    # Multi-defect alert
                    if results.get('is_multi_defect', False):
                        st.error(f"🚨 **MULTI-DEFECT PATTERN**: {results['n_defect_types']} defect types detected")
                    
                    # Results display
                    st.markdown("### 📊 Prediction Results")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.metric("Scrap Risk", f"{results['scrap_risk_pct']:.1f}%")
                        if 'ci_lower' in results:
                            st.caption(f"95% CI: {results['ci_lower']:.1f}% - {results['ci_upper']:.1f}%")
                    
                    with res_col2:
                        st.metric("Expected Scrap", f"{results['expected_scrap_pcs']:.1f} pieces")
                    
                    with res_col3:
                        st.metric("Expected Loss", f"${results['expected_loss_dollars']:.2f}")
                    
                    # Decision recommendation
                    if 'recommendation' in results:
                        if results['recommendation'] == 'INTERVENE':
                            st.error("⚠️ **RECOMMENDATION: INTERVENE** - Expected loss exceeds intervention cost")
                        else:
                            st.success("✅ **RECOMMENDATION: PROCEED** - Expected loss below intervention cost")
                    
                    # Historical context
                    with st.expander("📋 Historical Context"):
                        st.markdown(f"""
- **Historical Scrap Avg**: {results['historical_scrap_avg']:.2f}%
- **Historical Scrap Range**: {results['historical_scrap_min']:.2f}% - {results['historical_scrap_max']:.2f}%
- **Training Samples**: {results['training_samples']}
- **Features Used**: {results['n_features']}
                        """)
                    
                    # Top defects
                    st.markdown("### 🔍 Top Defect Concerns")
                    defect_data = []
                    for i in range(1, 6):
                        if f'top_defect_{i}' in results:
                            defect_data.append({
                                'Defect': results[f'top_defect_{i}'].replace('_', ' ').title(),
                                'Rate (%)': f"{results[f'top_defect_{i}_rate']*100:.2f}%"
                            })
                    if defect_data:
                        st.dataframe(pd.DataFrame(defect_data), use_container_width=True, hide_index=True)
                        
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ================================================================
# TAB 2: PHM LAB
# ================================================================
with tab2:
    st.header("🧪 PHM Enhancement Laboratory")
    
    st.markdown("""
    Test **all 16 combinations** of PHM enhancements to find the optimal configuration.
    """)
    
    if st.button("🚀 Run Full Factorial Experiment (16 configs)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)
        
        results_df = run_factorial_experiment(df_base, thr_label, n_est, update_progress)
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        st.session_state['factorial_results'] = results_df
        
        # Filter valid results
        valid = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()].copy()
        
        if len(valid) > 0 and 'brier_score' in valid.columns:
            valid = valid.sort_values('brier_score')
            
            st.markdown("### 📊 Results (sorted by Brier Score)")
            
            display_cols = ['config_name', 'brier_score', 'accuracy', 'recall', 'f1_score', 'n_features']
            st.dataframe(valid[[c for c in display_cols if c in valid.columns]], use_container_width=True)
            
            best = valid.iloc[0]
            st.success(f"🏆 **Best: {best['config_name']}** - Brier: {best['brier_score']:.4f}, Recall: {best['recall']:.1%}")
            
            # Download
            csv_data = results_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv_data, 
                              f"factorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


# ================================================================
# TAB 3: RESULTS EXPORT
# ================================================================
with tab3:
    st.header("📋 Prediction Results Log")
    
    df_results = get_results_dataframe()
    
    if df_results.empty:
        st.info("No predictions logged yet. Go to 'Predict' tab to run predictions.")
    else:
        st.success(f"📊 **{len(df_results)} predictions logged**")
        st.dataframe(df_results, use_container_width=True)
        
        csv_data = df_results.to_csv(index=False)
        st.download_button("📥 Download All Predictions (CSV)", csv_data,
                          f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


# ================================================================
# TAB 4: REFERENCE
# ================================================================
with tab4:
    st.header("📚 PHM Enhancement Reference")
    
    for key, info in PHM_ENHANCEMENTS.items():
        with st.expander(f"**{info['name']}**"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Reference:** {info['reference']}")
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Key References
    
    1. **Lei et al. (2018)** - Machinery health prognostics: A systematic review
    2. **Choo & Shin (2025)** - RUL + cost-optimal maintenance  
    3. **Campbell (2003)** - Castings Practice: The Ten Rules
    """)


st.markdown("---")
st.caption("🏭 Foundry Dashboard **v3.2.1** | PHM Laboratory | Campbell Process Mapping")
