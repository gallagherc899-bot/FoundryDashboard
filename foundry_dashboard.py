# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 3.2 - PHM ENHANCEMENT LABORATORY (FIXED)
# ================================================================
# 
# FEATURES:
# - Part-specific prediction with inputs (Part ID, Weight, Qty, Cost)
# - PHM Enhancement Toggles (Health Index, Temporal, Uncertainty, Decision)
# - Factorial Testing Mode (test all 16 combinations)
# - Single/Custom configuration testing
# - Comprehensive comparison results with downloadable reports
# - Multi-defect alerts and Campbell process diagnosis
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
    page_title="Foundry Dashboard v3.2 - PHM Laboratory", 
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
        <strong>Version 3.2 - PHM Enhancement Laboratory</strong> | 
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
# PHM FEATURE ENGINEERING FUNCTIONS
# ================================================================

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic multi-defect features (from v3.0)."""
    df = df.copy()
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    for col in defect_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Multi-defect features
    df['n_defect_types'] = (df[defect_cols] > 0).sum(axis=1)
    df['has_multiple_defects'] = (df['n_defect_types'] >= 2).astype(int)
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
    """PHM Enhancement 1: Health Index - Composite degradation score."""
    df = df.copy()
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    critical_defects = ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate']
    moderate_defects = ['shift_rate', 'core_rate', 'sand_rate', 'dross_rate']
    
    critical_score = 0
    moderate_score = 0
    minor_score = 0
    
    for col in defect_cols:
        if col in critical_defects:
            critical_score += df[col].fillna(0) * 3.0
        elif col in moderate_defects:
            moderate_score += df[col].fillna(0) * 2.0
        else:
            minor_score += df[col].fillna(0) * 1.0
    
    df['health_index'] = critical_score + moderate_score + minor_score
    max_hi = df['health_index'].max()
    df['health_index_norm'] = df['health_index'] / (max_hi + 0.001) if max_hi > 0 else 0
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """PHM Enhancement 2: Temporal Features - Trend detection."""
    df = df.copy()
    
    if 'week_ending' in df.columns:
        df = df.sort_values('week_ending').reset_index(drop=True)
    
    trend_cols = ['total_defect_rate', 'scrap_percent']
    if 'health_index' in df.columns:
        trend_cols.append('health_index')
    
    for col in trend_cols:
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


def calculate_uncertainty(model, X):
    """PHM Enhancement 3: Uncertainty Quantification."""
    if hasattr(model, 'estimators_'):
        predictions = []
        for tree in model.estimators_:
            pred = tree.predict_proba(X)
            if pred.shape[1] > 1:
                predictions.append(pred[:, 1])
            else:
                predictions.append(pred[:, 0])
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        ci_lower = np.clip(mean_pred - 1.96 * std_pred, 0, 1)
        ci_upper = np.clip(mean_pred + 1.96 * std_pred, 0, 1)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'uncertainty_width': ci_upper - ci_lower
        }
    else:
        pred = model.predict_proba(X)
        if pred.shape[1] > 1:
            pred = pred[:, 1]
        else:
            pred = pred[:, 0]
        return {
            'mean': pred,
            'std': np.zeros_like(pred),
            'ci_lower': pred,
            'ci_upper': pred,
            'uncertainty_width': np.zeros_like(pred)
        }


def add_decision_features(df: pd.DataFrame, cost_per_part: float = 10.0, 
                          intervention_cost: float = 50.0) -> pd.DataFrame:
    """PHM Enhancement 4: Decision Optimization Features."""
    df = df.copy()
    
    if 'scrap_percent' in df.columns:
        df['expected_loss'] = df['scrap_percent'] / 100 * df.get('order_quantity', 100) * cost_per_part
        df['cost_benefit_ratio'] = df['expected_loss'] / (intervention_cost + 0.001)
        df['recommend_intervention'] = (df['expected_loss'] > intervention_cost).astype(int)
    
    return df


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
        "part_id": "part_id", "work_order": "work_order",
        "work_order_number": "work_order", "work_order_#": "work_order",
        "order_quantity": "order_quantity", "order_qty": "order_quantity",
        "scrap%": "scrap_percent", "scrap_percent": "scrap_percent",
        "piece_weight_lbs": "piece_weight_lbs", "piece_weight": "piece_weight_lbs",
        "week_ending": "week_ending", "week_end": "week_ending",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
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
    
    # Add base features
    df = add_base_features(df)
    
    return df


def prepare_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare features based on PHM configuration."""
    df = df.copy()
    
    if config.get('health_index', False):
        df = add_health_index(df)
    
    if config.get('temporal', False):
        df = add_temporal_features(df)
    
    if config.get('decision', False):
        df = add_decision_features(df)
    
    return df


# ================================================================
# MODEL TRAINING AND EVALUATION
# ================================================================

def time_split_621(df: pd.DataFrame, train_ratio=0.6, calib_ratio=0.2):
    """6-2-1 Temporal Split."""
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:calib_end], df_sorted.iloc[calib_end:]


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    grp = df_train.groupby("part_id", dropna=False)["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp


def attach_train_features(df_sub: pd.DataFrame, mtbf_train: pd.DataFrame, 
                          part_freq_train: pd.Series, default_mtbf: float,
                          default_freq: float) -> pd.DataFrame:
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s


def get_feature_columns(df: pd.DataFrame, config: dict) -> list:
    """Get feature columns based on PHM configuration."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    # Multi-defect features
    multi_defect_feats = [
        "n_defect_types", "has_multiple_defects", "total_defect_rate",
        "max_defect_rate", "defect_concentration",
        "shift_x_tearup", "shrink_x_porosity", "core_x_sand"
    ]
    for f in multi_defect_feats:
        if f in df.columns:
            feats.append(f)
    
    # Health Index features
    if config.get('health_index', False):
        for f in ['health_index', 'health_index_norm']:
            if f in df.columns:
                feats.append(f)
    
    # Temporal features
    if config.get('temporal', False):
        temporal_feats = [c for c in df.columns if any(x in c for x in ['_trend', '_roll3'])]
        temporal_feats += ['month', 'quarter']
        for f in temporal_feats:
            if f in df.columns and f not in feats:
                feats.append(f)
    
    # Decision features
    if config.get('decision', False):
        for f in ['cost_benefit_ratio']:
            if f in df.columns:
                feats.append(f)
    
    # Defect rate columns
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    feats.extend(rate_cols)
    
    # Remove duplicates
    feats = list(dict.fromkeys(f for f in feats if f in df.columns))
    
    return feats


def train_and_evaluate(df: pd.DataFrame, config: dict, thr_label: float, n_est: int):
    """Train model with given PHM configuration and return metrics."""
    try:
        # Prepare features based on config
        df_prepared = prepare_features(df, config)
        
        # Split data
        df_train, df_calib, df_test = time_split_621(df_prepared)
        
        # Check minimum samples
        if len(df_train) < 20 or len(df_test) < 5:
            return {'error': 'Insufficient data for training'}, None, []
        
        # Compute MTBF features from training data
        mtbf_train = compute_mtbf_on_train(df_train, thr_label)
        part_freq_train = df_train["part_id"].value_counts(normalize=True)
        default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
        default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
        
        # Attach features
        df_train = attach_train_features(df_train.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_calib = attach_train_features(df_calib.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_test = attach_train_features(df_test.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
        
        # Get feature columns
        feats = get_feature_columns(df_train, config)
        
        # Ensure all features exist
        for f in feats:
            for d in [df_train, df_calib, df_test]:
                if f not in d.columns:
                    d[f] = 0.0
        
        # Prepare X and y
        X_train = df_train[feats].fillna(0)
        y_train = (df_train["scrap_percent"] > thr_label).astype(int)
        X_calib = df_calib[feats].fillna(0)
        y_calib = (df_calib["scrap_percent"] > thr_label).astype(int)
        X_test = df_test[feats].fillna(0)
        y_test = (df_test["scrap_percent"] > thr_label).astype(int)
        
        # Check class balance
        if y_train.nunique() < 2:
            return {'error': 'Only one class in training data'}, None, []
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=n_est,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(X_train, y_train)
        
        # Calibrate if possible
        pos = int(y_calib.sum())
        neg = int((y_calib == 0).sum())
        
        if pos >= 3 and neg >= 3:
            try:
                cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
                model = cal
                calibrated = True
            except:
                model = rf
                calibrated = False
        else:
            model = rf
            calibrated = False
        
        # Predictions - handle single class case
        preds_proba_raw = model.predict_proba(X_test)
        if preds_proba_raw.shape[1] > 1:
            preds_proba = preds_proba_raw[:, 1]
        else:
            preds_proba = preds_proba_raw[:, 0]
        
        preds_binary = (preds_proba > 0.5).astype(int)
        
        # Calculate metrics
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
            'test_positive': int(y_test.sum()),
            'test_negative': int((y_test == 0).sum()),
        }
        
        # Uncertainty quantification if enabled
        if config.get('uncertainty', False):
            uncertainty = calculate_uncertainty(rf, X_test)
            metrics['mean_uncertainty'] = float(np.mean(uncertainty['std']))
            metrics['mean_ci_width'] = float(np.mean(uncertainty['uncertainty_width']))
        
        return metrics, model, feats
        
    except Exception as e:
        return {'error': str(e)}, None, []


def run_factorial_experiment(df: pd.DataFrame, thr_label: float, n_est: int, 
                             progress_callback=None):
    """Run full factorial experiment testing all 16 PHM configurations."""
    enhancement_keys = ['health_index', 'temporal', 'uncertainty', 'decision']
    all_configs = list(product([False, True], repeat=4))
    
    results = []
    
    for i, config_tuple in enumerate(all_configs):
        config = dict(zip(enhancement_keys, config_tuple))
        
        active = [k[:2].upper() for k, v in config.items() if v]
        config_name = '+'.join(active) if active else 'Base'
        
        if progress_callback:
            progress_callback(i / len(all_configs), f"Testing: {config_name}")
        
        metrics, _, _ = train_and_evaluate(df, config, thr_label, n_est)
        
        result = {
            'config_id': i,
            'config_name': config_name,
            'health_index': config['health_index'],
            'temporal': config['temporal'],
            'uncertainty': config['uncertainty'],
            'decision': config['decision'],
        }
        
        if 'error' in metrics:
            result['error'] = metrics['error']
        else:
            result.update(metrics)
        
        results.append(result)
    
    return pd.DataFrame(results)


def run_single_experiment(df: pd.DataFrame, config: dict, thr_label: float, n_est: int):
    """Run single configuration experiment."""
    metrics, model, feats = train_and_evaluate(df, config, thr_label, n_est)
    
    active = [k[:2].upper() for k, v in config.items() if v]
    config_name = '+'.join(active) if active else 'Base'
    
    result = {'config_name': config_name, **config}
    
    if 'error' in metrics:
        result['error'] = metrics['error']
    else:
        result.update(metrics)
    
    return result, model, feats


# ================================================================
# PREDICTION FUNCTION FOR SPECIFIC PART
# ================================================================

def predict_for_part(df: pd.DataFrame, part_id: str, order_qty: int, 
                     piece_weight: float, cost_per_part: float,
                     config: dict, thr_label: float, n_est: int):
    """
    Run prediction for a specific part with given inputs.
    Returns prediction results with/without PHM enhancements.
    """
    results = {
        'part_id': part_id,
        'order_quantity': order_qty,
        'piece_weight_lbs': piece_weight,
        'cost_per_part': cost_per_part,
    }
    
    # Get part history
    part_history = df[df['part_id'] == part_id]
    results['part_historical_records'] = len(part_history)
    
    # Prepare data
    df_prepared = prepare_features(df.copy(), config)
    
    # Split and train
    df_train, df_calib, _ = time_split_621(df_prepared)
    
    if len(df_train) < 20:
        results['error'] = 'Insufficient training data'
        return results
    
    # Compute features from training data
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median())
    default_freq = float(part_freq_train.median())
    
    df_train = attach_train_features(df_train.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    feats = get_feature_columns(df_train, config)
    
    # Ensure features exist
    for f in feats:
        if f not in df_train.columns:
            df_train[f] = 0.0
        if f not in df_calib.columns:
            df_calib[f] = 0.0
    
    X_train = df_train[feats].fillna(0)
    y_train = (df_train["scrap_percent"] > thr_label).astype(int)
    X_calib = df_calib[feats].fillna(0)
    y_calib = (df_calib["scrap_percent"] > thr_label).astype(int)
    
    if y_train.nunique() < 2:
        results['error'] = 'Only one class in training data'
        return results
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=n_est,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    
    # Calibrate
    pos, neg = int(y_calib.sum()), int((y_calib == 0).sum())
    if pos >= 3 and neg >= 3:
        try:
            model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
        except:
            model = rf
    else:
        model = rf
    
    # Build input for prediction
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(part_history) > 0:
        defect_means = part_history[defect_cols].mean()
        hist_mttf = float(mtbf_train[mtbf_train["part_id"] == part_id]["mttf_scrap"].values[0]) \
            if part_id in mtbf_train["part_id"].values else default_mtbf
        hist_freq = float(part_freq_train.get(part_id, default_freq))
        historical_scrap = part_history['scrap_percent'].mean()
    else:
        defect_means = df[defect_cols].mean()
        hist_mttf = default_mtbf
        hist_freq = default_freq
        historical_scrap = df['scrap_percent'].mean()
    
    results['historical_scrap_avg'] = historical_scrap
    
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
    
    # Add multi-defect features
    defect_values = [defect_means.get(dc, 0.0) for dc in defect_cols]
    input_dict['n_defect_types'] = sum(1 for v in defect_values if v > 0)
    input_dict['has_multiple_defects'] = 1 if input_dict['n_defect_types'] >= 2 else 0
    input_dict['total_defect_rate'] = sum(defect_values)
    input_dict['max_defect_rate'] = max(defect_values) if defect_values else 0
    input_dict['defect_concentration'] = input_dict['max_defect_rate'] / (input_dict['total_defect_rate'] + 0.001)
    
    # Interaction terms
    input_dict['shift_x_tearup'] = defect_means.get('shift_rate', 0) * defect_means.get('tear_up_rate', 0)
    input_dict['shrink_x_porosity'] = defect_means.get('shrink_rate', 0) * defect_means.get('gas_porosity_rate', 0)
    input_dict['core_x_sand'] = defect_means.get('core_rate', 0) * defect_means.get('sand_rate', 0)
    
    # PHM features if enabled
    if config.get('health_index', False):
        critical = sum(defect_means.get(d, 0) * 3 for d in ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate'])
        moderate = sum(defect_means.get(d, 0) * 2 for d in ['shift_rate', 'core_rate', 'sand_rate', 'dross_rate'])
        minor = sum(defect_means.get(d, 0) for d in defect_cols if d not in ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate', 'shift_rate', 'core_rate', 'sand_rate', 'dross_rate'])
        input_dict['health_index'] = critical + moderate + minor
        input_dict['health_index_norm'] = min(1.0, input_dict['health_index'] / 5.0)
    
    if config.get('temporal', False):
        input_dict['total_defect_rate_trend'] = 0
        input_dict['total_defect_rate_roll3'] = input_dict['total_defect_rate']
        input_dict['scrap_percent_trend'] = 0
        input_dict['scrap_percent_roll3'] = historical_scrap
        input_dict['month'] = datetime.now().month
        input_dict['quarter'] = (datetime.now().month - 1) // 3 + 1
    
    if config.get('decision', False):
        input_dict['cost_benefit_ratio'] = (historical_scrap / 100 * order_qty * cost_per_part) / 50
    
    # Create input DataFrame with only the features the model expects
    X_input = pd.DataFrame([{f: input_dict.get(f, 0.0) for f in feats}])
    
    # Predict
    pred_proba_raw = model.predict_proba(X_input)
    if pred_proba_raw.shape[1] > 1:
        proba = pred_proba_raw[0, 1]
    else:
        proba = pred_proba_raw[0, 0]
    
    results['scrap_risk_pct'] = proba * 100
    results['reliability_pct'] = (1 - proba) * 100
    results['expected_scrap_pcs'] = order_qty * proba
    results['expected_loss_dollars'] = order_qty * proba * cost_per_part
    results['n_features'] = len(feats)
    results['n_defect_types'] = input_dict['n_defect_types']
    results['total_defect_rate'] = input_dict['total_defect_rate']
    
    # Uncertainty if enabled
    if config.get('uncertainty', False):
        uncertainty = calculate_uncertainty(rf, X_input)
        results['ci_lower'] = float(uncertainty['ci_lower'][0]) * 100
        results['ci_upper'] = float(uncertainty['ci_upper'][0]) * 100
        results['uncertainty_width'] = results['ci_upper'] - results['ci_lower']
    
    # Decision recommendation if enabled
    if config.get('decision', False):
        intervention_cost = 50
        if results['expected_loss_dollars'] > intervention_cost:
            results['recommendation'] = 'INTERVENE'
        else:
            results['recommendation'] = 'PROCEED'
    
    # Top defects for diagnosis
    top_defects = defect_means.sort_values(ascending=False).head(5)
    results['top_defects'] = top_defects.to_dict()
    
    return results


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

st.sidebar.header("🔬 PHM Enhancements")
st.sidebar.caption("Toggle individual enhancements")

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


# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", 
    "🧪 PHM Lab", 
    "📊 Results",
    "📚 Reference"
])


# ================================================================
# TAB 1: PREDICT (with inputs)
# ================================================================
with tab1:
    st.header("🔮 Scrap Risk Prediction")
    
    # Show active enhancements
    active_enhancements = [PHM_ENHANCEMENTS[k]['name'] for k, v in current_config.items() if v]
    if active_enhancements:
        st.success(f"**Active PHM Enhancements:** {', '.join(active_enhancements)}")
    else:
        st.info("**PHM Enhancements:** None (Base Model) - Enable in sidebar")
    
    st.markdown("---")
    
    # Input fields
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        part_id_input = st.text_input("Part ID", value="15")
    with col2:
        order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    with col3:
        piece_weight = st.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    with col4:
        cost_per_part = st.number_input("Cost per Part ($)", min_value=0.1, value=10.0)
    
    if st.button("🎯 Predict Scrap Risk"):
        with st.spinner("Running prediction..."):
            results = predict_for_part(
                df_base, part_id_input, order_qty, piece_weight, 
                cost_per_part, current_config, thr_label, n_est
            )
        
        if 'error' in results:
            st.error(f"❌ Error: {results['error']}")
        else:
            # Data confidence
            n_hist = results['part_historical_records']
            if n_hist >= 15:
                st.success(f"✅ **HIGH CONFIDENCE**: Part {part_id_input} has {n_hist} historical runs")
            elif n_hist >= 5:
                st.info(f"📊 **MEDIUM CONFIDENCE**: Part {part_id_input} has {n_hist} historical runs")
            else:
                st.warning(f"⚠️ **LOW CONFIDENCE**: Part {part_id_input} has only {n_hist} historical runs")
            
            # Multi-defect alert
            if results.get('n_defect_types', 0) >= 2:
                st.error(f"🚨 **MULTI-DEFECT PATTERN**: {results['n_defect_types']} defect types detected")
            
            # Main results
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
                    st.error(f"⚠️ **RECOMMENDATION: INTERVENE** - Expected loss exceeds intervention cost")
                else:
                    st.success(f"✅ **RECOMMENDATION: PROCEED** - Expected loss below intervention cost")
            
            # Top defects
            st.markdown("### 🔍 Top Defect Concerns")
            if 'top_defects' in results:
                defect_df = pd.DataFrame([
                    {'Defect': k.replace('_rate', '').replace('_', ' ').title(), 
                     'Rate': f"{v*100:.2f}%"}
                    for k, v in results['top_defects'].items()
                ])
                st.dataframe(defect_df, use_container_width=True, hide_index=True)
            
            # Store for download
            if 'prediction_log' not in st.session_state:
                st.session_state['prediction_log'] = []
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                **results,
                **{f'phm_{k}': v for k, v in current_config.items()}
            }
            # Remove non-serializable items
            if 'top_defects' in log_entry:
                for i, (k, v) in enumerate(log_entry['top_defects'].items()):
                    log_entry[f'defect_{i+1}'] = k
                    log_entry[f'defect_{i+1}_rate'] = v
                del log_entry['top_defects']
            
            st.session_state['prediction_log'].append(log_entry)
            
            st.success(f"✅ Prediction logged ({len(st.session_state['prediction_log'])} total)")


# ================================================================
# TAB 2: PHM LAB
# ================================================================
with tab2:
    st.header("🧪 PHM Enhancement Laboratory")
    
    st.markdown("""
    Test **all 16 combinations** of PHM enhancements to find the optimal configuration.
    """)
    
    test_mode = st.radio(
        "Test Mode",
        ["Single (current config)", "Full Factorial (all 16)"],
        horizontal=True
    )
    
    if test_mode == "Single (current config)":
        if st.button("🧪 Run Single Test"):
            with st.spinner("Training models..."):
                base_config = {'health_index': False, 'temporal': False, 'uncertainty': False, 'decision': False}
                base_results, _, _ = run_single_experiment(df_base, base_config, thr_label, n_est)
                current_results, _, _ = run_single_experiment(df_base, current_config, thr_label, n_est)
            
            if 'error' in base_results or 'error' in current_results:
                st.error(f"Error: {base_results.get('error', current_results.get('error', 'Unknown'))}")
            else:
                st.markdown("### 📊 Comparison")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.markdown("**Base Model**")
                    st.metric("Brier Score", f"{base_results['brier_score']:.4f}")
                    st.metric("Accuracy", f"{base_results['accuracy']:.1%}")
                    st.metric("Recall", f"{base_results['recall']:.1%}")
                
                with comp_col2:
                    st.markdown(f"**{current_results['config_name']}**")
                    st.metric("Brier Score", f"{current_results['brier_score']:.4f}")
                    st.metric("Accuracy", f"{current_results['accuracy']:.1%}")
                    st.metric("Recall", f"{current_results['recall']:.1%}")
                
                with comp_col3:
                    st.markdown("**Improvement**")
                    brier_imp = (base_results['brier_score'] - current_results['brier_score']) / base_results['brier_score'] * 100
                    st.metric("Brier Δ", f"{brier_imp:+.1f}%")
                    st.metric("Accuracy Δ", f"{(current_results['accuracy'] - base_results['accuracy'])*100:+.1f}%")
                    st.metric("Recall Δ", f"{(current_results['recall'] - base_results['recall'])*100:+.1f}%")
    
    else:  # Full Factorial
        if st.button("🚀 Run Full Factorial (16 configs)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            results_df = run_factorial_experiment(df_base, thr_label, n_est, update_progress)
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            st.session_state['factorial_results'] = results_df
            
            # Filter out errors
            valid_results = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()].copy()
            
            if len(valid_results) > 0 and 'brier_score' in valid_results.columns:
                valid_results = valid_results.sort_values('brier_score')
                
                st.markdown("### 📊 Results (sorted by Brier Score)")
                
                display_cols = ['config_name', 'brier_score', 'accuracy', 'recall', 'f1_score', 'n_features']
                st.dataframe(valid_results[display_cols], use_container_width=True)
                
                best = valid_results.iloc[0]
                st.success(f"🏆 **Best: {best['config_name']}** - Brier: {best['brier_score']:.4f}, Recall: {best['recall']:.1%}")
                
                # Download
                csv_data = results_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv_data, 
                                  f"factorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            else:
                st.warning("No valid results. Check data quality.")


# ================================================================
# TAB 3: RESULTS
# ================================================================
with tab3:
    st.header("📊 Logged Results")
    
    if 'prediction_log' not in st.session_state or len(st.session_state['prediction_log']) == 0:
        st.info("No predictions logged yet. Go to 'Predict' tab to run predictions.")
    else:
        pred_df = pd.DataFrame(st.session_state['prediction_log'])
        
        st.success(f"**{len(pred_df)} predictions logged**")
        
        # Display
        display_cols = [c for c in ['timestamp', 'part_id', 'order_quantity', 'scrap_risk_pct', 
                                    'expected_scrap_pcs', 'expected_loss_dollars', 'n_defect_types',
                                    'recommendation', 'phm_health_index', 'phm_temporal'] 
                       if c in pred_df.columns]
        
        st.dataframe(pred_df[display_cols], use_container_width=True)
        
        # Download
        csv_data = pred_df.to_csv(index=False)
        st.download_button("📥 Download All Predictions", csv_data,
                          f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        if st.button("🗑️ Clear Log"):
            st.session_state['prediction_log'] = []
            st.rerun()


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
    
    1. **Lei et al. (2018)** - Machinery health prognostics systematic review
    2. **Choo & Shin (2025)** - RUL + cost-optimal maintenance
    3. **Campbell (2003)** - Castings Practice: The Ten Rules
    """)


st.markdown("---")
st.caption("🏭 Foundry Dashboard **v3.2** | PHM Laboratory | Campbell Process Mapping")
