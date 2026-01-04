# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 3.2 - PHM ENHANCEMENT LABORATORY
# ================================================================
# 
# NEW IN V3.2:
# - PHM Enhancement Toggles (Health Index, Temporal, Uncertainty, Decision)
# - Factorial Testing Mode (test all 16 combinations)
# - Single/Custom configuration testing
# - Comprehensive comparison results with downloadable reports
# - Statistical improvement analysis
#
# RETAINED FROM V3.1:
# - Multi-defect feature engineering
# - Results export and logging
# - 6-2-1 Rolling Window
# - Data Confidence Indicators
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
    confusion_matrix,
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
    """
    PHM Enhancement 1: Health Index
    Composite degradation score combining all defect rates with process weights.
    Based on Lei et al. (2018) - degradation indicator concept.
    """
    df = df.copy()
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    # Weighted Health Index based on process criticality
    # Higher weights for defects that are harder to recover from
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
    
    # Composite Health Index (0 = healthy, higher = degraded)
    df['health_index'] = critical_score + moderate_score + minor_score
    
    # Normalized Health Index (0-1 scale)
    max_hi = df['health_index'].max()
    df['health_index_norm'] = df['health_index'] / (max_hi + 0.001)
    
    # Health Index categories
    df['health_category'] = pd.cut(
        df['health_index_norm'], 
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Healthy', 'Warning', 'Critical', 'Failure'],
        include_lowest=True
    )
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PHM Enhancement 2: Temporal Features
    Trend detection for degradation modeling.
    Captures whether quality is improving or degrading over time.
    """
    df = df.copy()
    
    # Ensure sorted by time
    if 'week_ending' in df.columns:
        df = df.sort_values('week_ending').reset_index(drop=True)
    
    # Key metrics to track trends
    trend_cols = ['total_defect_rate', 'scrap_percent', 'health_index'] if 'health_index' in df.columns else ['total_defect_rate', 'scrap_percent']
    
    for col in trend_cols:
        if col not in df.columns:
            continue
            
        # Part-level trends
        if 'part_id' in df.columns:
            # Trend (first derivative) - is it getting worse?
            df[f'{col}_trend'] = df.groupby('part_id')[col].diff().fillna(0)
            
            # Acceleration (second derivative) - is degradation accelerating?
            df[f'{col}_accel'] = df.groupby('part_id')[f'{col}_trend'].diff().fillna(0)
            
            # Rolling average (smoothed trend)
            df[f'{col}_roll3'] = df.groupby('part_id')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Rolling std (volatility)
            df[f'{col}_roll_std'] = df.groupby('part_id')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).std()
            ).fillna(0)
        else:
            # Global trends if no part_id
            df[f'{col}_trend'] = df[col].diff().fillna(0)
            df[f'{col}_accel'] = df[f'{col}_trend'].diff().fillna(0)
            df[f'{col}_roll3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_roll_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Time-based features
    if 'week_ending' in df.columns:
        df['week_of_year'] = pd.to_datetime(df['week_ending']).dt.isocalendar().week
        df['month'] = pd.to_datetime(df['week_ending']).dt.month
        df['quarter'] = pd.to_datetime(df['week_ending']).dt.quarter
    
    return df


def calculate_uncertainty(model, X, n_samples=100):
    """
    PHM Enhancement 3: Uncertainty Quantification
    Calculate prediction intervals using ensemble variance.
    """
    if hasattr(model, 'estimators_'):
        # Get predictions from each tree in the forest
        predictions = np.array([tree.predict_proba(X)[:, 1] for tree in model.estimators_])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval
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
        # Fallback for non-ensemble models
        pred = model.predict_proba(X)[:, 1]
        return {
            'mean': pred,
            'std': np.zeros_like(pred),
            'ci_lower': pred,
            'ci_upper': pred,
            'uncertainty_width': np.zeros_like(pred)
        }


def add_decision_features(df: pd.DataFrame, cost_per_part: float = 10.0, 
                          intervention_cost: float = 50.0) -> pd.DataFrame:
    """
    PHM Enhancement 4: Decision Optimization Features
    Cost-based intervention recommendations.
    Based on Choo & Shin (2025) - cost-optimal maintenance.
    """
    df = df.copy()
    
    if 'scrap_percent' in df.columns:
        # Expected loss without intervention
        df['expected_loss'] = df['scrap_percent'] / 100 * df.get('order_quantity', 100) * cost_per_part
        
        # Cost-benefit ratio
        df['cost_benefit_ratio'] = df['expected_loss'] / (intervention_cost + 0.001)
        
        # Decision threshold (intervene if expected loss > intervention cost)
        df['recommend_intervention'] = (df['expected_loss'] > intervention_cost).astype(int)
        
        # Priority score (higher = more urgent)
        df['intervention_priority'] = df['expected_loss'] * df.get('health_index_norm', 1.0)
    
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
    
    return df


def prepare_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare features based on PHM configuration.
    config = {'health_index': True/False, 'temporal': True/False, ...}
    """
    df = df.copy()
    
    # Always add base features
    df = add_base_features(df)
    
    # Add PHM enhancements based on config
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
    # Base features
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    # Multi-defect features (always included as base)
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
        hi_feats = ['health_index', 'health_index_norm']
        for f in hi_feats:
            if f in df.columns:
                feats.append(f)
    
    # Temporal features
    if config.get('temporal', False):
        temporal_feats = [c for c in df.columns if any(x in c for x in ['_trend', '_accel', '_roll3', '_roll_std'])]
        temporal_feats += ['week_of_year', 'month', 'quarter']
        for f in temporal_feats:
            if f in df.columns and f not in feats:
                feats.append(f)
    
    # Decision features
    if config.get('decision', False):
        decision_feats = ['cost_benefit_ratio', 'intervention_priority']
        for f in decision_feats:
            if f in df.columns:
                feats.append(f)
    
    # Defect rate columns
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    feats.extend(rate_cols)
    
    # Remove duplicates and non-existent columns
    feats = [f for f in dict.fromkeys(feats) if f in df.columns]
    
    return feats


def train_and_evaluate(df: pd.DataFrame, config: dict, thr_label: float, n_est: int):
    """
    Train model with given PHM configuration and return metrics.
    """
    # Prepare features based on config
    df_prepared = prepare_features(df, config)
    
    # Split data
    df_train, df_calib, df_test = time_split_621(df_prepared)
    
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
        if f not in df_train.columns:
            df_train[f] = 0.0
        if f not in df_calib.columns:
            df_calib[f] = 0.0
        if f not in df_test.columns:
            df_test[f] = 0.0
    
    # Prepare X and y
    X_train = df_train[feats].copy()
    y_train = (df_train["scrap_percent"] > thr_label).astype(int)
    X_calib = df_calib[feats].copy()
    y_calib = (df_calib["scrap_percent"] > thr_label).astype(int)
    X_test = df_test[feats].copy()
    y_test = (df_test["scrap_percent"] > thr_label).astype(int)
    
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
    
    # Predictions
    preds_proba = model.predict_proba(X_test)[:, 1]
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


def run_factorial_experiment(df: pd.DataFrame, thr_label: float, n_est: int, 
                             progress_callback=None):
    """
    Run full factorial experiment testing all 16 PHM configurations.
    """
    # All possible configurations
    enhancement_keys = ['health_index', 'temporal', 'uncertainty', 'decision']
    all_configs = list(product([False, True], repeat=4))
    
    results = []
    
    for i, config_tuple in enumerate(all_configs):
        config = dict(zip(enhancement_keys, config_tuple))
        
        # Create config name
        active = [k[:2].upper() for k, v in config.items() if v]
        config_name = '+'.join(active) if active else 'Base'
        
        if progress_callback:
            progress_callback(i / len(all_configs), f"Testing: {config_name}")
        
        try:
            metrics, _, _ = train_and_evaluate(df, config, thr_label, n_est)
            
            result = {
                'config_id': i,
                'config_name': config_name,
                'health_index': config['health_index'],
                'temporal': config['temporal'],
                'uncertainty': config['uncertainty'],
                'decision': config['decision'],
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            result = {
                'config_id': i,
                'config_name': config_name,
                'health_index': config['health_index'],
                'temporal': config['temporal'],
                'uncertainty': config['uncertainty'],
                'decision': config['decision'],
                'error': str(e)
            }
            results.append(result)
    
    return pd.DataFrame(results)


def run_single_experiment(df: pd.DataFrame, config: dict, thr_label: float, n_est: int):
    """Run single configuration experiment."""
    metrics, model, feats = train_and_evaluate(df, config, thr_label, n_est)
    
    active = [k[:2].upper() for k, v in config.items() if v]
    config_name = '+'.join(active) if active else 'Base'
    
    return {
        'config_name': config_name,
        **config,
        **metrics
    }, model, feats


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

st.sidebar.header("🔬 PHM Enhancements")
st.sidebar.caption("Toggle individual enhancements for testing")

enable_health_index = st.sidebar.checkbox("Health Index (HI)", value=False, 
    help="Composite degradation score combining all defect rates")
enable_temporal = st.sidebar.checkbox("Temporal Features (TF)", value=False,
    help="Trend detection: defect changes over time")
enable_uncertainty = st.sidebar.checkbox("Uncertainty Quantification (UQ)", value=False,
    help="Confidence intervals from ensemble variance")
enable_decision = st.sidebar.checkbox("Decision Optimization (DO)", value=False,
    help="Cost-based intervention recommendations")

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
    "🔬 Single Test", 
    "🧪 Factorial Experiment", 
    "📊 Results Analysis",
    "📚 PHM Reference"
])


# ================================================================
# TAB 1: SINGLE TEST
# ================================================================
with tab1:
    st.header("🔬 Single Configuration Test")
    
    st.markdown("""
    Test the current PHM configuration selected in the sidebar.
    Compare results against the base model (no enhancements).
    """)
    
    # Show current config
    active_enhancements = [k for k, v in current_config.items() if v]
    if active_enhancements:
        st.success(f"**Active Enhancements:** {', '.join([PHM_ENHANCEMENTS[k]['name'] for k in active_enhancements])}")
    else:
        st.info("**Active Enhancements:** None (Base Model)")
    
    if st.button("🧪 Run Single Test"):
        with st.spinner("Training models..."):
            # Run base model
            base_config = {'health_index': False, 'temporal': False, 'uncertainty': False, 'decision': False}
            base_results, _, _ = run_single_experiment(df_base, base_config, thr_label, n_est)
            
            # Run current config
            current_results, model, feats = run_single_experiment(df_base, current_config, thr_label, n_est)
        
        st.markdown("### 📊 Results Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Base Model**")
            st.metric("Brier Score", f"{base_results['brier_score']:.4f}")
            st.metric("Accuracy", f"{base_results['accuracy']:.1%}")
            st.metric("Recall", f"{base_results['recall']:.1%}")
            st.metric("F1 Score", f"{base_results['f1_score']:.3f}")
            st.metric("Features", base_results['n_features'])
        
        with col2:
            st.markdown(f"**{current_results['config_name']}**")
            st.metric("Brier Score", f"{current_results['brier_score']:.4f}")
            st.metric("Accuracy", f"{current_results['accuracy']:.1%}")
            st.metric("Recall", f"{current_results['recall']:.1%}")
            st.metric("F1 Score", f"{current_results['f1_score']:.3f}")
            st.metric("Features", current_results['n_features'])
        
        with col3:
            st.markdown("**Improvement**")
            brier_imp = (base_results['brier_score'] - current_results['brier_score']) / base_results['brier_score'] * 100
            acc_imp = (current_results['accuracy'] - base_results['accuracy']) * 100
            recall_imp = (current_results['recall'] - base_results['recall']) * 100
            f1_imp = (current_results['f1_score'] - base_results['f1_score'])
            
            st.metric("Brier Δ", f"{brier_imp:+.1f}%", delta_color="normal" if brier_imp > 0 else "inverse")
            st.metric("Accuracy Δ", f"{acc_imp:+.1f}%", delta_color="normal" if acc_imp > 0 else "inverse")
            st.metric("Recall Δ", f"{recall_imp:+.1f}%", delta_color="normal" if recall_imp > 0 else "inverse")
            st.metric("F1 Δ", f"{f1_imp:+.3f}", delta_color="normal" if f1_imp > 0 else "inverse")
        
        # Store results for download
        st.session_state['single_test_results'] = {
            'base': base_results,
            'current': current_results,
            'config': current_config
        }
        
        # Download button
        comparison_df = pd.DataFrame([base_results, current_results])
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Comparison (CSV)",
            csv_data,
            f"single_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


# ================================================================
# TAB 2: FACTORIAL EXPERIMENT
# ================================================================
with tab2:
    st.header("🧪 Full Factorial Experiment")
    
    st.markdown("""
    Test **all 16 possible combinations** of PHM enhancements to find the optimal configuration.
    
    | # | HI | TF | UQ | DO | Description |
    |---|----|----|----|----|-------------|
    | 0 | ❌ | ❌ | ❌ | ❌ | Base Model |
    | 1 | ✅ | ❌ | ❌ | ❌ | Health Index only |
    | ... | ... | ... | ... | ... | ... |
    | 15 | ✅ | ✅ | ✅ | ✅ | Full PHM Model |
    """)
    
    if st.button("🚀 Run Full Factorial Experiment (16 configurations)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)
        
        with st.spinner("Running factorial experiment..."):
            results_df = run_factorial_experiment(df_base, thr_label, n_est, update_progress)
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        # Store results
        st.session_state['factorial_results'] = results_df
        
        st.success(f"✅ Completed {len(results_df)} configuration tests!")
        
        # Display results table
        st.markdown("### 📊 Results Table")
        
        # Format for display
        display_cols = ['config_name', 'health_index', 'temporal', 'uncertainty', 'decision',
                       'brier_score', 'accuracy', 'recall', 'precision', 'f1_score', 'n_features']
        display_df = results_df[[c for c in display_cols if c in results_df.columns]].copy()
        
        # Add improvement vs base
        if 'brier_score' in results_df.columns:
            base_brier = results_df[results_df['config_name'] == 'Base']['brier_score'].values[0]
            display_df['brier_improvement_%'] = ((base_brier - results_df['brier_score']) / base_brier * 100).round(1)
        
        # Sort by Brier score (lower is better)
        display_df = display_df.sort_values('brier_score')
        
        st.dataframe(display_df, use_container_width=True)
        
        # Best configuration
        best_config = display_df.iloc[0]
        st.success(f"""
        🏆 **Best Configuration: {best_config['config_name']}**
        - Brier Score: {best_config['brier_score']:.4f}
        - Accuracy: {best_config['accuracy']:.1%}
        - Recall: {best_config['recall']:.1%}
        - Improvement vs Base: {best_config.get('brier_improvement_%', 'N/A')}%
        """)
        
        # Download
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Factorial Results (CSV)",
            csv_data,
            f"factorial_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


# ================================================================
# TAB 3: RESULTS ANALYSIS
# ================================================================
with tab3:
    st.header("📊 Results Analysis")
    
    if 'factorial_results' not in st.session_state:
        st.info("Run a factorial experiment first to see analysis.")
    else:
        results_df = st.session_state['factorial_results']
        
        # Bar chart of Brier scores
        st.markdown("### 📈 Brier Score by Configuration")
        
        fig_brier = px.bar(
            results_df.sort_values('brier_score'),
            x='config_name',
            y='brier_score',
            color='brier_score',
            color_continuous_scale='RdYlGn_r',
            title='Brier Score by PHM Configuration (Lower is Better)'
        )
        fig_brier.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_brier, use_container_width=True)
        
        # Recall comparison
        st.markdown("### 🎯 Recall by Configuration")
        
        fig_recall = px.bar(
            results_df.sort_values('recall', ascending=False),
            x='config_name',
            y='recall',
            color='recall',
            color_continuous_scale='RdYlGn',
            title='Recall by PHM Configuration (Higher is Better)'
        )
        fig_recall.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_recall, use_container_width=True)
        
        # Enhancement impact analysis
        st.markdown("### 🔍 Individual Enhancement Impact")
        
        # Calculate average impact of each enhancement
        impacts = {}
        for enhancement in ['health_index', 'temporal', 'uncertainty', 'decision']:
            with_enhancement = results_df[results_df[enhancement] == True]['brier_score'].mean()
            without_enhancement = results_df[results_df[enhancement] == False]['brier_score'].mean()
            impact = (without_enhancement - with_enhancement) / without_enhancement * 100
            impacts[PHM_ENHANCEMENTS[enhancement]['name']] = impact
        
        impact_df = pd.DataFrame([
            {'Enhancement': k, 'Brier Improvement %': v} 
            for k, v in impacts.items()
        ]).sort_values('Brier Improvement %', ascending=False)
        
        fig_impact = px.bar(
            impact_df,
            x='Enhancement',
            y='Brier Improvement %',
            color='Brier Improvement %',
            color_continuous_scale='RdYlGn',
            title='Average Impact of Each Enhancement on Brier Score'
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        st.dataframe(impact_df, use_container_width=True)
        
        # Synergy analysis
        st.markdown("### 🔗 Enhancement Synergies")
        
        # Compare individual vs combined effects
        base_brier = results_df[results_df['config_name'] == 'Base']['brier_score'].values[0]
        
        synergy_data = []
        
        # Two-way combinations
        combinations = [
            ('HI', 'TF', 'health_index', 'temporal'),
            ('HI', 'UQ', 'health_index', 'uncertainty'),
            ('HI', 'DO', 'health_index', 'decision'),
            ('TF', 'UQ', 'temporal', 'uncertainty'),
            ('TF', 'DO', 'temporal', 'decision'),
            ('UQ', 'DO', 'uncertainty', 'decision'),
        ]
        
        for name1, name2, key1, key2 in combinations:
            # Individual effects
            only1 = results_df[
                (results_df[key1] == True) & 
                (results_df[key2] == False) &
                (results_df[[k for k in ['health_index', 'temporal', 'uncertainty', 'decision'] if k not in [key1, key2]]].sum(axis=1) == 0)
            ]
            only2 = results_df[
                (results_df[key1] == False) & 
                (results_df[key2] == True) &
                (results_df[[k for k in ['health_index', 'temporal', 'uncertainty', 'decision'] if k not in [key1, key2]]].sum(axis=1) == 0)
            ]
            combined = results_df[
                (results_df[key1] == True) & 
                (results_df[key2] == True) &
                (results_df[[k for k in ['health_index', 'temporal', 'uncertainty', 'decision'] if k not in [key1, key2]]].sum(axis=1) == 0)
            ]
            
            if len(only1) > 0 and len(only2) > 0 and len(combined) > 0:
                imp1 = (base_brier - only1['brier_score'].values[0]) / base_brier * 100
                imp2 = (base_brier - only2['brier_score'].values[0]) / base_brier * 100
                imp_combined = (base_brier - combined['brier_score'].values[0]) / base_brier * 100
                expected = imp1 + imp2
                synergy = imp_combined - expected
                
                synergy_data.append({
                    'Combination': f"{name1} + {name2}",
                    f'{name1} alone': f"{imp1:.1f}%",
                    f'{name2} alone': f"{imp2:.1f}%",
                    'Expected (additive)': f"{expected:.1f}%",
                    'Actual combined': f"{imp_combined:.1f}%",
                    'Synergy': f"{synergy:+.1f}%"
                })
        
        if synergy_data:
            synergy_df = pd.DataFrame(synergy_data)
            st.dataframe(synergy_df, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - **Positive synergy**: Enhancements work better together than expected
            - **Negative synergy**: Enhancements may be redundant or conflicting
            - **Zero synergy**: Enhancements are independent
            """)


# ================================================================
# TAB 4: PHM REFERENCE
# ================================================================
with tab4:
    st.header("📚 PHM Enhancement Reference")
    
    st.markdown("""
    This section documents the PHM (Prognostics and Health Management) enhancements 
    implemented in this dashboard, their theoretical foundations, and expected benefits.
    """)
    
    for key, info in PHM_ENHANCEMENTS.items():
        with st.expander(f"**{info['name']}**"):
            st.markdown(f"""
            **Description:** {info['description']}
            
            **Literature Reference:** {info['reference']}
            """)
            
            if key == 'health_index':
                st.markdown("""
                **Implementation Details:**
                - Combines all defect rates into a single degradation score
                - Weights defects by criticality (critical: 3x, moderate: 2x, minor: 1x)
                - Normalizes to 0-1 scale for comparability
                - Categorizes into: Healthy, Warning, Critical, Failure
                
                **Expected Benefit:** Captures overall system health rather than individual symptoms
                """)
                
            elif key == 'temporal':
                st.markdown("""
                **Implementation Details:**
                - **Trend**: First derivative of defect rate (getting worse or better?)
                - **Acceleration**: Second derivative (is degradation accelerating?)
                - **Rolling Average**: 3-period smoothed trend
                - **Rolling Std**: Volatility of defect rates
                
                **Expected Benefit:** Detects degradation patterns before they become critical
                """)
                
            elif key == 'uncertainty':
                st.markdown("""
                **Implementation Details:**
                - Uses variance across Random Forest trees as uncertainty estimate
                - Calculates 95% confidence intervals for predictions
                - Reports mean uncertainty and CI width
                
                **Expected Benefit:** Indicates prediction reliability for decision-making
                """)
                
            elif key == 'decision':
                st.markdown("""
                **Implementation Details:**
                - Calculates expected loss = P(scrap) × quantity × cost
                - Compares to intervention cost
                - Recommends intervention when expected loss > intervention cost
                - Prioritizes by expected loss × health index
                
                **Expected Benefit:** Translates predictions into actionable decisions
                """)
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Key References
    
    1. **Lei, Y., et al. (2018).** Machinery health prognostics: A systematic review from data acquisition to RUL prediction. *Mechanical Systems and Signal Processing*, 104, 799–834.
    
    2. **Choo, Y.-S., & Shin, S.-J. (2025).** Integrating machine learning-based remaining useful life predictions with cost-optimal block replacement for industrial maintenance. *International Journal of Prognostics and Health Management*, 16(1).
    
    3. **Molęda, M. (2023).** From corrective to predictive maintenance: A review of approaches and state-of-the-art techniques. *IJERPH*, 20(1).
    
    4. **Campbell, J. (2003).** *Castings Practice: The Ten Rules of Castings*. Elsevier.
    """)


st.markdown("---")
st.caption("🏭 Foundry Scrap Risk Dashboard **v3.2 - PHM Enhancement Laboratory** | Factorial Testing | Based on Lei et al. (2018), Choo & Shin (2025), Campbell (2003)")
