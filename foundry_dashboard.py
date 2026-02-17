# ================================================================
# 🏭 FOUNDRY PROGNOSTIC RELIABILITY DASHBOARD
# THREE-STAGE HIERARCHICAL TRANSFER LEARNING VERSION
# ================================================================
#
# KEY INNOVATION: THREE-STAGE HIERARCHICAL LEARNING
# ====================================================
# Stage 1: FOUNDRY-WIDE - Train on ALL data with global threshold
#          → Learns patterns common across all parts
#          → Adds: global_scrap_probability feature
#
# Stage 2: DEFECT-CLUSTER - Train on TOP 5 PARETO defects
#          → Focuses on high-impact defects (~80% of scrap)
#          → Adds: defect_cluster_probability feature
#
# Stage 3: PART-SPECIFIC - Train on part's data with per-part threshold
#          → Inherits features from Stages 1 & 2
#          → Fine-tuned to detect deviation from part's baseline
#
# RESEARCH SUPPORT:
#   - Tercan et al. (2018): Multi-stage TL in injection molding
#   - Zhang et al. (2021): Hierarchical TL for semiconductor manufacturing
#   - Zhang H.B. et al. (2023): Hierarchical adaptive RUL prediction
#   - Agarwal & Chowdary (2020): Stacked ensemble learning
#
# ================================================================
# SCIKIT-LEARN APPLICATION OVERVIEW
# ================================================================
# This dashboard uses Python's Scikit-learn library (Pedregosa et al., 2011)
# for all machine learning operations. Here's how it's applied:
#
# TRAINING (train_global_model function):
#   - RandomForestClassifier: Ensemble learning with 180 decision trees
#   - CalibratedClassifierCV: Platt scaling for probability calibration
#
# PREDICTION (predict_for_part function):
#   - model.predict_proba(): Generate scrap probability for each record
#
# EVALUATION (used in Tabs 2, 3, 5):
#   - recall_score(): Measures % of actual failures correctly predicted
#   - precision_score(): Measures % of predictions that were correct
#   - roc_auc_score(): Measures model's discrimination ability
#   - brier_score_loss(): Measures probability calibration quality
#   - roc_curve(): Generates ROC curve data points
#   - calibration_curve(): Generates calibration curve data points
#   - confusion_matrix(): Generates TP, FP, TN, FN counts
#
# KEY REFERENCES:
#   - Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
#     Journal of Machine Learning Research, 12, 2825-2830.
#   - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
#   - Platt, J. (1999). Probabilistic outputs for support vector machines.
#     Advances in Large Margin Classifiers, 10(3), 61-74.
#
# KEY FEATURES MATCHING ENHANCED VERSION:
# 1. Multi-Defect Intelligence (n_defect_types, interactions)
# 2. Temporal Features (trends, rolling averages)
# 3. MTTS Reliability Features (hazard_rate, RUL proxy, etc.)
# 4. Global Model Training with 6-2-1 split
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================================================================
# SCIKIT-LEARN IMPORTS
# ================================================================
# These are the core ML functions from Scikit-learn used throughout
# this dashboard for model training, calibration, and evaluation.
#
# Reference: Pedregosa, F., et al. (2011). Scikit-learn: Machine 
# Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
# ================================================================
from sklearn.ensemble import RandomForestClassifier  # ML MODEL: Ensemble of 180 decision trees
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # CALIBRATION: Platt scaling
from sklearn.metrics import (
    brier_score_loss,      # EVALUATION: Probability calibration quality (lower = better)
    accuracy_score,        # EVALUATION: Overall correct predictions / total
    recall_score,          # EVALUATION: True Positives / (True Positives + False Negatives)
    precision_score,       # EVALUATION: True Positives / (True Positives + False Positives)
    f1_score,              # EVALUATION: Harmonic mean of precision and recall
    roc_auc_score,         # EVALUATION: Area Under ROC Curve (discrimination ability)
    roc_curve,             # VISUALIZATION: False Positive Rate vs True Positive Rate
    precision_recall_curve,# VISUALIZATION: Precision vs Recall tradeoff
    confusion_matrix       # EVALUATION: TP, FP, TN, FN matrix
)
from scipy import stats
from datetime import datetime

# ================================================================
# LIME - LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS
# ================================================================
# Reference: Ribeiro, M.T., Singh, S., & Guestrin, C. (2016).
# "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
# Proceedings of the 22nd ACM SIGKDD International Conference on
# Knowledge Discovery and Data Mining, 1135-1144.
#
# LIME explains individual predictions by:
# 1. Perturbing the input features around the instance of interest
# 2. Getting model predictions for each perturbation
# 3. Fitting a simple, interpretable model (linear regression) locally
# 4. Returning feature weights showing each feature's contribution
#
# This enables analysts to understand WHY the model made a specific
# prediction, supporting the NASA mission assurance principle of
# "dynamic, synthesizing feedback" for decision-making.
# ================================================================
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    # Fallback: LIME not installed

# ================================================================
# STREAMLIT CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Foundry Prognostic Reliability Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
# CONSTANTS - MATCHING ENHANCED VERSION
# ================================================================
RANDOM_STATE = 42
DEFAULT_CSV_PATH = "anonymized_parts.csv"
WEIGHT_TOLERANCE = 0.10
N_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 5
MULTI_DEFECT_THRESHOLD = 2  # >= 2 defects = multi-defect
ROLLING_WINDOW_SIZE = 3

# Feature toggles - ALL ENABLED to match enhanced version
MULTI_DEFECT_FEATURES_ENABLED = True
TEMPORAL_FEATURES_ENABLED = True
MTTS_FEATURES_ENABLED = True

# DOE Energy Benchmarks
DOE_BENCHMARKS = {
    'average': 22922,
    'best_practice': 18500,
    'theoretical_minimum': 5000
}
CO2_PER_MMBTU = 53.06

# RQ Validation Thresholds
RQ_THRESHOLDS = {
    'RQ1': {'recall': 0.80, 'precision': 0.70, 'f1': 0.70, 'auc': 0.80},
    'RQ2': {'sensor_benchmark': 0.90, 'phm_equivalence': 0.80},
    'RQ3': {'scrap_reduction_min': 0.10, 'scrap_reduction_max': 0.20, 'roi_min': 2.0}
}

# ================================================================
# PROCESS-DEFECT MAPPING
# ================================================================
# Derived from Campbell, J. (2003). Castings Practice: The 10 Rules
# of Castings. Elsevier Butterworth-Heinemann.
#
# Dataset defect terminology was aligned with Campbell's casting defect
# taxonomy. Each defect column mapped to the originating process stage
# per Campbell's 10 Rules. This is a proof-of-concept mapping —
# individual foundries must validate against their specific processes.
# See "Campbell Framework Reference" tab for full rationale.
# ================================================================
PROCESS_DEFECT_MAP = {
    "Melting": {"defects": ["dross_rate", "gas_porosity_rate"], 
                "description": "Metal preparation, temperature control",
                "campbell_rule": "Rule 1: Achieve a Good Quality Melt"},
    "Pouring": {"defects": ["misrun_rate", "missrun_rate", "short_pour_rate", "runout_rate"], 
                "description": "Pour temperature, rate control",
                "campbell_rule": "Rule 2: Avoid Turbulent Entrainment"},
    "Gating Design": {"defects": ["shrink_rate", "shrink_porosity_rate", "tear_up_rate"], 
                      "description": "Runner/riser sizing, feeding",
                      "campbell_rule": "Rule 6: Avoid Shrinkage Damage"},
    "Sand System": {"defects": ["sand_rate", "dirty_pattern_rate"], 
                    "description": "Sand preparation, binder ratio",
                    "campbell_rule": "Rules 2-3 (secondary)"},
    "Core Making": {"defects": ["core_rate", "crush_rate", "shift_rate"], 
                    "description": "Core integrity, venting",
                    "campbell_rule": "Rule 5: Avoid Core Blows"},
    "Shakeout": {"defects": ["bent_rate"], 
                 "description": "Casting extraction, cooling",
                 "campbell_rule": "Rule 9: Reduce Residual Stress"},
    "Pattern/Tooling": {"defects": ["gouged_rate"], 
                        "description": "Pattern accuracy, wear",
                        "campbell_rule": "Rule 10: Provide Location Points"},
    "Inspection": {"defects": ["outside_process_scrap_rate", "zyglo_rate", "failed_zyglo_rate"], 
                   "description": "Quality control, NDT",
                   "campbell_rule": "Detection stage (not process-origin)"},
    "Finishing": {"defects": ["over_grind_rate", "cut_into_rate"], 
                  "description": "Grinding, machining",
                  "campbell_rule": "Post-casting operations"}
}

# Hierarchical Pooling Configuration
POOLING_CONFIG = {
    'enabled': True,
    'min_part_level_data': 5,
    'weight_tolerance': 0.10,
    'min_runs_per_pooled_part': 5,  # Filter out parts with < 5 runs (reduces noise)
    'use_pooled_threshold': True,   # Use pooled avg+std instead of global threshold
    'confidence_thresholds': {
        'HIGH': 30,
        'MODERATE': 15,
        'LOW': 5,
    }
}

# Central Limit Theorem threshold: parts with ≥30 runs have statistically
# reliable sample means; parts below this show dual results (part-level +
# pooled comparison) so foundry managers can apply experienced judgment.
CLT_THRESHOLD = 30

# ================================================================
# THREE-STAGE HIERARCHICAL LEARNING CONFIGURATION
# ================================================================
# Stage 1: Foundry-Wide (all data, global threshold)
# Stage 2: Defect-Cluster (top 5 Pareto defects)
# Stage 3: Part-Specific (individual part, per-part threshold)
# ================================================================
THREE_STAGE_CONFIG = {
    'enabled': True,
    'stage1': {
        'name': 'Foundry-Wide',
        'description': 'Common patterns across all parts',
        'threshold_type': 'global_average',  # Uses dataset avg scrap %
    },
    'stage2': {
        'name': 'Defect-Cluster',
        'description': 'Patterns for top 5 Pareto defects',
        'threshold_type': 'cluster_average',  # Uses cluster avg scrap %
        'top_n_defects': 5,  # Focus on top 5 defects from Pareto
    },
    'stage3': {
        'name': 'Part-Specific',
        'description': 'Fine-tuned for individual part baseline',
        'threshold_type': 'per_part_average',  # Uses part's own avg scrap %
    }
}

# Top 5 Pareto Defects (from your Pareto chart - ~80% of scrap)
TOP_PARETO_DEFECTS = [
    'sand_rate',      # #1 - ~25% of scrap
    'shift_rate',     # #2 - ~10% of scrap
    'missrun_rate',   # #3 - ~10% of scrap  
    'gouged_rate',    # #4 - ~8% of scrap
    'dross_rate',     # #5 - ~8% of scrap
]

# Defect rate columns for pooling
DEFECT_RATE_COLUMNS = [
    'bent_rate', 'outside_process_scrap_rate', 'failed_zyglo_rate',
    'gouged_rate', 'shift_rate', 'missrun_rate', 'core_rate',
    'cut_into_rate', 'dirty_pattern_rate', 'crush_rate', 'zyglo_rate',
    'shrink_rate', 'short_pour_rate', 'runout_rate', 'shrink_porosity_rate',
    'gas_porosity_rate', 'over_grind_rate', 'sand_rate', 'tear_up_rate',
    'dross_rate'
]

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 25px; color: white;
    }
    .citation-box {
        background-color: #f0f7ff; border-left: 4px solid #1976D2;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
    }
    .hypothesis-pass {
        background-color: #e8f5e9; border-left: 4px solid #4CAF50;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
    }
    .hypothesis-fail {
        background-color: #fff3e0; border-left: 4px solid #FF9800;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# DATA LOADING
# ================================================================
@st.cache_data
def load_data(filepath):
    """Load and preprocess the foundry dataset."""
    if not os.path.exists(filepath):
        return None, None
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return None, None
    
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    col_map = {
        "part_id_anonymized": "part_id", "partid": "part_id",
        "quantity": "order_quantity", "order_qty": "order_quantity",
        "weight": "piece_weight_lbs", "piece_weight": "piece_weight_lbs",
        "piece_weight_(lbs)": "piece_weight_lbs",  # Handle parentheses in column name
        "scrap_%": "scrap_percent", "scrap": "scrap_percent",
        "scrap%": "scrap_percent",  # Handle this variant too
        "week_ending_date": "week_ending",
    }
    
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    # Ensure required columns
    if "part_id" not in df.columns:
        df["part_id"] = "UNKNOWN"
    if "order_quantity" not in df.columns:
        df["order_quantity"] = 100
    if "piece_weight_lbs" not in df.columns:
        df["piece_weight_lbs"] = 1.0
    if "scrap_percent" not in df.columns:
        if "scrap%" in df.columns:
            df["scrap_percent"] = pd.to_numeric(df["scrap%"], errors="coerce").fillna(0)
        else:
            df["scrap_percent"] = 0
    
    df["part_id"] = df["part_id"].astype(str).str.strip()
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(100)
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce").fillna(1.0)
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0)
    
    if "week_ending" in df.columns:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
        df = df.dropna(subset=["week_ending"])
        df = df.sort_values("week_ending").reset_index(drop=True)
    else:
        df["week_ending"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='W')
    
    defect_cols = [c for c in df.columns if c.endswith("_rate") and "total" not in c.lower()]
    
    return df, defect_cols


# ================================================================
# MULTI-DEFECT FEATURES (FROM ENHANCED V3.0)
# ================================================================
def add_multi_defect_features(df, defect_cols):
    """Add multi-defect intelligence features matching enhanced version."""
    df = df.copy()
    
    valid_defect_cols = [c for c in defect_cols if c in df.columns]
    if not valid_defect_cols:
        df['n_defect_types'] = 0
        df['has_multiple_defects'] = 0
        df['total_defect_rate'] = 0
        df['max_defect_rate'] = 0
        df['defect_concentration'] = 0
        return df
    
    # Ensure numeric
    for col in valid_defect_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Count defect types present
    df['n_defect_types'] = (df[valid_defect_cols] > 0).sum(axis=1)
    
    # Binary flag for multiple defects
    df['has_multiple_defects'] = (df['n_defect_types'] >= MULTI_DEFECT_THRESHOLD).astype(int)
    
    # Total defect burden
    df['total_defect_rate'] = df[valid_defect_cols].sum(axis=1)
    
    # Maximum single defect rate
    df['max_defect_rate'] = df[valid_defect_cols].max(axis=1)
    
    # Defect concentration
    df['defect_concentration'] = df['max_defect_rate'] / (df['total_defect_rate'] + 0.001)
    
    # Interaction terms
    if 'shift_rate' in df.columns and 'tear_up_rate' in df.columns:
        df['shift_x_tearup'] = df['shift_rate'] * df['tear_up_rate']
    if 'shrink_rate' in df.columns and 'gas_porosity_rate' in df.columns:
        df['shrink_x_porosity'] = df['shrink_rate'] * df['gas_porosity_rate']
    if 'shrink_rate' in df.columns and 'shrink_porosity_rate' in df.columns:
        df['shrink_x_shrink_porosity'] = df['shrink_rate'] * df['shrink_porosity_rate']
    if 'core_rate' in df.columns and 'sand_rate' in df.columns:
        df['core_x_sand'] = df['core_rate'] * df['sand_rate']
    
    return df


# ================================================================
# TEMPORAL FEATURES (FROM ENHANCED V3.1)
# ================================================================
def add_temporal_features(df):
    """Add temporal trend features matching enhanced version."""
    df = df.copy()
    
    if 'week_ending' in df.columns:
        df = df.sort_values('week_ending').reset_index(drop=True)
    
    for col in ['total_defect_rate', 'scrap_percent']:
        if col not in df.columns:
            continue
        
        if 'part_id' in df.columns:
            df[f'{col}_trend'] = df.groupby('part_id')[col].diff().fillna(0)
            df[f'{col}_roll3'] = df.groupby('part_id')[col].transform(
                lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
            )
        else:
            df[f'{col}_trend'] = df[col].diff().fillna(0)
            df[f'{col}_roll3'] = df[col].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    
    if 'week_ending' in df.columns:
        df['month'] = pd.to_datetime(df['week_ending']).dt.month
        df['quarter'] = pd.to_datetime(df['week_ending']).dt.quarter
    
    return df


# ================================================================
# MTTS COMPUTATION (FROM ENHANCED V3.2)
# ================================================================
def compute_mtts_metrics(df, threshold):
    """Compute MTTS metrics per part - matching enhanced version."""
    results = []
    df_sorted = df.sort_values(['part_id', 'week_ending']).copy()
    
    if 'order_quantity' not in df_sorted.columns:
        df_sorted['order_quantity'] = 1
    
    for part_id, group in df_sorted.groupby('part_id'):
        group = group.reset_index(drop=True)
        
        parts_since_last_failure = 0
        failure_cycles_parts = []
        failure_cycles_runs = []
        runs_since_last_failure = 0
        failure_count = 0
        
        for _, row in group.iterrows():
            order_qty = row.get('order_quantity', 1)
            parts_since_last_failure += order_qty
            runs_since_last_failure += 1
            
            if row['scrap_percent'] > threshold:
                failure_cycles_parts.append(parts_since_last_failure)
                failure_cycles_runs.append(runs_since_last_failure)
                failure_count += 1
                parts_since_last_failure = 0
                runs_since_last_failure = 0
        
        total_runs = len(group)
        total_parts = group['order_quantity'].sum() if 'order_quantity' in group.columns else total_runs
        avg_order_quantity = total_parts / total_runs if total_runs > 0 else 0
        
        if len(failure_cycles_parts) > 0:
            mtts_parts = np.mean(failure_cycles_parts)
            mtts_runs = np.mean(failure_cycles_runs)
        else:
            mtts_parts = total_parts
            mtts_runs = total_runs
        
        lambda_parts = failure_count / total_parts if total_parts > 0 else 0
        lambda_runs = failure_count / total_runs if total_runs > 0 else 0
        
        reliability_score = np.exp(-avg_order_quantity / mtts_parts) if mtts_parts > 0 else 0
        
        results.append({
            'part_id': part_id,
            'mtts_parts': mtts_parts,
            'mtts_runs': mtts_runs,
            'failure_count': failure_count,
            'total_runs': total_runs,
            'total_parts': total_parts,
            'avg_order_quantity': avg_order_quantity,
            'lambda_parts': lambda_parts,
            'lambda_runs': lambda_runs,
            'hazard_rate': lambda_runs,
            'reliability_score': reliability_score
        })
    
    return pd.DataFrame(results)


def add_mtts_features(df, threshold):
    """Add MTTS-based reliability features matching enhanced version."""
    df = df.copy()
    df = df.sort_values(['part_id', 'week_ending']).reset_index(drop=True)
    
    df['runs_since_last_failure'] = 0
    df['cumulative_scrap_in_cycle'] = 0.0
    df['degradation_velocity'] = 0.0
    df['degradation_acceleration'] = 0.0
    df['cycle_hazard_indicator'] = 0.0
    
    mtts_df = compute_mtts_metrics(df, threshold)
    
    for part_id, group in df.groupby('part_id'):
        idx_list = group.index.tolist()
        
        runs_since_failure = 0
        cumulative_scrap = 0.0
        prev_scrap = 0.0
        prev_velocity = 0.0
        
        part_mtts = mtts_df[mtts_df['part_id'] == part_id]
        part_mtts_runs = part_mtts['mtts_runs'].values[0] if len(part_mtts) > 0 else 10
        
        for idx in idx_list:
            runs_since_failure += 1
            current_scrap = df.loc[idx, 'scrap_percent']
            cumulative_scrap += current_scrap
            
            df.loc[idx, 'runs_since_last_failure'] = runs_since_failure
            df.loc[idx, 'cumulative_scrap_in_cycle'] = cumulative_scrap
            
            velocity = current_scrap - prev_scrap
            df.loc[idx, 'degradation_velocity'] = velocity
            df.loc[idx, 'degradation_acceleration'] = velocity - prev_velocity
            
            cycle_position = runs_since_failure / part_mtts_runs if part_mtts_runs > 0 else 0
            df.loc[idx, 'cycle_hazard_indicator'] = min(cycle_position, 2.0)
            
            prev_scrap = current_scrap
            prev_velocity = velocity
            
            if current_scrap > threshold:
                runs_since_failure = 0
                cumulative_scrap = 0.0
    
    # Merge part-level MTTS metrics
    merge_cols = ['part_id', 'mtts_parts', 'mtts_runs', 'lambda_parts', 'lambda_runs',
                  'hazard_rate', 'reliability_score', 'failure_count']
    df = df.merge(mtts_df[merge_cols], on='part_id', how='left')
    
    # Fill missing
    df['mtts_parts'] = df['mtts_parts'].fillna(df['mtts_parts'].median() if df['mtts_parts'].notna().any() else 1000)
    df['mtts_runs'] = df['mtts_runs'].fillna(df['mtts_runs'].median() if df['mtts_runs'].notna().any() else 10)
    df['hazard_rate'] = df['hazard_rate'].fillna(0.1)
    df['reliability_score'] = df['reliability_score'].fillna(0.5)
    
    # RUL proxy
    df['rul_proxy'] = (df['mtts_runs'] - df['runs_since_last_failure']).clip(lower=0)
    
    return df


# ================================================================
# HIERARCHICAL POOLING SYSTEM (FROM ENHANCED VERSION)
# ================================================================
def identify_part_defects(df, part_id):
    """Identify which defect types are present for a part."""
    part_id = str(part_id)
    part_data = df[df['part_id'] == part_id]
    present_defects = []
    
    for col in DEFECT_RATE_COLUMNS:
        if col in df.columns and (part_data[col] > 0).any():
            present_defects.append(col)
    
    return present_defects


def get_part_weight(df, part_id):
    """
    Get the most representative weight for a part.
    Uses mode (most common value) if available, otherwise median.
    This handles cases where a part may have multiple weight entries.
    """
    part_data = df[df['part_id'] == str(part_id)]
    if len(part_data) == 0:
        return None
    
    weights = part_data['piece_weight_lbs'].dropna()
    if len(weights) == 0:
        return None
    
    # Use mode (most common weight) - more likely to be the correct value
    mode_vals = weights.mode()
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]
    
    # Fallback to median if no mode
    return weights.median()


def filter_by_weight(df, target_weight, tolerance=None):
    """Filter parts by weight within ±10% tolerance."""
    if tolerance is None:
        tolerance = POOLING_CONFIG['weight_tolerance']
    
    weight_min = target_weight * (1 - tolerance)
    weight_max = target_weight * (1 + tolerance)
    
    weight_col = 'piece_weight_lbs'
    
    # Use mode/median for each part's weight instead of first()
    # This is more robust when parts have multiple records with potentially different weights
    part_weights = df.groupby('part_id')[weight_col].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
    )
    
    matching_parts = part_weights[
        (part_weights >= weight_min) & (part_weights <= weight_max)
    ].index.tolist()
    
    weight_range = f"{weight_min:.1f} - {weight_max:.1f}"
    
    return matching_parts, weight_range


def filter_by_exact_defects(df, part_ids, target_defects):
    """Filter parts that have at least one of the SAME defect types."""
    if not target_defects:
        return part_ids
    
    matching_parts = []
    
    for pid in part_ids:
        part_data = df[df['part_id'] == pid]
        for defect_col in target_defects:
            if defect_col in df.columns and (part_data[defect_col] > 0).any():
                matching_parts.append(pid)
                break
    
    return list(set(matching_parts))


def filter_by_any_defect(df, part_ids):
    """Filter parts that have ANY defect type."""
    matching_parts = []
    
    for pid in part_ids:
        part_data = df[df['part_id'] == pid]
        has_any_defect = False
        
        for defect_col in DEFECT_RATE_COLUMNS:
            if defect_col in df.columns and (part_data[defect_col] > 0).any():
                has_any_defect = True
                break
        
        if has_any_defect:
            matching_parts.append(pid)
    
    return list(set(matching_parts))


def get_confidence_tier(n):
    """Get confidence tier based on sample size."""
    thresholds = POOLING_CONFIG['confidence_thresholds']
    if n >= thresholds['HIGH']:
        return f"HIGH ({n} ≥ {thresholds['HIGH']})"
    elif n >= thresholds['MODERATE']:
        return f"MODERATE ({n} ≥ {thresholds['MODERATE']})"
    elif n >= thresholds['LOW']:
        return f"LOW ({n} ≥ {thresholds['LOW']})"
    else:
        return f"INSUFFICIENT ({n} < {thresholds['LOW']})"


def compute_pooled_prediction(df, part_id, threshold_pct):
    """
    Compute reliability prediction using hierarchical pooling.
    
    Cascading strategy:
    1. Check if part-level data is sufficient (n ≥ 5)
    2. If not, try Weight ±10% + Exact Defect matching
    3. If that doesn't work, try Weight ±10% + Any Defect
    4. Return best available prediction with full transparency
    
    IMPORTANT: Uses mode/median for weight to ensure correct weight matching.
    
    MTTS CALCULATION (CORRECTED):
    - MTTS (runs) = Total Runs / Failures
    - MTTS (parts) = Total Parts Produced / Failures
    - Failure = any run where scrap % > foundry-wide threshold (systemic approach)
    """
    thresholds = POOLING_CONFIG['confidence_thresholds']
    min_part_data = POOLING_CONFIG['min_part_level_data']
    weight_tolerance = POOLING_CONFIG['weight_tolerance']
    
    part_id = str(part_id)
    part_data = df[df['part_id'] == part_id]
    part_n = len(part_data)
    
    # Use the robust get_part_weight function instead of .iloc[0]
    target_weight = get_part_weight(df, part_id)
    if target_weight is None:
        target_weight = 0
    
    target_defects = identify_part_defects(df, part_id)
    target_defects_clean = [d.replace('_rate', '').replace('_', ' ').title() for d in target_defects]
    
    # Calculate total parts produced (sum of order quantities)
    total_parts_produced = part_data['order_quantity'].sum() if 'order_quantity' in part_data.columns else part_n
    
    result = {
        'part_id': part_id,
        'part_level_n': part_n,
        'part_level_sufficient': part_n >= min_part_data,
        'target_weight': target_weight,
        'weight_used_for_prediction': target_weight,
        'target_defects': target_defects_clean,
        'pooling_used': False,
        'show_dual': False,
        'total_parts_produced': total_parts_produced,
        'threshold_used': threshold_pct,
    }
    
    # CASE 1: CLT satisfied — part-level data is statistically reliable (≥30 runs)
    if part_n >= CLT_THRESHOLD:
        confidence = get_confidence_tier(part_n)
        failures = (part_data['scrap_percent'] > threshold_pct).sum()
        failure_rate = failures / part_n if part_n > 0 else 0
        
        # MTTS (runs) - average runs between failures
        if failures > 0:
            mtts_runs = part_n / failures
        else:
            mtts_runs = part_n * 10  # No failures observed, estimate high MTTS
        
        # MTTS (parts) - average parts produced between failures
        if failures > 0:
            mtts_parts = total_parts_produced / failures
        else:
            mtts_parts = total_parts_produced * 10
        
        reliability = np.exp(-1 / mtts_runs) if mtts_runs > 0 else 0
        
        result.update({
            'pooling_method': 'Part-Level (No Pooling Required)',
            'pooling_used': False,
            'show_dual': False,
            'weight_range': 'N/A',
            'pooled_n': part_n,
            'pooled_parts_count': 1,
            'included_part_ids': [part_id],
            'confidence': confidence,
            'mtts_runs': mtts_runs,
            'mtts_parts': mtts_parts,
            'total_parts_produced': total_parts_produced,
            'reliability_next_run': reliability,
            'failure_count': failures,
            'failure_rate': failure_rate,
        })
        return result
    
    # CASE 2: Below CLT (<30 runs) — compute BOTH part-level and pooled comparison
    # Part-level metrics from whatever data exists (even 1-4 records)
    if part_n > 0:
        pl_failures = (part_data['scrap_percent'] > threshold_pct).sum()
        pl_failure_rate = pl_failures / part_n if part_n > 0 else 0
        if pl_failures > 0:
            pl_mtts_runs = part_n / pl_failures
            pl_mtts_parts = total_parts_produced / pl_failures
        else:
            pl_mtts_runs = part_n * 2
            pl_mtts_parts = total_parts_produced * 2
        pl_reliability = np.exp(-1 / pl_mtts_runs) if pl_mtts_runs > 0 else 0
        pl_confidence = 'VERY LOW' if part_n < 5 else get_confidence_tier(part_n)
        pl_avg_scrap = part_data['scrap_percent'].mean()
    else:
        pl_failures = 0
        pl_failure_rate = 0
        pl_mtts_runs = 10
        pl_mtts_parts = 1000
        pl_reliability = 0
        pl_confidence = 'VERY LOW'
        pl_avg_scrap = 0
    
    # Store part-level metrics in result (these are the PRIMARY prediction)
    result.update({
        'pooling_method': f'Part-Level (Dual: {part_n} runs, below CLT)',
        'pooling_used': False,
        'show_dual': True,
        'weight_range': 'N/A',
        'pooled_n': part_n,
        'pooled_parts_count': 1,
        'included_part_ids': [part_id],
        'confidence': pl_confidence,
        'mtts_runs': pl_mtts_runs,
        'mtts_parts': pl_mtts_parts,
        'total_parts_produced': total_parts_produced,
        'reliability_next_run': pl_reliability,
        'failure_count': pl_failures,
        'failure_rate': pl_failure_rate,
        'part_level_avg_scrap': pl_avg_scrap,
    })
    
    # Now compute pooled COMPARISON (secondary reference)
    
    # Get config values
    min_runs_per_part = POOLING_CONFIG.get('min_runs_per_pooled_part', 5)
    use_pooled_threshold = POOLING_CONFIG.get('use_pooled_threshold', True)
    
    # Step 1: Weight filter
    weight_matched_parts, weight_range = filter_by_weight(df, target_weight, weight_tolerance)
    
    # Step 2: Exact defect filter
    exact_matched_parts = filter_by_exact_defects(df, weight_matched_parts, target_defects)
    exact_pooled_df = df[df['part_id'].isin(exact_matched_parts)]
    
    # Step 3: Any defect filter
    any_matched_parts = filter_by_any_defect(df, weight_matched_parts)
    any_pooled_df = df[df['part_id'].isin(any_matched_parts)]
    
    # Step 4: Weight only
    weight_only_df = df[df['part_id'].isin(weight_matched_parts)]
    
    # Helper function to apply minimum runs filter
    def apply_min_runs_filter(pool_df, min_runs):
        """Filter pool to only include parts with >= min_runs records."""
        if len(pool_df) == 0:
            return pool_df, [], []
        
        part_run_counts = pool_df.groupby('part_id').size().reset_index(name='runs')
        qualifying = part_run_counts[part_run_counts['runs'] >= min_runs]['part_id'].tolist()
        excluded = part_run_counts[part_run_counts['runs'] < min_runs]['part_id'].tolist()
        
        filtered_df = pool_df[pool_df['part_id'].isin(qualifying)]
        return filtered_df, qualifying, excluded
    
    # Apply minimum runs filter to each pooling method
    exact_filtered_df, exact_qualifying, exact_excluded = apply_min_runs_filter(exact_pooled_df, min_runs_per_part)
    any_filtered_df, any_qualifying, any_excluded = apply_min_runs_filter(any_pooled_df, min_runs_per_part)
    weight_filtered_df, weight_qualifying, weight_excluded = apply_min_runs_filter(weight_only_df, min_runs_per_part)
    
    # Select best pooling method (now using filtered counts)
    exact_filtered_n = len(exact_filtered_df)
    any_filtered_n = len(any_filtered_df)
    weight_filtered_n = len(weight_filtered_df)
    
    if exact_filtered_n >= thresholds['HIGH']:
        final_df = exact_filtered_df
        final_parts = exact_qualifying
        excluded_parts = exact_excluded
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_filtered_n >= thresholds['HIGH']:
        final_df = any_filtered_df
        final_parts = any_qualifying
        excluded_parts = any_excluded
        pooling_method = 'Weight ±10% + Any Defect'
    elif exact_filtered_n >= thresholds['MODERATE']:
        final_df = exact_filtered_df
        final_parts = exact_qualifying
        excluded_parts = exact_excluded
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_filtered_n >= thresholds['MODERATE']:
        final_df = any_filtered_df
        final_parts = any_qualifying
        excluded_parts = any_excluded
        pooling_method = 'Weight ±10% + Any Defect'
    elif exact_filtered_n >= thresholds['LOW']:
        final_df = exact_filtered_df
        final_parts = exact_qualifying
        excluded_parts = exact_excluded
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_filtered_n >= thresholds['LOW']:
        final_df = any_filtered_df
        final_parts = any_qualifying
        excluded_parts = any_excluded
        pooling_method = 'Weight ±10% + Any Defect'
    elif weight_filtered_n >= thresholds['LOW']:
        final_df = weight_filtered_df
        final_parts = weight_qualifying
        excluded_parts = weight_excluded
        pooling_method = 'Weight ±10% Only'
    else:
        # Insufficient data even with pooling — dual display still shows part-level
        result['pooled_comparison'] = None
        return result
    
    # Compute metrics from filtered pooled data
    pooled_n = len(final_df)
    confidence = get_confidence_tier(pooled_n)
    
    # Calculate total parts produced from pooled data
    pooled_total_parts = final_df['order_quantity'].sum() if 'order_quantity' in final_df.columns else pooled_n
    
    # STRATEGY C: Use pooled threshold (avg + std) instead of global threshold
    if use_pooled_threshold and pooled_n > 1:
        pooled_avg_scrap = final_df['scrap_percent'].mean()
        pooled_std_scrap = final_df['scrap_percent'].std()
        effective_threshold = pooled_avg_scrap + pooled_std_scrap
        threshold_source = 'pooled'
    else:
        pooled_avg_scrap = final_df['scrap_percent'].mean()
        pooled_std_scrap = final_df['scrap_percent'].std() if pooled_n > 1 else 0
        effective_threshold = threshold_pct
        threshold_source = 'global'
    
    # Calculate failures using effective threshold
    failures = (final_df['scrap_percent'] > effective_threshold).sum()
    failure_rate = failures / pooled_n if pooled_n > 0 else 0
    
    # MTTS (runs) = Total Runs / Failures
    if failures > 0:
        mtts_runs = pooled_n / failures
    else:
        mtts_runs = pooled_n * 2  # Conservative multiplier when no failures (was 10)
    
    # MTTS (parts) = Total Parts Produced / Failures
    if failures > 0:
        mtts_parts = pooled_total_parts / failures
    else:
        mtts_parts = pooled_total_parts * 2  # Conservative multiplier when no failures (was 10)
    
    reliability = np.exp(-1 / mtts_runs) if mtts_runs > 0 else 0
    
    # Build excluded parts info for disclaimer
    excluded_parts_info = []
    for exc_part in excluded_parts:
        exc_data = df[df['part_id'] == exc_part]
        if len(exc_data) > 0:
            excluded_parts_info.append({
                'part_id': exc_part,
                'runs': len(exc_data),
                'avg_scrap': exc_data['scrap_percent'].mean()
            })
    
    result['pooled_comparison'] = {
        'pooling_method': pooling_method,
        'weight_range': weight_range,
        'n_records': pooled_n,
        'n_parts': len(final_parts),
        'included_part_ids': final_parts,
        'excluded_part_ids': excluded_parts,
        'excluded_parts_info': excluded_parts_info,
        'min_runs_filter': min_runs_per_part,
        'confidence': get_confidence_tier(pooled_n),
        'mtts_runs': mtts_runs,
        'mtts_parts': mtts_parts,
        'total_parts_produced': pooled_total_parts,
        'reliability_next_run': reliability,
        'failure_count': failures,
        'failure_rate': failure_rate,
        'pooled_avg_scrap': pooled_avg_scrap,
        'pooled_std_scrap': pooled_std_scrap,
        'effective_threshold': effective_threshold,
        'threshold_source': threshold_source,
    }
    
    return result


# ================================================================
# 6-2-1 TEMPORAL SPLIT
# ================================================================
def time_split_621(df):
    """Split data temporally: 60% train, 20% calibration, 20% test."""
    df = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    
    return df.iloc[:train_end].copy(), df.iloc[train_end:calib_end].copy(), df.iloc[calib_end:].copy()


# ================================================================
# FEATURE ENGINEERING - MTBF ON TRAIN
# ================================================================
def compute_mtbf_on_train(df_train, threshold):
    """Compute MTTF proxy from training data only."""
    grp = df_train.groupby("part_id")["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= threshold, 1.0, grp["mttf_scrap"])
    return grp


def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    """Attach training-derived features to prevent leakage."""
    df_sub = df_sub.merge(mtbf_train, on="part_id", how="left")
    df_sub["mttf_scrap"] = df_sub["mttf_scrap"].fillna(default_mtbf)
    df_sub = df_sub.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    df_sub["part_freq"] = df_sub["part_freq"].fillna(default_freq)
    return df_sub


# ================================================================
# MAKE X, Y - MATCHING ENHANCED VERSION FEATURES
# ================================================================
def make_xy(df, threshold, defect_cols, use_multi_defect=True, use_temporal=True, use_mtts=True):
    """Prepare features matching enhanced version exactly."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    # Multi-defect features
    if use_multi_defect and MULTI_DEFECT_FEATURES_ENABLED:
        multi_feats = ["n_defect_types", "has_multiple_defects", "total_defect_rate",
                       "max_defect_rate", "defect_concentration",
                       "shift_x_tearup", "shrink_x_porosity", "shrink_x_shrink_porosity", "core_x_sand"]
        for f in multi_feats:
            if f in df.columns:
                feats.append(f)
    
    # Temporal features
    if use_temporal and TEMPORAL_FEATURES_ENABLED:
        temporal_feats = ["total_defect_rate_trend", "total_defect_rate_roll3",
                         "scrap_percent_trend", "scrap_percent_roll3", "month", "quarter"]
        for f in temporal_feats:
            if f in df.columns:
                feats.append(f)
    
    # MTTS features
    if use_mtts and MTTS_FEATURES_ENABLED:
        mtts_feats = ["mtts_runs", "hazard_rate", "reliability_score",
                      "runs_since_last_failure", "cumulative_scrap_in_cycle",
                      "degradation_velocity", "degradation_acceleration",
                      "cycle_hazard_indicator", "rul_proxy"]
        for f in mtts_feats:
            if f in df.columns:
                feats.append(f)
    
    # Add defect rate columns
    feats += [c for c in defect_cols if c in df.columns]
    
    # ================================================================
    # THREE-STAGE INHERITED FEATURES (if present)
    # Only include numeric probability features, NOT categorical tiers
    # ================================================================
    stage_feats = ["global_scrap_probability", "defect_cluster_probability"]
    for f in stage_feats:
        if f in df.columns:
            feats.append(f)
    
    # NOTE: Excluding global_risk_tier and cluster_risk_tier as they are 
    # categorical strings and would cause "could not convert string to float" errors
    
    # Ensure all features exist
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    
    y = (df["scrap_percent"] > threshold).astype(int)
    X = df[feats].fillna(0).copy()
    
    return X, y, feats


# ================================================================
# THREE-STAGE HIERARCHICAL LEARNING FUNCTIONS
# ================================================================
# Research Support:
#   - Tercan et al. (2018): Multi-stage TL reduces data requirements by 64%
#   - Zhang et al. (2021): Hierarchical TL extracts common then specific features
#   - Zhang H.B. et al. (2023): Hierarchical adaptive RUL prediction
#   - Agarwal & Chowdary (2020): Stacked ensemble learning
# ================================================================

def train_stage1_foundry_wide(df, defect_cols):
    """
    STAGE 1: FOUNDRY-WIDE MODEL
    
    Trains on ALL data using GLOBAL threshold (dataset avg scrap %).
    Purpose: Learn patterns common across all parts.
    Output: Adds 'global_scrap_probability' feature to each record.
    
    Reference: "Extract common characteristics of manufacturing system"
               - Zhang et al. (2021)
    """
    global_threshold = df["scrap_percent"].mean()
    
    # Add features
    df_stage1 = df.copy()
    df_stage1 = add_multi_defect_features(df_stage1, defect_cols)
    df_stage1 = add_temporal_features(df_stage1)
    df_stage1 = add_mtts_features(df_stage1, global_threshold)
    
    # 6-2-1 split
    df_train, df_calib, df_test = time_split_621(df_stage1)
    
    # MTBF from training only
    mtbf_train = compute_mtbf_on_train(df_train, global_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # Prepare X, y
    X_train, y_train, feats = make_xy(df_train, global_threshold, defect_cols)
    X_calib, y_calib, _ = make_xy(df_calib, global_threshold, defect_cols)
    X_test, y_test, _ = make_xy(df_test, global_threshold, defect_cols)
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Calibrate
    pos, neg = int(y_calib.sum()), int((y_calib == 0).sum())
    if pos >= 3 and neg >= 3:
        try:
            cal_model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            cal_model.fit(X_calib, y_calib)
        except:
            cal_model = rf
    else:
        cal_model = rf
    
    # Evaluate on test set
    if len(X_test) > 0 and y_test.nunique() == 2:
        y_prob = cal_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob),
            "brier": brier_score_loss(y_test, y_prob),
        }
    else:
        metrics = {"recall": 0, "precision": 0, "auc": 0.5, "brier": 0.25}
    
    return {
        "model": cal_model,
        "rf": rf,
        "features": feats,
        "threshold": global_threshold,
        "metrics": metrics,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "mtbf_train": mtbf_train,
        "part_freq_train": part_freq_train,
        "default_mtbf": default_mtbf,
        "default_freq": default_freq,
    }


def train_stage2_defect_cluster(df, defect_cols, stage1_model):
    """
    STAGE 2: DEFECT-CLUSTER MODEL
    
    Trains on records with TOP 5 PARETO defects using cluster threshold.
    Purpose: Learn patterns specific to high-impact defects (~80% of scrap).
    Output: Adds 'defect_cluster_probability' feature to matching records.
    
    Reference: "Multi-stage TL approach provides better predictions"
               - Tercan et al. (2018)
    """
    # Filter to records with any top Pareto defect present
    top_defects = TOP_PARETO_DEFECTS
    
    # Create mask for records with any top defect > 0
    defect_mask = pd.Series([False] * len(df), index=df.index)
    for defect in top_defects:
        if defect in df.columns:
            defect_mask = defect_mask | (df[defect] > 0)
    
    df_cluster = df[defect_mask].copy()
    
    # If not enough data, use all data
    if len(df_cluster) < 50:
        df_cluster = df.copy()
    
    cluster_threshold = df_cluster["scrap_percent"].mean()
    
    # Add features
    df_cluster = add_multi_defect_features(df_cluster, defect_cols)
    df_cluster = add_temporal_features(df_cluster)
    df_cluster = add_mtts_features(df_cluster, cluster_threshold)
    
    # Add Stage 1 predictions as feature
    df_cluster = add_stage1_features(df_cluster, stage1_model, defect_cols)
    
    # 6-2-1 split
    df_train, df_calib, df_test = time_split_621(df_cluster)
    
    # MTBF
    mtbf_train = compute_mtbf_on_train(df_train, cluster_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # Prepare X, y
    X_train, y_train, feats = make_xy(df_train, cluster_threshold, defect_cols)
    X_calib, y_calib, _ = make_xy(df_calib, cluster_threshold, defect_cols)
    X_test, y_test, _ = make_xy(df_test, cluster_threshold, defect_cols)
    
    # Train
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Calibrate
    pos, neg = int(y_calib.sum()), int((y_calib == 0).sum())
    if pos >= 3 and neg >= 3:
        try:
            cal_model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            cal_model.fit(X_calib, y_calib)
        except:
            cal_model = rf
    else:
        cal_model = rf
    
    # Evaluate
    if len(X_test) > 0 and y_test.nunique() == 2:
        y_prob = cal_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob),
        }
    else:
        metrics = {"recall": 0, "precision": 0, "auc": 0.5}
    
    return {
        "model": cal_model,
        "rf": rf,
        "features": feats,
        "threshold": cluster_threshold,
        "metrics": metrics,
        "n_records": len(df_cluster),
        "top_defects": top_defects,
    }


def add_stage1_features(df, stage1_result, defect_cols):
    """Add Stage 1 global predictions as features."""
    df = df.copy()
    
    # Prepare features for prediction
    temp_df = df.copy()
    
    # Ensure required features exist
    for f in stage1_result["features"]:
        if f not in temp_df.columns:
            temp_df[f] = 0.0
    
    X = temp_df[stage1_result["features"]].fillna(0)
    
    try:
        # Get global scrap probability from Stage 1 model
        probs = stage1_result["model"].predict_proba(X)[:, 1]
        df["global_scrap_probability"] = probs
        
        # Create risk tier
        df["global_risk_tier"] = pd.cut(
            df["global_scrap_probability"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        ).astype(str)
    except Exception as e:
        df["global_scrap_probability"] = 0.5
        df["global_risk_tier"] = "Medium"
    
    return df


def add_stage2_features(df, stage2_result, defect_cols):
    """Add Stage 2 defect cluster predictions as features."""
    df = df.copy()
    
    # Prepare features for prediction
    temp_df = df.copy()
    
    for f in stage2_result["features"]:
        if f not in temp_df.columns:
            temp_df[f] = 0.0
    
    X = temp_df[stage2_result["features"]].fillna(0)
    
    try:
        probs = stage2_result["model"].predict_proba(X)[:, 1]
        df["defect_cluster_probability"] = probs
        
        df["cluster_risk_tier"] = pd.cut(
            df["defect_cluster_probability"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        ).astype(str)
    except:
        df["defect_cluster_probability"] = 0.5
        df["cluster_risk_tier"] = "Medium"
    
    return df


def train_three_stage_model(df, defect_cols):
    """
    THREE-STAGE HIERARCHICAL TRAINING
    
    Stage 1: Foundry-Wide (global threshold)
    Stage 2: Defect-Cluster (top 5 Pareto defects)
    Stage 3: Combined model with inherited features
    
    Returns model with all stages integrated.
    """
    # ================================================================
    # STAGE 1: FOUNDRY-WIDE
    # ================================================================
    stage1_result = train_stage1_foundry_wide(df, defect_cols)
    
    # ================================================================
    # STAGE 2: DEFECT-CLUSTER (TOP 5 PARETO)
    # ================================================================
    stage2_result = train_stage2_defect_cluster(df, defect_cols, stage1_result)
    
    # ================================================================
    # STAGE 3: FINAL MODEL WITH INHERITED FEATURES
    # ================================================================
    # Add Stage 1 and Stage 2 features to full dataset
    df_enhanced = df.copy()
    df_enhanced = add_multi_defect_features(df_enhanced, defect_cols)
    df_enhanced = add_temporal_features(df_enhanced)
    
    # Use GLOBAL threshold for MTTS (Stage 1 alignment)
    global_threshold = df["scrap_percent"].mean()
    df_enhanced = add_mtts_features(df_enhanced, global_threshold)
    
    # Add inherited features from Stage 1 and Stage 2
    df_enhanced = add_stage1_features(df_enhanced, stage1_result, defect_cols)
    df_enhanced = add_stage2_features(df_enhanced, stage2_result, defect_cols)
    
    # 6-2-1 split
    df_train, df_calib, df_test = time_split_621(df_enhanced)
    
    # MTBF
    mtbf_train = compute_mtbf_on_train(df_train, global_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # Prepare X, y with GLOBAL threshold for final evaluation
    X_train, y_train, feats = make_xy(df_train, global_threshold, defect_cols)
    X_calib, y_calib, _ = make_xy(df_calib, global_threshold, defect_cols)
    X_test, y_test, _ = make_xy(df_test, global_threshold, defect_cols)
    
    # Train final Stage 3 model
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Calibrate
    pos, neg = int(y_calib.sum()), int((y_calib == 0).sum())
    if pos >= 3 and neg >= 3:
        try:
            cal_model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            cal_model.fit(X_calib, y_calib)
            calibration_method = "calibrated (sigmoid, cv=3)"
        except:
            cal_model = rf
            calibration_method = "uncalibrated"
    else:
        cal_model = rf
        calibration_method = "uncalibrated"
    
    # Final evaluation metrics
    if len(X_test) > 0 and y_test.nunique() == 2:
        y_prob = cal_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "brier": brier_score_loss(y_test, y_prob),
            "y_test": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics["roc_fpr"], metrics["roc_tpr"] = fpr, tpr
        
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            metrics["cal_true"], metrics["cal_pred"] = prob_true, prob_pred
        except:
            metrics["cal_true"], metrics["cal_pred"] = [0, 1], [0, 1]
    else:
        metrics = {"recall": 0, "precision": 0, "f1": 0, "accuracy": 0, "auc": 0.5, "brier": 0.25}
    
    return {
        "rf": rf,
        "cal_model": cal_model,
        "calibration_method": calibration_method,
        "features": feats,
        "df_train": df_train,
        "df_calib": df_calib,
        "df_test": df_test,
        "df_enhanced": df_enhanced,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "mtbf_train": mtbf_train,
        "part_freq_train": part_freq_train,
        "default_mtbf": default_mtbf,
        "default_freq": default_freq,
        "metrics": metrics,
        "n_train": len(df_train),
        "n_calib": len(df_calib),
        "n_test": len(df_test),
        "global_threshold": global_threshold,
        # Stage results for transparency
        "stage1": stage1_result,
        "stage2": stage2_result,
        "three_stage_enabled": True,
    }


# ================================================================
# GLOBAL MODEL TRAINING - MATCHING ENHANCED VERSION
# ================================================================
# SCIKIT-LEARN APPLICATION: MODEL TRAINING & CALIBRATION
# ================================================================
# This function uses Scikit-learn to:
#   1. Train a RandomForestClassifier (180 trees, balanced class weights)
#   2. Calibrate probabilities using CalibratedClassifierCV (Platt scaling)
#   3. Evaluate performance using recall_score, precision_score, etc.
#
# The trained model is then used in Tab 1 for predictions and
# Tabs 2, 3, 5 for validation metrics.
# ================================================================
def train_global_model(df, threshold, defect_cols):
    """
    Train global model with ALL enhanced features.
    
    SCIKIT-LEARN FUNCTIONS USED:
    - RandomForestClassifier(): Creates ensemble of 180 decision trees
    - CalibratedClassifierCV(): Calibrates probabilities using Platt scaling
    - recall_score(): Calculates True Positives / All Actual Positives
    - precision_score(): Calculates True Positives / All Predicted Positives
    - roc_auc_score(): Calculates Area Under ROC Curve
    - brier_score_loss(): Measures probability calibration quality
    - roc_curve(): Generates FPR, TPR points for ROC plot
    - calibration_curve(): Generates points for calibration plot
    """
    
    # Add ALL enhanced features BEFORE split
    df = add_multi_defect_features(df, defect_cols)
    df = add_temporal_features(df)
    df = add_mtts_features(df, threshold)
    
    # 6-2-1 temporal split
    df_train, df_calib, df_test = time_split_621(df)
    
    # MTBF from training only
    mtbf_train = compute_mtbf_on_train(df_train, threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # Prepare X, y with ALL features
    X_train, y_train, feats = make_xy(df_train, threshold, defect_cols)
    X_calib, y_calib, _ = make_xy(df_calib, threshold, defect_cols)
    X_test, y_test, _ = make_xy(df_test, threshold, defect_cols)
    
    # ================================================================
    # SCIKIT-LEARN: TRAIN RANDOM FOREST CLASSIFIER
    # ================================================================
    # Reference: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    # 
    # Parameters:
    #   n_estimators=180: Number of trees in the forest
    #   min_samples_leaf=5: Minimum samples required at leaf node
    #   class_weight="balanced": Adjusts weights inversely proportional to class frequencies
    #   random_state=42: Ensures reproducibility
    # ================================================================
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)  # SCIKIT-LEARN: Trains the model on training data
    
    # ================================================================
    # SCIKIT-LEARN: PROBABILITY CALIBRATION (PLATT SCALING)
    # ================================================================
    # Reference: Platt, J. (1999). Probabilistic outputs for support vector machines.
    #
    # CalibratedClassifierCV fits a sigmoid function to map raw scores to
    # well-calibrated probabilities. This ensures that when the model predicts
    # 70% probability, approximately 70% of those cases are actually positive.
    # ================================================================
    pos, neg = int(y_calib.sum()), int((y_calib == 0).sum())
    
    if pos >= 3 and neg >= 3:
        try:
            cal_model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            cal_model.fit(X_calib, y_calib)
            calibration_method = "calibrated (sigmoid, cv=3)"
        except:
            cal_model = rf
            calibration_method = "uncalibrated"
    else:
        cal_model = rf
        calibration_method = "uncalibrated"
    
    # ================================================================
    # SCIKIT-LEARN: EVALUATION METRICS ON TEST SET
    # ================================================================
    # These metrics are displayed in Tabs 2, 3, and 5
    # All functions come from sklearn.metrics module
    # ================================================================
    if len(X_test) > 0 and y_test.nunique() == 2:
        # SCIKIT-LEARN: Generate probability predictions
        y_prob = cal_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (scrap)
        y_pred = (y_prob >= 0.5).astype(int)  # Convert to binary prediction
        
        # ================================================================
        # SCIKIT-LEARN METRICS EXPLAINED:
        # ================================================================
        # recall_score: Of all actual failures, what % did we catch?
        #               Formula: TP / (TP + FN) = 98.6%
        #
        # precision_score: Of all predicted failures, what % were correct?
        #                  Formula: TP / (TP + FP) = 97.2%
        #
        # roc_auc_score: Probability that a randomly chosen positive ranks
        #                higher than a randomly chosen negative = 0.999
        #
        # brier_score_loss: Mean squared error of probability predictions
        #                   Range: 0 (perfect) to 1 (worst) = 0.012
        # ================================================================
        metrics = {
            "recall": recall_score(y_test, y_pred, zero_division=0),      # Used in Tab 2, 3, 5
            "precision": precision_score(y_test, y_pred, zero_division=0), # Used in Tab 2, 5
            "f1": f1_score(y_test, y_pred, zero_division=0),              # F1 score
            "accuracy": accuracy_score(y_test, y_pred),                    # Overall accuracy
            "auc": roc_auc_score(y_test, y_prob),                          # Used in Tab 2, 5
            "brier": brier_score_loss(y_test, y_prob),                     # Used in Tab 2
            "y_test": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred
        }
        
        # SCIKIT-LEARN: Generate ROC curve points for Tab 2 visualization
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics["roc_fpr"], metrics["roc_tpr"] = fpr, tpr
        
        # SCIKIT-LEARN: Generate calibration curve points for Tab 2 visualization
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            metrics["cal_true"], metrics["cal_pred"] = prob_true, prob_pred
        except:
            metrics["cal_true"], metrics["cal_pred"] = [0, 1], [0, 1]
    else:
        metrics = {"recall": 0, "precision": 0, "f1": 0, "accuracy": 0, "auc": 0.5, "brier": 0.25}
    
    return {
        "rf": rf, "cal_model": cal_model, "calibration_method": calibration_method,
        "features": feats, "df_train": df_train, "df_calib": df_calib, "df_test": df_test,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "mtbf_train": mtbf_train, "part_freq_train": part_freq_train,
        "default_mtbf": default_mtbf, "default_freq": default_freq,
        "metrics": metrics, "n_train": len(df_train), "n_calib": len(df_calib), "n_test": len(df_test)
    }


# ================================================================
# PART STATS AND DIAGNOSIS
# ================================================================
def get_part_stats(df, part_id):
    part_data = df[df["part_id"] == str(part_id)]
    if len(part_data) == 0:
        return None
    return {
        "n_records": len(part_data),
        "avg_scrap": part_data["scrap_percent"].mean(),
        "max_scrap": part_data["scrap_percent"].max(),
        "total_parts": part_data["order_quantity"].sum(),
        "avg_order_qty": part_data["order_quantity"].mean(),
        "piece_weight": part_data["piece_weight_lbs"].mode().iloc[0] if len(part_data["piece_weight_lbs"].mode()) > 0 else part_data["piece_weight_lbs"].median()
    }


def diagnose_processes(df, part_id, defect_cols):
    part_data = df[df["part_id"] == str(part_id)]
    if len(part_data) == 0:
        return None, None
    
    defect_rates = {col: part_data[col].mean() for col in defect_cols if col in part_data.columns}
    
    process_scores = {}
    for process, info in PROCESS_DEFECT_MAP.items():
        score = sum(defect_rates.get(d, 0) for d in info["defects"])
        process_scores[process] = score
    
    total = sum(process_scores.values())
    if total > 0:
        process_contributions = {p: (s / total) * 100 for p, s in process_scores.items()}
    else:
        process_contributions = {p: 0 for p in process_scores}
    
    sorted_processes = sorted(process_contributions.items(), key=lambda x: x[1], reverse=True)
    sorted_defects = sorted(defect_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return sorted_processes, sorted_defects


# ================================================================
# LIME LOCAL EXPLANATION FUNCTION
# ================================================================
# Reference: Ribeiro, M.T., Singh, S., & Guestrin, C. (2016).
# "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
#
# This function provides instance-level explanations for individual
# predictions, answering "WHY did the model predict this specific value?"
# ================================================================
def explain_prediction_lime(model, X_train, feature_names, instance, num_features=10):
    """
    Generate LIME explanation for a single prediction instance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict_proba method
    X_train : np.array or pd.DataFrame
        Training data used to fit LIME explainer
    feature_names : list
        Names of features
    instance : np.array or pd.DataFrame
        Single instance to explain (1D array or single-row DataFrame)
    num_features : int
        Number of top features to return
    
    Returns:
    --------
    dict with:
        - 'explanation': list of (feature, weight) tuples
        - 'prediction': model's prediction for this instance
        - 'prediction_proba': probability prediction
        - 'intercept': base value from local linear model
        - 'error': error message if any, None otherwise
    """
    if not LIME_AVAILABLE:
        return {
            'explanation': [],
            'prediction': None,
            'prediction_proba': None,
            'intercept': 0,
            'error': 'LIME not installed. Run: pip install lime'
        }
    
    try:
        # Convert X_train to numpy array
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values.astype(np.float64)
        else:
            X_train_array = np.array(X_train).astype(np.float64)
        
        # Handle NaN values in training data
        X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate training data
        if X_train_array.shape[0] < 10:
            return {
                'explanation': [],
                'prediction': None,
                'prediction_proba': None,
                'intercept': 0,
                'error': f'Insufficient training data: {X_train_array.shape[0]} samples (need at least 10)'
            }
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=X_train_array,
            feature_names=list(feature_names),
            class_names=['Good', 'Scrap'],
            mode='classification',
            discretize_continuous=True,
            random_state=42,
            verbose=False
        )
        
        # Ensure instance is 1D numpy array
        if hasattr(instance, 'values'):
            instance_array = instance.values.flatten().astype(np.float64)
        elif hasattr(instance, 'iloc'):
            instance_array = instance.iloc[0].values.astype(np.float64) if len(instance.shape) > 1 else instance.values.astype(np.float64)
        else:
            instance_array = np.array(instance).flatten().astype(np.float64)
        
        # Handle NaN values in instance
        instance_array = np.nan_to_num(instance_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate instance shape
        if len(instance_array) != X_train_array.shape[1]:
            return {
                'explanation': [],
                'prediction': None,
                'prediction_proba': None,
                'intercept': 0,
                'error': f'Feature mismatch: instance has {len(instance_array)} features, expected {X_train_array.shape[1]}'
            }
        
        # Generate explanation
        exp = explainer.explain_instance(
            data_row=instance_array,
            predict_fn=model.predict_proba,
            num_features=min(num_features, len(feature_names)),
            num_samples=500  # Reduced for faster computation
        )
        
        # Get model prediction for this instance
        proba = model.predict_proba(instance_array.reshape(1, -1))[0]
        pred_class = int(np.argmax(proba))
        
        # Extract explanation as list of (feature, weight) tuples
        explanation_list = exp.as_list()
        
        # Get intercept (base prediction from local model)
        # NOTE: exp.intercept is a DICTIONARY keyed by class label, not a list/array
        # For binary classification explaining class 1 (Scrap), structure is {1: value}
        intercept = 0.0
        if hasattr(exp, 'intercept') and exp.intercept:
            if isinstance(exp.intercept, dict):
                # LIME returns intercept as dict keyed by class label
                # Prefer class 1 (Scrap) intercept, fall back to any available
                if 1 in exp.intercept:
                    intercept = float(exp.intercept[1])
                elif exp.intercept:
                    # Use first available value if class 1 not present
                    intercept = float(list(exp.intercept.values())[0])
            else:
                # Fallback for unexpected format (array-like)
                try:
                    intercept = float(exp.intercept[1]) if len(exp.intercept) > 1 else float(exp.intercept[0])
                except (IndexError, KeyError, TypeError):
                    intercept = 0.0
        
        # Get local prediction value safely
        local_pred_value = None
        if hasattr(exp, 'local_pred'):
            try:
                if isinstance(exp.local_pred, dict):
                    # Handle dict format (keyed by class label)
                    local_pred_value = float(exp.local_pred.get(1, list(exp.local_pred.values())[0]))
                elif hasattr(exp.local_pred, '__len__') and len(exp.local_pred) > 0:
                    local_pred_value = float(exp.local_pred[0])
                else:
                    local_pred_value = float(exp.local_pred)
            except (IndexError, KeyError, TypeError, ValueError):
                local_pred_value = None
        
        return {
            'explanation': explanation_list,
            'prediction': pred_class,
            'prediction_proba': float(proba[1]),  # Probability of scrap (class 1)
            'intercept': intercept,
            'local_pred': local_pred_value,
            'error': None
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        # Get more detailed error info
        tb = traceback.format_exc()
        return {
            'explanation': [],
            'prediction': None,
            'prediction_proba': None,
            'intercept': 0,
            'error': error_msg,
            'traceback': tb
        }


def format_lime_feature(feature_str):
    """
    Format LIME feature string for display.
    LIME returns strings like "hazard_rate <= 0.15" or "shrink_rate > 2.30"
    """
    # Clean up feature names for display
    feature_str = feature_str.replace('_rate', '')
    feature_str = feature_str.replace('_', ' ')
    return feature_str.title()


def get_lime_color(weight):
    """Return color based on weight direction."""
    if weight > 0:
        return '#FFCDD2'  # Light red - increases scrap risk
    else:
        return '#C8E6C9'  # Light green - decreases scrap risk


def calculate_tte_savings(current_scrap, target_scrap, annual_production, energy_per_lb=DOE_BENCHMARKS['average']):
    reduction_pct = (current_scrap - target_scrap) / current_scrap if current_scrap > 0 else 0
    avoided_lbs = annual_production * (current_scrap - target_scrap) / 100
    tte_mmbtu = avoided_lbs * energy_per_lb / 1_000_000
    co2_tons = tte_mmbtu * CO2_PER_MMBTU / 1000
    return {"scrap_reduction_pct": reduction_pct, "avoided_scrap_lbs": avoided_lbs, "tte_savings_mmbtu": tte_mmbtu, "co2_savings_tons": co2_tons}


# ================================================================
# ACTION PLAN REPORT GENERATOR
# ================================================================
def generate_action_plan_report(
    part_id,
    current_state,
    target_state,
    defect_targets,
    lime_insights=None,
    timestamp=None
):
    """
    Generate a printable action plan report for foundry managers.
    
    Parameters:
    -----------
    part_id : str/int
        The part identifier
    current_state : dict
        Current metrics including threshold, reliability, mtts, failures, scrap_rate
    target_state : dict
        Target metrics including threshold, reliability, mtts, failures
    defect_targets : list of dict
        List of defects with current_rate, target_rate, process, estimated_reduction
    lime_insights : list of dict, optional
        LIME explanation features with feature, weight, direction
    timestamp : datetime, optional
        Report generation timestamp
    
    Returns:
    --------
    str : Markdown-formatted report
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Determine overall status
    if target_state['reliability'] >= 90:
        status = "🏆 WORLD-CLASS TARGET"
        status_desc = "Targeting elite manufacturing performance"
    elif target_state['reliability'] >= 80:
        status = "✅ ACCEPTABLE TARGET"
        status_desc = "Targeting industry-standard reliability"
    elif target_state['reliability'] >= 70:
        status = "⚠️ IMPROVEMENT TARGET"
        status_desc = "Working toward acceptable performance"
    else:
        status = "🔴 CRITICAL INTERVENTION"
        status_desc = "Immediate action required"
    
    # Build the report
    report = f"""
# 🏭 FOUNDRY SCRAP REDUCTION ACTION PLAN
## Part: {part_id}
**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M')}

---

## 📊 EXECUTIVE SUMMARY

| Metric | Current State | Target State | Change |
|--------|---------------|--------------|--------|
| **Scrap Threshold** | {current_state['threshold']:.2f}% | {target_state['threshold']:.2f}% | {target_state['threshold'] - current_state['threshold']:+.2f}% |
| **Reliability R(n)** | {current_state['reliability']:.1f}% | {target_state['reliability']:.1f}% | {target_state['reliability'] - current_state['reliability']:+.1f}% |
| **MTTS (parts)** | {current_state['mtts']:,.0f} | {target_state['mtts']:,.0f} | {target_state['mtts'] - current_state['mtts']:+,.0f} |
| **Failure Events** | {current_state['failures']:.0f} | {target_state['failures']:.0f} | {target_state['failures'] - current_state['failures']:+.0f} |

**Status:** {status}
*{status_desc}*

---

## 🎯 DEFECT REDUCTION TARGETS

| Priority | Defect | Process Area | Current Rate | Target Rate | Reduction | Est. Failures Avoided |
|----------|--------|--------------|--------------|-------------|-----------|----------------------|
"""
    
    # Add defect rows
    for i, defect in enumerate(defect_targets):
        priority = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        reduction_pct = ((defect['current_rate'] - defect['target_rate']) / defect['current_rate'] * 100) if defect['current_rate'] > 0 else 0
        report += f"| {priority} | {defect['defect_name']} | {defect['process']} | {defect['current_rate']:.2f}% | {defect['target_rate']:.2f}% | {reduction_pct:.0f}% | {defect['estimated_reduction']:.1f} |\n"
    
    report += """
---

## 🔧 ACTION ITEMS BY PROCESS AREA

"""
    
    # Group defects by process area
    process_actions = {}
    for defect in defect_targets:
        proc = defect['process']
        if proc not in process_actions:
            process_actions[proc] = []
        if defect['target_rate'] < defect['current_rate']:
            process_actions[proc].append(defect)
    
    # Process-specific recommendations based on Campbell's framework
    process_recommendations = {
        "Melting": [
            "Review melt temperature control (±10°F tolerance)",
            "Check degassing time and procedure",
            "Verify hydrogen content testing frequency",
            "Inspect furnace refractory condition"
        ],
        "Pouring": [
            "Calibrate pour temperature measurement",
            "Review pour rate and stream continuity",
            "Check ladle condition and preheating",
            "Verify mold fill time targets"
        ],
        "Gating Design": [
            "Review riser sizing calculations",
            "Check feeding distance requirements",
            "Verify chilling placement",
            "Run solidification simulation"
        ],
        "Sand System": [
            "Test sand AFS grain fineness",
            "Check binder ratio and mixing time",
            "Verify compaction/ramming pressure",
            "Review sand temperature control"
        ],
        "Core Making": [
            "Inspect core box venting",
            "Check core strength (dog bone test)",
            "Verify core coating application",
            "Review core storage conditions"
        ],
        "Shakeout": [
            "Review cooling time before shakeout",
            "Check shakeout equipment settings",
            "Verify handling procedures",
            "Inspect for thermal stress"
        ],
        "Pattern/Tooling": [
            "Measure pattern dimensional accuracy",
            "Check pattern wear and surface finish",
            "Verify draft angles",
            "Inspect loose piece fit"
        ],
        "Inspection": [
            "Review NDT procedure compliance",
            "Calibrate inspection equipment",
            "Verify acceptance criteria documentation",
            "Check inspector training records"
        ],
        "Finishing": [
            "Review grinding templates/fixtures",
            "Check grinder wheel specifications",
            "Verify dimensional inspection after grinding",
            "Document acceptable surface finish standards"
        ]
    }
    
    for proc, defects in process_actions.items():
        if defects:
            report += f"### {proc}\n\n"
            report += f"**Targeted Defects:** {', '.join([d['defect_name'] for d in defects])}\n\n"
            report += "**Recommended Actions:**\n"
            
            recommendations = process_recommendations.get(proc, ["Review process parameters", "Consult process engineer"])
            for rec in recommendations[:4]:  # Limit to top 4 recommendations
                report += f"- [ ] {rec}\n"
            report += "\n"
    
    # Add LIME insights if available
    if lime_insights and len(lime_insights) > 0:
        report += """---

## 🔍 ML MODEL INSIGHTS (LIME Analysis)

**Why the model predicts scrap risk for this part:**

| Feature Condition | Impact | Direction | Interpretation |
|-------------------|--------|-----------|----------------|
"""
        for insight in lime_insights[:8]:  # Top 8 features
            direction = "↑ Increases Risk" if insight['weight'] > 0 else "↓ Decreases Risk"
            impact = "HIGH" if abs(insight['weight']) > 0.1 else "MEDIUM" if abs(insight['weight']) > 0.05 else "LOW"
            report += f"| {insight['feature']} | {impact} | {direction} | Weight: {insight['weight']:+.3f} |\n"
        
        report += """
**Key Insight:** Features with positive weights (↑) are pushing predictions toward scrap. 
Focus improvement efforts on controlling these factors.
"""
    
    report += f"""
---

## 📋 IMPLEMENTATION CHECKLIST

### Immediate Actions (This Week)
- [ ] Review this plan with shift supervisors
- [ ] Assign process owners to each action item
- [ ] Establish baseline measurements for target defects
- [ ] Schedule follow-up review meeting

### Short-Term Actions (This Month)
- [ ] Implement process adjustments for top 2 defects
- [ ] Begin data collection on adjusted parameters
- [ ] Compare actual vs. predicted defect rates

### Verification (Next Month)
- [ ] Calculate new reliability metrics
- [ ] Compare to targets in this plan
- [ ] Document lessons learned
- [ ] Generate updated action plan if needed

---

## 📈 SUCCESS CRITERIA

This plan is successful when:
1. **Reliability reaches {target_state['reliability']:.0f}%** (from {current_state['reliability']:.0f}%)
2. **Failures reduce to {target_state['failures']:.0f}** (from {current_state['failures']:.0f})
3. **Each targeted defect meets its rate target** (see table above)

---

*Generated by Foundry Prognostic Reliability Dashboard*
*Reference: Campbell, J. (2015). Complete Casting Handbook. Butterworth-Heinemann.*
"""
    
    return report


def generate_scenario_comparison_report(part_id, scenarios, timestamp=None):
    """
    Generate a comparison report for multiple threshold scenarios.
    
    Parameters:
    -----------
    part_id : str/int
        The part identifier
    scenarios : list of dict
        Each scenario contains: name, threshold, reliability, mtts, failures, scrap_risk
    timestamp : datetime, optional
        Report generation timestamp
    
    Returns:
    --------
    str : Markdown-formatted comparison report
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    report = f"""
# 🏭 THRESHOLD SENSITIVITY ANALYSIS
## Part: {part_id}
**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M')}

---

## 📊 SCENARIO COMPARISON

| Scenario | Threshold | Scrap Risk | Reliability | MTTS (parts) | Failures | Assessment |
|----------|-----------|------------|-------------|--------------|----------|------------|
"""
    
    for scenario in scenarios:
        if scenario['reliability'] >= 80:
            assessment = "✅ Acceptable"
        elif scenario['reliability'] >= 70:
            assessment = "⚠️ Warning"
        else:
            assessment = "🔴 Critical"
        
        report += f"| {scenario['name']} | {scenario['threshold']:.2f}% | {scenario['scrap_risk']:.1f}% | {scenario['reliability']:.1f}% | {scenario['mtts']:,.0f} | {scenario['failures']:.0f} | {assessment} |\n"
    
    report += """
---

## 🎯 INTERPRETATION GUIDE

**Understanding the Trade-offs:**

- **Stricter Threshold (Lower %)**: More runs classified as "failures" → Lower MTTS → Lower Reliability
  - *Use for:* Safety-critical parts, high-value castings, strict customer requirements
  - *Trade-off:* Requires more aggressive process improvement to achieve targets

- **Lenient Threshold (Higher %)**: Fewer runs classified as "failures" → Higher MTTS → Higher Reliability  
  - *Use for:* General production, cost-sensitive orders, parts with inherent variability
  - *Trade-off:* May mask opportunities for improvement

---

## 📋 RECOMMENDATION

"""
    
    # Find the scenario closest to 80% reliability
    best_scenario = min(scenarios, key=lambda x: abs(x['reliability'] - 80))
    
    report += f"""
**Recommended Operating Point:** {best_scenario['name']} ({best_scenario['threshold']:.2f}% threshold)

This threshold achieves {best_scenario['reliability']:.1f}% reliability, which is closest to the 
industry-standard 80% target while being practically achievable.

**Next Steps:**
1. Adopt {best_scenario['threshold']:.2f}% as the working threshold for this part
2. Use the Reliability Improvement Planner to identify defect reduction targets
3. Generate an Action Plan with specific process improvements

---

*Generated by Foundry Prognostic Reliability Dashboard*
"""
    
    return report


# ================================================================
# MAIN APPLICATION
# ================================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Foundry Prognostic Reliability Dashboard</h1>
        <p>Three-Stage Hierarchical Learning | MTTS-Integrated ML | DOE-Aligned Impact Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    data_path = st.text_input("Data File Path", value=DEFAULT_CSV_PATH)
    
    result = load_data(data_path)
    if result is None or result[0] is None:
        st.error(f"❌ Could not load data from: {data_path}")
        return
    
    df, defect_cols = result
    st.success(f"✅ Loaded {len(df):,} records | {df['part_id'].nunique()} parts | {len(defect_cols)} defect types")
    
    threshold = df["scrap_percent"].mean()
    
    # ================================================================
    # THREE-STAGE HIERARCHICAL TRAINING
    # ================================================================
    st.info(f"""📊 **Three-Stage Hierarchical Learning Mode**
    - **Stage 1**: Foundry-Wide patterns (threshold: {threshold:.2f}%)
    - **Stage 2**: Top 5 Pareto defects (Sand, Shift, Misrun, Gouged, Dross)
    - **Stage 3**: Final model with inherited features
    """)
    
    with st.spinner("Training Three-Stage Hierarchical Model..."):
        global_model = train_three_stage_model(df, defect_cols)
    
    # Display stage results
    stage1_metrics = global_model['stage1']['metrics']
    stage2_metrics = global_model['stage2']['metrics']
    final_metrics = global_model['metrics']
    
    st.success(f"""✅ **Three-Stage Training Complete** ({global_model['calibration_method']})
    | Stage | Focus | Recall | Precision | AUC |
    |-------|-------|--------|-----------|-----|
    | Stage 1 | Foundry-Wide | {stage1_metrics['recall']*100:.1f}% | {stage1_metrics['precision']*100:.1f}% | {stage1_metrics['auc']:.3f} |
    | Stage 2 | Top 5 Defects | {stage2_metrics['recall']*100:.1f}% | {stage2_metrics['precision']*100:.1f}% | {stage2_metrics['auc']:.3f} |
    | **Final** | **Combined** | **{final_metrics['recall']*100:.1f}%** | **{final_metrics['precision']*100:.1f}%** | **{final_metrics['auc']:.3f}** |
    
    Features: {len(global_model['features'])} (including inherited: global_scrap_probability, defect_cluster_probability)
    """)
    
    # ================================================================
    # VITAL FEW — "MOST WANTED" PARTS
    # ================================================================
    # Compute vital few: top scrap-producing parts sorted by total scrap weight
    foundry_avg_scrap = df['scrap_percent'].mean()
    
    # Find the scrap weight column (handles parentheses variants)
    scrap_weight_col = None
    for col_name in ['total_scrap_weight_lbs', 'total_scrap_weight_(lbs)']:
        if col_name in df.columns:
            scrap_weight_col = col_name
            break
    
    # Find pieces scrapped column
    pieces_scrapped_col = 'pieces_scrapped' if 'pieces_scrapped' in df.columns else None
    
    if scrap_weight_col and pieces_scrapped_col:
        vital_few_parts = df.groupby('part_id').agg(
            runs=('scrap_percent', 'count'),
            avg_scrap=('scrap_percent', 'mean'),
            total_scrapped=(pieces_scrapped_col, 'sum'),
            total_scrap_weight=(scrap_weight_col, 'sum'),
            total_produced=('order_quantity', 'sum')
        ).reset_index()
    else:
        # Fallback: estimate weight from pieces_scrapped * piece_weight
        vital_few_parts = df.groupby('part_id').agg(
            runs=('scrap_percent', 'count'),
            avg_scrap=('scrap_percent', 'mean'),
            total_produced=('order_quantity', 'sum')
        ).reset_index()
        vital_few_parts['total_scrapped'] = 0
        vital_few_parts['total_scrap_weight'] = 0
    
    # Rank by total scrap weight descending
    vital_few_parts = vital_few_parts.sort_values('total_scrap_weight', ascending=False).reset_index(drop=True)
    vital_few_parts['cumul_weight'] = vital_few_parts['total_scrap_weight'].cumsum()
    total_scrap_weight_all = vital_few_parts['total_scrap_weight'].sum()
    vital_few_parts['cumul_pct'] = vital_few_parts['cumul_weight'] / total_scrap_weight_all * 100
    
    # Top 20 by scrap weight = "Most Wanted"
    most_wanted_df = vital_few_parts.head(20)
    most_wanted_ids = set(most_wanted_df['part_id'].values)
    
    # Pareto 80% threshold
    pareto_80_df = vital_few_parts[vital_few_parts['cumul_pct'] <= 80]
    n_pareto_80 = len(pareto_80_df) + 1  # +1 for the part that crosses 80%
    
    with st.expander(f"🎯 **VITAL FEW — Top 20 Most-Wanted Parts** (account for {most_wanted_df['total_scrap_weight'].sum()/total_scrap_weight_all*100:.0f}% of total scrap weight)", expanded=False):
        st.markdown(f"""
        **Pareto Analysis:** {n_pareto_80} parts ({n_pareto_80/len(vital_few_parts)*100:.0f}% of all parts) 
        produce 80% of total scrap weight. The top 20 parts below represent the highest-impact 
        intervention targets — these are where process improvements will most reduce foundry-wide 
        energy waste and material costs.
        
        *Foundry average scrap rate: {foundry_avg_scrap:.2f}% | DOE 10% target: {foundry_avg_scrap*0.90:.2f}% | DOE 20% target: {foundry_avg_scrap*0.80:.2f}%*
        """)
        
        mw_display = []
        for i, (_, row) in enumerate(most_wanted_df.iterrows()):
            mw_display.append({
                'Rank': f"#{i+1}",
                'Part ID': int(row['part_id']),
                'Runs': int(row['runs']),
                'Avg Scrap %': f"{row['avg_scrap']:.2f}%",
                'Total Scrapped (pcs)': f"{int(row['total_scrapped']):,}",
                'Total Scrap Weight (lbs)': f"{row['total_scrap_weight']:,.0f}",
                'Cumul % of Total': f"{row['cumul_pct']:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(mw_display), use_container_width=True, hide_index=True)
    
    # Part selection
    part_ids = sorted(df["part_id"].unique())
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_part = st.selectbox("Select Part ID", part_ids, index=0)
    
    part_stats = get_part_stats(df, selected_part)
    if part_stats:
        with col2:
            st.metric("📊 Avg Scrap %", f"{part_stats['avg_scrap']:.2f}%")
        with col3:
            st.metric("📋 Records", f"{part_stats['n_records']}")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Prognostic Model", "📊 RQ1: Model Validation", 
        "⚙️ RQ2: Reliability & PHM", "💰 RQ3: Operational Impact", "📈 All Parts Summary",
        "📖 Campbell Framework Reference"
    ])
    
    # TAB 1: PROGNOSTIC MODEL
    with tab1:
        st.header("Prognostic Model: Predict & Diagnose")
        
        # Calculate PART-SPECIFIC threshold for this part
        part_data = df[df['part_id'] == selected_part]
        part_threshold = part_data['scrap_percent'].mean()
        
        # Show three-stage info
        st.markdown(f"""
        <div class="citation-box">
            <strong>Three-Stage Hierarchical Prediction:</strong><br>
            This prediction combines knowledge from:<br>
            • <strong>Stage 1</strong>: Foundry-wide patterns (all {df['part_id'].nunique()} parts)<br>
            • <strong>Stage 2</strong>: Top 5 Pareto defect patterns (~80% of scrap)<br>
            • <strong>Stage 3</strong>: Part-specific calibration<br><br>
            <strong>Part-Specific Threshold:</strong> {part_threshold:.2f}% (Part {selected_part}'s average scrap rate)
        </div>
        """, unsafe_allow_html=True)
        
        # ---- MOST WANTED ALERT ----
        if selected_part in most_wanted_ids:
            # Get this part's rank and stats
            mw_row = most_wanted_df[most_wanted_df['part_id'] == selected_part].iloc[0]
            mw_rank = most_wanted_df.index[most_wanted_df['part_id'] == selected_part][0] + 1
            pct_of_total = mw_row['total_scrap_weight'] / total_scrap_weight_all * 100
            
            st.markdown(f"""
            <div style="background: #FFEBEE; border-left: 4px solid #D32F2F; padding: 15px; border-radius: 4px; margin: 10px 0;">
                <strong>🚨 VITAL FEW — Most Wanted #{mw_rank} of 20</strong><br><br>
                Part {selected_part} is ranked <strong>#{mw_rank}</strong> in total scrap weight contribution 
                ({mw_row['total_scrap_weight']:,.0f} lbs, <strong>{pct_of_total:.1f}%</strong> of foundry total). 
                With {int(mw_row['runs'])} production runs and {mw_row['avg_scrap']:.2f}% average scrap, 
                this part represents a <strong>high-impact intervention target</strong>.<br><br>
                <strong>Focus areas:</strong> Use the Failure-Conditional Pareto below to identify which defects 
                are elevated during high-scrap events, the Root Cause Process Diagnosis to trace defects to 
                Campbell's originating processes, and LIME to validate which features drive the model's 
                prediction. A 10% improvement in this part's scrap rate across future runs would save an 
                estimated <strong>{mw_row['total_scrap_weight'] * 0.10 / mw_row['runs']:,.0f} lbs per run</strong> — 
                compounding with each production cycle.
            </div>
            """, unsafe_allow_html=True)
        
        # Get pooled prediction for this part using PART-SPECIFIC threshold
        pooled_result = compute_pooled_prediction(df, selected_part, part_threshold)
        
        # Show CLT-based notification
        if pooled_result.get('show_dual'):
            # Below CLT: dual display
            part_n = pooled_result['part_level_n']
            pool_comp = pooled_result.get('pooled_comparison')
            
            st.info(f"ℹ️ **Part {selected_part} has {part_n} production runs** — below the Central Limit Theorem threshold of {CLT_THRESHOLD}. "
                    f"Showing both part-level prediction and pooled comparison for experienced judgment.")
            
            dual_col1, dual_col2 = st.columns(2)
            
            with dual_col1:
                st.markdown("##### 📊 Part-Level Prediction")
                st.caption(f"*Based on this part's {part_n} actual production run{'s' if part_n != 1 else ''}*")
                st.markdown(f"""
                | Part-Level Metrics | Value |
                |---------------------|-------|
                | **Part Weight** | **{pooled_result.get('weight_used_for_prediction', pooled_result['target_weight']):.2f} lbs** |
                | Records | {part_n} |
                | Confidence | {pooled_result['confidence']} |
                | Avg Scrap % | {pooled_result.get('part_level_avg_scrap', 0):.2f}% |
                | Failures | {pooled_result['failure_count']} |
                | Failure Rate | {pooled_result['failure_rate']*100:.1f}% |
                | MTTS (runs) | {pooled_result['mtts_runs']:.1f} |
                | MTTS (parts) | {pooled_result['mtts_parts']:,.0f} |
                | Reliability | {pooled_result['reliability_next_run']*100:.1f}% |
                """)
                st.caption(f"⚠️ < {CLT_THRESHOLD} runs — interpret with caution")
            
            with dual_col2:
                st.markdown("##### 🔄 Pooled Comparison")
                if pool_comp is not None:
                    st.caption(f"*{pool_comp['n_parts']} similar parts, {pool_comp['n_records']} total runs*")
                    st.markdown(f"""
                    | Pooled Details | Value |
                    |----------------|-------|
                    | **Pooling Method** | **{pool_comp['pooling_method']}** |
                    | Weight Range | {pool_comp['weight_range']} lbs |
                    | Total Records | {pool_comp['n_records']} |
                    | Confidence | {pool_comp['confidence']} |
                    | Pooled Threshold | {pool_comp['effective_threshold']:.2f}% (avg + 1σ) |
                    | Failures | {pool_comp['failure_count']} |
                    | Failure Rate | {pool_comp['failure_rate']*100:.1f}% |
                    | MTTS (runs) | {pool_comp['mtts_runs']:.1f} |
                    | MTTS (parts) | {pool_comp['mtts_parts']:,.0f} |
                    | Reliability | {pool_comp['reliability_next_run']*100:.1f}% |
                    """)
                    
                    # Show pooled parts
                    min_runs_filter = pool_comp.get('min_runs_filter', 5)
                    if pool_comp.get('included_part_ids'):
                        with st.expander(f"📋 View {pool_comp['n_parts']} Pooled Parts (≥{min_runs_filter} runs each)"):
                            for pid in pool_comp['included_part_ids'][:20]:
                                pid_weight = get_part_weight(df, pid)
                                if pid_weight is None:
                                    pid_weight = 0
                                pid_data = df[df['part_id'] == pid]
                                pid_n = len(pid_data)
                                pid_avg_scrap = pid_data['scrap_percent'].mean()
                                st.write(f"• Part {pid}: {pid_weight:.2f} lbs, {pid_n} runs, avg scrap {pid_avg_scrap:.1f}%")
                            if len(pool_comp['included_part_ids']) > 20:
                                st.write(f"... and {len(pool_comp['included_part_ids']) - 20} more parts")
                    
                    # Show excluded parts
                    excluded_info = pool_comp.get('excluded_parts_info', [])
                    if excluded_info:
                        with st.expander(f"🚫 {len(excluded_info)} Excluded Parts (<{min_runs_filter} runs)"):
                            for exc in excluded_info:
                                st.write(f"• Part {exc['part_id']}: {exc['runs']} runs, avg scrap {exc['avg_scrap']:.1f}%")
                else:
                    st.warning("⚠️ Insufficient similar parts found for pooled comparison. "
                              "Part-level prediction is the only available estimate.")
            
            st.markdown("""
            <div style="background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px; border-radius: 4px; margin: 10px 0; font-size: 0.9em;">
                <strong>How to interpret dual results:</strong> The <strong>Part-Level Prediction</strong> (left) reflects what the model learned from this specific part's history. The <strong>Pooled Comparison</strong> (right) shows what similar parts (matched by weight ±10% and defect profile) experienced historically, with confidence typically higher due to more data. When the two results agree, confidence is higher. When they diverge, experienced foundry judgment is essential — the part may have unique characteristics not captured by the pool, or the part's limited history may not yet reflect its true behavior.
            </div>
            """, unsafe_allow_html=True)
        else:
            # CLT satisfied (≥30 runs): full confidence in part-level prediction
            st.success(f"✅ **Part {selected_part} has sufficient data** ({pooled_result['part_level_n']} records — CLT satisfied)")
            st.markdown(f"""
            | Part Details | Value |
            |--------------|-------|
            | **Part Weight Used** | **{pooled_result.get('weight_used_for_prediction', pooled_result['target_weight']):.2f} lbs** |
            | Records | {pooled_result['part_level_n']} |
            | Confidence Level | {pooled_result['confidence']} |
            """)
        
        # ================================================================
        # LOW-SCRAP PART NOTIFICATION & POOLING MISMATCH WARNING
        # ================================================================
        effective_threshold_check = pooled_result.get('effective_threshold', part_threshold)
        pooling_used = pooled_result.get('pooling_used', False)
        
        # Check 1: Part is already at or below 1% scrap (gold standard)
        if part_threshold <= 1.0:
            total_parts_for_part = part_data['order_quantity'].sum()
            n_runs = len(part_data)
            st.markdown(f"""
            <div style="background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 15px; border-radius: 4px; margin: 10px 0;">
                <strong>✅ Low-Scrap Part — Not a Priority for Intervention</strong><br><br>
                Part {selected_part} averages <strong>{part_threshold:.2f}% scrap</strong> across 
                {n_runs} run{'s' if n_runs != 1 else ''} ({total_parts_for_part:,.0f} parts produced). 
                This is {'at or below' if part_threshold <= 0.5 else 'near'} the DOE 0.5% reduction target.<br><br>
                Until the majority of parts are below 1% scrap, this part is <strong>not driving costs 
                due to wasted energy and material</strong>. Improvement efforts should focus on the vital few 
                parts with scrap rates above the foundry average ({df['scrap_percent'].mean():.2f}%), 
                which account for the majority of total scrap weight.<br><br>
                <em>That said, the analysis below still identifies areas where this part's process 
                could be further refined. Results should be interpreted as optimization opportunities, 
                not urgent interventions.</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Check 2: Pooled comparison diverges — part's own rate is much lower than pooled rate
        pool_comp_check = pooled_result.get('pooled_comparison')
        if pooled_result.get('show_dual') and pool_comp_check is not None:
            pooled_avg_check = pool_comp_check.get('pooled_avg_scrap', 0)
            if part_threshold < pooled_avg_check and pooled_avg_check > 0:
                ratio = pooled_avg_check / part_threshold if part_threshold > 0 else float('inf')
                if ratio > 1.5:  # Pooled average is 50%+ higher than part's own average
                    st.markdown(f"""
                    <div style="background: #FFF3E0; border-left: 4px solid #FF9800; padding: 15px; border-radius: 4px; margin: 10px 0;">
                        <strong>⚠️ Pooled Comparison Diverges — Use Judgment</strong><br><br>
                        Part {selected_part}'s own average scrap rate is <strong>{part_threshold:.2f}%</strong>, 
                        but the pooled comparison population averages 
                        <strong>{pooled_avg_check:.2f}%</strong> — 
                        <strong>{ratio:.1f}× higher</strong>.<br><br>
                        This divergence may indicate this part has unique characteristics (tooling, alloy, 
                        operator experience) not captured by weight and defect matching alone. As more 
                        production runs accumulate, the part-level prediction will become increasingly reliable.
                    </div>
                    """, unsafe_allow_html=True)
        
        # Default order quantity to part's historical average
        avg_order_for_part = df[df['part_id'] == selected_part]['order_quantity'].mean()
        if pd.isna(avg_order_for_part) or avg_order_for_part < 1:
            avg_order_for_part = 100
        default_order_qty = int(min(5000, max(1, round(avg_order_for_part))))
        
        order_qty = st.slider("Order Quantity (parts)", 1, 5000, default_order_qty)
        
        # Use pooled MTTS (parts) if available - this is the CORRECTED calculation
        if pooled_result.get('mtts_parts') is not None:
            mtts_parts = pooled_result['mtts_parts']
            mtts_runs = pooled_result['mtts_runs']
            total_parts_produced = pooled_result.get('total_parts_produced', 0)
            failure_count = pooled_result['failure_count']
            failure_rate = pooled_result['failure_rate']
        elif pooled_result['mtts_runs'] is not None:
            # Fallback: convert runs to parts using average order quantity
            mtts_runs = pooled_result['mtts_runs']
            avg_qty = df[df['part_id'] == selected_part]['order_quantity'].mean()
            if pd.isna(avg_qty) or avg_qty == 0:
                avg_qty = 100
            mtts_parts = mtts_runs * avg_qty
            total_parts_produced = pooled_result.get('pooled_n', 0) * avg_qty
            failure_count = pooled_result['failure_count']
            failure_rate = pooled_result['failure_rate']
        else:
            mtts_parts = 1000
            mtts_runs = 10
            total_parts_produced = 0
            failure_count = 0
            failure_rate = 0
        
        # Calculate reliability using MTTS (parts)
        reliability = np.exp(-order_qty / mtts_parts) if mtts_parts > 0 else 0
        scrap_risk = 1 - reliability
        
        st.markdown("### 🎯 Prediction Summary")
        
        # Get threshold info for display
        effective_threshold = pooled_result.get('effective_threshold', part_threshold)
        threshold_source = pooled_result.get('threshold_source', 'part-specific')
        
        # Availability calculation (needed for both paths)
        avg_order_qty_avail = df[df['part_id'] == selected_part]['order_quantity'].mean()
        if pd.isna(avg_order_qty_avail) or avg_order_qty_avail == 0:
            avg_order_qty_avail = 100
        mttr_parts = DEFAULT_MTTR * avg_order_qty_avail
        availability = mtts_parts / (mtts_parts + mttr_parts) if mtts_parts > 0 else 0
        
        # === DUAL METRIC DISPLAY for parts below CLT ===
        if pooled_result.get('show_dual'):
            pool_comp = pooled_result.get('pooled_comparison')
            
            # Part-level metrics row
            st.markdown("**📊 Part-Level Prediction** *(from this part's data)*")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎲 Scrap Risk", f"{scrap_risk*100:.1f}%")
            m2.metric("📈 Reliability R(n)", f"{reliability*100:.1f}%")
            m3.metric("⚡ Availability", f"{availability*100:.1f}%")
            m4.metric("🔧 MTTS", f"{mtts_parts:,.0f} parts")
            
            # Pooled metrics row
            if pool_comp is not None:
                pool_mtts_p = pool_comp['mtts_parts']
                pool_rel = np.exp(-order_qty / pool_mtts_p) if pool_mtts_p > 0 else 0
                pool_scrap_risk = 1 - pool_rel
                pool_avail = pool_mtts_p / (pool_mtts_p + mttr_parts) if pool_mtts_p > 0 else 0
                
                st.markdown(f"**🔄 Pooled Comparison** *({pool_comp['n_parts']} similar parts, {pool_comp['n_records']} runs)*")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("🎲 Scrap Risk", f"{pool_scrap_risk*100:.1f}%",
                          delta=f"{(pool_scrap_risk - scrap_risk)*100:+.1f}%", delta_color="inverse")
                p2.metric("📈 Reliability R(n)", f"{pool_rel*100:.1f}%",
                          delta=f"{(pool_rel - reliability)*100:+.1f}%")
                p3.metric("⚡ Availability", f"{pool_avail*100:.1f}%",
                          delta=f"{(pool_avail - availability)*100:+.1f}%")
                p4.metric("🔧 MTTS", f"{pool_mtts_p:,.0f} parts",
                          delta=f"{pool_mtts_p - mtts_parts:+,.0f}")
                
                st.caption("*Deltas show how the pooled estimate differs from the part-level estimate. "
                           "When results diverge significantly, experienced foundry judgment should be applied.*")
        else:
            # Standard display for ≥30 run parts — no dual needed
            pass
        
        # MTTS info bar (shown for all parts)
        if failure_count > 0:
            st.success(f"""
            **MTTS (parts):** {total_parts_produced:,.0f} parts ÷ {failure_count} failures = **{mtts_parts:,.0f} parts** 
            — *On average, {mtts_parts:,.0f} parts produced between scrap events* 
            (Threshold: {effective_threshold:.2f}%, {threshold_source})
            """)
        else:
            st.warning(f"""
            **MTTS (parts):** {total_parts_produced:,.0f} parts × 2 (no failures) = **{mtts_parts:,.0f} parts** 
            — *No failures observed above {effective_threshold:.2f}% threshold — conservative estimate*
            """)
        
        # Generate curve data: n from 1 to max(2.5*MTTS, 5000, order_qty*1.5)
        curve_max_n = int(max(mtts_parts * 2.5, 5000, order_qty * 1.5))
        n_points = np.linspace(1, curve_max_n, 500)
        
        # Compute curves
        rel_curve = np.exp(-n_points / mtts_parts) * 100 if mtts_parts > 0 else np.zeros_like(n_points)
        risk_curve = 100 - rel_curve
        
        # e^(-1) point: n = MTTS, R(MTTS) = 36.8%, Risk = 63.2%
        e_inv = np.exp(-1) * 100  # 36.8%
        e_inv_risk = 100 - e_inv  # 63.2%
        
        # Current values at order_qty
        current_rel = reliability * 100
        current_risk = scrap_risk * 100
        
        # --- Two-column layout: Reliability | Scrap Risk ---
        pred_col1, pred_col2 = st.columns(2)
        
        # === RELIABILITY CHART ===
        with pred_col1:
            fig_rel = go.Figure()
            
            # Reliability curve
            fig_rel.add_trace(go.Scatter(
                x=n_points, y=rel_curve,
                mode='lines', line=dict(color='#1976D2', width=2.5),
                fill='tozeroy', fillcolor='rgba(25,118,210,0.1)',
                name='R(n)', showlegend=False
            ))
            
            # Vertical line at n = MTTS (e^-1 characteristic life)
            fig_rel.add_vline(
                x=mtts_parts, line_dash="dash", line_color="#E53935", line_width=1.5,
                annotation_text=f"e⁻¹ = {e_inv:.1f}% at n={mtts_parts:,.0f}",
                annotation_position="top right",
                annotation_font=dict(size=10, color="#E53935")
            )
            
            # Horizontal reference line at e^-1 = 36.8%
            fig_rel.add_hline(
                y=e_inv, line_dash="dot", line_color="#E53935", line_width=1, opacity=0.5
            )
            
            # Current order quantity marker
            fig_rel.add_trace(go.Scatter(
                x=[order_qty], y=[current_rel],
                mode='markers+text',
                marker=dict(size=12, color='#1976D2', symbol='diamond', 
                           line=dict(width=2, color='white')),
                text=[f'{current_rel:.1f}%'],
                textposition='top center',
                textfont=dict(size=11, color='#1976D2'),
                name='Current Order', showlegend=False
            ))
            
            fig_rel.update_layout(
                title=dict(text="<b>Reliability R(n)</b>", font=dict(size=14)),
                xaxis_title="Parts (n)",
                yaxis_title="Reliability (%)",
                yaxis=dict(range=[0, 105]),
                height=300, margin=dict(l=50, r=20, t=40, b=50),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0'),
                yaxis_gridcolor='#f0f0f0'
            )
            st.plotly_chart(fig_rel, use_container_width=True)
            
            # Current value
            st.metric("📈 Reliability R(n)", f"{current_rel:.1f}%")
            
            # Formula
            st.caption(f"R({order_qty:,}) = e^(-{order_qty:,} / {mtts_parts:,.0f}) = **{current_rel:.1f}%**")
            st.caption(f'*"Probability of completing {order_qty:,} parts without a scrap event"*')
        
        # === SCRAP RISK CHART ===
        with pred_col2:
            fig_risk = go.Figure()
            
            # Risk curve
            fig_risk.add_trace(go.Scatter(
                x=n_points, y=risk_curve,
                mode='lines', line=dict(color='#E53935', width=2.5),
                fill='tozeroy', fillcolor='rgba(229,57,53,0.1)',
                name='Scrap Risk', showlegend=False
            ))
            
            # Vertical line at n = MTTS (e^-1 characteristic life)
            fig_risk.add_vline(
                x=mtts_parts, line_dash="dash", line_color="#1976D2", line_width=1.5,
                annotation_text=f"e⁻¹ → Risk={e_inv_risk:.1f}% at n={mtts_parts:,.0f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#1976D2")
            )
            
            # Horizontal reference line at 1 - e^-1 = 63.2%
            fig_risk.add_hline(
                y=e_inv_risk, line_dash="dot", line_color="#1976D2", line_width=1, opacity=0.5
            )
            
            # Current order quantity marker
            fig_risk.add_trace(go.Scatter(
                x=[order_qty], y=[current_risk],
                mode='markers+text',
                marker=dict(size=12, color='#E53935', symbol='diamond',
                           line=dict(width=2, color='white')),
                text=[f'{current_risk:.1f}%'],
                textposition='top center',
                textfont=dict(size=11, color='#E53935'),
                name='Current Order', showlegend=False
            ))
            
            fig_risk.update_layout(
                title=dict(text="<b>Scrap Risk F(n)</b>", font=dict(size=14)),
                xaxis_title="Parts (n)",
                yaxis_title="Scrap Risk (%)",
                yaxis=dict(range=[0, 105]),
                height=300, margin=dict(l=50, r=20, t=40, b=50),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0'),
                yaxis_gridcolor='#f0f0f0'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Current value
            st.metric("🎲 Scrap Risk", f"{current_risk:.1f}%")
            
            # Formula
            st.caption(f"F({order_qty:,}) = 1 - R({order_qty:,}) = 1 - {current_rel:.1f}% = **{current_risk:.1f}%**")
            st.caption(f'*"Probability of experiencing a scrap event during this order"*')
        
        
        st.markdown("---")
        
        # ================================================================
        # DETAILED DEFECT ANALYSIS
        # ================================================================
        # Historical Pareto: P(defect_rate | all runs)
        # Failure Pareto:    P(defect_rate | scrap > threshold)
        # The difference reveals which defects are disproportionately present
        # during failure events — the assignable causes driving scrap.
        # ================================================================
        st.markdown("### 📊 Detailed Defect Analysis")
        
        # Get data for analysis (use pooled parts if pooling was applied)
        if pooled_result['pooling_used'] and pooled_result.get('included_part_ids'):
            analysis_df = df[df['part_id'].isin(pooled_result['included_part_ids'])]
        else:
            analysis_df = df[df['part_id'] == selected_part]
        
        # User-adjustable threshold for failure-conditional Pareto
        default_threshold = pooled_result.get('effective_threshold', part_threshold)
        max_scrap = float(analysis_df['scrap_percent'].max()) if len(analysis_df) > 0 else 50.0
        
        # DOE/EPA benchmark quick-set buttons (10% and 20% relative reduction)
        reduction_10 = round(part_threshold * 0.90 * 2) / 2  # Round to nearest 0.5
        reduction_20 = round(part_threshold * 0.80 * 2) / 2  # Round to nearest 0.5
        
        # Initialize session state for slider if not set or out of bounds for current part
        slider_max = max(max_scrap, default_threshold + 1.0)
        if 'unified_threshold_slider' not in st.session_state:
            st.session_state.unified_threshold_slider = min(default_threshold, max_scrap)
        elif st.session_state.unified_threshold_slider > slider_max:
            st.session_state.unified_threshold_slider = min(default_threshold, max_scrap)
        
        st.markdown("#### ⚙️ Scrap Exceedance Threshold")
        st.caption("*Adjust to redefine what counts as a 'failure.' This threshold drives the Failure-Conditional "
                   "Defect Pareto, Root Cause Diagnosis, and Scrap Threshold Sensitivity Analysis below.*")
        
        # Quick-set buttons row
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with btn_col1:
            st.caption(f"Part Avg: **{part_threshold:.2f}%**")
        with btn_col2:
            if st.button(f"📉 10% Reduction → {reduction_10:.1f}%", 
                        key="btn_10pct",
                        help=f"DOE lower bound: 10% relative reduction from {part_threshold:.2f}% avg scrap"):
                st.session_state.unified_threshold_slider = max(0.5, reduction_10)
                st.rerun()
        with btn_col3:
            if st.button(f"📉 20% Reduction → {reduction_20:.1f}%",
                        key="btn_20pct", 
                        help=f"DOE upper bound: 20% relative reduction from {part_threshold:.2f}% avg scrap"):
                st.session_state.unified_threshold_slider = max(0.5, reduction_20)
                st.rerun()
        with btn_col4:
            if st.button(f"🔄 Reset to Part Avg",
                        key="btn_reset_threshold",
                        help=f"Reset to part-specific average: {part_threshold:.2f}%"):
                st.session_state.unified_threshold_slider = min(default_threshold, max_scrap)
                st.rerun()
        
        threshold_col1, threshold_col2, threshold_col3 = st.columns([2, 1, 1])
        with threshold_col1:
            unified_threshold = st.slider(
                "🎚️ Scrap % Threshold (Failure Definition)",
                min_value=0.5,
                max_value=max(max_scrap, default_threshold + 1.0),
                step=0.5,
                key="unified_threshold_slider",
                help="Runs with scrap above this threshold are treated as 'failure events' — affects Pareto, reliability metrics, and sensitivity analysis"
            )
        
        # Count runs at this threshold
        failure_df = analysis_df[analysis_df['scrap_percent'] > unified_threshold]
        n_failures = len(failure_df)
        n_total = len(analysis_df)
        
        with threshold_col2:
            st.metric("Failure Runs", f"{n_failures} of {n_total}")
        with threshold_col3:
            st.metric("Failure Rate", f"{n_failures/n_total*100:.0f}%" if n_total > 0 else "0%")
        
        # Compute defect rates: historical (all runs) vs. failure-conditional
        defect_data = []
        for col in defect_cols:
            if col in analysis_df.columns:
                hist_rate = analysis_df[col].mean() * 100
                if hist_rate > 0:
                    defect_name = col.replace('_rate', '').replace('_', ' ').title()
                    # Conditional rate: average defect rate during failure runs only
                    if n_failures > 0:
                        failure_rate = failure_df[col].mean() * 100
                    else:
                        failure_rate = hist_rate  # No failures: fall back to historical
                    
                    # Risk multiplier: how much more prevalent is this defect during failures?
                    risk_multiplier = failure_rate / hist_rate if hist_rate > 0 else 1.0
                    
                    defect_data.append({
                        'Defect': defect_name,
                        'Defect_Code': col,
                        'Historical Rate (%)': hist_rate,
                        'Failure Rate (%)': failure_rate,
                        'Risk Multiplier': risk_multiplier,
                        'Expected Count': hist_rate / 100 * order_qty
                    })
        
        if defect_data:
            defect_df = pd.DataFrame(defect_data)
            
            # Pareto Charts side by side
            pareto_col1, pareto_col2 = st.columns(2)
            
            with pareto_col1:
                st.markdown("#### 📊 Historical Defect Pareto")
                st.caption(f"*Average defect rates across ALL {n_total} runs*")
                hist_data = defect_df.sort_values('Historical Rate (%)', ascending=False).head(10).copy()
                
                # Create Pareto chart
                total = hist_data['Historical Rate (%)'].sum()
                hist_data['Cumulative %'] = (hist_data['Historical Rate (%)'].cumsum() / total * 100) if total > 0 else 0
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=hist_data['Defect'],
                    y=hist_data['Historical Rate (%)'],
                    name='Rate (%)',
                    marker_color='steelblue'
                ))
                fig_hist.add_trace(go.Scatter(
                    x=hist_data['Defect'],
                    y=hist_data['Cumulative %'],
                    name='Cumulative %',
                    marker_color='red',
                    yaxis='y2',
                    mode='lines+markers'
                ))
                fig_hist.update_layout(
                    title="Top 10 Historical Defects (All Runs)",
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(title='Rate (%)', side='left'),
                    yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
                    height=400,
                    showlegend=True,
                    legend=dict(x=0.7, y=1)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with pareto_col2:
                st.markdown("#### 🔮 Failure-Conditional Defect Pareto")
                if n_failures > 0:
                    st.caption(f"*Average defect rates during {n_failures} failure runs (scrap > {unified_threshold:.1f}%)*")
                else:
                    st.caption(f"*No failure runs above {unified_threshold:.1f}% threshold — showing historical rates*")
                
                # Sort by FAILURE rate — Pareto order may differ from historical
                pred_data = defect_df.sort_values('Failure Rate (%)', ascending=False).head(10).copy()
                
                total_pred = pred_data['Failure Rate (%)'].sum()
                pred_data['Cumulative %'] = (pred_data['Failure Rate (%)'].cumsum() / total_pred * 100) if total_pred > 0 else 0
                
                # Color bars by risk multiplier: red if elevated during failures, gray if same/lower
                bar_colors = []
                for _, row in pred_data.iterrows():
                    if row['Risk Multiplier'] > 1.5:
                        bar_colors.append('#C62828')   # Dark red — strongly elevated during failures
                    elif row['Risk Multiplier'] > 1.1:
                        bar_colors.append('#E53935')   # Red — moderately elevated
                    elif row['Risk Multiplier'] > 0.9:
                        bar_colors.append('#FF8A65')   # Orange — similar to historical
                    else:
                        bar_colors.append('#66BB6A')   # Green — lower during failures
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Bar(
                    x=pred_data['Defect'],
                    y=pred_data['Failure Rate (%)'],
                    name='Failure Rate (%)',
                    marker_color=bar_colors,
                    text=[f"{m:.1f}×" for m in pred_data['Risk Multiplier']],
                    textposition='outside'
                ))
                fig_pred.add_trace(go.Scatter(
                    x=pred_data['Defect'],
                    y=pred_data['Cumulative %'],
                    name='Cumulative %',
                    marker_color='orange',
                    yaxis='y2',
                    mode='lines+markers'
                ))
                fig_pred.update_layout(
                    title="Top 10 Defects During Failure Events",
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(title='Failure Rate (%)', side='left'),
                    yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
                    height=400,
                    showlegend=True,
                    legend=dict(x=0.7, y=1)
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Risk multiplier insight callout
            if n_failures > 0:
                top_risk = defect_df.sort_values('Risk Multiplier', ascending=False).head(3)
                elevated = top_risk[top_risk['Risk Multiplier'] > 1.1]
                if len(elevated) > 0:
                    risk_items = [f"**{row['Defect']}** ({row['Risk Multiplier']:.1f}× historical)" for _, row in elevated.iterrows()]
                    st.info(f"⚠️ **Defects disproportionately elevated during failure events:** {', '.join(risk_items)}. "
                           f"These defects are more prevalent when scrap exceeds {unified_threshold:.1f}% than during normal production, "
                           f"indicating assignable causes for targeted intervention.")
        
        st.markdown("---")
        
        # ================================================================
        # ROOT CAUSE PROCESS DIAGNOSIS
        # ================================================================
        st.markdown("### 🏭 Root Cause Process Diagnosis")
        st.caption("*Based on Campbell (2003) process-defect relationships*")
        st.info(f"📍 **Using threshold: {unified_threshold:.1f}%** — Process contributions computed from "
                f"**{n_failures} failure runs** (scrap > {unified_threshold:.1f}%), not historical averages.")
        
        # Compute process contributions from FAILURE-CONDITIONAL defect rates
        # (driven by unified threshold slider, not historical averages)
        if n_failures > 0 and defect_data:
            # Use failure rates from the already-computed defect_data
            failure_defect_rates = {d['Defect_Code']: d['Failure Rate (%)'] / 100 for d in defect_data}
        else:
            # Fallback to historical if no failures
            failure_defect_rates = {d['Defect_Code']: d['Historical Rate (%)'] / 100 for d in defect_data} if defect_data else {}
        
        # Compute process scores from failure-conditional rates
        process_scores = {}
        for process, info in PROCESS_DEFECT_MAP.items():
            score = sum(failure_defect_rates.get(d, 0) for d in info['defects'])
            process_scores[process] = score
        
        total_score = sum(process_scores.values())
        if total_score > 0:
            process_contributions = {p: (s / total_score) * 100 for p, s in process_scores.items()}
        else:
            process_contributions = {p: 0 for p in process_scores}
        
        process_ranking = sorted(process_contributions.items(), key=lambda x: x[1], reverse=True)
        
        if process_ranking:
            # Process contribution chart
            process_data = [p for p in process_ranking if p[1] > 0]
            
            if process_data:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### 🎯 Top Contributing Processes")
                    for i, (process, contribution) in enumerate(process_data[:5]):
                        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                        color = '#FFEBEE' if i == 0 else '#FFF3E0' if i == 1 else '#E3F2FD' if i == 2 else '#F5F5F5'
                        campbell_ref = PROCESS_DEFECT_MAP[process].get('campbell_rule', '')
                        st.markdown(f"""
                        <div style="background: {color}; padding: 10px; border-radius: 8px; margin: 5px 0;">
                            {icon} <strong>{process}</strong>: {contribution:.1f}%
                            <br><small>{PROCESS_DEFECT_MAP[process]['description']}</small>
                            <br><small><em>Campbell (2003): {campbell_ref}</em></small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig_process = go.Figure(go.Bar(
                        x=[p[0] for p in process_data],
                        y=[p[1] for p in process_data],
                        marker_color=['#D32F2F', '#F57C00', '#1976D2', '#388E3C', '#7B1FA2', 
                                     '#00796B', '#5D4037', '#455A64', '#C2185B'][:len(process_data)]
                    ))
                    fig_process.update_layout(
                        title=f"Process Contributions During Failure Events (scrap > {unified_threshold:.1f}%)",
                        xaxis_title="Process",
                        yaxis_title="Contribution (%)",
                        height=350,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_process, use_container_width=True)
                
                # Detailed Process Table
                st.markdown("#### 📋 Detailed Process Analysis")
                
                process_table = []
                for process, contribution in process_data:
                    risk_share = contribution / 100 * scrap_risk * 100
                    defects = PROCESS_DEFECT_MAP[process]['defects']
                    defect_names = [d.replace('_rate', '').replace('_', ' ').title() for d in defects]
                    process_table.append({
                        'Process': process,
                        'Campbell Rule': PROCESS_DEFECT_MAP[process].get('campbell_rule', ''),
                        'Contribution (%)': f"{contribution:.1f}",
                        f'Risk Share (of {scrap_risk*100:.1f}%)': f"{risk_share:.2f}",
                        'Related Defects': ', '.join(defect_names),
                        'Description': PROCESS_DEFECT_MAP[process]['description']
                    })
                
                process_df = pd.DataFrame(process_table)
                st.dataframe(process_df, use_container_width=True, hide_index=True)
                
                # Defect to Process Mapping
                st.markdown("#### 🔗 Defect → Process Mapping")
                st.caption(f"*How each defect contributes to the **{scrap_risk*100:.1f}% total scrap risk***")
                
                if defect_data:
                    mapping_data = []
                    total_defect_rate = sum(d['Historical Rate (%)'] for d in defect_data[:10])
                    
                    for d in defect_data[:10]:
                        defect_code = d['Defect_Code']
                        # Find which processes this defect maps to
                        related_processes = []
                        campbell_rules = []
                        for proc, info in PROCESS_DEFECT_MAP.items():
                            if defect_code in info['defects']:
                                related_processes.append(proc)
                                campbell_rules.append(info.get('campbell_rule', ''))
                        
                        if total_defect_rate > 0:
                            risk_share = (d['Historical Rate (%)'] / total_defect_rate) * scrap_risk * 100
                        else:
                            risk_share = 0
                        
                        mapping_data.append({
                            'Defect': d['Defect'],
                            'Historical Rate (%)': f"{d['Historical Rate (%)']:.2f}",
                            'Failure Rate (%)': f"{d['Failure Rate (%)']:.2f}",
                            'Risk Multiplier': f"{d['Risk Multiplier']:.1f}×",
                            'Risk Share (%)': f"{risk_share:.2f}",
                            'Expected Count': f"{d['Expected Count']:.1f}",
                            'Root Cause Process(es)': ', '.join(related_processes) if related_processes else 'Unknown',
                            'Campbell Rule': ', '.join(campbell_rules) if campbell_rules else ''
                        })
                    
                    mapping_df = pd.DataFrame(mapping_data)
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        
        # ================================================================
        # LIME: LOCAL INTERPRETABLE MODEL EXPLANATION
        # ================================================================
        # Reference: Ribeiro, M.T., Singh, S., & Guestrin, C. (2016).
        # "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
        #
        # This section provides instance-level explanations for WHY the model
        # made this specific prediction, supporting Zio's (2022) call for
        # PHM model interpretability and NASA's mission assurance principle
        # of "dynamic, synthesizing feedback" for decision-making.
        # ================================================================
        st.markdown("---")
        st.markdown("### 🔬 LIME: Local Model Explanation")
        st.caption("*Why did the model make THIS specific prediction?*")
        
        st.markdown("""
        <div class="citation-box">
            <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong><br>
            Unlike global feature importance, LIME explains <em>individual predictions</em> by:
            <ol>
                <li>Perturbing the input features around this specific instance</li>
                <li>Observing how the model's prediction changes</li>
                <li>Fitting a simple linear model to approximate the complex model locally</li>
                <li>Returning feature weights showing each feature's contribution</li>
            </ol>
            <strong>Reference:</strong> Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). KDD.
        </div>
        """, unsafe_allow_html=True)
        
        if LIME_AVAILABLE:
            # Get the most recent record for this part to explain
            part_recent = part_data.sort_values('run_date' if 'run_date' in part_data.columns else part_data.columns[0]).iloc[-1:]
            
            if len(part_recent) > 0 and global_model is not None:
                # Prepare features for this instance
                try:
                    model_features = global_model['features']
                    X_train = global_model['X_train']
                    cal_model = global_model['cal_model']
                    
                    # Ensure all features exist in part_recent
                    for f in model_features:
                        if f not in part_recent.columns:
                            part_recent[f] = 0.0
                    
                    instance = part_recent[model_features].fillna(0)
                    
                    # Generate LIME explanation
                    with st.spinner("Generating LIME explanation (perturbing ~1000 samples)..."):
                        lime_result = explain_prediction_lime(
                            model=cal_model,
                            X_train=X_train,
                            feature_names=model_features,
                            instance=instance,
                            num_features=10
                        )
                    
                    # Store LIME result in session state for action plan report
                    st.session_state.lime_result = lime_result
                    
                    if lime_result['error'] is None:
                        lime_col1, lime_col2 = st.columns([1, 2])
                        
                        with lime_col1:
                            st.markdown("#### 🎯 Prediction Explained")
                            pred_proba = lime_result['prediction_proba']
                            st.metric(
                                "ML Scrap Probability",
                                f"{pred_proba*100:.1f}%",
                                help="Model's predicted probability of scrap for this instance"
                            )
                            
                            st.markdown(f"""
                            **Interpretation:**
                            - 🔴 **Red** features INCREASE scrap risk
                            - 🟢 **Green** features DECREASE scrap risk
                            - Larger bars = stronger influence
                            """)
                        
                        with lime_col2:
                            st.markdown("#### 📊 Feature Contributions to This Prediction")
                            
                            # Create bar chart of LIME weights
                            if lime_result['explanation']:
                                lime_df = pd.DataFrame(lime_result['explanation'], columns=['Feature', 'Weight'])
                                lime_df['Abs_Weight'] = lime_df['Weight'].abs()
                                lime_df = lime_df.sort_values('Abs_Weight', ascending=True)
                                
                                # Color by direction
                                colors = ['#EF5350' if w > 0 else '#66BB6A' for w in lime_df['Weight']]
                                
                                fig_lime = go.Figure(go.Bar(
                                    x=lime_df['Weight'],
                                    y=lime_df['Feature'],
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f"{w:+.3f}" for w in lime_df['Weight']],
                                    textposition='outside'
                                ))
                                
                                fig_lime.update_layout(
                                    title="LIME Feature Weights (Local Linear Approximation)",
                                    xaxis_title="Weight (+ increases scrap risk, - decreases)",
                                    yaxis_title="Feature Condition",
                                    height=400,
                                    showlegend=False,
                                    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
                                )
                                
                                st.plotly_chart(fig_lime, use_container_width=True)
                        
                        # Detailed LIME table
                        with st.expander("📋 View Detailed LIME Explanation"):
                            st.markdown("""
                            **How to Read This Table:**
                            - **Feature Condition**: The feature value range for this instance
                            - **Weight**: Contribution to scrap probability (+ increases, - decreases)
                            - **Direction**: Whether this feature pushes toward scrap or away from scrap
                            """)
                            
                            lime_table = []
                            for feat, weight in lime_result['explanation']:
                                direction = "↑ Increases Risk" if weight > 0 else "↓ Decreases Risk"
                                impact = "High" if abs(weight) > 0.1 else "Medium" if abs(weight) > 0.05 else "Low"
                                lime_table.append({
                                    'Feature Condition': feat,
                                    'Weight': f"{weight:+.4f}",
                                    'Direction': direction,
                                    'Impact': impact
                                })
                            
                            lime_detail_df = pd.DataFrame(lime_table)
                            st.dataframe(lime_detail_df, use_container_width=True, hide_index=True)
                            
                            st.info(f"""
                            **LIME Local Model Summary:**
                            - Intercept (base prediction): {lime_result['intercept']:.4f}
                            - Sum of weights + intercept ≈ predicted probability
                            - This linear approximation is valid only LOCALLY around this instance
                            """)
                    
                    else:
                        st.error(f"LIME Error: {lime_result['error']}")
                        if 'traceback' in lime_result:
                            with st.expander("🔍 View Technical Details"):
                                st.code(lime_result['traceback'])
                        
                except Exception as e:
                    import traceback
                    st.warning(f"Could not generate LIME explanation: {type(e).__name__}: {str(e)}")
                    with st.expander("🔍 View Technical Details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("""
            ⚠️ **LIME not installed**
            
            To enable LIME explanations, install the library:
            ```
            pip install lime
            ```
            
            LIME (Local Interpretable Model-agnostic Explanations) provides instance-level 
            explanations showing WHY the model made each specific prediction.
            """)
        
        
        # ================================================================
        # SCRAP THRESHOLD SENSITIVITY ANALYSIS
        # ================================================================
        st.markdown("---")
        st.markdown("### 📊 Scrap Threshold Sensitivity Analysis")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Paradigm Shift Demonstration:</strong><br>
            This analysis shows how reliability metrics respond <em>continuously</em> to different 
            scrap % threshold definitions—unlike traditional SPC which provides only binary 
            "in control" / "out of control" assessments.<br><br>
            <strong>Key Insight:</strong> Reliability is a <em>continuous function</em> of your quality standard.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate reasonable threshold range
        part_avg_sens = part_data['scrap_percent'].mean()
        part_std_sens = part_data['scrap_percent'].std()
        part_min_sens = part_data['scrap_percent'].min()
        part_max_sens = part_data['scrap_percent'].max()
        global_avg_sens = df['scrap_percent'].mean()
        
        # Use the unified threshold slider from the Detailed Defect Analysis section
        threshold_slider = unified_threshold
        
        st.info(f"📍 **Using threshold: {threshold_slider:.1f}%** (set via the Scrap Exceedance Threshold slider above)")
        
        sens_stats_col1, sens_stats_col2 = st.columns([1, 1])
        with sens_stats_col1:
            st.markdown(f"""
            **Part Statistics:**
            - Part Avg: {part_avg_sens:.2f}%
            - Global Avg: {global_avg_sens:.2f}%
            - Min: {part_min_sens:.2f}%
            - Max: {part_max_sens:.2f}%
            """)
        
        # Function to compute sensitivity at a single threshold
        def compute_single_threshold_metrics(part_df, thresh, qty):
            """Compute metrics at a single threshold value.
            
            Uses Total Parts / Failures method for MTTS calculation,
            consistent with NASA RAM Training (1999) methodology.
            """
            part_df = part_df.sort_values('week_ending').copy()
            total_runs = len(part_df)
            total_parts_qty = part_df['order_quantity'].sum() if 'order_quantity' in part_df.columns else total_runs
            
            # Count failures (runs where scrap % exceeds threshold)
            failures = (part_df['scrap_percent'] > thresh).sum()
            
            # Compute MTTS using Total/Failures method (NASA RAM standard)
            # This is consistent with the main Prediction Summary calculation
            if failures > 0:
                mtts_p = total_parts_qty / failures  # Total Parts / Failures
                mtts_r = total_runs / failures       # Total Runs / Failures
            else:
                # No failures = very high MTTS (use 2x total as proxy)
                mtts_p = total_parts_qty * 2
                mtts_r = total_runs * 2
            
            rel = np.exp(-qty / mtts_p) if mtts_p > 0 else 0
            risk = 1 - rel
            
            return {
                'mtts_parts': mtts_p,
                'mtts_runs': mtts_r,
                'reliability': rel * 100,
                'scrap_risk': risk * 100,
                'failure_count': failures,
                'total_runs': total_runs
            }
        
        # Compute at selected threshold
        sens_result = compute_single_threshold_metrics(part_data, threshold_slider, order_qty)
        
        # Display metrics at selected threshold
        st.markdown(f"#### 📍 Metrics at {threshold_slider:.1f}% Threshold")
        
        sens_m1, sens_m2, sens_m3, sens_m4 = st.columns(4)
        sens_m1.metric("🎲 Scrap Risk", f"{sens_result['scrap_risk']:.1f}%", 
                       delta=f"{sens_result['scrap_risk'] - scrap_risk*100:.1f}%" if threshold_slider != part_threshold else None)
        sens_m2.metric("📈 Reliability", f"{sens_result['reliability']:.1f}%",
                       delta=f"{sens_result['reliability'] - reliability*100:.1f}%" if threshold_slider != part_threshold else None)
        sens_m3.metric("🔧 MTTS (parts)", f"{sens_result['mtts_parts']:,.0f}")
        sens_m4.metric("⚠️ Failures", f"{sens_result['failure_count']:.0f} / {sens_result['total_runs']:.0f}")
        
        # Generate sensitivity curve data
        threshold_range = np.linspace(
            max(0.1, part_min_sens * 0.5),
            10.0,
            40
        )
        
        sens_data = []
        for t in threshold_range:
            res = compute_single_threshold_metrics(part_data, t, order_qty)
            sens_data.append({
                'threshold': t,
                'reliability': res['reliability'],
                'scrap_risk': res['scrap_risk'],
                'mtts_parts': res['mtts_parts'],
                'failure_count': res['failure_count']
            })
        sens_df = pd.DataFrame(sens_data)
        
        # Create 2x2 subplot visualization
        fig_sens = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "<b>Reliability vs. Threshold</b>",
                "<b>MTTS vs. Threshold</b>",
                "<b>Failure Count vs. Threshold</b>",
                "<b>Scrap Risk vs. Threshold</b>"
            ),
            vertical_spacing=0.18,
            horizontal_spacing=0.12
        )
        
        # 1. Reliability
        fig_sens.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['reliability'],
            mode='lines', line=dict(color='#1976D2', width=2),
            fill='tozeroy', fillcolor='rgba(25,118,210,0.2)',
            showlegend=False
        ), row=1, col=1)
        fig_sens.add_trace(go.Scatter(
            x=[threshold_slider], y=[sens_result['reliability']],
            mode='markers', marker=dict(size=12, color='#D32F2F', symbol='diamond'),
            showlegend=False
        ), row=1, col=1)
        fig_sens.add_hline(y=80, line_dash="dash", line_color="green", row=1, col=1)
        fig_sens.add_vline(x=part_threshold, line_dash="dot", line_color="gray", row=1, col=1)
        
        # 2. MTTS
        fig_sens.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['mtts_parts'],
            mode='lines', line=dict(color='#388E3C', width=2),
            showlegend=False
        ), row=1, col=2)
        fig_sens.add_trace(go.Scatter(
            x=[threshold_slider], y=[sens_result['mtts_parts']],
            mode='markers', marker=dict(size=12, color='#D32F2F', symbol='diamond'),
            showlegend=False
        ), row=1, col=2)
        fig_sens.add_vline(x=part_threshold, line_dash="dot", line_color="gray", row=1, col=2)
        
        # 3. Failure Count
        fig_sens.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['failure_count'],
            mode='lines', line=dict(color='#F57C00', width=2),
            showlegend=False
        ), row=2, col=1)
        fig_sens.add_trace(go.Scatter(
            x=[threshold_slider], y=[sens_result['failure_count']],
            mode='markers', marker=dict(size=12, color='#D32F2F', symbol='diamond'),
            showlegend=False
        ), row=2, col=1)
        fig_sens.add_vline(x=part_threshold, line_dash="dot", line_color="gray", row=2, col=1)
        
        # 4. Scrap Risk
        fig_sens.add_trace(go.Scatter(
            x=sens_df['threshold'], y=sens_df['scrap_risk'],
            mode='lines', line=dict(color='#D32F2F', width=2),
            fill='tozeroy', fillcolor='rgba(211,47,47,0.2)',
            showlegend=False
        ), row=2, col=2)
        fig_sens.add_trace(go.Scatter(
            x=[threshold_slider], y=[sens_result['scrap_risk']],
            mode='markers', marker=dict(size=12, color='#D32F2F', symbol='diamond'),
            showlegend=False
        ), row=2, col=2)
        fig_sens.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=2)
        fig_sens.add_vline(x=part_threshold, line_dash="dot", line_color="gray", row=2, col=2)
        
        # Update axes
        fig_sens.update_xaxes(title_text="Scrap % Threshold", row=1, col=1)
        fig_sens.update_xaxes(title_text="Scrap % Threshold", row=1, col=2)
        fig_sens.update_xaxes(title_text="Scrap % Threshold", row=2, col=1)
        fig_sens.update_xaxes(title_text="Scrap % Threshold", row=2, col=2)
        fig_sens.update_yaxes(title_text="Reliability (%)", row=1, col=1)
        fig_sens.update_yaxes(title_text="MTTS (parts)", row=1, col=2)
        fig_sens.update_yaxes(title_text="Failures", row=2, col=1)
        fig_sens.update_yaxes(title_text="Scrap Risk (%)", row=2, col=2)
        
        fig_sens.update_layout(
            height=600,
            title_text=f"<b>Threshold Sensitivity Analysis for Part {selected_part}</b>",
            title_font_size=18,
            showlegend=False,
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Make subplot titles larger and clearer
        for annotation in fig_sens['layout']['annotations']:
            annotation['font'] = dict(size=14, color='#333333')
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
        # Interpretation
        with st.expander("💡 Interpretation Guide"):
            st.markdown("""
            **📉 Lower Threshold (Stricter Standard):**
            - More events count as "failures" → Lower MTTS → Lower reliability
            - *Use for: Safety-critical parts, high-value castings, strict customers*
            
            **📈 Higher Threshold (Lenient Standard):**
            - Fewer events count as "failures" → Higher MTTS → Higher reliability
            - *Use for: General production, cost-sensitive orders*
            
            **Key Insight:** The gray dotted line shows the part's average scrap rate (current threshold).
            The green dashed line shows the 80% reliability target.
            """)
        
        # Comparison table at key thresholds
        with st.expander("📋 Comparison at Key Thresholds"):
            key_thresh = sorted(set([
                max(0.5, part_avg_sens - part_std_sens),
                part_avg_sens,
                global_avg_sens,
                part_avg_sens + part_std_sens
            ]))
            
            comp_data = []
            for kt in key_thresh:
                kr = compute_single_threshold_metrics(part_data, kt, order_qty)
                label = "Part Avg" if abs(kt - part_avg_sens) < 0.3 else \
                        "Global Avg" if abs(kt - global_avg_sens) < 0.3 else \
                        "Strict" if kt < part_avg_sens else "Lenient"
                comp_data.append({
                    'Standard': label,
                    'Threshold (%)': f"{kt:.2f}",
                    'Reliability (%)': f"{kr['reliability']:.1f}",
                    'MTTS (parts)': f"{kr['mtts_parts']:,.0f}",
                    'Failures': kr['failure_count'],
                    'Scrap Risk (%)': f"{kr['scrap_risk']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        
        # Reliability Metrics Snapshot
        st.markdown("---")
        st.markdown("### 📋 Reliability Metrics Snapshot")
        st.info(f"📍 **Using threshold: {unified_threshold:.1f}%** (set via the Scrap Exceedance Threshold slider above)")
        
        # Recalculate MTTS based on unified threshold
        snapshot_part_data = part_data.sort_values('week_ending').copy() if 'week_ending' in part_data.columns else part_data.copy()
        snapshot_total_parts = int(snapshot_part_data['order_quantity'].sum()) if 'order_quantity' in snapshot_part_data.columns else len(snapshot_part_data)
        snapshot_total_runs = len(snapshot_part_data)
        snapshot_failures = int((snapshot_part_data['scrap_percent'] > unified_threshold).sum())
        
        if snapshot_failures > 0:
            snapshot_mtts_parts = snapshot_total_parts / snapshot_failures
            snapshot_mtts_runs = snapshot_total_runs / snapshot_failures
        else:
            snapshot_mtts_parts = snapshot_total_parts * 2
            snapshot_mtts_runs = snapshot_total_runs * 2
        
        mtts_parts_val = snapshot_mtts_parts
        
        # Calculate reliability at different order quantities (in parts)
        rel_100 = np.exp(-100 / mtts_parts_val) if mtts_parts_val > 0 else 0
        rel_500 = np.exp(-500 / mtts_parts_val) if mtts_parts_val > 0 else 0
        rel_1000 = np.exp(-1000 / mtts_parts_val) if mtts_parts_val > 0 else 0
        
        r1, r5, r10 = st.columns(3)
        r1.metric("R(100 parts)", f"{rel_100*100:.1f}%")
        r5.metric("R(500 parts)", f"{rel_500*100:.1f}%")
        r10.metric("R(1000 parts)", f"{rel_1000*100:.1f}%")
        
        # Prepare values for table
        mtts_runs_display = f"{snapshot_mtts_runs:.1f}"
        mtts_parts_display = f"{mtts_parts_val:,.0f}"
        lambda_display = f"{1/mtts_parts_val:.6f}" if mtts_parts_val > 0 else "0"
        total_parts_display = f"{snapshot_total_parts:,}"
        total_runs_display = f"{snapshot_total_runs:,}"
        data_source = f"Part-level ({pooled_result.get('part_level_n', 0)} records)" + (f" + Pooled comparison" if pooled_result.get('show_dual') else "")
        
        st.markdown(f"""
        | Metric | Value | Formula |
        |--------|-------|---------|
        | **MTTS (parts)** | **{mtts_parts_display}** | Total Parts ({total_parts_display}) ÷ Failures ({snapshot_failures}) |
        | MTTS (runs) | {mtts_runs_display} | Total Runs ({total_runs_display}) ÷ Failures ({snapshot_failures}) |
        | λ (failure rate) | {lambda_display} | 1 ÷ MTTS (parts) |
        | Failures observed | {snapshot_failures} | Runs where Scrap % > {unified_threshold:.2f}% |
        | **Threshold used** | **{unified_threshold:.2f}%** | **Unified scrap exceedance threshold** |
        | Data source | {data_source} | |
        """)
        
        # ================================================================
        # RELIABILITY TARGET CALCULATOR
        # ================================================================
        st.markdown("---")
        st.markdown("### 🧮 Reliability Target Calculator")
        st.caption("*What MTTS is needed to achieve a specific reliability target for this order size?*")
        
        if defect_data and snapshot_failures > 0:
            # Current state from threshold-adjusted values
            current_failures = snapshot_failures
            total_parts_produced = snapshot_total_parts
            current_mtts = mtts_parts_val if mtts_parts_val > 0 else 1000
            current_reliability = np.exp(-order_qty / current_mtts) * 100 if current_mtts > 0 else 0
            sorted_defects = sorted(defect_data, key=lambda x: x['Historical Rate (%)'], reverse=True)[:5]
            
            target_reliability = st.slider(
                "Target Reliability (%)",
                min_value=80,
                max_value=100,
                value=90,  # Default to world-class target
                key=f"target_rel_slider_{selected_part}"
            )
            
            # Calculate required MTTS for target reliability
            # R = e^(-n/MTTS) → MTTS = -n / ln(R)
            if target_reliability < 100:
                required_mtts = -order_qty / np.log(target_reliability / 100)
                required_failures = total_parts_produced / required_mtts if required_mtts > 0 else 1
                failures_to_eliminate = current_failures - required_failures
                
                # What percentage reduction is needed?
                pct_reduction_needed = (failures_to_eliminate / current_failures * 100) if current_failures > 0 else 0
                
                calc1, calc2, calc3 = st.columns(3)
                
                calc1.metric(
                    "Required MTTS",
                    f"{required_mtts:,.0f} parts",
                    delta=f"+{required_mtts - current_mtts:,.0f} from current"
                )
                
                calc2.metric(
                    "Max Failures Allowed",
                    f"{required_failures:.1f}",
                    delta=f"-{failures_to_eliminate:.1f} from current"
                )
                
                calc3.metric(
                    "Failure Reduction Needed",
                    f"{pct_reduction_needed:.0f}%",
                    delta=None
                )
                
                # Actionable recommendation
                if failures_to_eliminate > 0:
                    top_defect_names = ', '.join([d['Defect'] for d in sorted_defects[:3]])
                    st.info(f"""
                    **📋 Action Required:** To achieve **{target_reliability}% reliability** for {order_qty:,} part orders:
                    - Reduce high-scrap runs from **{current_failures:.0f}** to **{required_failures:.0f}** (eliminate {failures_to_eliminate:.1f} failure events)
                    - This requires approximately **{pct_reduction_needed:.0f}%** overall failure rate reduction
                    - Focus on top Pareto defects: **{top_defect_names}**
                    """)
                else:
                    st.success(f"""
                    **✅ Target Already Achieved:** Current reliability ({current_reliability:.1f}%) already meets 
                    the {target_reliability}% target for {order_qty:,} part orders.
                    """)
            else:
                # target_reliability == 100
                st.warning("""
                **⚠️ 100% Reliability is Theoretically Unachievable:** 
                Under the exponential reliability model, 100% reliability requires infinite MTTS (zero failures ever). 
                This is not practically achievable. Consider targeting 95-99% for near-perfect performance.
                """)
            
            # Literature reference
            with st.expander("📚 Literature References for Action Thresholds"):
                st.markdown("""
                **Industry-Standard Reliability Thresholds:**
                
                | Reliability Level | Interpretation | Source |
                |-------------------|----------------|--------|
                | **≥90%** | World-class performance | NASA (1999), Tractian (2024) |
                | **80-90%** | Acceptable/target zone | DoD MIL-STD, Industry benchmarks |
                | **70-80%** | Warning zone - plan intervention | Manufacturing best practices |
                | **<70%** | Critical - immediate action required | PHM literature |
                
                **Key References:**
                - NASA (1999). *Reliability, Availability & Maintainability Training*. NASA/TP-2000-207428.
                - DoD (2018). *DOT&E Reliability Course*. DOTE Reliability Training Materials.
                - Tractian (2024). *Preventive Maintenance Guide*. Manufacturing facilities operating below 70% preventive work struggle with reliability.
                
                **PHM Decision Framework:**
                - **Scheduling Threshold** → When to START planning (R < 80%)
                - **Maintenance Threshold** → When to EXECUTE action (R < 70%)  
                - **Failure Threshold** → Critical intervention required (R < 50%)
                """)
        
        else:
            st.warning("Insufficient defect data available for this part to generate reliability targets.")
    # ================================================================
    # TAB 2: RQ1 - MODEL VALIDATION
    # ================================================================
    # SCIKIT-LEARN APPLICATION: EVALUATION METRICS DISPLAY
    # ================================================================
    # This tab displays the metrics computed by Scikit-learn:
    #   - recall_score() → 98.6% (displayed as "Recall")
    #   - precision_score() → 97.2% (displayed as "Precision")
    #   - roc_auc_score() → 0.999 (displayed as "AUC-ROC")
    #   - brier_score_loss() → 0.012 (displayed as "Brier Score")
    #   - roc_curve() → Data for ROC Curve visualization
    #   - calibration_curve() → Data for Calibration Curve visualization
    # ================================================================
    with tab2:
        st.header("RQ1: Model Validation & Predictive Performance")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Research Question 1:</strong> Does MTTS-integrated ML achieve effective prognostic recall (≥80%)?
            <br><strong>Hypothesis H1:</strong> MTTS integration achieves ≥80% recall, consistent with effective PHM.
        </div>
        """, unsafe_allow_html=True)
        
        metrics = global_model["metrics"]
        
        st.markdown(f"### 📊 Model Performance Metrics")
        st.caption(f"*Evaluated on test set: {global_model['n_test']} samples*")
        
        c1, c2, c3, c4 = st.columns(4)
        
        recall_pass = metrics["recall"] >= RQ_THRESHOLDS['RQ1']['recall']
        precision_pass = metrics["precision"] >= RQ_THRESHOLDS['RQ1']['precision']
        auc_pass = metrics["auc"] >= RQ_THRESHOLDS['RQ1']['auc']
        
        # SCIKIT-LEARN: recall_score(y_test, y_pred) = TP / (TP + FN)
        c1.metric(f"{'✅' if recall_pass else '❌'} Recall", f"{metrics['recall']*100:.1f}%", f"{'Pass' if recall_pass else 'Below'} ≥80%")
        # SCIKIT-LEARN: precision_score(y_test, y_pred) = TP / (TP + FP)
        c2.metric(f"{'✅' if precision_pass else '❌'} Precision", f"{metrics['precision']*100:.1f}%", f"{'Pass' if precision_pass else 'Below'} ≥70%")
        # SCIKIT-LEARN: roc_auc_score(y_test, y_prob) = Area under ROC curve
        c3.metric(f"{'✅' if auc_pass else '❌'} AUC-ROC", f"{metrics['auc']:.3f}", f"{'Pass' if auc_pass else 'Below'} ≥0.80")
        # SCIKIT-LEARN: brier_score_loss(y_test, y_prob) = Mean squared error of probabilities
        c4.metric("📉 Brier Score", f"{metrics['brier']:.3f}")
        
        h1_pass = recall_pass and precision_pass and auc_pass
        
        if h1_pass:
            st.success(f"""
            ### ✅ Hypothesis H1: SUPPORTED
            
            The MTTS-integrated ML model achieves **{metrics['recall']*100:.1f}% recall**, 
            meeting the PHM performance benchmark (Lei et al., 2018).
            """)
        else:
            st.warning("### ⚠️ Hypothesis H1: Partially Supported")
        
        # ROC Curve
        col1, col2 = st.columns(2)
        with col1:
            if "roc_fpr" in metrics:
                st.markdown("#### ROC Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics["roc_fpr"], y=metrics["roc_tpr"], mode='lines', name=f'Model (AUC={metrics["auc"]:.3f})'))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Calibration Curve
        with col2:
            if "cal_true" in metrics:
                st.markdown("#### Calibration Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics["cal_pred"], y=metrics["cal_true"], mode='lines+markers', name='Model'))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash')))
                fig.update_layout(xaxis_title="Predicted Probability", yaxis_title="Actual Frequency", height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    # ================================================================
    # TAB 3: RQ2 - PHM EQUIVALENCE
    # ================================================================
    # SCIKIT-LEARN APPLICATION: RECALL COMPARISON
    # ================================================================
    # This tab uses the recall_score computed by Scikit-learn and
    # compares it to the sensor-based PHM benchmark (90%).
    #
    # PHM Equivalence = (sklearn recall_score / 0.90) × 100
    #                 = (0.986 / 0.90) × 100 = 109.5%
    # ================================================================
    with tab3:
        st.header("RQ2: Reliability & PHM Equivalence")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Research Question 2:</strong> Can sensor-free ML achieve ≥80% of sensor-based PHM performance?
            <br><strong>Hypothesis H2:</strong> SPC-native ML achieves ≥80% PHM-equivalent recall without sensors.
        </div>
        """, unsafe_allow_html=True)
        
        sensor_benchmark = RQ_THRESHOLDS['RQ2']['sensor_benchmark']
        # SCIKIT-LEARN: This recall value comes from recall_score(y_test, y_pred)
        model_recall = global_model["metrics"]["recall"]
        phm_equiv = (model_recall / sensor_benchmark) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Sensor Benchmark", f"{sensor_benchmark*100:.0f}%", help="Typical sensor-based PHM recall (Lei et al., 2018)")
        c2.metric("🤖 Our Model Recall", f"{model_recall*100:.1f}%", help="Model's recall on test data")
        
        phm_pass = phm_equiv >= RQ_THRESHOLDS['RQ2']['phm_equivalence'] * 100
        c3.metric(f"{'✅' if phm_pass else '❌'} PHM Equivalence", f"{phm_equiv:.1f}%", f"{'Pass' if phm_pass else 'Below'} ≥80%")
        
        if phm_pass:
            st.success(f"### ✅ Hypothesis H2: SUPPORTED\n\nPHM Equivalence: **{phm_equiv:.1f}%** (≥80%)")
        else:
            st.warning(f"### ⚠️ Hypothesis H2: Partially Supported")
    
    # TAB 4: RQ3 - OPERATIONAL IMPACT
    with tab4:
        st.header("RQ3: Operational Impact Analysis")
        
        total_weight = (df["order_quantity"] * df["piece_weight_lbs"]).sum()
        date_range = (df["week_ending"].max() - df["week_ending"].min()).days
        annual_prod = total_weight * (365 / date_range) if date_range > 0 else total_weight
        current_scrap = df["scrap_percent"].mean()
        
        c1, c2 = st.columns(2)
        with c1:
            annual_production = st.number_input("Annual Production (lbs)", value=int(annual_prod), min_value=1000)
            current_scrap = st.number_input("Current Scrap Rate (%)", value=float(current_scrap))
            material_cost = st.number_input("Material Cost ($/lb)", value=2.50)
        with c2:
            target_scrap = st.number_input("Target Scrap Rate (%)", value=max(0, current_scrap * 0.85))
            implementation_cost = st.number_input("Implementation Cost ($)", value=2000.0)
            energy_cost = st.number_input("Energy Cost ($/MMBtu)", value=12.0)
        
        tte = calculate_tte_savings(current_scrap, target_scrap, annual_production)
        material_savings = tte["avoided_scrap_lbs"] * material_cost
        energy_savings = tte["tte_savings_mmbtu"] * energy_cost
        total_savings = material_savings + energy_savings
        roi = total_savings / implementation_cost if implementation_cost > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📉 Scrap Reduction", f"{tte['scrap_reduction_pct']*100:.1f}%")
        m2.metric("⚡ TTE Savings", f"{tte['tte_savings_mmbtu']:,.1f} MMBtu")
        m3.metric("🌿 CO₂ Avoided", f"{tte['co2_savings_tons']:,.2f} tons")
        m4.metric("💰 ROI", f"{roi:.1f}×")
    
    # ================================================================
    # TAB 5: ALL PARTS SUMMARY
    # ================================================================
    # SCIKIT-LEARN APPLICATION: AGGREGATE METRICS DISPLAY
    # ================================================================
    # This tab displays the aggregated Scikit-learn metrics:
    #   - recall_score() → Recall (98.6%)
    #   - precision_score() → Precision (97.2%)
    #   - roc_auc_score() → AUC-ROC (0.999)
    #
    # These are the SAME metrics computed in train_global_model()
    # and represent the model's performance on the held-out test set.
    #
    # The per-part analysis shows how the global model performs
    # when applied to each individual part.
    # ================================================================
    with tab5:
        st.header("📈 All Parts Summary: Global Model Performance")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Validation Methodology:</strong> 6-2-1 temporal split (60% train, 20% calibration, 20% test)
            <br><strong>Model:</strong> Random Forest with probability calibration (Platt scaling)
            <br><strong>Library:</strong> Scikit-learn (Pedregosa et al., 2011) - Industry standard ML library
        </div>
        """, unsafe_allow_html=True)
        
        metrics = global_model["metrics"]
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Train (60%)", f"{global_model['n_train']:,}", help="Records used to train the model")
        c2.metric("Calib (20%)", f"{global_model['n_calib']:,}", help="Records used for probability calibration")
        c3.metric("Test (20%)", f"{global_model['n_test']:,}", help="Held-out records for final evaluation")
        c4.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c5.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c6.metric("AUC-ROC", f"{metrics['auc']:.3f}")
        
        st.markdown("### ✅ Model Validation Summary")
        
        h1_pass = (metrics["recall"] >= 0.80 and metrics["precision"] >= 0.70 and metrics["auc"] >= 0.80)
        phm_equiv = (metrics["recall"] / 0.90) * 100
        h2_pass = phm_equiv >= 80
        
        c1, c2 = st.columns(2)
        with c1:
            if h1_pass:
                st.success(f"""### ✅ H1: SUPPORTED — Model Predictions Are Validated
                
This model was validated using **industry-standard machine learning evaluation methods** implemented in Scikit-learn (Pedregosa et al., 2011).

**Performance Metrics (Industry Standards from PHM Literature):**
- **Recall: {metrics['recall']*100:.1f}%** — Of all actual scrap events, the model predicted {metrics['recall']*100:.1f}%
  - *Target: ≥80% per Lei et al. (2018) PHM benchmarks* ✓
- **Precision: {metrics['precision']*100:.1f}%** — Of all predicted scrap events, {metrics['precision']*100:.1f}% were correct
  - *Target: ≥70%* ✓
- **AUC-ROC: {metrics['auc']:.3f}** — Model's ability to distinguish scrap from non-scrap
  - *Target: ≥0.80 (1.0 = perfect)* ✓
                """)
            else:
                st.warning(f"### ⚠️ H1: Partially Supported")
        with c2:
            if h2_pass:
                st.success(f"""### ✅ H2: SUPPORTED — Matches Sensor-Based Performance

**PHM Equivalence: {phm_equiv:.1f}%**

This metric compares our sensor-free, SPC-based model against traditional sensor-based Prognostics and Health Management (PHM) systems that typically achieve ~90% recall (Lei et al., 2018).

**Calculation:** (Model Recall ÷ Sensor Benchmark) × 100
= ({metrics['recall']*100:.1f}% ÷ 90%) × 100 = **{phm_equiv:.1f}%**

**Result:** This dashboard **EXCEEDS** sensor-based performance using only existing SPC data—no new sensors or infrastructure required.
                """)
            else:
                st.warning(f"### ⚠️ H2: Partially Supported")
        
        st.markdown("---")
        
        # SYSTEMIC THRESHOLD EXPLANATION
        st.markdown("### 📊 Understanding H1 and H2 Pass Rates")
        
        st.info(f"""
**Why use a foundry-wide threshold ({threshold:.2f}%)?**

This dashboard treats scrap as a **systemic issue**—the result of interconnected foundry processes (Melting, Pouring, Sand System, etc.) rather than isolated part-specific problems. Therefore, all parts are evaluated against the **foundry-wide average scrap rate ({threshold:.2f}%)** as the common standard.

**The systemic approach:**
- The goal is to **lower total foundry scrap**, not just stabilize individual parts at their current (possibly poor) levels
- Parts exceeding the foundry average are pulling the entire system down
- Root causes often trace to **shared processes** that affect multiple parts
- Using a common threshold allows fair comparison across all parts

**What "failure" means:** Any production run where scrap % exceeds {threshold:.2f}% (the foundry average) is counted as a failure event. This systemic definition identifies parts that need process improvement to bring the entire foundry's performance up.
        """)
        
        # ================================================================
        # PER-PART ANALYSIS USING GLOBAL MODEL
        # ================================================================
        st.markdown("### 📊 Per-Part Reliability Distribution (All Parts)")
        st.caption(f"*Each part evaluated against the foundry-wide threshold of {threshold:.2f}%*")
        
        with st.spinner("Computing per-part metrics..."):
            # Compute metrics for each part using pooled predictions
            part_results = []
            
            for pid in df['part_id'].unique():
                # Get pooled prediction for this part
                pooled = compute_pooled_prediction(df, pid, threshold)
                
                # Get part stats
                part_data = df[df['part_id'] == pid]
                n_records = len(part_data)
                avg_scrap = part_data['scrap_percent'].mean()
                total_parts_prod = part_data['order_quantity'].sum() if 'order_quantity' in part_data.columns else n_records
                
                # Compute reliability metrics using MTTS (parts)
                mtts_parts = pooled.get('mtts_parts', 0)
                mtts_runs = pooled.get('mtts_runs', 0)
                
                if mtts_runs and mtts_runs > 0:
                    reliability = np.exp(-1 / mtts_runs)  # R(1 run)
                    failure_rate = pooled['failure_rate']
                else:
                    reliability = 0
                    failure_rate = 0
                
                # PHM equivalence for this part
                part_phm_equiv = (reliability / 0.90) * 100 if reliability > 0 else 0
                
                # Determine H1/H2 pass status
                h1_pass_part = reliability >= 0.80
                h2_pass_part = part_phm_equiv >= 80
                
                part_results.append({
                    'Part ID': pid,
                    'Records': n_records,
                    'Total Parts': total_parts_prod,
                    'Avg Scrap %': avg_scrap,
                    'Pooled Records': pooled['pooled_n'],
                    'Pooling Method': pooled['pooling_method'],
                    'MTTS (parts)': mtts_parts,
                    'MTTS (runs)': mtts_runs,
                    'Reliability R(1)': reliability * 100,
                    'Failure Rate': failure_rate * 100,
                    'PHM Equiv %': part_phm_equiv,
                    'H1 Pass': h1_pass_part,
                    'H2 Pass': h2_pass_part,
                    'Confidence': pooled['confidence']
                })
            
            results_df = pd.DataFrame(part_results)
        
        # Summary Statistics
        st.markdown("### 📈 Summary Statistics")
        
        total_parts = len(results_df)
        h1_pass_count = results_df['H1 Pass'].sum()
        h2_pass_count = results_df['H2 Pass'].sum()
        pooled_count = (results_df['Records'] < CLT_THRESHOLD).sum()
        
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Parts", f"{total_parts}")
        s2.metric("H1 Pass Rate", f"{h1_pass_count/total_parts*100:.1f}%", f"{h1_pass_count}/{total_parts}")
        s3.metric("H2 Pass Rate", f"{h2_pass_count/total_parts*100:.1f}%", f"{h2_pass_count}/{total_parts}")
        s4.metric("Below CLT (< 30)", f"{pooled_count}", "Dual results shown")
        s5.metric("Avg Reliability", f"{results_df['Reliability R(1)'].mean():.1f}%")
        
        # Interpretation of pass rates
        st.warning(f"""
**Interpreting H1/H2 Pass Rates:**

The low pass rates are **not a model failure**—the model is validated with {metrics['recall']*100:.1f}% recall. These rates reveal the **actual foundry performance**:

- **H1 Pass Rate ({h1_pass_count/total_parts*100:.1f}%):** Only {h1_pass_count} of {total_parts} parts consistently meet the foundry standard (Reliability ≥ 80%)
- **H2 Pass Rate ({h2_pass_count/total_parts*100:.1f}%):** Only {h2_pass_count} of {total_parts} parts achieve PHM-equivalent reliability

This means **{total_parts - h1_pass_count} parts ({(total_parts - h1_pass_count)/total_parts*100:.1f}%)** have chronic scrap issues exceeding the foundry average of {threshold:.2f}%, representing opportunities for systemic process improvement.
        """)
        
        # Distribution Charts
        st.markdown("### 📊 RQ1: Model Validation Distributions")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # Reliability Distribution
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Histogram(
                x=results_df['Reliability R(1)'],
                nbinsx=20,
                marker_color='#4CAF50',
                name='Reliability'
            ))
            fig_rel.add_vline(x=80, line_dash="dash", line_color="red",
                             annotation_text="80% Threshold")
            fig_rel.update_layout(
                title="Reliability R(1 run) Distribution",
                xaxis_title="Reliability (%)",
                yaxis_title="Number of Parts",
                height=350
            )
            st.plotly_chart(fig_rel, use_container_width=True)
            
            # Stats table
            st.markdown(f"""
            | Statistic | Value |
            |-----------|-------|
            | Mean | {results_df['Reliability R(1)'].mean():.1f}% |
            | Median | {results_df['Reliability R(1)'].median():.1f}% |
            | Std Dev | {results_df['Reliability R(1)'].std():.1f}% |
            | Min | {results_df['Reliability R(1)'].min():.1f}% |
            | Max | {results_df['Reliability R(1)'].max():.1f}% |
            | Parts ≥80% | {(results_df['Reliability R(1)'] >= 80).sum()} |
            """)
        
        with dist_col2:
            # Failure Rate Distribution
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Histogram(
                x=results_df['Failure Rate'],
                nbinsx=20,
                marker_color='#FF9800',
                name='Failure Rate'
            ))
            fig_fr.update_layout(
                title="Failure Rate Distribution",
                xaxis_title="Failure Rate (%)",
                yaxis_title="Number of Parts",
                height=350
            )
            st.plotly_chart(fig_fr, use_container_width=True)
            
            # Stats table
            st.markdown(f"""
            | Statistic | Value |
            |-----------|-------|
            | Mean | {results_df['Failure Rate'].mean():.2f}% |
            | Median | {results_df['Failure Rate'].median():.2f}% |
            | Std Dev | {results_df['Failure Rate'].std():.2f}% |
            | Min | {results_df['Failure Rate'].min():.2f}% |
            | Max | {results_df['Failure Rate'].max():.2f}% |
            """)
        
        # PHM Equivalence Distribution
        st.markdown("### 📊 RQ2: PHM Equivalence Distribution")
        
        phm_col1, phm_col2 = st.columns(2)
        
        with phm_col1:
            fig_phm = go.Figure()
            fig_phm.add_trace(go.Histogram(
                x=results_df['PHM Equiv %'],
                nbinsx=20,
                marker_color='#2196F3',
                name='PHM Equivalence'
            ))
            fig_phm.add_vline(x=80, line_dash="dash", line_color="red",
                             annotation_text="80% Threshold")
            fig_phm.update_layout(
                title="PHM Equivalence Distribution (Model Recall / 90%)",
                xaxis_title="PHM Equivalence (%)",
                yaxis_title="Number of Parts",
                height=350
            )
            st.plotly_chart(fig_phm, use_container_width=True)
        
        with phm_col2:
            # MTTS Distribution
            mtts_valid = results_df[results_df['MTTS (runs)'] > 0]['MTTS (runs)']
            fig_mtts = go.Figure()
            fig_mtts.add_trace(go.Histogram(
                x=mtts_valid,
                nbinsx=20,
                marker_color='#9C27B0',
                name='MTTS'
            ))
            fig_mtts.update_layout(
                title="MTTS (runs) Distribution",
                xaxis_title="MTTS (runs until failure)",
                yaxis_title="Number of Parts",
                height=350
            )
            st.plotly_chart(fig_mtts, use_container_width=True)
        
        # Data Quality / Pooling Analysis
        st.markdown("### 📊 Data Quality & Pooling Analysis")
        
        pool_col1, pool_col2 = st.columns(2)
        
        with pool_col1:
            # Records per part distribution
            fig_records = go.Figure()
            fig_records.add_trace(go.Histogram(
                x=results_df['Records'],
                nbinsx=30,
                marker_color='#607D8B',
                name='Records'
            ))
            fig_records.add_vline(x=5, line_dash="dash", line_color="red",
                                 annotation_text="Min for Part-Level (5)")
            fig_records.add_vline(x=30, line_dash="dash", line_color="green",
                                 annotation_text="HIGH Confidence (30)")
            fig_records.update_layout(
                title="Records per Part Distribution",
                xaxis_title="Number of Records",
                yaxis_title="Number of Parts",
                height=350
            )
            st.plotly_chart(fig_records, use_container_width=True)
        
        with pool_col2:
            # Pooling method breakdown
            pooling_counts = results_df['Pooling Method'].value_counts()
            fig_pooling = go.Figure(go.Pie(
                labels=pooling_counts.index,
                values=pooling_counts.values,
                hole=0.4
            ))
            fig_pooling.update_layout(
                title="Pooling Methods Used",
                height=350
            )
            st.plotly_chart(fig_pooling, use_container_width=True)
        
        # Detailed Results Table
        st.markdown("### 📋 Detailed Results Table")
        
        with st.expander("View All Parts Data"):
            # Format for display
            display_df = results_df.copy()
            display_df['Avg Scrap %'] = display_df['Avg Scrap %'].round(2)
            display_df['MTTS (runs)'] = display_df['MTTS (runs)'].round(1)
            display_df['Reliability R(1)'] = display_df['Reliability R(1)'].round(1)
            display_df['Failure Rate'] = display_df['Failure Rate'].round(2)
            display_df['PHM Equiv %'] = display_df['PHM Equiv %'].round(1)
            display_df['H1 Pass'] = display_df['H1 Pass'].map({True: '✅', False: '❌'})
            display_df['H2 Pass'] = display_df['H2 Pass'].map({True: '✅', False: '❌'})
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="all_parts_results.csv",
                mime="text/csv"
            )
    # ================================================================
    # TAB 6: CAMPBELL FRAMEWORK REFERENCE
    # ================================================================
    with tab6:
        st.header("Campbell's 10 Rules of Castings: Process-Defect Framework")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Source:</strong> Campbell, J. (2003). <em>Castings Practice: The 10 Rules of Castings</em>. 
            Elsevier Butterworth-Heinemann. ISBN 07506 4791 4.<br><br>
            <strong>Application:</strong> The process-defect mappings used in this dashboard's Root Cause Process 
            Diagnosis are derived from Campbell's framework. The dataset's defect terminology was aligned with 
            Campbell's casting defect taxonomy, and defects were mapped to the originating process stages he identifies.
            Individual foundries would need to validate these mappings against their specific process configurations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ---- Campbell's 10 Rules Overview ----
        st.markdown("### The 10 Rules of Castings")
        st.caption("*Campbell's framework identifies 10 critical rules governing casting quality. "
                   "Each rule maps to specific process stages and the defects that arise when the rule is violated.*")
        
        rules_data = [
            {"Rule": "1", "Name": "Achieve a Good Quality Melt", 
             "Process Stage": "Melting & Holding",
             "Defects When Violated": "Dross, gas porosity, bifilm inclusions, oxide skins",
             "Campbell's Guidance": "Melt must be substantially free from non-metallic inclusions and bifilms. "
                                    "Gas in solution and alloy impurities (e.g., Fe in Al) act as bifilm-opening agents."},
            {"Rule": "2", "Name": "Avoid Turbulent Entrainment", 
             "Process Stage": "Pouring & Filling System",
             "Defects When Violated": "Misruns, sand inclusions, random porosity, oxide folds",
             "Campbell's Guidance": "Critical velocity ~0.5 m/s must not be exceeded. Turbulent filling creates "
                                    "'the standard legacy: random inclusions, random porosity, and misruns.' "
                                    "Most so-called sand problems are actually filling system problems."},
            {"Rule": "3", "Name": "Avoid Laminar Entrainment of Surface Film", 
             "Process Stage": "Filling System Design",
             "Defects When Violated": "Surface oxide folds, lap defects, cold shuts",
             "Campbell's Guidance": "The liquid front must expand continuously. Progress only uphill in an "
                                    "uninterrupted advance. Only bottom gating is permissible; no falling or "
                                    "sliding downhill of liquid metal."},
            {"Rule": "4", "Name": "Avoid Bubble Damage", 
             "Process Stage": "Gating Design & Pouring",
             "Defects When Violated": "Gas bubbles, bubble trails, subsurface porosity",
             "Campbell's Guidance": "No bubbles of air entrained by the filling system should pass through the "
                                    "liquid metal. Requires proper offset step pouring basin, avoidance of wells "
                                    "or volume-increasing features."},
            {"Rule": "5", "Name": "Avoid Core Blows", 
             "Process Stage": "Core Making",
             "Defects When Violated": "Core gas porosity, blow holes, internal voids",
             "Campbell's Guidance": "Cores must be of sufficiently low gas content and/or adequately vented. "
                                    "No use of clay-based repair paste unless demonstrated fully dried."},
            {"Rule": "6", "Name": "Avoid Shrinkage Damage", 
             "Process Stage": "Gating/Feeding/Riser Design",
             "Defects When Violated": "Shrinkage porosity, shrinkage cavities, hot tears",
             "Campbell's Guidance": "No feeding uphill in larger sections due to unreliable pressure gradient "
                                    "and convection complications. Demonstrate good feeding by following all Feeding Rules."},
            {"Rule": "7", "Name": "Avoid Convection Damage", 
             "Process Stage": "Thermal Management",
             "Defects When Violated": "Channel segregation, convection-driven porosity redistribution",
             "Campbell's Guidance": "Assess freezing time vs. convection damage time. Thin and thick sections "
                                    "avoid problems automatically; intermediate sections require design intervention."},
            {"Rule": "8", "Name": "Reduce Segregation Damage", 
             "Process Stage": "Alloy & Process Control",
             "Defects When Violated": "Chemical segregation, out-of-spec composition zones",
             "Campbell's Guidance": "Predict segregation to be within specification limits. Avoid channel "
                                    "segregation formation if possible."},
            {"Rule": "9", "Name": "Reduce Residual Stress", 
             "Process Stage": "Shakeout & Heat Treatment",
             "Defects When Violated": "Distortion, cracking, residual stress failures",
             "Campbell's Guidance": "No quenching into water following solution treatment of light alloys. "
                                    "Polymer quenchant or forced air quench acceptable if stress shown negligible."},
            {"Rule": "10", "Name": "Provide Location Points", 
             "Process Stage": "Pattern/Tooling Design",
             "Defects When Violated": "Dimensional non-conformance, assembly failures",
             "Campbell's Guidance": "All castings to be provided with agreed location points for dimensional "
                                    "checking and machining pickup."}
        ]
        
        rules_df = pd.DataFrame(rules_data)
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ---- Dashboard Mapping ----
        st.markdown("### Dashboard Process-Defect Mapping")
        st.caption("*How dataset defect columns were mapped to Campbell's process stages*")
        
        st.info("**Methodology:** The anonymized dataset uses defect rate terminology (e.g., `dross_rate`, "
                "`gas_porosity_rate`, `sand_rate`) that aligns with Campbell's casting defect taxonomy. "
                "Each defect column was mapped to the originating process stage identified by Campbell's 10 Rules. "
                "This mapping is a **proof-of-concept** -- individual foundries must validate these mappings against "
                "their specific equipment, process flows, and defect classification systems.")
        
        mapping_data = [
            {"Dashboard Process": "Melting", "Dataset Defect Columns": "dross_rate, gas_porosity_rate",
             "Campbell Rule(s)": "Rule 1: Good Quality Melt",
             "Rationale": "Campbell identifies dross and gas porosity as direct consequences of melt quality -- "
                         "bifilms, oxide skins, and dissolved gas originate in the melting and holding stages (Ch. 1)."},
            {"Dashboard Process": "Pouring", "Dataset Defect Columns": "misrun_rate, missrun_rate, short_pour_rate, runout_rate",
             "Campbell Rule(s)": "Rule 2: Avoid Turbulent Entrainment",
             "Rationale": "Campbell states misruns result from 'unpredictable ebb and flow during filling' -- "
                         "turbulent entrainment during pouring. Short pours and runouts are filling interruptions (Ch. 2)."},
            {"Dashboard Process": "Gating Design", "Dataset Defect Columns": "shrink_rate, shrink_porosity_rate, tear_up_rate",
             "Campbell Rule(s)": "Rule 6: Avoid Shrinkage Damage",
             "Rationale": "Shrinkage porosity and hot tears arise from inadequate feeding system design -- "
                         "runner/riser sizing and feeding path geometry (Ch. 6, Feeding Rules 1-7)."},
            {"Dashboard Process": "Sand System", "Dataset Defect Columns": "sand_rate, dirty_pattern_rate",
             "Campbell Rule(s)": "Rules 2-3 (secondary)",
             "Rationale": "Campbell notes most 'sand problems such as mould erosion and sand inclusions' are "
                         "actually consequences of poor filling systems, but sand preparation (binder ratio, "
                         "moisture content) remains an independent process variable (Ch. 2)."},
            {"Dashboard Process": "Core Making", "Dataset Defect Columns": "core_rate, crush_rate, shift_rate",
             "Campbell Rule(s)": "Rule 5: Avoid Core Blows",
             "Rationale": "Core integrity, venting, and gas content directly control core-related defects. "
                         "Crush and shift are mechanical core failures during mould assembly or pouring (Ch. 5)."},
            {"Dashboard Process": "Shakeout", "Dataset Defect Columns": "bent_rate",
             "Campbell Rule(s)": "Rule 9: Reduce Residual Stress",
             "Rationale": "Bent castings result from mechanical damage during extraction or residual stress "
                         "from premature shakeout before adequate cooling (Ch. 9)."},
            {"Dashboard Process": "Pattern/Tooling", "Dataset Defect Columns": "gouged_rate",
             "Campbell Rule(s)": "Rule 10: Provide Location Points (extended)",
             "Rationale": "Gouging is a surface defect consistent with pattern wear, damage, or dimensional "
                         "degradation. Campbell emphasizes pattern/tooling accuracy for dimensional conformance (Ch. 10)."},
            {"Dashboard Process": "Inspection", "Dataset Defect Columns": "outside_process_scrap_rate, zyglo_rate, failed_zyglo_rate",
             "Campbell Rule(s)": "Detection, not origination",
             "Rationale": "These are detection-stage classifications, not process-origin defects. Zyglo (fluorescent "
                         "penetrant) reveals subsurface cracks that may originate from Rules 1, 2, 6, or 9. "
                         "'Outside process scrap' is vendor/customer-attributed. Mapped to Inspection as the "
                         "classifying stage -- this is a known limitation of the dataset's terminology."},
            {"Dashboard Process": "Finishing", "Dataset Defect Columns": "over_grind_rate, cut_into_rate",
             "Campbell Rule(s)": "Post-casting operations",
             "Rationale": "Over-grinding and cut-into defects occur during finishing operations (grinding, machining) "
                         "after the casting is solidified. These are process-induced defects not directly covered by "
                         "Campbell's 10 Rules, which focus on the casting process itself."}
        ]
        
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ---- Limitations and Validation ----
        st.markdown("### Mapping Limitations & Validation Requirements")
        
        col_lim1, col_lim2 = st.columns(2)
        
        with col_lim1:
            st.markdown("""
            **What This Mapping IS:**
            - A proof-of-concept alignment between dataset terminology and Campbell's (2003) established 
              casting defect taxonomy
            - A demonstration that ML-predicted defect patterns can be traced to originating process stages 
              using domain knowledge frameworks
            - A template that any foundry can adapt by validating mappings against their specific processes
            """)
        
        with col_lim2:
            st.markdown("""
            **What This Mapping IS NOT:**
            - A validated mapping for any specific foundry's process configuration
            - A claim that every defect has a single originating process (many defects have multiple 
              potential causes -- e.g., porosity can originate from Rules 1, 2, 4, 5, or 6)
            - A substitute for foundry-specific root cause analysis using their own process data and 
              engineering expertise
            """)
        
        st.markdown("---")
        
        st.markdown("### Foundry Implementation Guidance")
        st.info("""
        **For foundries adopting this framework:**
        
        1. **Map your defect codes** to Campbell's taxonomy -- your classification system may use different 
           terminology than this dataset
        2. **Validate process-defect relationships** using your own historical data -- run failure-conditional 
           Pareto analysis to confirm which defects are actually elevated during your high-scrap events
        3. **Iterate the mapping** -- Campbell's Rules provide the theoretical foundation, but your specific 
           equipment, alloys, and process parameters determine the actual defect-process relationships
        4. **Use LIME explanations** to validate -- if the model identifies sand_rate as a top contributor but 
           your sand system is well-controlled, the mapping may need adjustment for your foundry
        """)
        
        st.markdown("---")
        st.caption("*Campbell, J. (2003). Castings Practice: The 10 Rules of Castings. Elsevier Butterworth-Heinemann. "
                   "Process-defect mappings derived from Campbell's framework as a proof-of-concept. "
                   "Individual foundries must validate mappings against their specific process configurations.*")
    
    st.markdown("---")
    st.caption("🏭 Foundry Dashboard V3 | Global Model with Multi-Defect + Temporal + MTTS Features | 6-2-1 Split")


if __name__ == "__main__":
    main()
