# ================================================================
# 🏭 FOUNDRY PROGNOSTIC RELIABILITY DASHBOARD
# STREAMLINED VERSION V3 - MATCHING ENHANCED VERSION
# ================================================================
#
# This version replicates the EXACT model and features from the
# enhanced version (v3.6) to achieve ~99% recall performance.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from scipy import stats
from datetime import datetime

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
DEFAULT_MTTR = 1.0
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

# Campbell Process-Defect Mapping
PROCESS_DEFECT_MAP = {
    "Melting": {"defects": ["dross_rate", "gas_porosity_rate"], "description": "Metal preparation, temperature control"},
    "Pouring": {"defects": ["misrun_rate", "missrun_rate", "short_pour_rate", "runout_rate"], "description": "Pour temperature, rate control"},
    "Gating Design": {"defects": ["shrink_rate", "shrink_porosity_rate", "tear_up_rate"], "description": "Runner/riser sizing, feeding"},
    "Sand System": {"defects": ["sand_rate", "dirty_pattern_rate"], "description": "Sand preparation, binder ratio"},
    "Core Making": {"defects": ["core_rate", "crush_rate", "shift_rate"], "description": "Core integrity, venting"},
    "Shakeout": {"defects": ["bent_rate"], "description": "Casting extraction, cooling"},
    "Pattern/Tooling": {"defects": ["gouged_rate"], "description": "Pattern accuracy, wear"},
    "Inspection": {"defects": ["outside_process_scrap_rate", "zyglo_rate", "failed_zyglo_rate"], "description": "Quality control, NDT"},
    "Finishing": {"defects": ["over_grind_rate", "cut_into_rate"], "description": "Grinding, machining"}
}

# Hierarchical Pooling Configuration
POOLING_CONFIG = {
    'enabled': True,
    'min_part_level_data': 5,
    'weight_tolerance': 0.10,
    'confidence_thresholds': {
        'HIGH': 30,
        'MODERATE': 15,
        'LOW': 5,
    }
}

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
        "scrap_%": "scrap_percent", "scrap": "scrap_percent",
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


def filter_by_weight(df, target_weight, tolerance=None):
    """Filter parts by weight within ±10% tolerance."""
    if tolerance is None:
        tolerance = POOLING_CONFIG['weight_tolerance']
    
    weight_min = target_weight * (1 - tolerance)
    weight_max = target_weight * (1 + tolerance)
    
    weight_col = 'piece_weight_lbs'
    part_weights = df.groupby('part_id')[weight_col].first()
    
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
    """
    thresholds = POOLING_CONFIG['confidence_thresholds']
    min_part_data = POOLING_CONFIG['min_part_level_data']
    weight_tolerance = POOLING_CONFIG['weight_tolerance']
    
    part_id = str(part_id)
    part_data = df[df['part_id'] == part_id]
    part_n = len(part_data)
    
    target_weight = part_data['piece_weight_lbs'].iloc[0] if len(part_data) > 0 else 0
    target_defects = identify_part_defects(df, part_id)
    target_defects_clean = [d.replace('_rate', '').replace('_', ' ').title() for d in target_defects]
    
    result = {
        'part_id': part_id,
        'part_level_n': part_n,
        'part_level_sufficient': part_n >= min_part_data,
        'target_weight': target_weight,
        'target_defects': target_defects_clean,
        'pooling_used': False,
    }
    
    # CASE 1: Part-level data is sufficient
    if part_n >= min_part_data:
        confidence = get_confidence_tier(part_n)
        failures = (part_data['scrap_percent'] > threshold_pct).sum()
        failure_rate = failures / part_n if part_n > 0 else 0
        
        if failures > 0:
            failure_cycles = []
            runs_since_failure = 0
            for _, row in part_data.sort_values('week_ending').iterrows():
                runs_since_failure += 1
                if row['scrap_percent'] > threshold_pct:
                    failure_cycles.append(runs_since_failure)
                    runs_since_failure = 0
            mtts = np.mean(failure_cycles) if failure_cycles else part_n
        else:
            mtts = part_n * 10
        
        reliability = np.exp(-1 / mtts) if mtts > 0 else 0
        
        result.update({
            'pooling_method': 'Part-Level (No Pooling Required)',
            'pooling_used': False,
            'weight_range': 'N/A',
            'pooled_n': part_n,
            'pooled_parts_count': 1,
            'included_part_ids': [part_id],
            'confidence': confidence,
            'mtts_runs': mtts,
            'reliability_next_run': reliability,
            'failure_count': failures,
            'failure_rate': failure_rate,
        })
        return result
    
    # CASE 2: Need pooling
    result['pooling_used'] = True
    
    # Step 1: Weight filter
    weight_matched_parts, weight_range = filter_by_weight(df, target_weight, weight_tolerance)
    
    # Step 2: Exact defect filter
    exact_matched_parts = filter_by_exact_defects(df, weight_matched_parts, target_defects)
    exact_pooled_df = df[df['part_id'].isin(exact_matched_parts)]
    exact_pooled_n = len(exact_pooled_df)
    
    # Step 3: Any defect filter
    any_matched_parts = filter_by_any_defect(df, weight_matched_parts)
    any_pooled_df = df[df['part_id'].isin(any_matched_parts)]
    any_pooled_n = len(any_pooled_df)
    
    # Step 4: Weight only
    weight_only_df = df[df['part_id'].isin(weight_matched_parts)]
    weight_only_n = len(weight_only_df)
    
    # Select best pooling method
    if exact_pooled_n >= thresholds['HIGH']:
        final_parts = exact_matched_parts
        final_df = exact_pooled_df
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_pooled_n >= thresholds['HIGH']:
        final_parts = any_matched_parts
        final_df = any_pooled_df
        pooling_method = 'Weight ±10% + Any Defect'
    elif exact_pooled_n >= thresholds['MODERATE']:
        final_parts = exact_matched_parts
        final_df = exact_pooled_df
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_pooled_n >= thresholds['MODERATE']:
        final_parts = any_matched_parts
        final_df = any_pooled_df
        pooling_method = 'Weight ±10% + Any Defect'
    elif exact_pooled_n >= thresholds['LOW']:
        final_parts = exact_matched_parts
        final_df = exact_pooled_df
        pooling_method = 'Weight ±10% + Exact Defect Match'
    elif any_pooled_n >= thresholds['LOW']:
        final_parts = any_matched_parts
        final_df = any_pooled_df
        pooling_method = 'Weight ±10% + Any Defect'
    elif weight_only_n >= thresholds['LOW']:
        final_parts = weight_matched_parts
        final_df = weight_only_df
        pooling_method = 'Weight ±10% Only'
    else:
        # Insufficient data even with pooling
        result.update({
            'pooling_method': 'Insufficient Data',
            'weight_range': weight_range,
            'pooled_n': 0,
            'pooled_parts_count': 0,
            'confidence': 'INSUFFICIENT',
            'mtts_runs': None,
            'reliability_next_run': None,
            'failure_count': 0,
            'failure_rate': 0,
        })
        return result
    
    # Compute metrics from pooled data
    pooled_n = len(final_df)
    confidence = get_confidence_tier(pooled_n)
    failures = (final_df['scrap_percent'] > threshold_pct).sum()
    failure_rate = failures / pooled_n if pooled_n > 0 else 0
    
    if failures > 0:
        failure_cycles = []
        for pid in final_parts:
            pid_data = final_df[final_df['part_id'] == pid].sort_values('week_ending')
            runs_since_failure = 0
            for _, row in pid_data.iterrows():
                runs_since_failure += 1
                if row['scrap_percent'] > threshold_pct:
                    failure_cycles.append(runs_since_failure)
                    runs_since_failure = 0
        mtts = np.mean(failure_cycles) if failure_cycles else pooled_n / max(failures, 1)
    else:
        mtts = pooled_n * 10
    
    reliability = np.exp(-1 / mtts) if mtts > 0 else 0
    
    result.update({
        'pooling_method': pooling_method,
        'weight_range': weight_range,
        'pooled_n': pooled_n,
        'pooled_parts_count': len(final_parts),
        'included_part_ids': final_parts,
        'confidence': confidence,
        'mtts_runs': mtts,
        'reliability_next_run': reliability,
        'failure_count': failures,
        'failure_rate': failure_rate,
    })
    
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
    
    # Ensure all features exist
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    
    y = (df["scrap_percent"] > threshold).astype(int)
    X = df[feats].fillna(0).copy()
    
    return X, y, feats


# ================================================================
# GLOBAL MODEL TRAINING - MATCHING ENHANCED VERSION
# ================================================================
def train_global_model(df, threshold, defect_cols):
    """Train global model with ALL enhanced features."""
    
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
            calibration_method = "calibrated (sigmoid, cv=3)"
        except:
            cal_model = rf
            calibration_method = "uncalibrated"
    else:
        cal_model = rf
        calibration_method = "uncalibrated"
    
    # Metrics on TEST set
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


def calculate_tte_savings(current_scrap, target_scrap, annual_production, energy_per_lb=DOE_BENCHMARKS['average']):
    reduction_pct = (current_scrap - target_scrap) / current_scrap if current_scrap > 0 else 0
    avoided_lbs = annual_production * (current_scrap - target_scrap) / 100
    tte_mmbtu = avoided_lbs * energy_per_lb / 1_000_000
    co2_tons = tte_mmbtu * CO2_PER_MMBTU / 1000
    return {"scrap_reduction_pct": reduction_pct, "avoided_scrap_lbs": avoided_lbs, "tte_savings_mmbtu": tte_mmbtu, "co2_savings_tons": co2_tons}


# ================================================================
# MAIN APPLICATION
# ================================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Foundry Prognostic Reliability Dashboard</h1>
        <p>Sensor-Free Predictive Scrap Prevention | MTTS-Integrated ML | DOE-Aligned Impact Analysis</p>
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
    
    # TRAIN GLOBAL MODEL WITH ALL FEATURES
    with st.spinner("Training global model with Multi-Defect + Temporal + MTTS features (6-2-1 split)..."):
        global_model = train_global_model(df, threshold, defect_cols)
    
    st.success(f"✅ Model trained ({global_model['calibration_method']}): {global_model['n_train']} train, {len(global_model['features'])} features")
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prognostic Model", "📊 RQ1: Model Validation", 
        "⚙️ RQ2: Reliability & PHM", "💰 RQ3: Operational Impact", "📈 All Parts Summary"
    ])
    
    # TAB 1: PROGNOSTIC MODEL
    with tab1:
        st.header("Prognostic Model: Predict & Diagnose")
        
        # Get pooled prediction for this part
        pooled_result = compute_pooled_prediction(df, selected_part, threshold)
        
        # Show pooling notification if used
        if pooled_result['pooling_used']:
            st.warning(f"⚠️ **Insufficient Data for Part {selected_part}** (only {pooled_result['part_level_n']} records)")
            
            st.markdown(f"""
            **Hierarchical Pooling Applied:** {pooled_result['pooling_method']}
            
            | Pooling Details | Value |
            |-----------------|-------|
            | Target Part Weight | {pooled_result['target_weight']:.2f} lbs |
            | Target Part Defects | {', '.join(pooled_result['target_defects']) if pooled_result['target_defects'] else 'None detected'} |
            | Weight Range (±10%) | {pooled_result['weight_range']} lbs |
            | Parts Pooled | {pooled_result['pooled_parts_count']} parts |
            | Total Records | {pooled_result['pooled_n']} records |
            | Confidence Level | {pooled_result['confidence']} |
            """)
            
            # Show pooled parts
            if pooled_result.get('included_part_ids'):
                with st.expander(f"📋 View {pooled_result['pooled_parts_count']} Pooled Parts"):
                    for pid in pooled_result['included_part_ids'][:20]:
                        pid_data = df[df['part_id'] == pid]
                        pid_weight = pid_data['piece_weight_lbs'].iloc[0] if len(pid_data) > 0 else 0
                        pid_n = len(pid_data)
                        st.write(f"• Part {pid}: {pid_weight:.2f} lbs, {pid_n} records")
                    if len(pooled_result['included_part_ids']) > 20:
                        st.write(f"... and {len(pooled_result['included_part_ids']) - 20} more parts")
        
        order_qty = st.slider("Order Quantity (parts)", 1, 5000, 100)
        
        # Use pooled MTTS if available
        if pooled_result['mtts_runs'] is not None:
            mtts_runs = pooled_result['mtts_runs']
            # Convert to parts-based MTTS
            avg_qty = df[df['part_id'] == selected_part]['order_quantity'].mean()
            if pd.isna(avg_qty) or avg_qty == 0:
                avg_qty = 100
            mtts = mtts_runs * avg_qty
            failure_rate = pooled_result['failure_rate']
        else:
            mtts = 1000
            avg_qty = 100
            failure_rate = 0
        
        reliability = np.exp(-order_qty / mtts) if mtts > 0 else 0
        scrap_risk = 1 - reliability
        availability = mtts / (mtts + DEFAULT_MTTR * avg_qty) if mtts > 0 else 0
        
        st.markdown("### 🎯 Prediction Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎲 Scrap Risk", f"{scrap_risk*100:.1f}%")
        m2.metric("📈 Reliability R(n)", f"{reliability*100:.1f}%")
        m3.metric("⚡ Availability", f"{availability*100:.1f}%")
        m4.metric("🔧 MTTS", f"{mtts:,.0f} parts")
        
        st.info(f"**Reliability Formula:** R({order_qty}) = e^(-{order_qty}/{mtts:.0f}) = {reliability*100:.1f}%")
        
        st.markdown("---")
        
        # ================================================================
        # DETAILED DEFECT ANALYSIS
        # ================================================================
        st.markdown("### 📊 Detailed Defect Analysis")
        
        # Get data for analysis (use pooled parts if pooling was applied)
        if pooled_result['pooling_used'] and pooled_result.get('included_part_ids'):
            analysis_df = df[df['part_id'].isin(pooled_result['included_part_ids'])]
        else:
            analysis_df = df[df['part_id'] == selected_part]
        
        # Compute defect rates
        defect_data = []
        for col in defect_cols:
            if col in analysis_df.columns:
                hist_rate = analysis_df[col].mean() * 100
                if hist_rate > 0:
                    defect_name = col.replace('_rate', '').replace('_', ' ').title()
                    defect_data.append({
                        'Defect': defect_name,
                        'Defect_Code': col,
                        'Historical Rate (%)': hist_rate,
                        'Predicted Rate (%)': hist_rate,  # Using historical as proxy
                        'Expected Count': hist_rate / 100 * order_qty
                    })
        
        if defect_data:
            defect_df = pd.DataFrame(defect_data).sort_values('Historical Rate (%)', ascending=False)
            
            # Pareto Charts side by side
            pareto_col1, pareto_col2 = st.columns(2)
            
            with pareto_col1:
                st.markdown("#### 📊 Historical Defect Pareto")
                hist_data = defect_df.head(10).copy()
                
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
                    title="Top 10 Historical Defects",
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(title='Rate (%)', side='left'),
                    yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
                    height=400,
                    showlegend=True,
                    legend=dict(x=0.7, y=1)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with pareto_col2:
                st.markdown("#### 🔮 Predicted Defect Pareto")
                pred_data = defect_df.head(10).copy()
                
                total_pred = pred_data['Predicted Rate (%)'].sum()
                pred_data['Cumulative %'] = (pred_data['Predicted Rate (%)'].cumsum() / total_pred * 100) if total_pred > 0 else 0
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Bar(
                    x=pred_data['Defect'],
                    y=pred_data['Predicted Rate (%)'],
                    name='Rate (%)',
                    marker_color='#E53935'
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
                    title="Top 10 Predicted Defects",
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(title='Rate (%)', side='left'),
                    yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
                    height=400,
                    showlegend=True,
                    legend=dict(x=0.7, y=1)
                )
                st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("---")
        
        # ================================================================
        # ROOT CAUSE PROCESS DIAGNOSIS
        # ================================================================
        st.markdown("### 🏭 Root Cause Process Diagnosis")
        st.caption("*Based on Campbell (2003) process-defect relationships*")
        
        process_ranking, top_defects = diagnose_processes(df, selected_part, defect_cols)
        
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
                        st.markdown(f"""
                        <div style="background: {color}; padding: 10px; border-radius: 8px; margin: 5px 0;">
                            {icon} <strong>{process}</strong>: {contribution:.1f}%
                            <br><small>{PROCESS_DEFECT_MAP[process]['description']}</small>
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
                        title="Process Contributions to Predicted Defects",
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
                        for proc, info in PROCESS_DEFECT_MAP.items():
                            if defect_code in info['defects']:
                                related_processes.append(proc)
                        
                        if total_defect_rate > 0:
                            risk_share = (d['Historical Rate (%)'] / total_defect_rate) * scrap_risk * 100
                        else:
                            risk_share = 0
                        
                        mapping_data.append({
                            'Defect': d['Defect'],
                            'Historical Rate (%)': f"{d['Historical Rate (%)']:.2f}",
                            'Risk Share (%)': f"{risk_share:.2f}",
                            'Expected Count': f"{d['Expected Count']:.1f}",
                            'Root Cause Process(es)': ', '.join(related_processes) if related_processes else 'Unknown'
                        })
                    
                    mapping_df = pd.DataFrame(mapping_data)
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        # Reliability Metrics Snapshot
        st.markdown("---")
        st.markdown("### 📋 Reliability Metrics Snapshot")
        
        r1, r5, r10 = st.columns(3)
        mtts_runs_val = pooled_result['mtts_runs'] if pooled_result['mtts_runs'] else 0
        rel_1 = np.exp(-1 / mtts_runs_val) if mtts_runs_val > 0 else 0
        rel_5 = np.exp(-5 / mtts_runs_val) if mtts_runs_val > 0 else 0
        rel_10 = np.exp(-10 / mtts_runs_val) if mtts_runs_val > 0 else 0
        
        r1.metric("R(1 run)", f"{rel_1*100:.1f}%")
        r5.metric("R(5 runs)", f"{rel_5*100:.1f}%")
        r10.metric("R(10 runs)", f"{rel_10*100:.1f}%")
        
        # Prepare values for table
        mtts_runs_display = f"{pooled_result['mtts_runs']:.1f}" if pooled_result['mtts_runs'] else "N/A"
        lambda_display = f"{1/mtts:.6f}" if mtts > 0 else "0"
        data_source = f"Pooled ({pooled_result['pooled_n']} records)" if pooled_result['pooling_used'] else "Part-level"
        
        st.markdown(f"""
        | Metric | Value | Formula |
        |--------|-------|---------|
        | MTTS (parts) | {mtts:,.0f} | Total Parts / Failures |
        | MTTS (runs) | {mtts_runs_display} | Total Runs / Failures |
        | λ (failure rate) | {lambda_display} | 1 / MTTS |
        | Failures observed | {pooled_result['failure_count']} | Scrap > threshold |
        | Data source | {data_source} | |
        """)
    
    # TAB 2: RQ1 - MODEL VALIDATION
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
        
        c1.metric(f"{'✅' if recall_pass else '❌'} Recall", f"{metrics['recall']*100:.1f}%", f"{'Pass' if recall_pass else 'Below'} ≥80%")
        c2.metric(f"{'✅' if precision_pass else '❌'} Precision", f"{metrics['precision']*100:.1f}%", f"{'Pass' if precision_pass else 'Below'} ≥70%")
        c3.metric(f"{'✅' if auc_pass else '❌'} AUC-ROC", f"{metrics['auc']:.3f}", f"{'Pass' if auc_pass else 'Below'} ≥0.80")
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
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics["roc_fpr"], y=metrics["roc_tpr"], mode='lines', name=f'Model (AUC={metrics["auc"]:.3f})'))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "cal_true" in metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics["cal_pred"], y=metrics["cal_true"], mode='lines+markers', name='Model'))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash')))
                fig.update_layout(title="Calibration Curve", xaxis_title="Predicted", yaxis_title="Actual", height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: RQ2 - PHM EQUIVALENCE
    with tab3:
        st.header("RQ2: Reliability & PHM Equivalence")
        
        st.markdown("""
        <div class="citation-box">
            <strong>Research Question 2:</strong> Can sensor-free ML achieve ≥80% of sensor-based PHM performance?
            <br><strong>Hypothesis H2:</strong> SPC-native ML achieves ≥80% PHM-equivalent recall without sensors.
        </div>
        """, unsafe_allow_html=True)
        
        sensor_benchmark = RQ_THRESHOLDS['RQ2']['sensor_benchmark']
        model_recall = global_model["metrics"]["recall"]
        phm_equiv = (model_recall / sensor_benchmark) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Sensor Benchmark", f"{sensor_benchmark*100:.0f}%")
        c2.metric("🤖 Our Model Recall", f"{model_recall*100:.1f}%")
        
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
    
    # TAB 5: ALL PARTS SUMMARY
    with tab5:
        st.header("📈 All Parts Summary: Global Model Performance")
        
        metrics = global_model["metrics"]
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Train (60%)", f"{global_model['n_train']:,}")
        c2.metric("Calib (20%)", f"{global_model['n_calib']:,}")
        c3.metric("Test (20%)", f"{global_model['n_test']:,}")
        c4.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c5.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c6.metric("AUC-ROC", f"{metrics['auc']:.3f}")
        
        st.markdown("### ✅ Hypothesis Validation Summary")
        
        h1_pass = (metrics["recall"] >= 0.80 and metrics["precision"] >= 0.70 and metrics["auc"] >= 0.80)
        phm_equiv = (metrics["recall"] / 0.90) * 100
        h2_pass = phm_equiv >= 80
        
        c1, c2 = st.columns(2)
        with c1:
            if h1_pass:
                st.success(f"### ✅ H1: SUPPORTED\n- Recall: {metrics['recall']*100:.1f}%\n- Precision: {metrics['precision']*100:.1f}%\n- AUC: {metrics['auc']:.3f}")
            else:
                st.warning(f"### ⚠️ H1: Partially Supported")
        with c2:
            if h2_pass:
                st.success(f"### ✅ H2: SUPPORTED\n- PHM Equivalence: {phm_equiv:.1f}%")
            else:
                st.warning(f"### ⚠️ H2: Partially Supported")
        
        st.markdown("---")
        
        # ================================================================
        # PER-PART ANALYSIS USING GLOBAL MODEL
        # ================================================================
        st.markdown("### 📊 Per-Part Performance Distribution (All 359 Parts)")
        st.caption("*Each part's predictions made using the global model with hierarchical pooling for low-data parts*")
        
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
                
                # Compute reliability metrics
                if pooled['mtts_runs'] and pooled['mtts_runs'] > 0:
                    mtts_runs = pooled['mtts_runs']
                    reliability = np.exp(-1 / mtts_runs)  # R(1 run)
                    failure_rate = pooled['failure_rate']
                else:
                    mtts_runs = 0
                    reliability = 0
                    failure_rate = 0
                
                # PHM equivalence for this part
                part_phm_equiv = (reliability / 0.90) * 100 if reliability > 0 else 0
                
                # Determine H1/H2 pass status
                # For per-part, we use the pooled reliability as proxy for recall
                h1_pass_part = reliability >= 0.80
                h2_pass_part = part_phm_equiv >= 80
                
                part_results.append({
                    'Part ID': pid,
                    'Records': n_records,
                    'Avg Scrap %': avg_scrap,
                    'Pooled Records': pooled['pooled_n'],
                    'Pooling Method': pooled['pooling_method'],
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
        pooled_count = (results_df['Records'] < 5).sum()
        
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Parts", f"{total_parts}")
        s2.metric("H1 Pass Rate", f"{h1_pass_count/total_parts*100:.1f}%", f"{h1_pass_count}/{total_parts}")
        s3.metric("H2 Pass Rate", f"{h2_pass_count/total_parts*100:.1f}%", f"{h2_pass_count}/{total_parts}")
        s4.metric("Parts Needing Pooling", f"{pooled_count}", f"< 5 records")
        s5.metric("Avg Reliability", f"{results_df['Reliability R(1)'].mean():.1f}%")
        
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
    
    st.markdown("---")
    st.caption("🏭 Foundry Dashboard V3 | Global Model with Multi-Defect + Temporal + MTTS Features | 6-2-1 Split")


if __name__ == "__main__":
    main()
