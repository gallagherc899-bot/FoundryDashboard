# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 3.1 - TEMPORAL FEATURES ENHANCEMENT
# ================================================================
# 
# NEW IN V3.1:
# - Temporal Features (rolling averages, trends, seasonality)
# - Based on PHM study findings: +1-4% F1 improvement
# - Optimal threshold: 5.0% (configurable)
#
# RETAINED FROM V3.0:
# - Multi-defect feature engineering (n_defect_types, total_defect_rate)
# - Multi-defect interaction detection
# - Multi-defect alerts in predictions
# - Performance comparison: with vs without multi-defect features
# - 6-2-1 Rolling Window + Data Confidence Indicators
#
# Based on Campbell (2003) "Castings Practice: The Ten Rules"
# PHM Enhancement: Lei et al. (2018) - Temporal degradation patterns
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
    classification_report,
    recall_score,
    precision_score,
    f1_score,
)
from datetime import datetime
import json

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard v3.1 - Temporal Features", 
    layout="wide"
)

# ================================================================
# VERSION BANNER
# ================================================================
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
            padding: 10px 20px; border-radius: 10px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">🏭 Foundry Scrap Risk Dashboard</h2>
    <p style="color: #a8d0ff; margin: 5px 0 0 0;">
        <strong>Version 3.1 - Temporal Features Enhancement</strong> | 
        6-2-1 Rolling Window | Campbell Process Mapping | PHM Optimized
    </p>
</div>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 5.0  # Optimal threshold based on PHM study
MIN_SAMPLES_LEAF = 2
MIN_SAMPLES_PER_CLASS = 5

# ================================================================
# DATA CONFIDENCE CONFIGURATION
# ================================================================
RECOMMENDED_MIN_RECORDS = 5
HIGH_CONFIDENCE_RECORDS = 15
DATA_CONFIDENCE_COLORS = {
    'high': '#28a745',
    'medium': '#ffc107',
    'low': '#dc3545'
}

# ================================================================
# ROLLING WINDOW CONFIGURATION
# ================================================================
ROLLING_WINDOW_ENABLED = True
RETRAIN_THRESHOLD = 50
WINDOW_TRAIN_RATIO = 0.6
WINDOW_CALIB_RATIO = 0.2
WINDOW_TEST_RATIO = 0.2
OUTCOMES_FILE = "prediction_outcomes.csv"

# ================================================================
# MULTI-DEFECT CONFIGURATION (FROM V3.0)
# ================================================================
MULTI_DEFECT_THRESHOLD = 2  # Alert when >= this many defect types
MULTI_DEFECT_FEATURES_ENABLED = True  # Toggle for comparison

# ================================================================
# TEMPORAL FEATURES CONFIGURATION (NEW IN V3.1)
# Based on PHM study: Temporal Features provide +1-4% F1 improvement
# ================================================================
TEMPORAL_FEATURES_ENABLED = True  # Master toggle for temporal features
ROLLING_WINDOW_SIZE = 3  # Number of periods for rolling average

# ================================================================
# CAMPBELL PROCESS-DEFECT MAPPING
# Based on Campbell (2003) "Castings Practice: The Ten Rules"
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

# Create reverse mapping: defect -> processes
DEFECT_TO_PROCESS = {}
for process, info in PROCESS_DEFECT_MAP.items():
    for defect in info["defects"]:
        if defect not in DEFECT_TO_PROCESS:
            DEFECT_TO_PROCESS[defect] = []
        DEFECT_TO_PROCESS[defect].append(process)


# ================================================================
# MULTI-DEFECT FEATURE ENGINEERING (NEW IN V3.0)
# ================================================================
def add_multi_defect_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-defect intelligence features to the dataframe.
    
    NEW FEATURES:
    - n_defect_types: Count of defect types present (rate > 0)
    - has_multiple_defects: Binary flag (1 if >= 2 defect types)
    - total_defect_rate: Sum of all defect rates
    - max_defect_rate: Maximum single defect rate
    - defect_concentration: How concentrated defects are (max/total)
    - shift_x_tearup: Interaction term for common co-occurring defects
    - shrink_x_porosity: Interaction term for metallurgical defects
    """
    df = df.copy()
    
    # Identify defect columns
    defect_cols = [c for c in df.columns if c.endswith('_rate')]
    
    if len(defect_cols) == 0:
        return df
    
    # Ensure numeric
    for col in defect_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Count of defect types present
    df['n_defect_types'] = (df[defect_cols] > 0).sum(axis=1)
    
    # Binary flag for multiple defects
    df['has_multiple_defects'] = (df['n_defect_types'] >= MULTI_DEFECT_THRESHOLD).astype(int)
    
    # Total defect burden
    df['total_defect_rate'] = df[defect_cols].sum(axis=1)
    
    # Maximum single defect rate
    df['max_defect_rate'] = df[defect_cols].max(axis=1)
    
    # Defect concentration (high = concentrated in few types, low = spread across many)
    df['defect_concentration'] = df['max_defect_rate'] / (df['total_defect_rate'] + 0.001)
    
    # Interaction terms for common defect pairs
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
# TEMPORAL FEATURES (NEW IN V3.1)
# Based on PHM study: Provides +1-4% F1 improvement
# ================================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for trend detection and seasonality.
    PHM Enhancement based on Lei et al. (2018) degradation modeling.
    """
    df = df.copy()
    
    # Ensure sorted by time
    if 'week_ending' in df.columns:
        df = df.sort_values('week_ending').reset_index(drop=True)
    
    # Add trend features for key metrics
    for col in ['total_defect_rate', 'scrap_percent']:
        if col not in df.columns:
            continue
        
        # Per-part trend (rate of change)
        if 'part_id' in df.columns:
            df[f'{col}_trend'] = df.groupby('part_id')[col].diff().fillna(0)
            # Rolling average (smoothed signal)
            df[f'{col}_roll3'] = df.groupby('part_id')[col].transform(
                lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
            )
        else:
            df[f'{col}_trend'] = df[col].diff().fillna(0)
            df[f'{col}_roll3'] = df[col].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    
    # Seasonal features (month, quarter)
    if 'week_ending' in df.columns:
        df['month'] = pd.to_datetime(df['week_ending']).dt.month
        df['quarter'] = pd.to_datetime(df['week_ending']).dt.quarter
    
    return df


def get_multi_defect_analysis(df_row: pd.Series, defect_cols: list) -> dict:
    """
    Analyze multi-defect patterns for a single row.
    Returns detailed breakdown for display.
    """
    # Get non-zero defects
    active_defects = []
    for col in defect_cols:
        if col in df_row.index and df_row[col] > 0:
            active_defects.append({
                'defect': col,
                'rate': df_row[col],
                'processes': DEFECT_TO_PROCESS.get(col, ['Unknown'])
            })
    
    # Sort by rate descending
    active_defects.sort(key=lambda x: x['rate'], reverse=True)
    
    # Analyze patterns
    n_defects = len(active_defects)
    total_rate = sum(d['rate'] for d in active_defects)
    
    # Find process overlaps
    all_processes = []
    for d in active_defects:
        all_processes.extend(d['processes'])
    
    from collections import Counter
    process_counts = Counter(all_processes)
    
    return {
        'n_defect_types': n_defects,
        'active_defects': active_defects,
        'total_rate': total_rate,
        'is_multi_defect': n_defects >= MULTI_DEFECT_THRESHOLD,
        'process_overlap': process_counts,
        'primary_processes': [p for p, c in process_counts.most_common(3)]
    }


def display_multi_defect_alert(analysis: dict):
    """Display multi-defect alert banner if applicable."""
    if not analysis['is_multi_defect']:
        return
    
    n_defects = analysis['n_defect_types']
    active = analysis['active_defects']
    
    # Create defect list
    defect_list = ", ".join([
        f"{d['defect'].replace('_rate', '').replace('_', ' ').title()} ({d['rate']*100:.1f}%)"
        for d in active[:5]
    ])
    
    # Find common processes
    common_processes = analysis['primary_processes']
    
    st.error(f"""
🚨 **MULTI-DEFECT ALERT: {n_defects} Defect Types Detected**

**Active Defects:** {defect_list}

**Common Root Cause Processes:** {', '.join(common_processes)}

⚠️ **Interpretation:** Multiple concurrent defects on a single work order often indicate 
systemic process issues rather than isolated incidents. When defects like Shift and Tear-Up 
co-occur (as in your data), they typically affect the **same casting**, compounding scrap risk.

💡 **Recommended Action:** Focus on the overlapping processes ({', '.join(common_processes[:2])}) 
as they likely share a common root cause.
    """)


# ================================================================
# DATA CONFIDENCE FUNCTIONS
# ================================================================
def get_data_confidence_level(n_records):
    """Determine confidence level based on number of records."""
    if n_records >= HIGH_CONFIDENCE_RECORDS:
        return 'high', 'HIGH', DATA_CONFIDENCE_COLORS['high']
    elif n_records >= RECOMMENDED_MIN_RECORDS:
        return 'medium', 'MEDIUM', DATA_CONFIDENCE_COLORS['medium']
    else:
        return 'low', 'LOW', DATA_CONFIDENCE_COLORS['low']


def get_confidence_percentage(n_records):
    """Calculate confidence percentage (0-100) based on records."""
    if n_records >= HIGH_CONFIDENCE_RECORDS:
        return 100
    else:
        return min(100, (n_records / HIGH_CONFIDENCE_RECORDS) * 100)


def display_data_confidence_banner(n_records, part_id):
    """Display appropriate confidence banner based on data availability."""
    level, level_text, color = get_data_confidence_level(n_records)
    confidence_pct = get_confidence_percentage(n_records)
    
    if level == 'low':
        st.warning(f"""
⚠️ **DATA NOTICE: LOW CONFIDENCE**

This prediction for **Part {part_id}** is based on only **{n_records} historical run(s)**.

- Recommended minimum: {RECOMMENDED_MIN_RECORDS} runs
- High confidence threshold: {HIGH_CONFIDENCE_RECORDS} runs

**Interpret this prediction with caution.** The model uses similar parts data to supplement, 
but part-specific predictions improve significantly with more historical data.

💡 *Log outcomes after production runs to build part history and improve future predictions.*
        """)
    elif level == 'medium':
        st.info(f"""
📊 **DATA NOTICE: MEDIUM CONFIDENCE**

This prediction for **Part {part_id}** is based on **{n_records} historical runs**.

- Meets minimum threshold: ✅ ({RECOMMENDED_MIN_RECORDS} runs)
- High confidence threshold: {HIGH_CONFIDENCE_RECORDS} runs

Prediction reliability is acceptable but will improve with additional historical data.
        """)
    else:
        st.success(f"""
✅ **DATA CONFIDENCE: HIGH**

This prediction for **Part {part_id}** is based on **{n_records} historical runs**.

Sufficient historical data exists for reliable ML-based prediction.
        """)
    
    return level, n_records


def display_confidence_meter(n_records):
    """Display a visual confidence meter."""
    confidence_pct = get_confidence_percentage(n_records)
    level, level_text, color = get_data_confidence_level(n_records)
    
    filled_blocks = int(confidence_pct / 5)
    empty_blocks = 20 - filled_blocks
    
    if level == 'high':
        bar_color = "🟩"
    elif level == 'medium':
        bar_color = "🟨"
    else:
        bar_color = "🟥"
    
    progress_bar = bar_color * filled_blocks + "⬜" * empty_blocks
    
    st.markdown(f"""
**📊 Data Confidence: {level_text}** ({n_records} of {HIGH_CONFIDENCE_RECORDS} recommended runs)

{progress_bar} {confidence_pct:.0f}%
    """)


# -------------------------------
# Utilities
# -------------------------------
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
    """Rename columns to canonical names."""
    rename_map = {
        "part_id": "part_id",
        "work_order": "work_order",
        "work_order_number": "work_order",
        "work_order_#": "work_order",
        "work_order_num": "work_order",
        "order_quantity": "order_quantity",
        "order_qty": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "scrap%": "scrap_percent",
        "scrap_percent": "scrap_percent",
        "scrap": "scrap_percent",
        "piece_weight_lbs": "piece_weight_lbs",
        "piece_weight": "piece_weight_lbs",
        "week_ending": "week_ending",
        "week_end": "week_ending",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


# -------------------------------
# Data loading and cleaning
# -------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str, add_multi_defect: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != "None"]).strip() 
                      for col in df.columns.values]

    df.columns = _normalize_headers(df.columns)
    df = _canonical_rename(df)

    if df.columns.duplicated().any():
        st.warning(f"⚠️ Detected {df.columns.duplicated().sum()} duplicate column names - keeping first occurrence")
        df = df.loc[:, ~df.columns.duplicated()]
    
    for c in ["part_id", "order_quantity", "piece_weight_lbs", "week_ending", "scrap_percent"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "week_ending" else pd.NaT

    if "part_id" in df.columns:
        if isinstance(df["part_id"], pd.DataFrame):
            df["part_id"] = df["part_id"].iloc[:, 0]
        df["part_id"] = df["part_id"].fillna("Unknown").astype(str)
        df["part_id"] = df["part_id"].str.strip()
        df["part_id"] = df["part_id"].replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
    
    if "work_order" in df.columns:
        if isinstance(df["work_order"], pd.DataFrame):
            df["work_order"] = df["work_order"].iloc[:, 0]
        df["work_order"] = df["work_order"].fillna("Unknown").astype(str)
        df["work_order"] = df["work_order"].str.strip()
    
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(0)
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce").fillna(0.0)
    
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in defect_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)
    df = df.sort_values("week_ending").reset_index(drop=True)

    # ADD MULTI-DEFECT FEATURES (FROM V3.0)
    if add_multi_defect:
        df = add_multi_defect_features(df)
    
    # ADD TEMPORAL FEATURES (NEW IN V3.1)
    if TEMPORAL_FEATURES_ENABLED:
        df = add_temporal_features(df)

    n_parts = df["part_id"].nunique() if "part_id" in df.columns else 0
    n_work_orders = df["work_order"].nunique() if "work_order" in df.columns else len(df)
    temporal_status = "ON" if TEMPORAL_FEATURES_ENABLED else "OFF"
    st.info(f"✅ Loaded {len(df):,} rows | {n_parts} unique parts | {n_work_orders} work orders | {len(defect_cols)} defect columns | Temporal features: {temporal_status}")
    return df


# ================================================================
# OUTCOME LOGGING FOR ROLLING WINDOW
# ================================================================
def load_outcomes(filepath=OUTCOMES_FILE):
    """Load previously logged prediction outcomes."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()


def save_outcome(outcome_data: dict, filepath=OUTCOMES_FILE):
    """Save a new prediction outcome."""
    df_existing = load_outcomes(filepath)
    
    outcome_data['timestamp'] = datetime.now().isoformat()
    df_new = pd.DataFrame([outcome_data])
    
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(filepath, index=False)
    
    return len(df_combined)


def get_outcomes_count():
    """Get count of logged outcomes since last retrain."""
    df = load_outcomes()
    if df.empty:
        return 0
    if 'retrain_marker' in df.columns:
        last_retrain = df[df['retrain_marker'] == True]
        if not last_retrain.empty:
            last_retrain_idx = last_retrain.index[-1]
            return len(df) - last_retrain_idx - 1
    return len(df)


def mark_retrain():
    """Mark that retraining has occurred."""
    df = load_outcomes()
    if not df.empty:
        marker = {
            'timestamp': datetime.now().isoformat(),
            'retrain_marker': True,
            'part_id': 'RETRAIN_MARKER',
            'predicted_scrap': 0,
            'actual_scrap': 0
        }
        df_new = pd.DataFrame([marker])
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv(OUTCOMES_FILE, index=False)


# ================================================================
# 6-2-1 TEMPORAL SPLIT
# ================================================================
def time_split_621(df: pd.DataFrame, train_ratio=0.6, calib_ratio=0.2):
    """TRUE 6-2-1 Temporal Split."""
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:calib_end], df_sorted.iloc[calib_end:]


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    """Compute Mean Time To Failure proxy for each PART from training data only."""
    grp = df_train.groupby("part_id", dropna=False)["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp


def attach_train_features(df_sub: pd.DataFrame, mtbf_train: pd.DataFrame, 
                          part_freq_train: pd.Series, default_mtbf: float,
                          default_freq: float) -> pd.DataFrame:
    """Attach MTTF and part frequency features computed from training data."""
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s


def make_xy(df: pd.DataFrame, thr_label: float, use_rate_cols: bool, use_multi_defect: bool = True, use_temporal: bool = True):
    """Prepare features (X) and labels (y)."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    # Add multi-defect features (FROM V3.0)
    if use_multi_defect:
        multi_defect_feats = [
            "n_defect_types", "has_multiple_defects", "total_defect_rate",
            "max_defect_rate", "defect_concentration",
            "shift_x_tearup", "shrink_x_porosity", "shrink_x_shrink_porosity", "core_x_sand"
        ]
        for f in multi_defect_feats:
            if f in df.columns:
                feats.append(f)
    
    # Add temporal features (NEW IN V3.1)
    if use_temporal and TEMPORAL_FEATURES_ENABLED:
        temporal_feats = [
            "total_defect_rate_trend", "total_defect_rate_roll3",
            "scrap_percent_trend", "scrap_percent_roll3",
            "month", "quarter"
        ]
        for f in temporal_feats:
            if f in df.columns:
                feats.append(f)
    
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]

    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    y = (df["scrap_percent"] > thr_label).astype(int)
    X = df[feats].copy()
    return X, y, feats


def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators):
    """Train Random Forest and calibrate with Platt scaling."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    pos = int(y_calib.sum())
    neg = int((y_calib == 0).sum())
    
    if pos < 3 or neg < 3:
        return rf, rf, "uncalibrated (insufficient calibration samples)"
    
    try:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
        return rf, cal, "calibrated (sigmoid, cv=3)"
    except ValueError as e:
        return rf, rf, f"uncalibrated ({e})"


# ================================================================
# ROLLING WINDOW MODEL TRAINING
# ================================================================
def train_model_with_rolling_window(df_base, df_outcomes, thr_label, use_rate_cols, n_est, use_multi_defect=True):
    """Train model using 6-2-1 rolling window on combined base + outcome data."""
    
    if df_outcomes is not None and len(df_outcomes) > 0:
        df_outcomes_valid = df_outcomes[df_outcomes['part_id'] != 'RETRAIN_MARKER'].copy()
        
        if len(df_outcomes_valid) > 0:
            if 'actual_scrap' in df_outcomes_valid.columns and 'scrap_percent' not in df_outcomes_valid.columns:
                df_outcomes_valid['scrap_percent'] = df_outcomes_valid['actual_scrap']
            
            if 'timestamp' in df_outcomes_valid.columns and 'week_ending' not in df_outcomes_valid.columns:
                df_outcomes_valid['week_ending'] = pd.to_datetime(df_outcomes_valid['timestamp'])
            
            if 'part_id' in df_outcomes_valid.columns and 'scrap_percent' in df_outcomes_valid.columns:
                for col in df_base.columns:
                    if col not in df_outcomes_valid.columns:
                        df_outcomes_valid[col] = 0.0 if col != 'week_ending' else pd.NaT
                
                df_combined = pd.concat([df_base, df_outcomes_valid[df_base.columns]], ignore_index=True)
                df_combined = df_combined.sort_values('week_ending').reset_index(drop=True)
                
                st.success(f"🔄 Rolling Window: Combined {len(df_base)} base + {len(df_outcomes_valid)} outcome records = {len(df_combined)} total")
            else:
                df_combined = df_base.copy()
        else:
            df_combined = df_base.copy()
    else:
        df_combined = df_base.copy()
    
    df_train, df_calib, df_test = time_split_621(df_combined)
    
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols, use_multi_defect)
    X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols, use_multi_defect)
    rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)
    
    return rf, cal_model, method, feats, df_train, df_calib, df_test, mtbf_train, part_freq_train, default_mtbf, default_freq


# ================================================================
# MODEL COMPARISON (NEW IN V3.0)
# ================================================================
def compare_models_with_without_multi_defect(df_base, thr_label, use_rate_cols, n_est):
    """
    Compare model performance WITH vs WITHOUT multi-defect features.
    Returns comparison metrics for display.
    """
    results = {}
    
    # Model WITHOUT multi-defect features
    df_train, df_calib, df_test = time_split_621(df_base)
    
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    df_train_f = attach_train_features(df_train.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib_f = attach_train_features(df_calib.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test_f = attach_train_features(df_test.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # WITHOUT multi-defect
    X_train_no, y_train_no, feats_no = make_xy(df_train_f.copy(), thr_label, use_rate_cols, use_multi_defect=False)
    X_calib_no, y_calib_no, _ = make_xy(df_calib_f.copy(), thr_label, use_rate_cols, use_multi_defect=False)
    X_test_no, y_test_no, _ = make_xy(df_test_f.copy(), thr_label, use_rate_cols, use_multi_defect=False)
    
    _, cal_no, _ = train_and_calibrate(X_train_no, y_train_no, X_calib_no, y_calib_no, n_est)
    
    preds_no = cal_no.predict_proba(X_test_no)[:, 1]
    pred_binary_no = (preds_no > 0.5).astype(int)
    
    results['without'] = {
        'brier': brier_score_loss(y_test_no, preds_no),
        'accuracy': accuracy_score(y_test_no, pred_binary_no),
        'recall': recall_score(y_test_no, pred_binary_no, zero_division=0),
        'precision': precision_score(y_test_no, pred_binary_no, zero_division=0),
        'f1': f1_score(y_test_no, pred_binary_no, zero_division=0),
        'n_features': len(feats_no)
    }
    
    # WITH multi-defect
    X_train_yes, y_train_yes, feats_yes = make_xy(df_train_f.copy(), thr_label, use_rate_cols, use_multi_defect=True)
    X_calib_yes, y_calib_yes, _ = make_xy(df_calib_f.copy(), thr_label, use_rate_cols, use_multi_defect=True)
    X_test_yes, y_test_yes, _ = make_xy(df_test_f.copy(), thr_label, use_rate_cols, use_multi_defect=True)
    
    _, cal_yes, _ = train_and_calibrate(X_train_yes, y_train_yes, X_calib_yes, y_calib_yes, n_est)
    
    preds_yes = cal_yes.predict_proba(X_test_yes)[:, 1]
    pred_binary_yes = (preds_yes > 0.5).astype(int)
    
    results['with'] = {
        'brier': brier_score_loss(y_test_yes, preds_yes),
        'accuracy': accuracy_score(y_test_yes, pred_binary_yes),
        'recall': recall_score(y_test_yes, pred_binary_yes, zero_division=0),
        'precision': precision_score(y_test_yes, pred_binary_yes, zero_division=0),
        'f1': f1_score(y_test_yes, pred_binary_yes, zero_division=0),
        'n_features': len(feats_yes)
    }
    
    # Calculate improvements
    results['improvement'] = {
        'brier': (results['without']['brier'] - results['with']['brier']) / results['without']['brier'] * 100,
        'accuracy': (results['with']['accuracy'] - results['without']['accuracy']) * 100,
        'recall': (results['with']['recall'] - results['without']['recall']) * 100,
        'precision': (results['with']['precision'] - results['without']['precision']) * 100,
        'f1': (results['with']['f1'] - results['without']['f1']) * 100,
        'n_features': results['with']['n_features'] - results['without']['n_features']
    }
    
    return results


# ================================================================
# PROCESS ROOT CAUSE DIAGNOSIS
# ================================================================
def calculate_process_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Campbell process indices from defect rates"""
    df = df.copy()
    
    for process, info in PROCESS_DEFECT_MAP.items():
        present_cols = [c for c in info["defects"] if c in df.columns]
        if present_cols:
            df[f"{process}_index"] = df[present_cols].mean(axis=1)
        else:
            df[f"{process}_index"] = 0.0
    
    return df


def diagnose_root_causes(defect_predictions: pd.DataFrame) -> pd.DataFrame:
    """Map predicted defects to root cause processes."""
    if defect_predictions.empty:
        return pd.DataFrame(columns=["Process", "Contribution (%)", "Description"])
    
    process_scores = {}
    
    for _, row in defect_predictions.iterrows():
        defect = row.get("Defect_Code", "")
        likelihood = row.get("Predicted Rate (%)", 0.0)
        
        if defect in DEFECT_TO_PROCESS:
            processes = DEFECT_TO_PROCESS[defect]
            contribution = likelihood / len(processes) if len(processes) > 0 else 0.0
            
            for process in processes:
                if process not in process_scores:
                    process_scores[process] = 0.0
                process_scores[process] += contribution
    
    if not process_scores:
        return pd.DataFrame(columns=["Process", "Contribution (%)", "Description"])
    
    diagnosis = pd.DataFrame([
        {
            "Process": process,
            "Contribution (%)": score,
            "Description": PROCESS_DEFECT_MAP.get(process, {}).get("description", "")
        }
        for process, score in process_scores.items()
    ])
    
    if not diagnosis.empty:
        diagnosis = diagnosis.sort_values("Contribution (%)", ascending=False)
    
    return diagnosis


def create_pareto_chart(data: pd.DataFrame, value_col: str, label_col: str, title: str):
    """Create interactive Pareto chart with cumulative percentage"""
    data = data.sort_values(value_col, ascending=False).head(10)
    
    total = data[value_col].sum()
    data["Cumulative %"] = (data[value_col].cumsum() / total * 100) if total > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data[label_col],
        y=data[value_col],
        name=value_col,
        marker_color='steelblue',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=data[label_col],
        y=data["Cumulative %"],
        name='Cumulative %',
        marker_color='red',
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=label_col, tickangle=-45),
        yaxis=dict(title=value_col, side='left'),
        yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
        legend=dict(x=0.7, y=1),
        height=400,
        hovermode='x unified'
    )
    
    return fig


# ================================================================
# DYNAMIC PART-SPECIFIC DATA PREPARATION
# ================================================================
def prepare_part_specific_data(df_full: pd.DataFrame, target_part: str, 
                                piece_weight: float, thr_label: float, 
                                min_samples: int = 30):
    """Prepare part-specific dataset with similarity-based expansion."""
    st.info(f"🔍 Preparing data for Part {target_part}...")
    
    df_part = df_full[df_full["part_id"] == target_part].copy()
    
    if len(df_part) < min_samples:
        st.info(f"⚠️ Only {len(df_part)} samples for Part {target_part}. Expanding search...")
        
        weight_tolerance = 0.1
        max_tolerance = 0.5
        
        while len(df_part) < min_samples and weight_tolerance <= max_tolerance:
            lower = piece_weight * (1 - weight_tolerance)
            upper = piece_weight * (1 + weight_tolerance)
            
            df_similar = df_full[
                (df_full["piece_weight_lbs"] >= lower) & 
                (df_full["piece_weight_lbs"] <= upper)
            ].copy()
            
            df_part = df_similar.copy()
            weight_tolerance += 0.1
        
        st.info(f"✅ Found {len(df_part)} similar samples (±{(weight_tolerance-0.1)*100:.0f}% weight tolerance)")
    
    df_part["temp_label"] = (df_part["scrap_percent"] > thr_label).astype(int)
    label_counts = df_part["temp_label"].value_counts()
    
    if len(label_counts) < 2 or label_counts.min() < MIN_SAMPLES_PER_CLASS:
        st.warning(f"⚠️ Insufficient label diversity. Expanding to broader dataset...")
        lower = piece_weight * 0.5
        upper = piece_weight * 1.5
        df_part = df_full[
            (df_full["piece_weight_lbs"] >= lower) & 
            (df_full["piece_weight_lbs"] <= upper)
        ].copy()
        df_part["temp_label"] = (df_part["scrap_percent"] > thr_label).astype(int)
        label_counts = df_part["temp_label"].value_counts()
        
        if len(label_counts) < 2 or label_counts.min() < MIN_SAMPLES_PER_CLASS:
            st.error("❌ Cannot find sufficient label diversity even with broad search.")
            return None
    
    df_part = df_part.drop(columns=["temp_label"])
    
    st.success(f"✅ Dataset prepared: {len(df_part)} samples, Labels: {label_counts.to_dict()}")
    return df_part


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

st.sidebar.header("🧬 Multi-Defect Intelligence (V3.0)")
use_multi_defect = st.sidebar.checkbox("Enable Multi-Defect Features", True)
st.sidebar.caption("Adds n_defect_types, interaction terms, and alerts")

st.sidebar.header("🔄 Rolling Window")
rolling_enabled = st.sidebar.checkbox("Enable 6-2-1 Rolling Window", ROLLING_WINDOW_ENABLED)
if rolling_enabled:
    retrain_thresh = st.sidebar.number_input("Retrain after N outcomes", min_value=10, max_value=200, value=RETRAIN_THRESHOLD)
    outcomes_count = get_outcomes_count()
    st.sidebar.metric("Outcomes since last retrain", outcomes_count)
    if outcomes_count >= retrain_thresh:
        st.sidebar.warning(f"⚠️ {outcomes_count} outcomes logged - retrain recommended!")

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()


# -------------------------------
# Load and prepare data
# -------------------------------
df_base = load_and_clean(csv_path, add_multi_defect=use_multi_defect)
df_base = calculate_process_indices(df_base)

df_outcomes = load_outcomes() if rolling_enabled else None

# Display data summary in sidebar
st.sidebar.header("📊 Data Summary")
n_parts = df_base["part_id"].nunique()
n_work_orders = df_base["work_order"].nunique() if "work_order" in df_base.columns else len(df_base)
avg_runs_per_part = len(df_base) / n_parts if n_parts > 0 else 0

st.sidebar.metric("Unique Parts", f"{n_parts:,}")
st.sidebar.metric("Work Orders", f"{n_work_orders:,}")
st.sidebar.metric("Avg Runs/Part", f"{avg_runs_per_part:.1f}")

# Multi-defect stats
if use_multi_defect and 'n_defect_types' in df_base.columns:
    multi_defect_pct = (df_base['has_multiple_defects'].sum() / len(df_base) * 100)
    st.sidebar.metric("Multi-Defect Work Orders", f"{multi_defect_pct:.1f}%")

st.sidebar.caption("ℹ️ Model analyzes by Part ID, not Work Order")

# Train model with rolling window
(rf_base, cal_model_base, method_base, feats_base, 
 df_train_base, df_calib_base, df_test_base,
 mtbf_train_base, part_freq_train_base, 
 default_mtbf_base, default_freq_base) = train_model_with_rolling_window(
    df_base, df_outcomes, thr_label, use_rate_cols, n_est, use_multi_defect
)

feature_label = "with Multi-Defect features" if use_multi_defect else "without Multi-Defect features"
if rolling_enabled:
    st.success(f"✅ Model trained ({feature_label}) with 6-2-1 Rolling Window: {method_base}, {len(df_train_base)} train samples, {len(feats_base)} features")
else:
    st.success(f"✅ Base model loaded ({feature_label}): {method_base}, {len(df_train_base)} samples, {len(feats_base)} features")


# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict & Diagnose", "📏 Validation", "📊 Model Comparison", "📝 Log Outcome"])

# ================================================================
# TAB 1: PREDICTION & PROCESS DIAGNOSIS
# ================================================================
with tab1:
    st.header("🔮 Scrap Risk Prediction & Root Cause Analysis")
    
    if rolling_enabled:
        st.info("🔄 **Rolling Window Mode**: Model trains on combined historical + logged outcome data using 6-2-1 temporal split")
    
    if use_multi_defect:
        st.info("🧬 **Multi-Defect Intelligence**: Enabled - model includes defect count, interactions, and concentration features")
    
    col1, col2, col3, col4 = st.columns(4)
    part_id_input = col1.text_input("Part ID", value="Unknown")
    order_qty = col2.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = col3.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = col4.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

    if st.button("🎯 Predict Risk & Diagnose Process"):
        try:
            st.cache_data.clear()
            
            st.info("🔄 Retraining model with part-specific dataset...")
            
            # Reload and combine with outcomes
            df_full = load_and_clean(csv_path, add_multi_defect=use_multi_defect)
            df_full = calculate_process_indices(df_full)
            
            # If rolling window, combine with outcomes
            if rolling_enabled:
                df_outcomes_current = load_outcomes()
                if df_outcomes_current is not None and len(df_outcomes_current) > 0:
                    df_outcomes_valid = df_outcomes_current[df_outcomes_current['part_id'] != 'RETRAIN_MARKER'].copy()
                    if len(df_outcomes_valid) > 0 and 'actual_scrap' in df_outcomes_valid.columns:
                        df_outcomes_valid['scrap_percent'] = df_outcomes_valid['actual_scrap']
                        if 'timestamp' in df_outcomes_valid.columns:
                            df_outcomes_valid['week_ending'] = pd.to_datetime(df_outcomes_valid['timestamp'])
                        for col in df_full.columns:
                            if col not in df_outcomes_valid.columns:
                                df_outcomes_valid[col] = 0.0
                        df_full = pd.concat([df_full, df_outcomes_valid[df_full.columns]], ignore_index=True)
                        df_full = df_full.sort_values('week_ending').reset_index(drop=True)
            
            # Count part-specific records
            part_specific_records = len(df_full[df_full["part_id"] == part_id_input])
            
            # Get part history for multi-defect analysis
            part_history = df_full[df_full["part_id"] == part_id_input]
            defect_cols = [c for c in df_full.columns if c.endswith("_rate")]
            
            # MULTI-DEFECT ANALYSIS (NEW IN V3.0)
            if len(part_history) > 0 and use_multi_defect:
                # Get most recent row for analysis
                latest_row = part_history.iloc[-1]
                multi_defect_analysis = get_multi_defect_analysis(latest_row, defect_cols)
                
                if multi_defect_analysis['is_multi_defect']:
                    display_multi_defect_alert(multi_defect_analysis)
            
            df_part = prepare_part_specific_data(
                df_full, 
                part_id_input, 
                piece_weight, 
                thr_label, 
                min_samples=30
            )
            
            if df_part is None:
                st.error("❌ Cannot proceed with prediction - insufficient data diversity")
                st.stop()
            
            # 6-2-1 temporal split
            df_train, df_calib, df_test = time_split_621(df_part)
            
            for split_name, split_df in [("train", df_train), ("calib", df_calib)]:
                split_labels = (split_df["scrap_percent"] > thr_label).astype(int)
                if split_labels.nunique() < 2:
                    st.warning(f"⚠️ {split_name} split lacks diversity. Using stratified split...")
                    from sklearn.model_selection import train_test_split
                    y_stratify = (df_part["scrap_percent"] > thr_label).astype(int)
                    train_temp, test_temp = train_test_split(
                        df_part, test_size=0.4, stratify=y_stratify, random_state=RANDOM_STATE
                    )
                    calib_temp, test_temp = train_test_split(
                        test_temp, test_size=0.5, stratify=(test_temp["scrap_percent"] > thr_label).astype(int),
                        random_state=RANDOM_STATE
                    )
                    df_train, df_calib, df_test = train_temp, calib_temp, test_temp
                    break

            mtbf_train = compute_mtbf_on_train(df_train, thr_label)
            part_freq_train = df_train["part_id"].value_counts(normalize=True)
            default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
            default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
            
            df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
            df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
            
            X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols, use_multi_defect)
            X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols, use_multi_defect)
            
            rf_part, cal_part, method_part = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)
            
            st.success(f"✅ Part-specific model trained: {method_part}, {len(X_train)} samples, {len(feats)} features")
            st.info(f"📊 Training labels: Scrap=1: {y_train.sum()}, Scrap=0: {(y_train == 0).sum()}")
            
            # Create prediction input
            if len(part_history) > 0:
                hist_mttf = float(mtbf_train[mtbf_train["part_id"] == part_id_input]["mttf_scrap"].values[0]) \
                    if part_id_input in mtbf_train["part_id"].values else default_mtbf
                hist_freq = float(part_freq_train.get(part_id_input, default_freq))
                defect_means = part_history[defect_cols].mean()
                
                # Multi-defect features from history
                if use_multi_defect:
                    n_defect_types = (part_history[defect_cols] > 0).sum(axis=1).mean()
                    has_multiple = 1 if n_defect_types >= MULTI_DEFECT_THRESHOLD else 0
                    total_defect = part_history[defect_cols].sum(axis=1).mean()
                    max_defect = part_history[defect_cols].max(axis=1).mean()
                    defect_conc = max_defect / (total_defect + 0.001)
            else:
                hist_mttf = default_mtbf
                hist_freq = default_freq
                defect_means = df_full[defect_cols].mean()
                
                if use_multi_defect:
                    n_defect_types = df_full['n_defect_types'].mean() if 'n_defect_types' in df_full.columns else 0
                    has_multiple = 0
                    total_defect = df_full['total_defect_rate'].mean() if 'total_defect_rate' in df_full.columns else 0
                    max_defect = df_full['max_defect_rate'].mean() if 'max_defect_rate' in df_full.columns else 0
                    defect_conc = max_defect / (total_defect + 0.001)

            input_dict = {
                "order_quantity": order_qty,
                "piece_weight_lbs": piece_weight,
                "mttf_scrap": hist_mttf,
                "part_freq": hist_freq,
            }
            
            # Add multi-defect features
            if use_multi_defect:
                input_dict["n_defect_types"] = n_defect_types
                input_dict["has_multiple_defects"] = has_multiple
                input_dict["total_defect_rate"] = total_defect
                input_dict["max_defect_rate"] = max_defect
                input_dict["defect_concentration"] = defect_conc
                
                # Interaction terms
                if 'shift_rate' in defect_means.index and 'tear_up_rate' in defect_means.index:
                    input_dict["shift_x_tearup"] = defect_means['shift_rate'] * defect_means['tear_up_rate']
                if 'shrink_rate' in defect_means.index and 'gas_porosity_rate' in defect_means.index:
                    input_dict["shrink_x_porosity"] = defect_means['shrink_rate'] * defect_means['gas_porosity_rate']
                if 'shrink_rate' in defect_means.index and 'shrink_porosity_rate' in defect_means.index:
                    input_dict["shrink_x_shrink_porosity"] = defect_means['shrink_rate'] * defect_means['shrink_porosity_rate']
                if 'core_rate' in defect_means.index and 'sand_rate' in defect_means.index:
                    input_dict["core_x_sand"] = defect_means['core_rate'] * defect_means['sand_rate']
            
            for dc in defect_cols:
                input_dict[dc] = defect_means.get(dc, 0.0)

            X_input = pd.DataFrame([input_dict])[feats]
            
            proba = cal_part.predict_proba(X_input)[0, 1]
            scrap_risk = proba * 100
            reliability = (1 - proba) * 100
            expected_scrap_pcs = order_qty * proba
            expected_loss = expected_scrap_pcs * cost_per_part

            # Store prediction for potential outcome logging
            st.session_state['last_prediction'] = {
                'part_id': part_id_input,
                'order_quantity': order_qty,
                'piece_weight_lbs': piece_weight,
                'predicted_scrap': scrap_risk,
                'cost_per_part': cost_per_part,
                'part_records': part_specific_records,
                'multi_defect_enabled': use_multi_defect
            }

            # Display data confidence banner
            st.markdown("---")
            display_data_confidence_banner(part_specific_records, part_id_input)
            display_confidence_meter(part_specific_records)
            
            # Display results
            st.markdown(f"### 🎯 Risk Assessment for Part {part_id_input}")
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Scrap Risk", f"{scrap_risk:.1f}%")
            r2.metric("Expected Scrap", f"{expected_scrap_pcs:.1f} pieces")
            r3.metric("Expected Loss", f"${expected_loss:,.2f}")
            r4.metric("Reliability", f"{reliability:.1f}%")

            # Multi-defect summary (NEW IN V3.0)
            if use_multi_defect:
                st.markdown("#### 🧬 Multi-Defect Intelligence Summary")
                md_col1, md_col2, md_col3, md_col4 = st.columns(4)
                md_col1.metric("Avg Defect Types", f"{n_defect_types:.1f}")
                md_col2.metric("Multi-Defect Pattern", "Yes" if has_multiple else "No")
                md_col3.metric("Total Defect Rate", f"{total_defect:.2f}")
                md_col4.metric("Defect Concentration", f"{defect_conc:.2f}")

            # Data source explanation
            with st.expander("📋 Data Sources Used for This Prediction"):
                st.markdown(f"""
**Part-Specific Data:**
- Historical runs for Part {part_id_input}: **{part_specific_records}**
- Defect rates based on: {'Part-specific history' if part_specific_records > 0 else 'Dataset average (no part history)'}

**Model Features ({len(feats)} total):**
- Base features: order_quantity, piece_weight_lbs, mttf_scrap, part_freq
- Defect rate features: {len(defect_cols)} columns
- Multi-defect features: {'Enabled' if use_multi_defect else 'Disabled'}

**Confidence Assessment:**
- Data confidence: **{get_data_confidence_level(part_specific_records)[1]}**
- Recommended minimum: {RECOMMENDED_MIN_RECORDS} runs
- High confidence threshold: {HIGH_CONFIDENCE_RECORDS} runs
                """)

            # Defect analysis
            if defect_cols:
                st.markdown("---")
                st.markdown("### 📋 Detailed Defect Analysis")
                
                defect_predictions = []
                for dc in defect_cols:
                    hist_rate = defect_means.get(dc, 0.0)
                    pred_rate = hist_rate * proba
                    expected_count = order_qty * pred_rate
                    
                    defect_name = dc.replace("_rate", "").replace("_", " ").title()
                    defect_predictions.append({
                        "Defect": defect_name,
                        "Defect_Code": dc,
                        "Historical Rate (%)": hist_rate * 100,
                        "Predicted Rate (%)": pred_rate * 100,
                        "Expected Count": expected_count
                    })
                
                defect_df = pd.DataFrame(defect_predictions).sort_values("Predicted Rate (%)", ascending=False)
                
                # Pareto charts
                pareto_col1, pareto_col2 = st.columns(2)
                
                with pareto_col1:
                    st.markdown("#### 📊 Historical Defect Pareto")
                    hist_data = defect_df[["Defect", "Historical Rate (%)"]].copy()
                    hist_chart = create_pareto_chart(
                        hist_data, 
                        "Historical Rate (%)", 
                        "Defect",
                        "Top 10 Historical Defects"
                    )
                    st.plotly_chart(hist_chart, use_container_width=True)
                
                with pareto_col2:
                    st.markdown("#### 🔮 Predicted Defect Pareto")
                    pred_data = defect_df[["Defect", "Predicted Rate (%)"]].copy()
                    pred_chart = create_pareto_chart(
                        pred_data,
                        "Predicted Rate (%)",
                        "Defect",
                        "Top 10 Predicted Defects"
                    )
                    st.plotly_chart(pred_chart, use_container_width=True)

                # Process diagnosis
                st.markdown("### 🏭 Root Cause Process Diagnosis")
                st.markdown("*Based on Campbell (2003) process-defect relationships*")
                
                top_defects = defect_df.head(10).copy()
                diagnosis = diagnose_root_causes(top_defects)
                
                if not diagnosis.empty:
                    fig_process = px.bar(
                        diagnosis,
                        x="Process",
                        y="Contribution (%)",
                        color="Contribution (%)",
                        color_continuous_scale="Reds",
                        title="Process Contributions to Predicted Defects"
                    )
                    fig_process.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_process, use_container_width=True)
                    
                    st.markdown("#### 📋 Detailed Process Analysis")
                    diagnosis_display = diagnosis.copy()
                    diagnosis_display["Contribution (%)"] = diagnosis_display["Contribution (%)"].round(2)
                    
                    st.dataframe(
                        diagnosis_display.style.background_gradient(
                            subset=["Contribution (%)"], 
                            cmap="Reds"
                        ),
                        use_container_width=True
                    )
                    
                    # Defect-to-Process mapping
                    st.markdown("#### 🔗 Defect → Process Mapping")
                    
                    mapping_data = []
                    for _, row in top_defects.head(10).iterrows():
                        defect_code = row["Defect_Code"]
                        if defect_code in DEFECT_TO_PROCESS:
                            processes = DEFECT_TO_PROCESS[defect_code]
                            mapping_data.append({
                                "Defect": row["Defect"],
                                "Predicted Rate (%)": f"{row['Predicted Rate (%)']:.2f}",
                                "Expected Count": f"{row['Expected Count']:.1f}",
                                "Root Cause Process(es)": ", ".join(processes),
                                "# Processes": len(processes)
                            })
                    
                    if mapping_data:
                        mapping_df = pd.DataFrame(mapping_data)
                        st.dataframe(mapping_df, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommended Actions")
                    
                    top_process = diagnosis.iloc[0]["Process"]
                    top_contribution = diagnosis.iloc[0]["Contribution (%)"]
                    
                    st.info(f"""
**Primary Focus: {top_process}** ({top_contribution:.1f}% contribution)

**Description:** {PROCESS_DEFECT_MAP[top_process]["description"]}

**Recommended Actions:**
- Review SPC charts for {top_process} parameters
- Increase inspection frequency for related defects
- Consider process capability study for {top_process}
- Check if {top_process} parameters are within specification limits
                    """)
                    
                    if len(diagnosis) > 1:
                        st.warning(f"""
**Secondary Concern: {diagnosis.iloc[1]["Process"]}** ({diagnosis.iloc[1]["Contribution (%)"]:.1f}% contribution)

Monitor this process as well, as it contributes significantly to the predicted defect profile.
                        """)

            else:
                st.warning("⚠️ No defect rate columns found in dataset")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ================================================================
# TAB 2: VALIDATION
# ================================================================
with tab2:
    st.header("📏 Model Validation (6-2-1 Rolling Window)")
    
    if rolling_enabled:
        st.info("🔄 **Rolling Window Mode**: Validation metrics reflect combined historical + outcome data")
    
    if use_multi_defect:
        st.info("🧬 **Multi-Defect Intelligence**: Enabled in current model")
    
    try:
        X_test, y_test, _ = make_xy(df_test_base, thr_label, use_rate_cols, use_multi_defect)
        preds = cal_model_base.predict_proba(X_test)[:, 1]
        pred_binary = (preds > 0.5).astype(int)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, pred_binary)
        
        col1.metric("Brier Score", f"{brier:.4f}")
        col2.metric("Accuracy", f"{acc:.3f}")
        
        if y_test.sum() > 0:
            recall = recall_score(y_test, pred_binary, zero_division=0)
            prec = precision_score(y_test, pred_binary, zero_division=0)
            col3.metric("Recall (High Scrap)", f"{recall:.3f}")
            col4.metric("Precision", f"{prec:.3f}")
        
        # Data split info
        st.markdown("### 📊 Data Split (6-2-1 Temporal)")
        split_col1, split_col2, split_col3 = st.columns(3)
        split_col1.metric("Train (oldest 60%)", f"{len(df_train_base):,} rows")
        split_col2.metric("Calibration (middle 20%)", f"{len(df_calib_base):,} rows")
        split_col3.metric("Test (newest 20%)", f"{len(df_test_base):,} rows")
        
        # Confusion matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, pred_binary)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual OK", "Actual Scrap"],
            columns=["Pred OK", "Pred Scrap"]
        )
        st.dataframe(cm_df)
        
        # Classification report
        st.markdown("### Classification Report")
        st.text(classification_report(y_test, pred_binary))
        
        # Feature importances
        st.markdown("### 🔍 Feature Importances")
        try:
            if hasattr(cal_model_base, "calibrated_classifiers_"):
                base = cal_model_base.calibrated_classifiers_[0].estimator
                importances = base.feature_importances_
            elif hasattr(cal_model_base, "base_estimator"):
                base = cal_model_base.base_estimator
                if isinstance(base, list):
                    importances = base[0].feature_importances_
                else:
                    importances = base.feature_importances_
            else:
                importances = cal_model_base.feature_importances_
            
            feat_imp = pd.DataFrame({
                "Feature": feats_base,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(20)
            
            # Highlight multi-defect features
            multi_defect_feats = ["n_defect_types", "has_multiple_defects", "total_defect_rate",
                                  "max_defect_rate", "defect_concentration", "shift_x_tearup",
                                  "shrink_x_porosity", "shrink_x_shrink_porosity", "core_x_sand"]
            feat_imp["Type"] = feat_imp["Feature"].apply(
                lambda x: "Multi-Defect (V3.0)" if x in multi_defect_feats else "Standard"
            )
            
            fig_imp = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                color="Type",
                orientation='h',
                title="Top 20 Features (Multi-Defect features highlighted)",
                color_discrete_map={"Multi-Defect (V3.0)": "#ff6b6b", "Standard": "steelblue"}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not extract feature importances: {e}")
        
    except Exception as e:
        st.warning(f"⚠️ Validation failed: {e}")


# ================================================================
# TAB 3: MODEL COMPARISON (NEW IN V3.0)
# ================================================================
with tab3:
    st.header("📊 Model Comparison: With vs Without Multi-Defect Features")
    
    st.markdown("""
    This comparison shows the impact of the **Multi-Defect Intelligence** features 
    introduced in Version 3.0 on model performance.
    """)
    
    if st.button("🔬 Run Comparison"):
        with st.spinner("Training both models for comparison..."):
            try:
                comparison = compare_models_with_without_multi_defect(
                    df_base, thr_label, use_rate_cols, n_est
                )
                
                st.markdown("### 📈 Performance Comparison")
                
                # Create comparison table
                comp_data = {
                    "Metric": ["Brier Score ↓", "Accuracy ↑", "Recall ↑", "Precision ↑", "F1 Score ↑", "# Features"],
                    "Without Multi-Defect": [
                        f"{comparison['without']['brier']:.4f}",
                        f"{comparison['without']['accuracy']:.3f}",
                        f"{comparison['without']['recall']:.3f}",
                        f"{comparison['without']['precision']:.3f}",
                        f"{comparison['without']['f1']:.3f}",
                        f"{comparison['without']['n_features']}"
                    ],
                    "With Multi-Defect (V3.0)": [
                        f"{comparison['with']['brier']:.4f}",
                        f"{comparison['with']['accuracy']:.3f}",
                        f"{comparison['with']['recall']:.3f}",
                        f"{comparison['with']['precision']:.3f}",
                        f"{comparison['with']['f1']:.3f}",
                        f"{comparison['with']['n_features']}"
                    ],
                    "Change": [
                        f"{comparison['improvement']['brier']:+.1f}% {'✅' if comparison['improvement']['brier'] > 0 else '❌'}",
                        f"{comparison['improvement']['accuracy']:+.1f}% {'✅' if comparison['improvement']['accuracy'] > 0 else '❌'}",
                        f"{comparison['improvement']['recall']:+.1f}% {'✅' if comparison['improvement']['recall'] > 0 else '❌'}",
                        f"{comparison['improvement']['precision']:+.1f}% {'✅' if comparison['improvement']['precision'] > 0 else '❌'}",
                        f"{comparison['improvement']['f1']:+.1f}% {'✅' if comparison['improvement']['f1'] > 0 else '❌'}",
                        f"+{comparison['improvement']['n_features']}"
                    ]
                }
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True)
                
                # Visual comparison
                st.markdown("### 📊 Visual Comparison")
                
                fig = go.Figure()
                
                metrics = ["Accuracy", "Recall", "Precision", "F1 Score"]
                without_vals = [
                    comparison['without']['accuracy'],
                    comparison['without']['recall'],
                    comparison['without']['precision'],
                    comparison['without']['f1']
                ]
                with_vals = [
                    comparison['with']['accuracy'],
                    comparison['with']['recall'],
                    comparison['with']['precision'],
                    comparison['with']['f1']
                ]
                
                fig.add_trace(go.Bar(
                    name='Without Multi-Defect',
                    x=metrics,
                    y=without_vals,
                    marker_color='lightgray'
                ))
                
                fig.add_trace(go.Bar(
                    name='With Multi-Defect (V3.0)',
                    x=metrics,
                    y=with_vals,
                    marker_color='#ff6b6b'
                ))
                
                fig.update_layout(
                    barmode='group',
                    title="Model Performance Comparison",
                    yaxis_title="Score",
                    yaxis_range=[0, 1]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                st.markdown("### 📋 Summary")
                
                improvements = []
                if comparison['improvement']['brier'] > 0:
                    improvements.append(f"Brier Score improved by {comparison['improvement']['brier']:.1f}%")
                if comparison['improvement']['accuracy'] > 0:
                    improvements.append(f"Accuracy improved by {comparison['improvement']['accuracy']:.1f}%")
                if comparison['improvement']['recall'] > 0:
                    improvements.append(f"Recall improved by {comparison['improvement']['recall']:.1f}%")
                if comparison['improvement']['f1'] > 0:
                    improvements.append(f"F1 Score improved by {comparison['improvement']['f1']:.1f}%")
                
                if improvements:
                    st.success(f"""
✅ **Multi-Defect Features Improve Model Performance**

{chr(10).join(['• ' + imp for imp in improvements])}

The multi-defect features capture important patterns like:
- **Defect co-occurrence** (multiple defects on same work order)
- **Defect interactions** (e.g., Shift × Tear-Up combination)
- **Overall defect burden** (total and max defect rates)
                    """)
                else:
                    st.info("""
📊 **Results Mixed or No Improvement**

The multi-defect features did not show clear improvement on this dataset split.
This could be due to:
- Limited multi-defect cases in test set
- Existing features already capturing the patterns
- Need for more data to see the benefit
                    """)
                
            except Exception as e:
                st.error(f"❌ Comparison failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ================================================================
# TAB 4: LOG OUTCOME
# ================================================================
with tab4:
    st.header("📝 Log Production Outcome")
    
    if not rolling_enabled:
        st.warning("⚠️ Rolling window is disabled. Enable it in the sidebar to log outcomes.")
    else:
        st.markdown("""
        **Log actual outcomes to enable rolling window retraining.**
        
        After a production run completes, enter the actual scrap results here.
        The model will retrain on updated data after enough outcomes are logged.
        """)
        
        if 'last_prediction' in st.session_state:
            last_pred = st.session_state['last_prediction']
            st.info(f"📌 Last prediction: Part {last_pred['part_id']}, Predicted Scrap: {last_pred['predicted_scrap']:.1f}% (based on {last_pred.get('part_records', 'N/A')} historical runs)")
            use_last = st.checkbox("Use last prediction details", value=True)
        else:
            use_last = False
            last_pred = None
        
        out_col1, out_col2, out_col3, out_col4 = st.columns(4)
        
        if use_last and last_pred:
            outcome_part_id = out_col1.text_input("Part ID", value=last_pred['part_id'], key="outcome_part")
            outcome_qty = out_col2.number_input("Order Quantity", value=int(last_pred['order_quantity']), key="outcome_qty")
            outcome_weight = out_col3.number_input("Piece Weight", value=float(last_pred['piece_weight_lbs']), key="outcome_weight")
        else:
            outcome_part_id = out_col1.text_input("Part ID", value="", key="outcome_part")
            outcome_qty = out_col2.number_input("Order Quantity", value=100, key="outcome_qty")
            outcome_weight = out_col3.number_input("Piece Weight", value=5.0, key="outcome_weight")
        
        actual_scrap_pct = out_col4.number_input("Actual Scrap %", min_value=0.0, max_value=100.0, value=0.0, key="actual_scrap")
        
        if st.button("💾 Log Outcome"):
            if outcome_part_id and outcome_part_id != "":
                outcome_data = {
                    'part_id': outcome_part_id,
                    'order_quantity': outcome_qty,
                    'piece_weight_lbs': outcome_weight,
                    'actual_scrap': actual_scrap_pct,
                    'predicted_scrap': last_pred['predicted_scrap'] if use_last and last_pred else None,
                }
                
                count = save_outcome(outcome_data)
                st.success(f"✅ Outcome logged! Total outcomes: {count}")
                
                outcomes_since_retrain = get_outcomes_count()
                if outcomes_since_retrain >= RETRAIN_THRESHOLD:
                    st.warning(f"⚠️ {outcomes_since_retrain} outcomes logged since last retrain. Consider retraining the model.")
                    if st.button("🔄 Retrain Model Now"):
                        mark_retrain()
                        st.cache_data.clear()
                        st.experimental_rerun()
            else:
                st.error("❌ Please enter a Part ID")
        
        st.markdown("### 📋 Recent Outcomes")
        df_outcomes_display = load_outcomes()
        if not df_outcomes_display.empty:
            df_display = df_outcomes_display[df_outcomes_display['part_id'] != 'RETRAIN_MARKER'].tail(20)
            if not df_display.empty:
                st.dataframe(df_display[['timestamp', 'part_id', 'order_quantity', 'actual_scrap', 'predicted_scrap']].tail(10))
            else:
                st.info("No outcomes logged yet.")
        else:
            st.info("No outcomes logged yet.")
        
        st.markdown("### 🔄 Manual Retrain")
        if st.button("Force Retrain Model"):
            mark_retrain()
            st.cache_data.clear()
            st.success("✅ Retrain marker set. Refresh the page to retrain with latest data.")


st.markdown("---")
st.caption("🏭 Foundry Scrap Risk Dashboard **v3.0 - Multi-Defect Intelligence** | Based on Campbell (2003) | 6-2-1 Rolling Window | Data Confidence Indicators")
