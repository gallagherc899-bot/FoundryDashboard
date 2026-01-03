# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 2.0 - TRUE 6-2-1 ROLLING WINDOW
# Features: Prediction, Process Root Cause Analysis, Pareto Charts,
#           Rolling Window Retraining, Outcome Logging
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
)
from datetime import datetime
import json

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard (Rolling Window)", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2
MIN_SAMPLES_PER_CLASS = 5

# ================================================================
# ROLLING WINDOW CONFIGURATION
# ================================================================
ROLLING_WINDOW_ENABLED = True      # Toggle rolling window on/off
RETRAIN_THRESHOLD = 50             # Retrain after N new outcomes logged
WINDOW_TRAIN_RATIO = 0.6           # 60% for training
WINDOW_CALIB_RATIO = 0.2           # 20% for calibration
WINDOW_TEST_RATIO = 0.2            # 20% for testing
OUTCOMES_FILE = "prediction_outcomes.csv"  # File to store outcomes

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
def load_and_clean(csv_path: str) -> pd.DataFrame:
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
    
    # Sort by time for temporal split
    df = df.sort_values("week_ending").reset_index(drop=True)

    n_parts = df["part_id"].nunique() if "part_id" in df.columns else 0
    n_work_orders = df["work_order"].nunique() if "work_order" in df.columns else len(df)
    st.info(f"✅ Loaded {len(df):,} rows | {n_parts} unique parts | {n_work_orders} work orders | {len(defect_cols)} defect columns")
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
    # Count outcomes since last retrain marker
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
        # Add a marker row
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
# 6-2-1 TEMPORAL SPLIT (Rolling Window Compatible)
# ================================================================
def time_split_621(df: pd.DataFrame, train_ratio=0.6, calib_ratio=0.2):
    """
    TRUE 6-2-1 Temporal Split.
    
    Data MUST be sorted by week_ending before calling.
    - Train: oldest 60%
    - Calibration: middle 20%  
    - Test: newest 20%
    """
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


def make_xy(df: pd.DataFrame, thr_label: float, use_rate_cols: bool):
    """Prepare features (X) and labels (y)."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
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
def train_model_with_rolling_window(df_base, df_outcomes, thr_label, use_rate_cols, n_est):
    """
    Train model using 6-2-1 rolling window on combined base + outcome data.
    
    If outcomes exist, they are appended to base data (sorted by time),
    then 6-2-1 split is applied to the combined dataset.
    """
    
    # Combine base data with logged outcomes
    if df_outcomes is not None and len(df_outcomes) > 0:
        # Filter valid outcome rows (not markers)
        df_outcomes_valid = df_outcomes[df_outcomes['part_id'] != 'RETRAIN_MARKER'].copy()
        
        if len(df_outcomes_valid) > 0:
            # Ensure outcomes have required columns
            required_cols = ['part_id', 'order_quantity', 'piece_weight_lbs', 'scrap_percent', 'week_ending']
            
            # Check which columns exist
            available_cols = [c for c in required_cols if c in df_outcomes_valid.columns]
            
            if 'actual_scrap' in df_outcomes_valid.columns and 'scrap_percent' not in df_outcomes_valid.columns:
                df_outcomes_valid['scrap_percent'] = df_outcomes_valid['actual_scrap']
            
            if 'timestamp' in df_outcomes_valid.columns and 'week_ending' not in df_outcomes_valid.columns:
                df_outcomes_valid['week_ending'] = pd.to_datetime(df_outcomes_valid['timestamp'])
            
            # Only merge if we have the minimum required data
            if 'part_id' in df_outcomes_valid.columns and 'scrap_percent' in df_outcomes_valid.columns:
                # Get columns from base that outcomes might be missing
                for col in df_base.columns:
                    if col not in df_outcomes_valid.columns:
                        df_outcomes_valid[col] = 0.0 if col != 'week_ending' else pd.NaT
                
                # Combine
                df_combined = pd.concat([df_base, df_outcomes_valid[df_base.columns]], ignore_index=True)
                df_combined = df_combined.sort_values('week_ending').reset_index(drop=True)
                
                st.success(f"🔄 Rolling Window: Combined {len(df_base)} base + {len(df_outcomes_valid)} outcome records = {len(df_combined)} total")
            else:
                df_combined = df_base.copy()
        else:
            df_combined = df_base.copy()
    else:
        df_combined = df_base.copy()
    
    # Apply 6-2-1 temporal split
    df_train, df_calib, df_test = time_split_621(df_combined)
    
    # Compute features from training data only
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # Train model
    X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
    X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
    rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)
    
    return rf, cal_model, method, feats, df_train, df_calib, df_test, mtbf_train, part_freq_train, default_mtbf, default_freq


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
df_base = load_and_clean(csv_path)
df_base = calculate_process_indices(df_base)

# Load outcomes for rolling window
df_outcomes = load_outcomes() if rolling_enabled else None

# Display data summary in sidebar
st.sidebar.header("📊 Data Summary")
n_parts = df_base["part_id"].nunique()
n_work_orders = df_base["work_order"].nunique() if "work_order" in df_base.columns else len(df_base)
avg_runs_per_part = len(df_base) / n_parts if n_parts > 0 else 0

st.sidebar.metric("Unique Parts", f"{n_parts:,}")
st.sidebar.metric("Work Orders", f"{n_work_orders:,}")
st.sidebar.metric("Avg Runs/Part", f"{avg_runs_per_part:.1f}")
st.sidebar.caption("ℹ️ Model analyzes by Part ID, not Work Order")

# Train model with rolling window
(rf_base, cal_model_base, method_base, feats_base, 
 df_train_base, df_calib_base, df_test_base,
 mtbf_train_base, part_freq_train_base, 
 default_mtbf_base, default_freq_base) = train_model_with_rolling_window(
    df_base, df_outcomes, thr_label, use_rate_cols, n_est
)

if rolling_enabled:
    st.success(f"✅ Model trained with 6-2-1 Rolling Window: {method_base}, {len(df_train_base)} train samples")
else:
    st.success(f"✅ Base model loaded: {method_base}, {len(df_train_base)} samples")


# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict & Diagnose", "📏 Validation", "📝 Log Outcome"])

# ================================================================
# TAB 1: PREDICTION & PROCESS DIAGNOSIS
# ================================================================
with tab1:
    st.header("🔮 Scrap Risk Prediction & Root Cause Analysis")
    
    if rolling_enabled:
        st.info("🔄 **Rolling Window Mode**: Model trains on combined historical + logged outcome data using 6-2-1 temporal split")
    
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
            df_full = load_and_clean(csv_path)
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
            
            X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
            X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
            
            rf_part, cal_part, method_part = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)
            
            st.success(f"✅ Part-specific model trained: {method_part}, {len(X_train)} samples")
            st.info(f"📊 Training labels: Scrap=1: {y_train.sum()}, Scrap=0: {(y_train == 0).sum()}")
            
            # Create prediction input
            part_history = df_full[df_full["part_id"] == part_id_input]
            if len(part_history) > 0:
                hist_mttf = float(mtbf_train[mtbf_train["part_id"] == part_id_input]["mttf_scrap"].values[0]) \
                    if part_id_input in mtbf_train["part_id"].values else default_mtbf
                hist_freq = float(part_freq_train.get(part_id_input, default_freq))
            else:
                hist_mttf = default_mtbf
                hist_freq = default_freq

            defect_cols = [c for c in df_full.columns if c.endswith("_rate")]
            if len(part_history) > 0:
                defect_means = part_history[defect_cols].mean()
            else:
                defect_means = df_full[defect_cols].mean()

            input_dict = {
                "order_quantity": order_qty,
                "piece_weight_lbs": piece_weight,
                "mttf_scrap": hist_mttf,
                "part_freq": hist_freq,
            }
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
                'cost_per_part': cost_per_part
            }

            # Display results
            st.markdown("---")
            st.markdown(f"### 🎯 Risk Assessment for Part {part_id_input}")
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Scrap Risk", f"{scrap_risk:.1f}%")
            r2.metric("Expected Scrap", f"{expected_scrap_pcs:.1f} pieces")
            r3.metric("Expected Loss", f"${expected_loss:,.2f}")
            r4.metric("Reliability", f"{reliability:.1f}%")

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
                    st.markdown("Shows which processes are responsible for each predicted defect")
                    
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
    
    try:
        X_test, y_test, _ = make_xy(df_test_base, thr_label, use_rate_cols)
        preds = cal_model_base.predict_proba(X_test)[:, 1]
        pred_binary = (preds > 0.5).astype(int)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, pred_binary)
        
        col1.metric("Brier Score", f"{brier:.4f}")
        col2.metric("Accuracy", f"{acc:.3f}")
        
        if y_test.sum() > 0:
            recall = recall_score(y_test, pred_binary, zero_division=0)
            col3.metric("Recall (High Scrap)", f"{recall:.3f}")
        
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
            }).sort_values("Importance", ascending=False).head(15)
            
            fig_imp = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation='h',
                title="Top 15 Features"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not extract feature importances: {e}")
        
    except Exception as e:
        st.warning(f"⚠️ Validation failed: {e}")


# ================================================================
# TAB 3: LOG OUTCOME (For Rolling Window)
# ================================================================
with tab3:
    st.header("📝 Log Production Outcome")
    
    if not rolling_enabled:
        st.warning("⚠️ Rolling window is disabled. Enable it in the sidebar to log outcomes.")
    else:
        st.markdown("""
        **Log actual outcomes to enable rolling window retraining.**
        
        After a production run completes, enter the actual scrap results here.
        The model will retrain on updated data after enough outcomes are logged.
        """)
        
        # Check if there's a recent prediction to reference
        if 'last_prediction' in st.session_state:
            last_pred = st.session_state['last_prediction']
            st.info(f"📌 Last prediction: Part {last_pred['part_id']}, Predicted Scrap: {last_pred['predicted_scrap']:.1f}%")
            use_last = st.checkbox("Use last prediction details", value=True)
        else:
            use_last = False
            last_pred = None
        
        # Input form
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
                
                # Check if retrain needed
                outcomes_since_retrain = get_outcomes_count()
                if outcomes_since_retrain >= RETRAIN_THRESHOLD:
                    st.warning(f"⚠️ {outcomes_since_retrain} outcomes logged since last retrain. Consider retraining the model.")
                    if st.button("🔄 Retrain Model Now"):
                        mark_retrain()
                        st.cache_data.clear()
                        st.experimental_rerun()
            else:
                st.error("❌ Please enter a Part ID")
        
        # Show recent outcomes
        st.markdown("### 📋 Recent Outcomes")
        df_outcomes_display = load_outcomes()
        if not df_outcomes_display.empty:
            # Filter out markers
            df_display = df_outcomes_display[df_outcomes_display['part_id'] != 'RETRAIN_MARKER'].tail(20)
            if not df_display.empty:
                st.dataframe(df_display[['timestamp', 'part_id', 'order_quantity', 'actual_scrap', 'predicted_scrap']].tail(10))
            else:
                st.info("No outcomes logged yet.")
        else:
            st.info("No outcomes logged yet.")
        
        # Manual retrain button
        st.markdown("### 🔄 Manual Retrain")
        if st.button("Force Retrain Model"):
            mark_retrain()
            st.cache_data.clear()
            st.success("✅ Retrain marker set. Refresh the page to retrain with latest data.")


st.markdown("---")
st.caption("Based on Campbell (2003) *Castings Practice: The Ten Rules of Castings* | 6-2-1 Rolling Window Enabled")
