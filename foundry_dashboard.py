# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# Features: Prediction, Process Root Cause Analysis, Pareto Charts
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
)

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2
MIN_SAMPLES_PER_CLASS = 5

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
    rename_map = {
        "work_order": "part_id",
        "work_order_number": "part_id",
        "work_order_#": "part_id",
        "order_quantity": "order_quantity",
        "order_qty": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "piece_weight_lbs": "piece_weight_lbs",
        "piece_weight": "piece_weight_lbs",
        "week_ending": "week_ending",
        "week_end": "week_ending",
        "scrap%": "scrap_percent",
        "scrap_percent": "scrap_percent",
        "scrap": "scrap_percent",
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

    # Handle duplicate columns (keep first occurrence only)
    if df.columns.duplicated().any():
        st.warning(f"⚠️ Detected {df.columns.duplicated().sum()} duplicate column names - keeping first occurrence")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure key columns exist
    for c in ["part_id", "order_quantity", "piece_weight_lbs", "week_ending", "scrap_percent"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "week_ending" else pd.NaT

    # Clean part_id FIRST (before other conversions)
    if "part_id" in df.columns:
        # Force to single column if somehow still a DataFrame
        if isinstance(df["part_id"], pd.DataFrame):
            df["part_id"] = df["part_id"].iloc[:, 0]
        df["part_id"] = df["part_id"].fillna("Unknown").astype(str)
        df["part_id"] = df["part_id"].str.strip()
        df["part_id"] = df["part_id"].replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
    
    # Data type conversions
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(0)
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce").fillna(0.0)
    
    # Normalize defect columns
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in defect_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Drop invalid dates
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    st.info(f"✅ Loaded {len(df):,} rows, {len(defect_cols)} defect rate columns")
    return df


def time_split(df: pd.DataFrame, train_ratio=0.6, calib_ratio=0.2):
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


def make_xy(df: pd.DataFrame, thr_label: float, use_rate_cols: bool):
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
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    pos = int(y_calib.sum())
    neg = int((y_calib == 0).sum())
    
    # Need at least 3 samples per class for 3-fold CV
    if pos < 3 or neg < 3:
        st.warning(f"⚠️ Insufficient samples for calibration (Scrap=1: {pos}, Scrap=0: {neg}). Using uncalibrated model.")
        return rf, rf, "uncalibrated (insufficient calibration samples)"
    
    try:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
        return rf, cal, "calibrated (sigmoid, cv=3)"
    except ValueError as e:
        st.warning(f"⚠️ Calibration failed: {e}. Using uncalibrated model.")
        return rf, rf, "uncalibrated (calibration failed)"


# ================================================================
# DYNAMIC PART-SPECIFIC DATA PREPARATION
# ================================================================
def prepare_part_specific_data(df_full: pd.DataFrame, target_part: str, 
                                piece_weight: float, thr_label: float, 
                                min_samples: int = 30):
    """
    Prepare part-specific dataset with similarity-based expansion.
    Ensures label diversity for training.
    """
    st.info(f"🔍 Preparing data for Part {target_part}...")
    
    # Start with exact part matches
    df_part = df_full[df_full["part_id"] == target_part].copy()
    
    # If we don't have enough samples, expand by similarity
    if len(df_part) < min_samples:
        st.info(f"⚠️ Only {len(df_part)} samples for Part {target_part}. Expanding search...")
        
        # Find similar parts by weight
        weight_tolerance = 0.1  # Start with ±10%
        max_tolerance = 0.5     # Max ±50%
        
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
    
    # Check label diversity
    df_part["temp_label"] = (df_part["scrap_percent"] > thr_label).astype(int)
    label_counts = df_part["temp_label"].value_counts()
    
    if len(label_counts) < 2 or label_counts.min() < MIN_SAMPLES_PER_CLASS:
        st.warning(f"⚠️ Insufficient label diversity. Expanding to broader dataset...")
        # Use much broader weight range
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
    """
    Map predicted defects to their root cause processes.
    Returns a ranked list of processes by contribution.
    """
    if defect_predictions.empty:
        return pd.DataFrame(columns=["Process", "Contribution (%)", "Description"])
    
    process_scores = {}
    
    for _, row in defect_predictions.iterrows():
        defect = row.get("Defect_Code", "")
        likelihood = row.get("Predicted Rate (%)", 0.0)
        
        # Find which processes could cause this defect
        if defect in DEFECT_TO_PROCESS:
            processes = DEFECT_TO_PROCESS[defect]
            # Distribute the defect likelihood across responsible processes
            contribution = likelihood / len(processes) if len(processes) > 0 else 0.0
            
            for process in processes:
                if process not in process_scores:
                    process_scores[process] = 0.0
                process_scores[process] += contribution
    
    # Convert to DataFrame and sort
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
    
    # Calculate cumulative percentage
    total = data[value_col].sum()
    data["Cumulative %"] = (data[value_col].cumsum() / total * 100) if total > 0 else 0
    
    fig = go.Figure()
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=data[label_col],
        y=data[value_col],
        name=value_col,
        marker_color='steelblue',
        yaxis='y'
    ))
    
    # Cumulative line
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


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# -------------------------------
# Load base dataset (for validation tab only)
# -------------------------------
df_base = load_and_clean(csv_path)
df_base = calculate_process_indices(df_base)

# Split for validation metrics
df_train_base, df_calib_base, df_test_base = time_split(df_base)

# Train base model for validation tab
mtbf_train_base = compute_mtbf_on_train(df_train_base, thr_label)
part_freq_train_base = df_train_base["part_id"].value_counts(normalize=True)
default_mtbf_base = float(mtbf_train_base["mttf_scrap"].median()) if len(mtbf_train_base) else 1.0
default_freq_base = float(part_freq_train_base.median()) if len(part_freq_train_base) else 0.0

df_train_base = attach_train_features(df_train_base, mtbf_train_base, part_freq_train_base, default_mtbf_base, default_freq_base)
df_calib_base = attach_train_features(df_calib_base, mtbf_train_base, part_freq_train_base, default_mtbf_base, default_freq_base)
df_test_base = attach_train_features(df_test_base, mtbf_train_base, part_freq_train_base, default_mtbf_base, default_freq_base)

X_train_base, y_train_base, feats_base = make_xy(df_train_base, thr_label, use_rate_cols)
X_calib_base, y_calib_base, _ = make_xy(df_calib_base, thr_label, use_rate_cols)
rf_base, cal_model_base, method_base = train_and_calibrate(X_train_base, y_train_base, X_calib_base, y_calib_base, n_est)

st.success(f"✅ Base model loaded: {method_base}, {len(X_train_base)} samples")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["🔮 Predict & Diagnose", "📏 Validation"])

# ================================================================
# TAB 1: PREDICTION & PROCESS DIAGNOSIS
# ================================================================
with tab1:
    st.header("🔮 Scrap Risk Prediction & Root Cause Analysis")
    
    # Input form
    col1, col2, col3, col4 = st.columns(4)
    part_id_input = col1.text_input("Part ID", value="Unknown")
    order_qty = col2.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = col3.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = col4.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

    if st.button("🎯 Predict Risk & Diagnose Process"):
        try:
            # 🔥 CLEAR ALL CACHES FOR FRESH TRAINING
            st.cache_data.clear()
            
            st.info("🔄 Retraining model with part-specific dataset...")
            
            # Reload full dataset
            df_full = load_and_clean(csv_path)
            df_full = calculate_process_indices(df_full)
            
            # Prepare part-specific dataset
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
            
            # Time-based split
            df_train, df_calib, df_test = time_split(df_part)
            
            # Check each split has both labels
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
            
            # Feature engineering
            mtbf_train = compute_mtbf_on_train(df_train, thr_label)
            part_freq_train = df_train["part_id"].value_counts(normalize=True)
            default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
            default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
            
            df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
            df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
            df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
            
            # Train part-specific model
            X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
            X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
            
            if y_train.nunique() < 2:
                st.error("❌ Training set has only one class after feature engineering")
                st.stop()
            
            rf_part, cal_model_part, method_part = train_and_calibrate(
                X_train, y_train, X_calib, y_calib, n_est
            )
            
            st.success(f"✅ Part-specific model trained: {method_part}, {len(X_train)} samples")
            st.info(f"📊 Training labels: Scrap=1: {y_train.sum()}, Scrap=0: {(y_train==0).sum()}")
            
            # Prepare input
            input_df = pd.DataFrame(
                [[part_id_input, order_qty, piece_weight, default_mtbf, default_freq]],
                columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
            )
            
            X_input = input_df.drop(columns=["part_id"])
            for col in feats:
                if col not in X_input.columns:
                    X_input[col] = 0.0
            X_input = X_input[feats]

            # Make prediction
            prob = float(cal_model_part.predict_proba(X_input)[0, 1])
            adj_prob = np.clip(prob, 0.0, 1.0)
            exp_scrap = order_qty * adj_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1.0 - adj_prob

            # Display overall metrics
            st.markdown(f"### 🎯 Risk Assessment for Part **{part_id_input}**")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Scrap Risk", f"{adj_prob*100:.1f}%")
            metric_col2.metric("Expected Scrap", f"{exp_scrap:.1f} pieces")
            metric_col3.metric("Expected Loss", f"${exp_loss:,.2f}")
            metric_col4.metric("Reliability", f"{reliability*100:.1f}%")

            st.markdown("---")

            # ================================================================
            # DEFECT-LEVEL PREDICTIONS
            # ================================================================
            st.markdown("### 🔬 Detailed Defect Analysis")
            
            defect_cols = [c for c in df_full.columns if c.endswith("_rate")]
            
            if len(defect_cols) > 0:
                # Get historical baseline for this part (or similar parts)
                similar_parts = df_train[
                    (df_train["piece_weight_lbs"] >= piece_weight * 0.9) & 
                    (df_train["piece_weight_lbs"] <= piece_weight * 1.1)
                ]
                
                if len(similar_parts) < 10:
                    similar_parts = df_train  # Fallback to all training data
                
                # Calculate defect predictions
                defect_predictions = []
                for defect_col in defect_cols:
                    historical_rate = similar_parts[defect_col].mean()
                    # Adjust by overall risk prediction
                    predicted_rate = historical_rate * (1 + adj_prob)
                    expected_defects = order_qty * predicted_rate / 100
                    
                    defect_predictions.append({
                        "Defect": defect_col.replace("_rate", "").replace("_", " ").title(),
                        "Defect_Code": defect_col,
                        "Historical Rate (%)": historical_rate,
                        "Predicted Rate (%)": predicted_rate,
                        "Expected Count": expected_defects
                    })
                
                defect_df = pd.DataFrame(defect_predictions).sort_values("Predicted Rate (%)", ascending=False)
                
                # ================================================================
                # PARETO CHARTS
                # ================================================================
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

                # ================================================================
                # PROCESS ROOT CAUSE DIAGNOSIS
                # ================================================================
                st.markdown("### 🏭 Root Cause Process Diagnosis")
                st.markdown("*Based on Campbell (2003) process-defect relationships*")
                
                # Get top defects for diagnosis
                top_defects = defect_df.head(10).copy()
                
                # Diagnose processes
                diagnosis = diagnose_root_causes(top_defects)
                
                if not diagnosis.empty:
                    # Display process contribution chart
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
                    
                    # Display detailed process table
                    st.markdown("#### 📋 Detailed Process Analysis")
                    
                    # Format the diagnosis table
                    diagnosis_display = diagnosis.copy()
                    diagnosis_display["Contribution (%)"] = diagnosis_display["Contribution (%)"].round(2)
                    
                    st.dataframe(
                        diagnosis_display.style.background_gradient(
                            subset=["Contribution (%)"], 
                            cmap="Reds"
                        ),
                        use_container_width=True
                    )
                    
                    # ================================================================
                    # DEFECT-TO-PROCESS MAPPING TABLE
                    # ================================================================
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
                    
                    # ================================================================
                    # ACTIONABLE RECOMMENDATIONS
                    # ================================================================
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
    st.header("📏 Model Validation (6-2-1 Split)")
    
    try:
        X_test, y_test, _ = make_xy(df_test_base, thr_label, use_rate_cols)
        preds = cal_model_base.predict_proba(X_test)[:, 1]
        pred_binary = (preds > 0.5).astype(int)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Brier Score", f"{brier_score_loss(y_test, preds):.4f}")
        col2.metric("Accuracy", f"{accuracy_score(y_test, pred_binary):.3f}")
        
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
            if hasattr(cal_model_base, "base_estimator"):
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

st.markdown("---")
st.caption("Based on Campbell (2003) *Castings Practice: The Ten Rules of Castings*")
