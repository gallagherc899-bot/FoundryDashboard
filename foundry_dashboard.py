# ================================================================
# 🏭 Foundry Scrap Risk Dashboard with Process Diagnosis
# VERSION 3.1 - MULTI-DEFECT INTELLIGENCE + RESULTS EXPORT
# ================================================================
# 
# NEW IN V3.1:
# - Comprehensive prediction results logging
# - Side-by-side WITH/WITHOUT multi-defect comparison per prediction
# - Downloadable CSV export of all predictions
# - Batch analysis support
#
# RETAINED FROM V3.0:
# - Multi-defect feature engineering (n_defect_types, total_defect_rate)
# - Multi-defect interaction detection
# - Multi-defect alerts in predictions
# - Performance comparison: with vs without multi-defect features
# - 6-2-1 Rolling Window + Data Confidence Indicators
#
# Based on Campbell (2003) "Castings Practice: The Ten Rules"
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

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard v3.1 - Results Export", 
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
        <strong>Version 3.1 - Multi-Defect Intelligence + Results Export</strong> | 
        6-2-1 Rolling Window | Campbell Process Mapping | Batch Analysis
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
RESULTS_LOG_FILE = "prediction_results_log.csv"

# ================================================================
# MULTI-DEFECT CONFIGURATION
# ================================================================
MULTI_DEFECT_THRESHOLD = 2
MULTI_DEFECT_FEATURES_ENABLED = True

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
# PREDICTION RESULTS LOGGING (NEW IN V3.1)
# ================================================================
def initialize_results_log():
    """Initialize the session state for results logging."""
    if 'prediction_results' not in st.session_state:
        st.session_state['prediction_results'] = []


def log_prediction_result(result_dict):
    """Add a prediction result to the session log."""
    initialize_results_log()
    result_dict['timestamp'] = datetime.now().isoformat()
    result_dict['prediction_id'] = len(st.session_state['prediction_results']) + 1
    st.session_state['prediction_results'].append(result_dict)


def get_results_dataframe():
    """Convert logged results to a DataFrame."""
    initialize_results_log()
    if not st.session_state['prediction_results']:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state['prediction_results'])


def clear_results_log():
    """Clear all logged results."""
    st.session_state['prediction_results'] = []


def save_results_to_file(filepath=RESULTS_LOG_FILE):
    """Save results log to CSV file."""
    df = get_results_dataframe()
    if not df.empty:
        df.to_csv(filepath, index=False)
    return df


def load_results_from_file(filepath=RESULTS_LOG_FILE):
    """Load results log from CSV file."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            st.session_state['prediction_results'] = df.to_dict('records')
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()


# ================================================================
# MULTI-DEFECT FEATURE ENGINEERING
# ================================================================
def add_multi_defect_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-defect intelligence features to the dataframe."""
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
    
    if 'shift_rate' in df.columns and 'tear_up_rate' in df.columns:
        df['shift_x_tearup'] = df['shift_rate'] * df['tear_up_rate']
    
    if 'shrink_rate' in df.columns and 'gas_porosity_rate' in df.columns:
        df['shrink_x_porosity'] = df['shrink_rate'] * df['gas_porosity_rate']
    
    if 'shrink_rate' in df.columns and 'shrink_porosity_rate' in df.columns:
        df['shrink_x_shrink_porosity'] = df['shrink_rate'] * df['shrink_porosity_rate']
    
    if 'core_rate' in df.columns and 'sand_rate' in df.columns:
        df['core_x_sand'] = df['core_rate'] * df['sand_rate']
    
    return df


def get_multi_defect_analysis(df_row: pd.Series, defect_cols: list) -> dict:
    """Analyze multi-defect patterns for a single row."""
    active_defects = []
    for col in defect_cols:
        if col in df_row.index and df_row[col] > 0:
            active_defects.append({
                'defect': col,
                'rate': df_row[col],
                'processes': DEFECT_TO_PROCESS.get(col, ['Unknown'])
            })
    
    active_defects.sort(key=lambda x: x['rate'], reverse=True)
    
    n_defects = len(active_defects)
    total_rate = sum(d['rate'] for d in active_defects)
    
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
    
    defect_list = ", ".join([
        f"{d['defect'].replace('_rate', '').replace('_', ' ').title()} ({d['rate']*100:.1f}%)"
        for d in active[:5]
    ])
    
    common_processes = analysis['primary_processes']
    
    st.error(f"""
🚨 **MULTI-DEFECT ALERT: {n_defects} Defect Types Detected**

**Active Defects:** {defect_list}

**Common Root Cause Processes:** {', '.join(common_processes)}

⚠️ **Interpretation:** Multiple concurrent defects often indicate systemic process issues.
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
    
    if level == 'low':
        st.warning(f"⚠️ **LOW CONFIDENCE**: Part {part_id} has only **{n_records}** historical runs (recommended: {RECOMMENDED_MIN_RECORDS}+)")
    elif level == 'medium':
        st.info(f"📊 **MEDIUM CONFIDENCE**: Part {part_id} has **{n_records}** historical runs")
    else:
        st.success(f"✅ **HIGH CONFIDENCE**: Part {part_id} has **{n_records}** historical runs")
    
    return level, n_records


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


@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str, add_multi_defect: bool = True) -> pd.DataFrame:
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

    if add_multi_defect:
        df = add_multi_defect_features(df)

    return df


# ================================================================
# MODEL TRAINING FUNCTIONS
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


def make_xy(df: pd.DataFrame, thr_label: float, use_rate_cols: bool, use_multi_defect: bool = True):
    """Prepare features (X) and labels (y)."""
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    
    if use_multi_defect:
        multi_defect_feats = [
            "n_defect_types", "has_multiple_defects", "total_defect_rate",
            "max_defect_rate", "defect_concentration",
            "shift_x_tearup", "shrink_x_porosity", "shrink_x_shrink_porosity", "core_x_sand"
        ]
        for f in multi_defect_feats:
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
        return rf, rf, "uncalibrated"
    
    try:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
        return rf, cal, "calibrated"
    except ValueError as e:
        return rf, rf, f"uncalibrated"


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


def prepare_part_specific_data(df_full: pd.DataFrame, target_part: str, 
                                piece_weight: float, thr_label: float, 
                                min_samples: int = 30):
    """Prepare part-specific dataset with similarity-based expansion."""
    df_part = df_full[df_full["part_id"] == target_part].copy()
    
    if len(df_part) < min_samples:
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
    
    df_part["temp_label"] = (df_part["scrap_percent"] > thr_label).astype(int)
    label_counts = df_part["temp_label"].value_counts()
    
    if len(label_counts) < 2 or label_counts.min() < MIN_SAMPLES_PER_CLASS:
        lower = piece_weight * 0.5
        upper = piece_weight * 1.5
        df_part = df_full[
            (df_full["piece_weight_lbs"] >= lower) & 
            (df_full["piece_weight_lbs"] <= upper)
        ].copy()
        df_part["temp_label"] = (df_part["scrap_percent"] > thr_label).astype(int)
        label_counts = df_part["temp_label"].value_counts()
        
        if len(label_counts) < 2 or label_counts.min() < MIN_SAMPLES_PER_CLASS:
            return None
    
    df_part = df_part.drop(columns=["temp_label"])
    return df_part


# ================================================================
# DUAL PREDICTION FUNCTION (WITH and WITHOUT Multi-Defect)
# ================================================================
def run_dual_prediction(df_full, part_id_input, order_qty, piece_weight, cost_per_part, thr_label, n_est):
    """
    Run prediction both WITH and WITHOUT multi-defect features.
    Returns comprehensive results dictionary for logging.
    """
    results = {
        'part_id': part_id_input,
        'order_quantity': order_qty,
        'piece_weight_lbs': piece_weight,
        'cost_per_part': cost_per_part,
        'threshold': thr_label,
    }
    
    # Get part-specific record count
    part_specific_records = len(df_full[df_full["part_id"] == part_id_input])
    results['part_historical_records'] = part_specific_records
    results['data_confidence'] = get_data_confidence_level(part_specific_records)[1]
    
    # Get defect columns
    defect_cols = [c for c in df_full.columns if c.endswith("_rate")]
    
    # Part history
    part_history = df_full[df_full["part_id"] == part_id_input]
    
    # Calculate historical defect rates
    if len(part_history) > 0:
        defect_means = part_history[defect_cols].mean()
        historical_scrap_avg = part_history['scrap_percent'].mean()
        historical_scrap_max = part_history['scrap_percent'].max()
        historical_scrap_min = part_history['scrap_percent'].min()
    else:
        defect_means = df_full[defect_cols].mean()
        historical_scrap_avg = df_full['scrap_percent'].mean()
        historical_scrap_max = df_full['scrap_percent'].max()
        historical_scrap_min = df_full['scrap_percent'].min()
    
    results['historical_scrap_avg'] = historical_scrap_avg
    results['historical_scrap_max'] = historical_scrap_max
    results['historical_scrap_min'] = historical_scrap_min
    
    # Add top defect rates to results
    top_defects = defect_means.sort_values(ascending=False).head(5)
    for i, (defect, rate) in enumerate(top_defects.items(), 1):
        results[f'top_defect_{i}_name'] = defect.replace('_rate', '')
        results[f'top_defect_{i}_rate'] = rate
    
    # Multi-defect analysis
    if len(part_history) > 0:
        latest_row = part_history.iloc[-1]
        multi_analysis = get_multi_defect_analysis(latest_row, defect_cols)
        results['n_defect_types'] = multi_analysis['n_defect_types']
        results['is_multi_defect'] = multi_analysis['is_multi_defect']
        results['total_defect_rate'] = multi_analysis['total_rate']
        results['primary_processes'] = ', '.join(multi_analysis['primary_processes'][:3])
    else:
        results['n_defect_types'] = 0
        results['is_multi_defect'] = False
        results['total_defect_rate'] = 0
        results['primary_processes'] = ''
    
    # Prepare data
    df_part = prepare_part_specific_data(df_full, part_id_input, piece_weight, thr_label, min_samples=30)
    
    if df_part is None:
        results['error'] = 'Insufficient data diversity'
        return results
    
    results['training_samples'] = len(df_part)
    
    # ============================================================
    # PREDICTION WITHOUT MULTI-DEFECT FEATURES
    # ============================================================
    df_train, df_calib, df_test = time_split_621(df_part)
    
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    df_train_f = attach_train_features(df_train.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib_f = attach_train_features(df_calib.copy(), mtbf_train, part_freq_train, default_mtbf, default_freq)
    
    # WITHOUT multi-defect
    X_train_no, y_train_no, feats_no = make_xy(df_train_f.copy(), thr_label, True, use_multi_defect=False)
    X_calib_no, y_calib_no, _ = make_xy(df_calib_f.copy(), thr_label, True, use_multi_defect=False)
    
    _, cal_no, method_no = train_and_calibrate(X_train_no, y_train_no, X_calib_no, y_calib_no, n_est)
    
    # Build input for prediction (WITHOUT)
    if len(part_history) > 0:
        hist_mttf = float(mtbf_train[mtbf_train["part_id"] == part_id_input]["mttf_scrap"].values[0]) \
            if part_id_input in mtbf_train["part_id"].values else default_mtbf
        hist_freq = float(part_freq_train.get(part_id_input, default_freq))
    else:
        hist_mttf = default_mtbf
        hist_freq = default_freq
    
    input_dict_no = {
        "order_quantity": order_qty,
        "piece_weight_lbs": piece_weight,
        "mttf_scrap": hist_mttf,
        "part_freq": hist_freq,
    }
    for dc in defect_cols:
        input_dict_no[dc] = defect_means.get(dc, 0.0)
    
    X_input_no = pd.DataFrame([input_dict_no])[feats_no]
    proba_no = cal_no.predict_proba(X_input_no)[0, 1]
    
    results['WITHOUT_scrap_risk_pct'] = proba_no * 100
    results['WITHOUT_reliability_pct'] = (1 - proba_no) * 100
    results['WITHOUT_expected_scrap_pcs'] = order_qty * proba_no
    results['WITHOUT_expected_loss_dollars'] = order_qty * proba_no * cost_per_part
    results['WITHOUT_n_features'] = len(feats_no)
    results['WITHOUT_calibration'] = method_no
    
    # ============================================================
    # PREDICTION WITH MULTI-DEFECT FEATURES
    # ============================================================
    X_train_yes, y_train_yes, feats_yes = make_xy(df_train_f.copy(), thr_label, True, use_multi_defect=True)
    X_calib_yes, y_calib_yes, _ = make_xy(df_calib_f.copy(), thr_label, True, use_multi_defect=True)
    
    _, cal_yes, method_yes = train_and_calibrate(X_train_yes, y_train_yes, X_calib_yes, y_calib_yes, n_est)
    
    # Build input for prediction (WITH)
    input_dict_yes = input_dict_no.copy()
    
    # Add multi-defect features
    if len(part_history) > 0:
        n_defect_types = (part_history[defect_cols] > 0).sum(axis=1).mean()
        has_multiple = 1 if n_defect_types >= MULTI_DEFECT_THRESHOLD else 0
        total_defect = part_history[defect_cols].sum(axis=1).mean()
        max_defect = part_history[defect_cols].max(axis=1).mean()
        defect_conc = max_defect / (total_defect + 0.001)
    else:
        n_defect_types = df_full['n_defect_types'].mean() if 'n_defect_types' in df_full.columns else 0
        has_multiple = 0
        total_defect = df_full['total_defect_rate'].mean() if 'total_defect_rate' in df_full.columns else 0
        max_defect = df_full['max_defect_rate'].mean() if 'max_defect_rate' in df_full.columns else 0
        defect_conc = max_defect / (total_defect + 0.001)
    
    input_dict_yes["n_defect_types"] = n_defect_types
    input_dict_yes["has_multiple_defects"] = has_multiple
    input_dict_yes["total_defect_rate"] = total_defect
    input_dict_yes["max_defect_rate"] = max_defect
    input_dict_yes["defect_concentration"] = defect_conc
    
    # Interaction terms
    if 'shift_rate' in defect_means.index and 'tear_up_rate' in defect_means.index:
        input_dict_yes["shift_x_tearup"] = defect_means['shift_rate'] * defect_means['tear_up_rate']
    if 'shrink_rate' in defect_means.index and 'gas_porosity_rate' in defect_means.index:
        input_dict_yes["shrink_x_porosity"] = defect_means['shrink_rate'] * defect_means['gas_porosity_rate']
    if 'shrink_rate' in defect_means.index and 'shrink_porosity_rate' in defect_means.index:
        input_dict_yes["shrink_x_shrink_porosity"] = defect_means['shrink_rate'] * defect_means['shrink_porosity_rate']
    if 'core_rate' in defect_means.index and 'sand_rate' in defect_means.index:
        input_dict_yes["core_x_sand"] = defect_means['core_rate'] * defect_means['sand_rate']
    
    X_input_yes = pd.DataFrame([input_dict_yes])[feats_yes]
    proba_yes = cal_yes.predict_proba(X_input_yes)[0, 1]
    
    results['WITH_scrap_risk_pct'] = proba_yes * 100
    results['WITH_reliability_pct'] = (1 - proba_yes) * 100
    results['WITH_expected_scrap_pcs'] = order_qty * proba_yes
    results['WITH_expected_loss_dollars'] = order_qty * proba_yes * cost_per_part
    results['WITH_n_features'] = len(feats_yes)
    results['WITH_calibration'] = method_yes
    
    # ============================================================
    # CALCULATE DIFFERENCES
    # ============================================================
    results['DIFF_scrap_risk_pct'] = results['WITH_scrap_risk_pct'] - results['WITHOUT_scrap_risk_pct']
    results['DIFF_expected_scrap_pcs'] = results['WITH_expected_scrap_pcs'] - results['WITHOUT_expected_scrap_pcs']
    results['DIFF_expected_loss_dollars'] = results['WITH_expected_loss_dollars'] - results['WITHOUT_expected_loss_dollars']
    
    # Process diagnosis (using WITH model prediction)
    defect_predictions = []
    for dc in defect_cols:
        hist_rate = defect_means.get(dc, 0.0)
        pred_rate = hist_rate * proba_yes
        defect_name = dc.replace("_rate", "").replace("_", " ").title()
        defect_predictions.append({
            "Defect": defect_name,
            "Defect_Code": dc,
            "Historical Rate (%)": hist_rate * 100,
            "Predicted Rate (%)": pred_rate * 100,
        })
    
    defect_df = pd.DataFrame(defect_predictions).sort_values("Predicted Rate (%)", ascending=False)
    diagnosis = diagnose_root_causes(defect_df.head(10))
    
    if not diagnosis.empty:
        results['top_process_1'] = diagnosis.iloc[0]['Process'] if len(diagnosis) > 0 else ''
        results['top_process_1_contribution'] = diagnosis.iloc[0]['Contribution (%)'] if len(diagnosis) > 0 else 0
        results['top_process_2'] = diagnosis.iloc[1]['Process'] if len(diagnosis) > 1 else ''
        results['top_process_2_contribution'] = diagnosis.iloc[1]['Contribution (%)'] if len(diagnosis) > 1 else 0
        results['top_process_3'] = diagnosis.iloc[2]['Process'] if len(diagnosis) > 2 else ''
        results['top_process_3_contribution'] = diagnosis.iloc[2]['Contribution (%)'] if len(diagnosis) > 2 else 0
    
    return results


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
n_est = st.sidebar.slider("Number of trees", 50, 300, DEFAULT_ESTIMATORS, 10)

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


# -------------------------------
# Load data
# -------------------------------
df_base = load_and_clean(csv_path, add_multi_defect=True)
df_base = calculate_process_indices(df_base)

n_parts = df_base["part_id"].nunique()
n_work_orders = len(df_base)
st.info(f"✅ Loaded {n_work_orders:,} work orders | {n_parts} unique parts")


# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict & Log", "📋 Results Export", "📊 Analysis"])


# ================================================================
# TAB 1: PREDICT & LOG
# ================================================================
with tab1:
    st.header("🔮 Predict & Log Results")
    
    st.markdown("""
    Each prediction runs **TWO models** (WITH and WITHOUT multi-defect features) 
    and logs comprehensive results for later analysis.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    part_id_input = col1.text_input("Part ID", value="15")
    order_qty = col2.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = col3.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = col4.number_input("Cost per Part ($)", min_value=0.1, value=10.0)
    
    if st.button("🎯 Run Dual Prediction & Log Results"):
        with st.spinner("Running predictions..."):
            try:
                results = run_dual_prediction(
                    df_base, part_id_input, order_qty, piece_weight, 
                    cost_per_part, thr_label, n_est
                )
                
                # Log the results
                log_prediction_result(results)
                
                # Display summary
                st.success(f"✅ Prediction logged! (#{len(st.session_state['prediction_results'])})")
                
                # Show data confidence
                display_data_confidence_banner(results['part_historical_records'], part_id_input)
                
                # Multi-defect alert
                if results.get('is_multi_defect', False):
                    st.error(f"🚨 **MULTI-DEFECT PATTERN**: {results['n_defect_types']} defect types detected")
                
                # Comparison display
                st.markdown("### 📊 Prediction Comparison")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.markdown("**WITHOUT Multi-Defect**")
                    st.metric("Scrap Risk", f"{results['WITHOUT_scrap_risk_pct']:.1f}%")
                    st.metric("Expected Scrap", f"{results['WITHOUT_expected_scrap_pcs']:.1f} pcs")
                    st.metric("Expected Loss", f"${results['WITHOUT_expected_loss_dollars']:.2f}")
                
                with comp_col2:
                    st.markdown("**WITH Multi-Defect (V3.1)**")
                    st.metric("Scrap Risk", f"{results['WITH_scrap_risk_pct']:.1f}%")
                    st.metric("Expected Scrap", f"{results['WITH_expected_scrap_pcs']:.1f} pcs")
                    st.metric("Expected Loss", f"${results['WITH_expected_loss_dollars']:.2f}")
                
                with comp_col3:
                    st.markdown("**DIFFERENCE**")
                    diff_risk = results['DIFF_scrap_risk_pct']
                    st.metric("Risk Δ", f"{diff_risk:+.1f}%", 
                             delta_color="inverse" if diff_risk > 0 else "normal")
                    st.metric("Scrap Δ", f"{results['DIFF_expected_scrap_pcs']:+.1f} pcs")
                    st.metric("Loss Δ", f"${results['DIFF_expected_loss_dollars']:+.2f}")
                
                # Process diagnosis
                st.markdown("### 🏭 Top Process Concerns (Campbell)")
                proc_data = []
                for i in range(1, 4):
                    proc = results.get(f'top_process_{i}', '')
                    contrib = results.get(f'top_process_{i}_contribution', 0)
                    if proc:
                        proc_data.append({'Process': proc, 'Contribution (%)': contrib})
                
                if proc_data:
                    st.dataframe(pd.DataFrame(proc_data), use_container_width=True)
                
                # Historical context
                with st.expander("📋 Historical Context"):
                    st.markdown(f"""
- **Historical Scrap Avg**: {results['historical_scrap_avg']:.2f}%
- **Historical Scrap Range**: {results['historical_scrap_min']:.2f}% - {results['historical_scrap_max']:.2f}%
- **Training Samples Used**: {results['training_samples']}
- **Features (WITHOUT)**: {results['WITHOUT_n_features']}
- **Features (WITH)**: {results['WITH_n_features']}
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ================================================================
# TAB 2: RESULTS EXPORT
# ================================================================
with tab2:
    st.header("📋 Prediction Results Log")
    
    df_results = get_results_dataframe()
    
    if df_results.empty:
        st.info("No predictions logged yet. Go to the 'Predict & Log' tab to run predictions.")
    else:
        st.success(f"📊 **{len(df_results)} predictions logged**")
        
        # Column selection
        st.markdown("### Select Columns to Display/Export")
        
        all_columns = df_results.columns.tolist()
        
        # Group columns by category
        input_cols = ['prediction_id', 'timestamp', 'part_id', 'order_quantity', 
                      'piece_weight_lbs', 'cost_per_part', 'threshold']
        context_cols = ['part_historical_records', 'data_confidence', 'training_samples',
                       'historical_scrap_avg', 'historical_scrap_max', 'historical_scrap_min']
        defect_cols = [c for c in all_columns if 'top_defect' in c or 'n_defect' in c or 
                      'is_multi_defect' in c or 'total_defect_rate' in c]
        without_cols = [c for c in all_columns if c.startswith('WITHOUT_')]
        with_cols = [c for c in all_columns if c.startswith('WITH_')]
        diff_cols = [c for c in all_columns if c.startswith('DIFF_')]
        process_cols = [c for c in all_columns if 'top_process' in c or 'primary_processes' in c]
        
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            show_inputs = st.checkbox("Input Parameters", value=True)
            show_context = st.checkbox("Data Context", value=True)
            show_defects = st.checkbox("Defect Analysis", value=True)
        
        with col_select2:
            show_without = st.checkbox("WITHOUT Multi-Defect Results", value=True)
            show_with = st.checkbox("WITH Multi-Defect Results", value=True)
            show_diff = st.checkbox("Differences", value=True)
            show_process = st.checkbox("Process Diagnosis", value=True)
        
        # Build selected columns
        selected_cols = []
        if show_inputs:
            selected_cols.extend([c for c in input_cols if c in all_columns])
        if show_context:
            selected_cols.extend([c for c in context_cols if c in all_columns])
        if show_defects:
            selected_cols.extend([c for c in defect_cols if c in all_columns])
        if show_without:
            selected_cols.extend([c for c in without_cols if c in all_columns])
        if show_with:
            selected_cols.extend([c for c in with_cols if c in all_columns])
        if show_diff:
            selected_cols.extend([c for c in diff_cols if c in all_columns])
        if show_process:
            selected_cols.extend([c for c in process_cols if c in all_columns])
        
        # Remove duplicates while preserving order
        selected_cols = list(dict.fromkeys(selected_cols))
        
        if selected_cols:
            df_display = df_results[selected_cols]
            
            st.markdown("### 📊 Results Table")
            st.dataframe(df_display, use_container_width=True)
            
            # Download buttons
            st.markdown("### 📥 Download Options")
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                csv_selected = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Selected Columns (CSV)",
                    data=csv_selected,
                    file_name=f"prediction_results_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with dl_col2:
                csv_all = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download ALL Columns (CSV)",
                    data=csv_all,
                    file_name=f"prediction_results_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with dl_col3:
                # JSON download (no extra dependencies needed)
                json_data = df_results.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON (All Data)",
                    data=json_data,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("Please select at least one column category to display.")
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        if 'WITH_scrap_risk_pct' in df_results.columns and 'WITHOUT_scrap_risk_pct' in df_results.columns:
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("**WITHOUT Multi-Defect**")
                st.write(f"- Mean Scrap Risk: {df_results['WITHOUT_scrap_risk_pct'].mean():.2f}%")
                st.write(f"- Std Dev: {df_results['WITHOUT_scrap_risk_pct'].std():.2f}%")
                st.write(f"- Range: {df_results['WITHOUT_scrap_risk_pct'].min():.2f}% - {df_results['WITHOUT_scrap_risk_pct'].max():.2f}%")
            
            with summary_col2:
                st.markdown("**WITH Multi-Defect**")
                st.write(f"- Mean Scrap Risk: {df_results['WITH_scrap_risk_pct'].mean():.2f}%")
                st.write(f"- Std Dev: {df_results['WITH_scrap_risk_pct'].std():.2f}%")
                st.write(f"- Range: {df_results['WITH_scrap_risk_pct'].min():.2f}% - {df_results['WITH_scrap_risk_pct'].max():.2f}%")
            
            st.markdown("**Difference (WITH - WITHOUT)**")
            mean_diff = df_results['DIFF_scrap_risk_pct'].mean()
            st.write(f"- Mean Risk Difference: {mean_diff:+.2f}%")
            st.write(f"- Predictions where WITH > WITHOUT: {(df_results['DIFF_scrap_risk_pct'] > 0).sum()} / {len(df_results)}")


# ================================================================
# TAB 3: ANALYSIS
# ================================================================
with tab3:
    st.header("📊 Comparative Analysis")
    
    df_results = get_results_dataframe()
    
    if df_results.empty or len(df_results) < 2:
        st.info("Need at least 2 predictions for analysis. Run more predictions in the 'Predict & Log' tab.")
    else:
        st.markdown("### 📈 Visual Comparison")
        
        # Scatter plot: WITH vs WITHOUT
        if 'WITH_scrap_risk_pct' in df_results.columns and 'WITHOUT_scrap_risk_pct' in df_results.columns:
            fig = px.scatter(
                df_results,
                x='WITHOUT_scrap_risk_pct',
                y='WITH_scrap_risk_pct',
                color='data_confidence',
                hover_data=['part_id', 'part_historical_records'],
                title='Scrap Risk: WITH vs WITHOUT Multi-Defect Features',
                labels={
                    'WITHOUT_scrap_risk_pct': 'WITHOUT Multi-Defect (%)',
                    'WITH_scrap_risk_pct': 'WITH Multi-Defect (%)'
                }
            )
            
            # Add diagonal line (y = x)
            max_val = max(df_results['WITHOUT_scrap_risk_pct'].max(), df_results['WITH_scrap_risk_pct'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='No Difference Line',
                line=dict(dash='dash', color='gray')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Points **above** the diagonal: WITH model predicts **higher** risk
            - Points **below** the diagonal: WITHOUT model predicts **higher** risk
            - Points **on** the diagonal: Both models agree
            """)
        
        # Bar chart by part
        st.markdown("### 📊 Risk by Part ID")
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='WITHOUT Multi-Defect',
            x=df_results['part_id'].astype(str),
            y=df_results['WITHOUT_scrap_risk_pct'],
            marker_color='lightgray'
        ))
        fig_bar.add_trace(go.Bar(
            name='WITH Multi-Defect',
            x=df_results['part_id'].astype(str),
            y=df_results['WITH_scrap_risk_pct'],
            marker_color='#ff6b6b'
        ))
        
        fig_bar.update_layout(
            barmode='group',
            title='Scrap Risk Comparison by Part',
            xaxis_title='Part ID',
            yaxis_title='Scrap Risk (%)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation with historical records
        st.markdown("### 📉 Impact of Data Availability")
        
        if 'part_historical_records' in df_results.columns:
            fig_records = px.scatter(
                df_results,
                x='part_historical_records',
                y='DIFF_scrap_risk_pct',
                color='is_multi_defect',
                hover_data=['part_id'],
                title='Risk Difference vs Historical Records',
                labels={
                    'part_historical_records': 'Number of Historical Records',
                    'DIFF_scrap_risk_pct': 'Risk Difference (WITH - WITHOUT) %'
                }
            )
            fig_records.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_records, use_container_width=True)


st.markdown("---")
st.caption("🏭 Foundry Scrap Risk Dashboard **v3.1** | Multi-Defect Intelligence + Results Export | Campbell (2003)")
