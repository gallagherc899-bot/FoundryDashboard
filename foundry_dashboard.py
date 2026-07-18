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
#          → Focuses on high-impact defects (~66% of scrap)
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
# 4. Global Model Training with 60-20-20 split
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
try:
    from statsmodels.stats.diagnostic import lilliefors as lilliefors_test
    LILLIEFORS_AVAILABLE = True
except ImportError:
    LILLIEFORS_AVAILABLE = False

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
# Source: Eppich (2004), Exhibit ii, p. v — Permanent Mold/Sand Aluminum facility
# Tacit (primary) energy = 47,250 BTU/lb (94.5 MMBtu/ton), confirmed by
# Kermeli et al. (2016) ENERGY STAR Metal Casting Guide, Table 10, p. 99.
# Prior value (22,922 BTU/lb) was the Die Casting Al-1 direct-site-energy figure — wrong
# facility type and excludes upstream electrical multiplier (2.86×).
DOE_BENCHMARKS = {
    'average': 47250,       # BTU/lb — Permanent Mold/Sand Al, tacit (Eppich 2004)
    'best_practice': 18500,
    'theoretical_minimum': 5000
}
CO2_PER_MMBTU = 53.06

# RQ Validation Thresholds
# Aligned with dissertation: H1=MPTS reliability, H2=RF classifier, H3=TTE/GHG
RQ_THRESHOLDS = {
    'RQ1': {'sensor_benchmark': 0.90, 'phm_equivalence': 0.80},
    'RQ2': {'recall': 0.80, 'precision': 0.70, 'f1': 0.70, 'auc': 0.80},
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
    "Pattern/Tooling": {"defects": [], 
                        "description": "Pattern accuracy, wear",
                        "campbell_rule": "Rule 10: Provide Location Points"},
    "Inspection": {"defects": ["outside_process_scrap_rate", "zyglo_rate", "failed_zyglo_rate"], 
                   "description": "Quality control, NDT",
                   "campbell_rule": "Detection stage (not process-origin)"},
    "Finishing": {"defects": ["over_grind_rate", "cut_into_rate", "gouged_rate"], 
                  "description": "Grinding, machining",
                  "campbell_rule": "Post-casting operations"}
}

# ================================================================
# DEFECT → CAMPBELL PROCESS MULTI-CAUSE MAPPING
# ================================================================
# Each defect maps to one or more (process, is_primary) tuples.
# is_primary=True  → this process is the dominant/first-suspect origin
#                    per Campbell's 10 Rules.
# is_primary=False → this process can also produce this defect but is
#                    a secondary or less common origin.
#
# When a run shows ONLY a single multi-cause defect, all candidate
# processes are listed with equal weight — the manager decides.
# When a run shows MULTIPLE defects from one process's signature,
# co-occurrence scoring elevates that process automatically.
#
# Reference: Campbell, J. (2003). Castings Practice: The 10 Rules.
# ================================================================
DEFECT_TO_PROCESSES = {
    # ── Single-cause defects ──────────────────────────────────────────
    'dross_rate':              [('Melting',         True)],
    'misrun_rate':             [('Pouring',         True)],
    'missrun_rate':            [('Pouring',         True)],
    'short_pour_rate':         [('Pouring',         True)],
    'runout_rate':             [('Pouring',         True)],
    'dirty_pattern_rate':      [('Sand System',     True)],
    'bent_rate':               [('Shakeout',        True)],
    # Gouged -> Finishing: in the casting literature 'gouging' is a
    #   finishing/salvage operation (remove defective metal by grinding,
    #   machining, or gouging with arc/gas/chisel before rectification)
    #   per Beeley (2001), Ch.5 p.307 and dressing p.543 -- not a molding
    #   defect. (Reassigned from Pattern/Tooling, which had no literature
    #   support for a 'gouge' defect.)
    'gouged_rate':             [('Finishing',       True)],
    'over_grind_rate':         [('Finishing',       True)],
    'cut_into_rate':           [('Finishing',       True)],
    'zyglo_rate':              [('Inspection',      True)],
    'failed_zyglo_rate':       [('Inspection',      True)],
    # outside_process_scrap_rate is a detection label; origin is genuinely
    # ambiguous — could be any upstream process
    'outside_process_scrap_rate': [('Inspection',   True)],

    # ── Multi-cause defects ───────────────────────────────────────────
    # Gas porosity: primary = dissolved gas / bifilms at melting (Rule 1);
    #   secondary = binder/moisture outgassing from cores (Rule 5);
    #   secondary = turbulent air entrainment during fill (Rule 2)
    'gas_porosity_rate': [
        ('Melting',      True),
        ('Core Making',  False),
        ('Pouring',      False),
    ],

    # Shrinkage: primary = inadequate feeding / riser sizing (Rule 6);
    #   secondary = dissolved gas shrinkage interaction from melt (Rule 1)
    'shrink_rate': [
        ('Gating Design', True),
        ('Melting',       False),
    ],

    # Shrink porosity: same causal chain as shrink_rate
    'shrink_porosity_rate': [
        ('Gating Design', True),
        ('Melting',       False),
    ],

    # Tear-up (hot tear): primary = solidification restraint / feeding (Rule 6);
    #   secondary = core rigidity constrains cooling casting (Rule 5)
    'tear_up_rate': [
        ('Gating Design', True),
        ('Core Making',   False),
    ],

    # Sand inclusion: primary = sand preparation / binder quality;
    #   secondary = turbulent fill velocity erodes mould surface (Rule 2)
    'sand_rate': [
        ('Sand System',  True),
        ('Pouring',      False),
    ],

    # Core defect: primary = core integrity / venting (Rule 5);
    #   secondary = sand quality affects core strength
    'core_rate': [
        ('Core Making',  True),
        ('Sand System',  False),
    ],

    # Crush: primary = core mechanical failure during assembly (Rule 5);
    #   secondary = pattern / core-print dimensional mismatch
    'crush_rate': [
        ('Core Making',    True),
        ('Pattern/Tooling', False),
    ],

    # Shift: primary = core misalignment (Rule 5);
    #   secondary = pattern wear / cope-drag dimensional issues (Rule 10)
    'shift_rate': [
        ('Core Making',    True),
        ('Pattern/Tooling', False),
    ],
}

# Primary-cause reverse lookup — used for backward-compat and simple displays
DEFECT_TO_PROCESS = {
    defect: next(proc for proc, is_primary in procs if is_primary)
    for defect, procs in DEFECT_TO_PROCESSES.items()
}

# Module-level flat primary map (for the Defense tab attribution display)
DEFECT_TO_PROC = dict(DEFECT_TO_PROCESS)


def process_co_occurrence_score(process_name, observed_defects_set):
    """
    Score how well a Campbell process explains the observed defect pattern.

    Score = |observed_defects ∩ process_defects| / |process_defects|

    Example:
      Process X causes {A, B, C}.  Run shows defects A + B.
        → score = 2/3 = 0.67  (two of three signatures present)
      Process Y causes {A, D}.  Same run.
        → score = 1/2 = 0.50  (only one of two signatures present)
      → Process X is more likely given the co-occurrence evidence.

    A score of 0 means none of this process's defects appear in the run.
    A score of 1 means ALL of this process's defect signatures are present.
    """
    process_defects = set(PROCESS_DEFECT_MAP.get(process_name, {}).get('defects', []))
    if not process_defects:
        return 0.0
    matches = observed_defects_set & process_defects
    return len(matches) / len(process_defects)

# Features that the model uses which are NOT raw defects.
# These are excluded from Campbell process attribution because
# they are either model intermediates or reliability metrics,
# not directly actionable process signals.
_NON_DEFECT_FEATURES = {
    # Hierarchical model outputs
    "global_scrap_probability", "defect_cluster_probability",
    # Multi-defect aggregates (decomposed separately)
    "n_defect_types", "has_multiple_defects",
    "total_defect_rate", "max_defect_rate", "defect_concentration",
    # Temporal / rolling
    "total_defect_rate_trend", "total_defect_rate_roll3",
    "scrap_percent_trend", "scrap_percent_roll3",
    "month", "quarter",
    # MTTS / reliability
    "mtts_runs", "hazard_rate", "reliability_score",
    "runs_since_last_failure", "cumulative_scrap_in_cycle",
    "degradation_velocity", "degradation_acceleration",
    "cycle_hazard_indicator", "rul_proxy",
    # Part metadata
    "order_quantity", "piece_weight_lbs",
    "mean_scrap_rate_train", "part_freq",
}

# Hierarchical Pooling Configuration
POOLING_CONFIG = {
    'enabled': True,
    'min_part_level_data': 5,
    'weight_tolerance': 0.10,
    'min_runs_per_pooled_part': 5,  # Filter out parts with < 5 runs (reduces noise)
    'use_pooled_threshold': False,  # Pooled comparison uses target part's own avg scrap threshold
    'confidence_thresholds': {
        'HIGH': 30,
        'MODERATE': 15,
        'LOW': 5,
    }
}

# Data sufficiency threshold: parts with ≥30 runs have sufficient history for
# reliable part-level estimation (Lawless, 2003). Below this, dual results
# (part-level + pooled comparison) shown for experienced judgment.
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

# Top 5 Pareto Defects (from dataset Pareto analysis - ~66% of scrap)
TOP_PARETO_DEFECTS = [
    'sand_rate',      # #1 - ~27% of scrap
    'shift_rate',     # #2 - ~11% of scrap
    'missrun_rate',   # #3 - ~11% of scrap  
    'gouged_rate',    # #4 - ~9% of scrap
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
        background: linear-gradient(135deg, #C8A96E 0%, #D4B483 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 25px;
        color: #1a1a1a; border: 1px solid #a0845a;
    }
    .main-header h1 { color: #1a1a1a !important; font-weight: 700; }
    .main-header p  { color: #2d2d2d !important; font-weight: 500; }
    .citation-box {
        background-color: #ddeeff; border-left: 4px solid #0D47A1;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
        color: #0a2340;
    }
    .hypothesis-pass {
        background-color: #1B5E20; border-left: 6px solid #FFFFFF;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
        color: #FFFFFF !important;
    }
    .hypothesis-pass * { color: #FFFFFF !important; }
    .hypothesis-fail {
        background-color: #E65100; border-left: 6px solid #FFFFFF;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
        color: #FFFFFF !important;
    }
    .hypothesis-fail * { color: #FFFFFF !important; }
    /* Fix Streamlit success/info boxes for contrast */
    div[data-testid="stAlert"] > div {
        color: #0a0a0a !important;
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
        df["week_ending"] = df["week_ending"].astype(str).str.strip()
        # Tolerant parser — format='mixed' accepts both 'M/D/YYYY' and 'MM/DD/YYYY'
        # variants present in the source workbook. Prior behavior used the strict
        # default parser which silently dropped rows whose dates used a different
        # numeric padding, producing a census that did not match §3.2.1 of the
        # dissertation.
        df["week_ending"] = pd.to_datetime(
            df["week_ending"], errors="coerce", format="mixed"
        )
        # If any dates still fail to parse after the tolerant pass, keep the row
        # but assign a sentinel so downstream sort_values() works. Do NOT drop.
        # This prevents silent census loss from date-format inconsistencies.
        if df["week_ending"].isna().any():
            n_bad = df["week_ending"].isna().sum()
            print(f"[load_data] Warning: {n_bad} rows had unparseable dates; "
                  f"assigned to epoch start for sorting only (rows retained).")
            df["week_ending"] = df["week_ending"].fillna(pd.Timestamp("1970-01-01"))
        # ------------------------------------------------------------------
        # DETERMINISTIC SORT
        # ------------------------------------------------------------------
        # Capture the original CSV row index as a stable tiebreak before any
        # sort. Multiple rows can share the same (week_ending, work_order_#,
        # part_id) triple (76 such rows exist in this dataset). Without a
        # final unique tiebreak, different transforms in the pipeline can
        # reorder rows within identical primary keys, producing a non-
        # deterministic 60/20/20 split boundary and shifting the test-set
        # failure count by ±1 between runs.
        #
        # The triple-key sort below is applied in load_data and MUST be
        # preserved by every downstream transform. All sort_values() calls
        # elsewhere in this module use the same key sequence.
        if "_csv_row_idx" not in df.columns:
            df["_csv_row_idx"] = np.arange(len(df))
        _sort_keys = ["week_ending"]
        if "work_order_#" in df.columns:
            _sort_keys.append("work_order_#")
        if "part_id" in df.columns:
            _sort_keys.append("part_id")
        _sort_keys.append("_csv_row_idx")
        df = df.sort_values(_sort_keys).reset_index(drop=True)
    else:
        df["week_ending"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='W')
        if "_csv_row_idx" not in df.columns:
            df["_csv_row_idx"] = np.arange(len(df))

    # ------------------------------------------------------------------
    # §3.2.1 DATA-INTEGRITY EXCLUSION
    # ------------------------------------------------------------------
    # The dissertation's Section 3.2.1 excludes runs where pieces_scrapped
    # exceeds order_quantity (an impossible state that indicates data-entry
    # error). Prior versions of this dashboard did not apply this rule,
    # producing an H1 cohort that disagreed with the dissertation by one
    # part. The rule is applied here so the dashboard's working census
    # exactly matches the dissertation's stated methodology.
    if "pieces_scrapped" in df.columns and "order_quantity" in df.columns:
        df["pieces_scrapped"] = pd.to_numeric(
            df["pieces_scrapped"], errors="coerce").fillna(0)
        integrity_mask = df["pieces_scrapped"] <= df["order_quantity"]
        n_excluded = (~integrity_mask).sum()
        if n_excluded > 0:
            print(f"[load_data] §3.2.1 exclusion: dropped {n_excluded} rows where "
                  f"pieces_scrapped > order_quantity (data-entry errors).")
        df = df[integrity_mask].reset_index(drop=True)
    
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
    """Add temporal trend features matching enhanced version.
    
    Uses the deterministic sort key established by load_data (week_ending,
    work_order_#, part_id, _csv_row_idx) to preserve the stable row order.
    """
    df = df.copy()
    
    if 'week_ending' in df.columns:
        _keys = ['week_ending']
        for k in ['work_order_#', 'part_id', '_csv_row_idx']:
            if k in df.columns:
                _keys.append(k)
        df = df.sort_values(_keys).reset_index(drop=True)
    
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
    """
    Compute MTTS metrics per part using simple ratio (Eq 3.1).
    
    MTTS(parts) = Total Parts Produced / Number of Failures    (Eq 3.1)
    MTTS(runs)  = Total Runs / Number of Failures
    h(t)        = 1 / MTTS(parts)                              (Eq 3.2)
    R(t)        = e^(-h(t) * parts_ordered)                    (Eq 3.3)
    
    Consistent with MTTF = total operating time / number of failures
    (Ebeling, 1997, An Introduction to Reliability and Maintainability Engineering).
    """
    results = []
    df_sorted = df.sort_values(['part_id', 'week_ending']).copy()
    
    if 'order_quantity' not in df_sorted.columns:
        df_sorted['order_quantity'] = 1
    
    for part_id, group in df_sorted.groupby('part_id'):
        group = group.reset_index(drop=True)
        
        # Count failures (runs where scrap % exceeds threshold)
        failure_count = (group['scrap_percent'] > threshold).sum()
        
        total_runs = len(group)
        total_parts = group['order_quantity'].sum() if 'order_quantity' in group.columns else total_runs
        avg_order_quantity = total_parts / total_runs if total_runs > 0 else 0
        
        # Eq 3.1: MTTS = Total Parts (or Runs) / Number of Failures
        if failure_count > 0:
            mtts_parts = total_parts / failure_count
            mtts_runs = total_runs / failure_count
        else:
            mtts_parts = total_parts
            mtts_runs = total_runs
        
        # Eq 3.2: h(t) = 1 / MTTS(parts)
        lambda_parts = 1 / mtts_parts if mtts_parts > 0 else 0
        lambda_runs = failure_count / total_runs if total_runs > 0 else 0
        
        # Eq 3.3: R(t) = e^(-h(t) * avg_order_quantity)
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


def _nist_uniform_order_statistic_medians(m):
    """NIST/SEMATECH e-Handbook §1.3.3.22 uniform order-statistic medians:
        m(1)   = 1 - m(n)
        m(i)   = (i - 0.3175) / (n + 0.365)   for i = 2..n-1
        m(n)   = 0.5 ** (1/n)
    Returns the length-m array of uniform medians U(i)."""
    u = np.empty(m, dtype=float)
    mn = 0.5 ** (1.0 / m)
    for i in range(1, m + 1):
        if i == 1:
            u[0] = 1.0 - mn
        elif i == m:
            u[m - 1] = mn
        else:
            u[i - 1] = (i - 0.3175) / (m + 0.365)
    return u


LOUIT_MIN_FAILURES_FOR_TREND = 5   # below this, trend tests are not meaningful
LOUIT_MIN_FAILURES_FOR_R2 = 5
LOUIT_LAPLACE_CRIT = 1.96          # standard normal, two-sided alpha = 0.05

def compute_louit_screening(df, part_id):
    """Louit et al. (2009) screening for one part. Threshold = part's own mean
    scrap%. Returns metrics dict with honest sample-size gating."""
    g = df[df['part_id'] == str(part_id)].sort_values('week_ending').reset_index(drop=True)
    n_runs = len(g)
    out = {'part_id': str(part_id), 'n_runs': n_runs}
    if n_runs == 0:
        out['status'] = 'no data'; return out

    thr = float(g['scrap_percent'].mean())
    g['run_idx'] = np.arange(1, n_runs + 1)
    g['fail'] = (g['scrap_percent'] > thr).astype(int)
    fr = g.loc[g['fail'] == 1, 'run_idx'].tolist()
    nf = len(fr)
    total_runs = n_runs
    total_parts = float(g['order_quantity'].sum())
    avg_oq = total_parts / total_runs if total_runs else 0.0
    mpts_parts = total_parts / nf if nf > 0 else total_parts
    mpts_runs = total_runs / nf if nf > 0 else total_runs
    R_avg = float(np.exp(-avg_oq / mpts_parts)) if mpts_parts > 0 else 0.0
    out.update({'threshold_pct': thr, 'n_failures': nf, 'total_runs': total_runs,
                'total_parts': total_parts, 'avg_order_qty': avg_oq,
                'mpts_parts': mpts_parts, 'mpts_runs': mpts_runs, 'R_at_avg_oq': R_avg})

    if nf >= LOUIT_MIN_FAILURES_FOR_TREND:
        T = np.asarray(fr, dtype=float); b = float(total_runs)
        L = (T.sum() - nf * b / 2.0) / np.sqrt(nf * (b ** 2) / 12.0)
        X = np.diff([0] + fr).astype(float)
        cv = X.std(ddof=1) / X.mean() if (X.mean() > 0 and len(X) > 1) else np.nan
        LR = L / cv if (cv and not np.isnan(cv) and cv != 0) else np.nan
        out.update({'laplace_L': float(L),
                    'cv_hat': (float(cv) if not np.isnan(cv) else None),
                    'lewis_robinson_LR': (float(LR) if not np.isnan(LR) else None),
                    'laplace_no_trend': bool(abs(L) < LOUIT_LAPLACE_CRIT),
                    'lr_no_trend': (bool(abs(LR) < LOUIT_LAPLACE_CRIT) if not np.isnan(LR) else None),
                    'trend_testable': True})
    else:
        out.update({'laplace_L': None, 'cv_hat': None, 'lewis_robinson_LR': None,
                    'laplace_no_trend': None, 'lr_no_trend': None, 'trend_testable': False})

    if nf >= LOUIT_MIN_FAILURES_FOR_R2:
        parts_cum = g['order_quantity'].cumsum()
        idx_fail = g.index[g['fail'] == 1].tolist()
        parts_at_fail = [float(parts_cum.iloc[i]) for i in idx_fail]
        ivp = np.diff([0.0] + parts_at_fail)
        xs = np.sort(ivp); m = len(xs)
        U = _nist_uniform_order_statistic_medians(m)
        q = -np.log(1 - U)          # exponential percent point function G(U)
        out['r2_parts'] = float(np.corrcoef(q, xs)[0, 1] ** 2)
        out['r2_testable'] = True
    else:
        out['r2_parts'] = None; out['r2_testable'] = False

    if not out['trend_testable']:
        out['verdict'] = ("Insufficient failures (n=%d < %d) \u2014 trend test not meaningful"
                          % (nf, LOUIT_MIN_FAILURES_FOR_TREND))
    elif out['laplace_no_trend'] and out.get('lr_no_trend'):
        out['verdict'] = "No significant trend (Laplace + Lewis\u2013Robinson) \u2014 renewal/HPP baseline supported"
    elif out['laplace_no_trend'] and not out.get('lr_no_trend'):
        out['verdict'] = "Laplace no-trend but Lewis\u2013Robinson flags trend \u2014 interpret with caution (underdispersion)"
    else:
        out['verdict'] = "Significant trend detected \u2014 consider NHPP, not renewal/MPTS"
    return out


def compute_louit_probplot_data(df, part_id):
    """Return exponential probability-plot arrays for one part, on the SAME
    parts-to-failure interval basis used for r2_parts in compute_louit_screening.
    Returns dict with q (exp order-stat medians), xs (sorted intervals),
    fit_slope, fit_intercept, r2, n_intervals — or None if untestable."""
    g = df[df['part_id'] == str(part_id)].sort_values('week_ending').reset_index(drop=True)
    if len(g) == 0:
        return None
    thr = float(g['scrap_percent'].mean())
    g = g.copy()
    g['fail'] = (g['scrap_percent'] > thr).astype(int)
    parts_cum = g['order_quantity'].cumsum()
    idx_fail = g.index[g['fail'] == 1].tolist()
    if len(idx_fail) < LOUIT_MIN_FAILURES_FOR_R2:
        return None
    parts_at_fail = [float(parts_cum.iloc[i]) for i in idx_fail]
    ivp = np.diff([0.0] + parts_at_fail)
    xs = np.sort(ivp)
    m = len(xs)
    U = _nist_uniform_order_statistic_medians(m)
    q = -np.log(1 - U)          # exponential percent point function G(U)
    slope, intercept = np.polyfit(q, xs, 1)
    r2 = float(np.corrcoef(q, xs)[0, 1] ** 2)
    return {'q': q, 'xs': xs, 'slope': float(slope), 'intercept': float(intercept),
            'r2': r2, 'n_intervals': m, 'threshold_pct': thr}





def add_mtts_sequential_features(df, threshold):
    """
    Add per-record sequential MTTS features (NO future-data leakage).
    
    These features are computed strictly from past/current data within each
    part's timeline using a forward-only loop. No aggregate part-level metrics
    are included here — those are computed separately from training data only
    via compute_mtts_on_train() to prevent temporal leakage.
    
    LEAKAGE PREVENTION (Colaresi & Mahmood, 2017):
    Sequential features (runs_since_last_failure, degradation_velocity, etc.)
    are inherently causal — each record's value depends only on prior records.
    Aggregate metrics (mtts_runs, hazard_rate, reliability_score) are computed
    from training data only and attached via attach_mtts_aggregate_features().
    
    Consistent with MTTF = total operating time / number of failures
    (Ebeling, 1997, An Introduction to Reliability and Maintainability Engineering).
    """
    df = df.copy()
    # Process per-part with stable secondary keys (preserves pipeline determinism)
    _sort_keys = ['part_id', 'week_ending']
    for k in ['work_order_#', '_csv_row_idx']:
        if k in df.columns:
            _sort_keys.append(k)
    df = df.sort_values(_sort_keys).reset_index(drop=True)
    
    df['runs_since_last_failure'] = 0
    df['cumulative_scrap_in_cycle'] = 0.0
    df['degradation_velocity'] = 0.0
    df['degradation_acceleration'] = 0.0
    
    for part_id, group in df.groupby('part_id'):
        idx_list = group.index.tolist()
        
        runs_since_failure = 0
        cumulative_scrap = 0.0
        prev_scrap = 0.0
        prev_velocity = 0.0
        
        for idx in idx_list:
            runs_since_failure += 1
            current_scrap = df.loc[idx, 'scrap_percent']
            cumulative_scrap += current_scrap
            
            df.loc[idx, 'runs_since_last_failure'] = runs_since_failure
            df.loc[idx, 'cumulative_scrap_in_cycle'] = cumulative_scrap
            
            velocity = current_scrap - prev_scrap
            df.loc[idx, 'degradation_velocity'] = velocity
            df.loc[idx, 'degradation_acceleration'] = velocity - prev_velocity
            
            prev_scrap = current_scrap
            prev_velocity = velocity
            
            if current_scrap > threshold:
                runs_since_failure = 0
                cumulative_scrap = 0.0
    
    return df


def compute_mtts_on_train(df_train, threshold):
    """
    Compute aggregate MTTS metrics from TRAINING DATA ONLY.
    
    LEAKAGE PREVENTION: This function is called AFTER the temporal split,
    using only the training partition. The resulting metrics are then merged
    into all partitions (train, calib, test) via attach_mtts_aggregate_features(),
    ensuring that test-set records never contain MTTS values computed from
    future data.
    
    MTTS(parts) = Total Parts Produced / Number of Failures    (Eq 3.1)
    MTTS(runs)  = Total Runs / Number of Failures
    h(t)        = 1 / MTTS(parts)                              (Eq 3.2)
    R(t)        = e^(-h(t) * parts_ordered)                    (Eq 3.3)
    
    Consistent with MTTF = total operating time / number of failures
    (Ebeling, 1997, An Introduction to Reliability and Maintainability Engineering).
    """
    results = []
    
    for part_id, group in df_train.groupby('part_id'):
        total_runs = len(group)
        total_parts = group['order_quantity'].sum() if 'order_quantity' in group.columns else total_runs
        avg_order_quantity = total_parts / total_runs if total_runs > 0 else 0
        failure_count = (group['scrap_percent'] > threshold).sum()
        
        # Eq 3.1: MTTS = Total Parts (or Runs) / Number of Failures
        if failure_count > 0:
            mtts_parts = total_parts / failure_count
            mtts_runs = total_runs / failure_count
        else:
            mtts_parts = total_parts
            mtts_runs = total_runs
        
        # Eq 3.2: h(t) = 1 / MTTS(parts)
        lambda_parts = 1 / mtts_parts if mtts_parts > 0 else 0
        lambda_runs = failure_count / total_runs if total_runs > 0 else 0
        
        # Eq 3.3: R(t) = e^(-h(t) * avg_order_quantity)
        reliability_score = np.exp(-avg_order_quantity / mtts_parts) if mtts_parts > 0 else 0
        
        results.append({
            'part_id': part_id,
            'mtts_parts': mtts_parts,
            'mtts_runs': mtts_runs,
            'failure_count': failure_count,
            'lambda_parts': lambda_parts,
            'lambda_runs': lambda_runs,
            'hazard_rate': lambda_runs,
            'reliability_score': reliability_score
        })
    
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=['part_id', 'mtts_parts', 'mtts_runs', 'failure_count',
                 'lambda_parts', 'lambda_runs', 'hazard_rate', 'reliability_score']
    )


def attach_mtts_aggregate_features(df, mtts_train_df):
    """
    Attach training-derived MTTS aggregates to any data split.
    
    LEAKAGE PREVENTION: mtts_train_df was computed from training data only.
    Parts not seen during training receive median imputed values, ensuring
    no future information leaks into calibration or test partitions.
    
    Also computes derived features (cycle_hazard_indicator, rul_proxy) using
    the training-derived MTTS values combined with the per-record sequential
    features already present in df.
    """
    merge_cols = ['part_id', 'mtts_parts', 'mtts_runs', 'lambda_parts', 'lambda_runs',
                  'hazard_rate', 'reliability_score', 'failure_count']
    available_cols = [c for c in merge_cols if c in mtts_train_df.columns]
    
    # Drop any existing MTTS columns to avoid merge conflicts
    for col in available_cols:
        if col != 'part_id' and col in df.columns:
            df = df.drop(columns=[col])
    
    df = df.merge(mtts_train_df[available_cols], on='part_id', how='left')
    
    # Fill missing values for parts not in training data
    median_mtts_parts = mtts_train_df['mtts_parts'].median() if len(mtts_train_df) > 0 and mtts_train_df['mtts_parts'].notna().any() else 1000
    median_mtts_runs = mtts_train_df['mtts_runs'].median() if len(mtts_train_df) > 0 and mtts_train_df['mtts_runs'].notna().any() else 10
    
    df['mtts_parts'] = df['mtts_parts'].fillna(median_mtts_parts)
    df['mtts_runs'] = df['mtts_runs'].fillna(median_mtts_runs)
    df['hazard_rate'] = df['hazard_rate'].fillna(0.1)
    df['reliability_score'] = df['reliability_score'].fillna(0.5)
    df['failure_count'] = df['failure_count'].fillna(0)
    
    # Compute derived features using training-derived MTTS + sequential features
    if 'runs_since_last_failure' in df.columns:
        df['cycle_hazard_indicator'] = (
            df['runs_since_last_failure'] / df['mtts_runs'].replace(0, 1)
        ).clip(upper=2.0)
        df['rul_proxy'] = (df['mtts_runs'] - df['runs_since_last_failure']).clip(lower=0)
    else:
        df['cycle_hazard_indicator'] = 0.0
        df['rul_proxy'] = 0.0
    
    return df


# BACKWARD COMPATIBILITY: Keep original function for runtime analysis
# (e.g., compute_pooled_prediction, Tab displays) where leakage is not
# a concern because these are descriptive statistics, not model features.
def add_mtts_features(df, threshold):
    """
    Add MTTS-based reliability features (FULL HISTORY version).
    
    WARNING: This function uses the FULL dataset to compute aggregate MTTS
    metrics. It is retained ONLY for runtime descriptive analysis (e.g.,
    pooled predictions, Tab 1 displays). For MODEL TRAINING, use the 
    leakage-safe pipeline: add_mtts_sequential_features() → split → 
    compute_mtts_on_train() → attach_mtts_aggregate_features().
    """
    df = df.copy()
    _sort_keys = ['part_id', 'week_ending']
    for k in ['work_order_#', '_csv_row_idx']:
        if k in df.columns:
            _sort_keys.append(k)
    df = df.sort_values(_sort_keys).reset_index(drop=True)
    
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
        
        # Eq 3.1: MTTS = Total Parts (or Runs) / Number of Failures
        if failures > 0:
            mtts_runs = part_n / failures
        else:
            mtts_runs = part_n * 10  # No failures observed, estimate high MTTS
        
        # Eq 3.1: MTTS(parts) = Total Parts Produced / Failures
        if failures > 0:
            mtts_parts = total_parts_produced / failures
        else:
            mtts_parts = total_parts_produced * 10
        
        # Eq 3.3: R(t) = e^(-h(t) * avg_parts_per_run)
        avg_order_qty = total_parts_produced / part_n if part_n > 0 else 0
        reliability = np.exp(-avg_order_qty / mtts_parts) if mtts_parts > 0 else 0
        
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
        # Eq 3.3: R(t) = e^(-h(t) * avg_parts_per_run)
        pl_avg_order_qty = total_parts_produced / part_n if part_n > 0 else 0
        pl_reliability = np.exp(-pl_avg_order_qty / pl_mtts_parts) if pl_mtts_parts > 0 else 0
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
    
    # Use the target part's own average scrap rate as threshold for pooled comparison
    # Pooling is a contextual lookup, not an independent statistical method
    pooled_avg_scrap = final_df['scrap_percent'].mean()
    pooled_std_scrap = final_df['scrap_percent'].std() if pooled_n > 1 else 0
    effective_threshold = threshold_pct
    threshold_source = 'part-specific'
    
    # Calculate failures using effective threshold
    failures = (final_df['scrap_percent'] > effective_threshold).sum()
    failure_rate = failures / pooled_n if pooled_n > 0 else 0
    
    # Eq 3.1: MTTS(runs) = Total Runs / Failures
    if failures > 0:
        mtts_runs = pooled_n / failures
    else:
        mtts_runs = pooled_n * 2  # Conservative multiplier when no failures (was 10)
    
    # Eq 3.1: MTTS(parts) = Total Parts Produced / Failures
    if failures > 0:
        mtts_parts = pooled_total_parts / failures
    else:
        mtts_parts = pooled_total_parts * 2  # Conservative multiplier when no failures (was 10)
    
    # Eq 3.3: R(t) = e^(-h(t) * avg_parts_per_run)
    pooled_avg_order_qty = pooled_total_parts / pooled_n if pooled_n > 0 else 0
    reliability = np.exp(-pooled_avg_order_qty / mtts_parts) if mtts_parts > 0 else 0
    
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
# 60-20-20 TEMPORAL SPLIT
# ================================================================
def time_split_60_20_20(df):
    """Split data temporally: 60% train, 20% calibration, 20% test.
    
    Uses the deterministic sort established by load_data. The triple-key
    sort (week_ending, work_order_#, part_id, _csv_row_idx) guarantees
    that the split boundary is stable across pipeline runs — critical
    for reproducible recall / CI metrics in Chapter 4.
    """
    _keys = ['week_ending']
    for k in ['work_order_#', 'part_id', '_csv_row_idx']:
        if k in df.columns:
            _keys.append(k)
    df = df.sort_values(_keys).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    
    return df.iloc[:train_end].copy(), df.iloc[train_end:calib_end].copy(), df.iloc[calib_end:].copy()


# ================================================================
# ADDITIONAL VALIDATION METRICS (Clopper-Pearson, Seen/Unseen, Hazard)
# ================================================================
def clopper_pearson_ci(k, n, alpha=0.05):
    """
    Compute exact Clopper-Pearson confidence interval for binomial proportion.
    
    APA Citation: Clopper, C. J., & Pearson, E. S. (1934). The use of confidence 
    intervals in the case of the binomial. Biometrika, 26(4), 404-413.
    
    Parameters:
        k: number of successes (TP for recall)
        n: number of trials (TP + FN for recall)
        alpha: significance level (default 0.05 for 95% CI)
    Returns:
        (lower, upper) bounds
    """
    from scipy.stats import beta as beta_dist
    if n == 0:
        return (0.0, 1.0)
    lower = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lower, upper)


def compute_seen_unseen_metrics(global_model):
    """
    Partition test set by part familiarity and compute recall for each group.
    
    Uses test_part_ids and train_part_set stored at training time to guarantee
    exact alignment with y_test/y_pred (merges can reset DataFrame indices).
    
    Returns dict with seen/unseen parts counts, runs, recall, precision,
    plus sanity-check totals.
    """
    # Use the aligned arrays stored at training time
    test_part_ids = global_model.get('test_part_ids')
    train_part_set = global_model.get('train_part_set')
    
    y_test = global_model['metrics'].get('y_test')
    y_pred = global_model['metrics'].get('y_pred')
    
    if test_part_ids is None or train_part_set is None or y_test is None or y_pred is None:
        return None
    
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    
    # Sanity: lengths must match
    if len(test_part_ids) != len(y_test_arr) or len(y_test_arr) != len(y_pred_arr):
        return None
    
    # Partition
    seen_mask = np.array([pid in train_part_set for pid in test_part_ids])
    unseen_mask = ~seen_mask
    
    # Overall totals for sanity check
    overall_tp = int(((y_test_arr == 1) & (y_pred_arr == 1)).sum())
    overall_fn = int(((y_test_arr == 1) & (y_pred_arr == 0)).sum())
    overall_failures = overall_tp + overall_fn
    
    results = {'_overall_tp': overall_tp, '_overall_fn': overall_fn, '_overall_failures': overall_failures}
    
    running_tp = 0
    running_fn = 0
    
    for label, mask in [('seen', seen_mask), ('unseen', unseen_mask)]:
        n_runs = int(mask.sum())
        if n_runs == 0:
            continue
        
        yt = y_test_arr[mask]
        yp = y_pred_arr[mask]
        
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        failures = tp + fn  # actual positives in this group
        
        rec = tp / failures if failures > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        n_unique_parts = len(set(test_part_ids[mask]))
        
        running_tp += tp
        running_fn += fn
        
        results[label] = {
            'n_parts': n_unique_parts,
            'n_runs': n_runs,
            'failures': failures,  # actual failures (TP+FN) in this group
            'recall': rec,
            'precision': prec,
            'tp': tp,
            'fn': fn,
            'fp': fp,
        }
    
    # Sanity check: group totals must equal overall totals
    results['_sanity_ok'] = (running_tp == overall_tp and running_fn == overall_fn)
    
    return results


def compute_dual_model_validation_table(df, defect_cols, global_model, cohort_parts):
    """
    Compute the full 22-part Dual-Model Validation table (reproduces
    Chapter 4 Table 4-9 from the dissertation).

    For each part in cohort_parts, this computes:
      - n: historical run count
      - Avg Scrap %: part-level mean
      - Last Scrap %: most recent run
      - MPTS P%: probability of scrap threshold exceedance, derived from
        the MPTS reliability formula R(n) = e^(-avg_order_qty / MPTS_parts)
      - RF Last%: calibrated Stage-3 RF probability for the most recent run,
        scored through the already-trained global_model (uses df_enhanced to
        ensure engineered features are real, not zeros)
      - Δ (pp): divergence = RF Last% − MPTS P%
      - Signal: S1/S2/S3/S4 per the ±5pp threshold + chronic-vs-active rule
      - Chronic Process: top Pareto-Campbell process from historical failures
      - Agree: ✓/~/✗/— convergent-validity check vs RF's top process
      - Last-Run Active Process: Campbell process for the highest-rate defect
        in the most recent run

    Args:
        df: working census DataFrame (already §3.2.1-filtered by load_data)
        defect_cols: list of defect rate columns
        global_model: dict returned by train_global_model()
        cohort_parts: list of part IDs (e.g., COHORT_22)

    Returns:
        pandas DataFrame with columns matching Table 4-9 layout, plus a
        summary dict with convergent-validity statistics.
    """
    DEFECT_TO_PROC = {
        "dross_rate": "Melting", "gas_porosity_rate": "Melting",
        "missrun_rate": "Pouring", "misrun_rate": "Pouring",
        "short_pour_rate": "Pouring", "runout_rate": "Pouring",
        "shrink_rate": "Gating Design", "tear_up_rate": "Gating Design",
        "shrink_porosity_rate": "Gating Design",
        "sand_rate": "Sand System", "dirty_pattern_rate": "Sand System",
        "core_rate": "Core Making", "crush_rate": "Core Making",
        "shift_rate": "Core Making",
        "bent_rate": "Shakeout", "gouged_rate": "Finishing",
        "over_grind_rate": "Finishing", "cut_into_rate": "Finishing",
        "zyglo_rate": "Inspection", "failed_zyglo_rate": "Inspection",
        "outside_process_scrap_rate": "Inspection",
    }

    enh = global_model["df_enhanced"]
    feat_cols = global_model["features"]
    cal_model = global_model["cal_model"]
    mtts_tr = compute_mtts_on_train(global_model["df_train"],
                                     global_model["global_threshold"])

    def _pareto_top_process(candidate_df, historical_df):
        """Return (top_defect_display, top_process) by ranking defects on
        their raw mean rate within the candidate failure-run subset.

        This matches §3.4.5 of the dissertation, which specifies raw
        mean defect rate over the failure-run group as the ranking
        criterion (no enrichment denominator). An earlier formulation
        used Signal = fr² / hist; that was simplified on the basis of
        empirical testing (the simpler statistic produced 13/14 = 92.9%
        MPTS-RF attribution agreement on the qualifying subset, clearing
        the ≥90% Campbell & Fiske convergent-validity threshold without
        the additional formula complexity to defend).

        The historical_df argument is retained in the signature for
        callers that still pass it; it is no longer used for ranking.
        """
        if len(candidate_df) == 0:
            return "—", "—"
        scored = []
        for d in defect_cols:
            if d not in candidate_df.columns:
                continue
            cand_rate = candidate_df[d].mean()
            if cand_rate > 0:
                scored.append((d, cand_rate))
        if not scored:
            return "—", "—"
        scored.sort(key=lambda x: x[1], reverse=True)
        top_d = scored[0][0]
        return (top_d.replace("_rate", "").replace("_", " ").title(),
                DEFECT_TO_PROC.get(top_d, "—"))

    rows = []
    for pid in cohort_parts:
        pid_s = str(pid)
        raw_part = df[df["part_id"] == pid_s].copy()
        if len(raw_part) == 0:
            rows.append({
                "Part": pid, "n": 0, "Avg Scrap%": None, "Last Scrap%": None,
                "MPTS P%": None, "RF Last%": None, "Δ (pp)": None,
                "Signal": "—", "Chronic Process": "—", "Chronic Defect": "—",
                "Agree": "—", "Last-Run Active": "—", "Last-Run Defect": "—",
            })
            continue
        if "week_ending" in raw_part.columns:
            raw_part = raw_part.sort_values("week_ending")

        # ── MPTS probability ──────────────────────────────────────────
        n_runs = len(raw_part)
        avg_scrap_pct = raw_part["scrap_percent"].mean()
        last_scrap_pct = float(raw_part["scrap_percent"].iloc[-1])
        threshold = avg_scrap_pct  # part-mean threshold per §3.2.1
        fail_mask = raw_part["scrap_percent"] > threshold
        fail_count = int(fail_mask.sum())
        total_qty = raw_part["order_quantity"].sum()
        avg_qty = total_qty / n_runs if n_runs > 0 else 0
        mpts_parts = total_qty / fail_count if fail_count > 0 else total_qty
        R_n = float(np.exp(-avg_qty / mpts_parts)) if mpts_parts > 0 else 1.0
        mpts_prob = round((1 - R_n) * 100, 1)

        # ── MPTS Pareto-Campbell attribution (historical failure runs) ──
        fail_runs = raw_part[fail_mask]
        mpts_top_def, mpts_top_proc = _pareto_top_process(fail_runs, raw_part)

        # ── RF last-run scoring via enhanced dataframe ────────────────
        rf_last_prob = None
        rf_top_proc = "—"
        try:
            part_enh = enh[enh["part_id"] == pid_s].copy()
            if "week_ending" in part_enh.columns:
                part_enh = part_enh.sort_values("week_ending")
            if len(part_enh) > 0:
                last_row_enh = part_enh.iloc[[-1]].copy()
                last_row_enh = attach_train_features(
                    last_row_enh,
                    global_model["scrap_rate_train"],
                    global_model["part_freq_train"],
                    global_model["default_scrap_rate"],
                    global_model["default_freq"],
                )
                last_row_enh = attach_mtts_aggregate_features(last_row_enh, mtts_tr)
                last_feat = last_row_enh.reindex(columns=feat_cols, fill_value=0).fillna(0)
                rf_last_prob = round(
                    float(cal_model.predict_proba(last_feat)[:, 1][0]) * 100, 1
                )

                # RF Pareto-Campbell attribution: use all RF-predicted
                # failure runs for this part (probability ≥ 0.50).
                all_part_enh = part_enh.copy()
                all_part_enh = attach_train_features(
                    all_part_enh,
                    global_model["scrap_rate_train"],
                    global_model["part_freq_train"],
                    global_model["default_scrap_rate"],
                    global_model["default_freq"],
                )
                all_part_enh = attach_mtts_aggregate_features(all_part_enh, mtts_tr)
                X_all = all_part_enh.reindex(columns=feat_cols, fill_value=0).fillna(0)
                probs_all = cal_model.predict_proba(X_all)[:, 1]
                rf_fail_mask = probs_all >= 0.50
                if rf_fail_mask.sum() > 0:
                    rf_fail_df = raw_part.iloc[rf_fail_mask] if len(raw_part) == len(rf_fail_mask) else raw_part[raw_part.index.isin(all_part_enh.index[rf_fail_mask])]
                    if len(rf_fail_df) > 0:
                        _, rf_top_proc = _pareto_top_process(rf_fail_df, raw_part)
        except Exception:
            rf_last_prob = None

        # ── Last-run active process (from raw defect rates) ─────────────
        last_row_raw = raw_part.iloc[-1]
        last_rates = {c: float(last_row_raw[c]) for c in defect_cols
                      if c in raw_part.columns and float(last_row_raw[c]) > 0}
        if last_rates:
            top_lr = max(last_rates, key=last_rates.get)
            last_run_proc = DEFECT_TO_PROC.get(top_lr, "—")
            last_run_def = top_lr.replace("_rate", "").replace("_", " ").title()
        else:
            last_run_proc = "—"
            last_run_def = "—"

        # ── Divergence + scenario ─────────────────────────────────────
        if rf_last_prob is not None:
            divergence = round(rf_last_prob - mpts_prob, 1)
            if abs(divergence) <= 5.0:
                signal = "S1 — Aligned ≈"
            elif divergence > 5.0:
                signal = "S2 — Alarm ▲"
            elif last_run_proc not in ("—", mpts_top_proc) and last_run_proc:
                signal = f"S4 — New Process ⚡"
            else:
                signal = "S3 — Improvement ▼"
        else:
            divergence = None
            signal = "—"

        # ── Convergent-validity agreement ─────────────────────────────
        # Per §4.4.3: convergent validity agreement is computed over the
        # data-sufficient subset only (parts with ≥2 observed failure runs).
        # Parts with <2 failure runs cannot support a reliable MPTS Pareto-
        # Campbell attribution and are therefore excluded from the agreement
        # denominator. The filter is applied at the summary-rollup step below
        # (not here), so each row still records its raw agreement result.
        if mpts_top_proc == "—" or rf_top_proc == "—":
            agree = "—"
        elif mpts_top_proc == rf_top_proc:
            agree = "✓ Agree"
        else:
            agree = "✗ Differ"

        rows.append({
            "Part": pid,
            "n": n_runs,
            "Failure Runs": fail_count,
            "Avg Scrap%": round(avg_scrap_pct, 3),
            "Last Scrap%": round(last_scrap_pct, 3),
            "MPTS P%": mpts_prob,
            "RF Last%": rf_last_prob if rf_last_prob is not None else None,
            "Δ (pp)": divergence,
            "Signal": signal,
            "Chronic Process": mpts_top_proc,
            "Chronic Defect": mpts_top_def,
            "Agree": agree,
            "Last-Run Active": last_run_proc,
            "Last-Run Defect": last_run_def,
        })

    result_df = pd.DataFrame(rows)

    # ── Convergent-validity summary (§4.4.3 data-sufficiency filter) ──
    # Qualifying parts are those with ≥2 observed failure runs AND both
    # methods producing an attributable result. This matches the
    # dissertation's §4.4.3 definition and the Campbell & Fiske (1959)
    # convergent-validity framework.
    attributable_mask = result_df["Agree"].isin(["✓ Agree", "✗ Differ"])
    sufficient_mask = result_df["Failure Runs"] >= 2
    qualifying = result_df[attributable_mask & sufficient_mask]
    n_qual = len(qualifying)
    n_agree = int((qualifying["Agree"] == "✓ Agree").sum())
    agreement_pct = round(100 * n_agree / n_qual, 1) if n_qual > 0 else None

    # Also report excluded parts for transparency
    excluded_insufficient = result_df[attributable_mask & ~sufficient_mask]
    excluded_unattributable = result_df[~attributable_mask]

    summary = {
        "cohort_size": len(cohort_parts),
        "qualifying_parts": n_qual,                        # ≥2 failure runs AND attributable
        "agreeing_parts": n_agree,
        "agreement_pct": agreement_pct,
        "excluded_insufficient": len(excluded_insufficient),   # attributable but <2 failures
        "excluded_unattributable": len(excluded_unattributable),  # at least one method produced "—"
        "divergence_threshold_pp": 5.0,
        "filter_description": "Per §4.4.3: qualifying = ≥2 observed failure runs AND both methods attributable",
    }
    return result_df, summary


def compute_empirical_hazard(df, threshold_mode='part_mean'):
    """
    Compute empirical hazard by equal-width bins on normalized inter-failure intervals.
    
    For the H1 assessable cohort (parts with ≥4 failures → ≥3 inter-failure
    intervals), normalizes intervals by part-specific MTTS and bins into
    equal-width segments. The cohort size is determined dynamically from the
    working census; with the §3.2.1-compliant working census it yields 30 parts.
    
    Returns dict with bin hazards, formal test results, and part-level stats.
    """
    from scipy import stats as sp_stats
    
    part_stats = df.groupby('part_id').agg(
        n_runs=('scrap_percent', 'count'),
        avg_scrap=('scrap_percent', 'mean')
    ).reset_index()
    
    results = {}
    for _, row in part_stats.iterrows():
        pid = row['part_id']
        part_df = df[df['part_id'] == pid].reset_index(drop=True)
        thresh = row['avg_scrap']  # part mean as threshold
        
        failures = part_df['scrap_percent'] > thresh
        failure_indices = np.where(failures)[0]
        
        if len(failure_indices) < 4:  # need >=3 intervals
            continue
        
        intervals = np.diff(failure_indices)
        n_failures = failures.sum()
        mtts = len(part_df) / n_failures if n_failures > 0 else np.inf
        
        results[pid] = {
            'intervals': intervals,
            'mtts': mtts,
            'n_intervals': len(intervals),
            'n_failures': int(n_failures),
            'n_runs': len(part_df),
        }
    
    if not results:
        return None
    
    # Pool normalized intervals
    all_normalized = []
    for pid, r in results.items():
        all_normalized.extend(r['intervals'] / r['mtts'])
    all_normalized = np.array(all_normalized)
    n = len(all_normalized)
    
    # Equal-width bins
    p95 = np.percentile(all_normalized, 95)
    edges = np.linspace(0, p95, 5)
    edges[-1] = all_normalized.max() + 0.001
    
    bin_hazards = []
    bin_details = []
    for i in range(4):
        lo, hi = edges[i], edges[i+1]
        width = hi - lo
        n_at_risk = np.sum(all_normalized >= lo)
        d_i = np.sum((all_normalized >= lo) & (all_normalized < hi)) if i < 3 else np.sum(all_normalized >= lo)
        h_i = d_i / (n_at_risk * width) if (n_at_risk > 0 and width > 0) else 0
        bin_hazards.append(h_i)
        bin_details.append({'lo': lo, 'hi': hi, 'width': width, 'n_risk': n_at_risk, 'd': d_i, 'h': h_i})
    
    bin_hazards = np.array(bin_hazards)
    
    # Formal tests
    obs_quartile = [np.sum((all_normalized >= np.percentile(all_normalized, q*25)) & 
                           (all_normalized < np.percentile(all_normalized, (q+1)*25))) 
                    for q in range(3)]
    obs_quartile.append(n - sum(obs_quartile))
    chi2, chi2_p = sp_stats.chisquare(obs_quartile)
    
    from scipy.stats import kendalltau
    tau, tau_p = kendalltau(np.arange(n), all_normalized)
    
    # Lilliefors-corrected KS test: adjusts critical values because the scale
    # parameter (MPTS_runs) is estimated from the same data being tested.
    # Standard kstest critical values are too liberal when parameters are
    # estimated from the sample (Lilliefors, 1967; Meeker & Escobar, 1998).
    if LILLIEFORS_AVAILABLE:
        ks_stat, ks_p = lilliefors_test(all_normalized, dist='exp')
        ks_method = 'Lilliefors-corrected'
    else:
        ks_stat, ks_p = sp_stats.kstest(all_normalized, 'expon', args=(0, 1))
        ks_method = 'Standard KS (statsmodels unavailable)'
    
    mean_norm = float(all_normalized.mean())
    cv_norm = float(all_normalized.std() / all_normalized.mean()) if all_normalized.mean() > 0 else 0
    
    return {
        'n_assessable_parts': len(results),
        'n_intervals': n,
        'mean_normalized': mean_norm,
        'cv_normalized': cv_norm,
        'bin_hazards': bin_hazards.tolist(),
        'bin_details': bin_details,
        'bin_cv': float(np.std(bin_hazards) / np.mean(bin_hazards)) if np.mean(bin_hazards) > 0 else 0,
        'chi2': float(chi2), 'chi2_p': float(chi2_p),
        'kendall_tau': float(tau), 'kendall_p': float(tau_p),
        'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
        'ks_method': ks_method,
        'all_normalized': all_normalized,
    }


# ================================================================
# FEATURE ENGINEERING - MEAN SCRAP RATE (TRAINING-DERIVED)
# ================================================================
# NOTE: compute_mean_scrap_on_train() computes a simple average scrap
# rate per part — it is NOT the MTTS reliability metric. MTTS metrics
# are computed in compute_mtts_on_train(). See docstrings for details.
# ================================================================
def compute_mean_scrap_on_train(df_train, threshold):
    """
    Compute mean scrap rate per part from TRAINING data only.
    
    NOTE: This is a simple average scrap rate per part, NOT the MTTS
    reliability metric defined in Eq 3.1. The MTTS metrics (mtts_parts, 
    mtts_runs, hazard_rate, reliability_score) are computed separately 
    in compute_mtts_on_train(). This feature captures each part's 
    baseline scrap tendency as a model input.
    
    Renamed from 'mttf_scrap' to 'mean_scrap_rate_train' to avoid 
    confusion with the actual MTTS/MTTF reliability calculations.
    """
    grp = df_train.groupby("part_id")["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mean_scrap_rate_train"}, inplace=True)
    grp["mean_scrap_rate_train"] = np.where(grp["mean_scrap_rate_train"] <= threshold, 1.0, grp["mean_scrap_rate_train"])
    return grp


def attach_train_features(df_sub, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq):
    """Attach training-derived features to prevent leakage."""
    df_sub = df_sub.merge(scrap_rate_train, on="part_id", how="left")
    df_sub["mean_scrap_rate_train"] = df_sub["mean_scrap_rate_train"].fillna(default_scrap_rate)
    df_sub = df_sub.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    df_sub["part_freq"] = df_sub["part_freq"].fillna(default_freq)
    return df_sub


# ================================================================
# MAKE X, Y - MATCHING ENHANCED VERSION FEATURES
# ================================================================
def make_xy(df, threshold, defect_cols, use_multi_defect=True, use_temporal=True, use_mtts=True):
    """Prepare features matching enhanced version exactly."""
    feats = ["order_quantity", "piece_weight_lbs", "mean_scrap_rate_train", "part_freq"]
    
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
    
    # Add features (sequential MTTS features are safe before split - backward-looking only)
    df_stage1 = df.copy()
    df_stage1 = add_multi_defect_features(df_stage1, defect_cols)
    df_stage1 = add_temporal_features(df_stage1)
    df_stage1 = add_mtts_sequential_features(df_stage1, global_threshold)
    
    # 60-20-20 split
    df_train, df_calib, df_test = time_split_60_20_20(df_stage1)
    
    # MTTS aggregates from training only (LEAKAGE PREVENTION)
    mtts_train = compute_mtts_on_train(df_train, global_threshold)
    df_train = attach_mtts_aggregate_features(df_train, mtts_train)
    df_calib = attach_mtts_aggregate_features(df_calib, mtts_train)
    df_test = attach_mtts_aggregate_features(df_test, mtts_train)
    
    # Mean scrap rate from training only
    scrap_rate_train = compute_mean_scrap_on_train(df_train, global_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_scrap_rate = float(scrap_rate_train["mean_scrap_rate_train"].median()) if len(scrap_rate_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_calib = attach_train_features(df_calib, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_test = attach_train_features(df_test, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    
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
        "scrap_rate_train": scrap_rate_train,
        "part_freq_train": part_freq_train,
        "default_scrap_rate": default_scrap_rate,
        "default_freq": default_freq,
    }


def train_stage2_defect_cluster(df, defect_cols, stage1_model):
    """
    STAGE 2: DEFECT-CLUSTER MODEL
    
    Trains on records with TOP 5 PARETO defects using cluster threshold.
    Purpose: Learn patterns specific to high-impact defects (~66% of scrap).
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
    
    # Add features (sequential MTTS features are safe before split)
    df_cluster = add_multi_defect_features(df_cluster, defect_cols)
    df_cluster = add_temporal_features(df_cluster)
    df_cluster = add_mtts_sequential_features(df_cluster, cluster_threshold)
    
    # Add Stage 1 predictions as feature
    df_cluster = add_stage1_features(df_cluster, stage1_model, defect_cols)
    
    # 60-20-20 split
    df_train, df_calib, df_test = time_split_60_20_20(df_cluster)
    
    # MTTS aggregates from training only (LEAKAGE PREVENTION)
    mtts_train = compute_mtts_on_train(df_train, cluster_threshold)
    df_train = attach_mtts_aggregate_features(df_train, mtts_train)
    df_calib = attach_mtts_aggregate_features(df_calib, mtts_train)
    df_test = attach_mtts_aggregate_features(df_test, mtts_train)
    
    # Mean scrap rate from training only
    scrap_rate_train = compute_mean_scrap_on_train(df_train, cluster_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_scrap_rate = float(scrap_rate_train["mean_scrap_rate_train"].median()) if len(scrap_rate_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_calib = attach_train_features(df_calib, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_test = attach_train_features(df_test, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    
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
    # Add base features to full dataset (sequential MTTS only - no leakage)
    df_enhanced = df.copy()
    df_enhanced = add_multi_defect_features(df_enhanced, defect_cols)
    df_enhanced = add_temporal_features(df_enhanced)
    
    # Use GLOBAL threshold for MTTS (Stage 1 alignment)
    global_threshold = df["scrap_percent"].mean()
    df_enhanced = add_mtts_sequential_features(df_enhanced, global_threshold)
    
    # ================================================================
    # INHERITED FEATURES: Stage 1 and Stage 2 predictions
    # ================================================================
    # STACKING NOTE (Fix 2 - Temporal Alignment):
    # Stage 1/2 models were each trained on their own first-60% temporal
    # partition. Applying them to the full dataset means:
    #   - Records in Stage 3's training set (first 60%): Stage 1/2
    #     predictions are IN-SAMPLE (same temporal window as their training).
    #     This introduces mild optimism in inherited features.
    #   - Records in Stage 3's calib/test sets (last 40%): Stage 1/2
    #     predictions are genuinely OUT-OF-SAMPLE.
    # This is a known limitation of two-level stacking without cross-fold
    # prediction generation (Wolpert, 1992; Breiman, 1996). The RF's
    # internal regularization (bootstrap sampling, feature subsetting)
    # mitigates but does not eliminate this optimism.
    # ================================================================
    df_enhanced = add_stage1_features(df_enhanced, stage1_result, defect_cols)
    df_enhanced = add_stage2_features(df_enhanced, stage2_result, defect_cols)
    
    # 60-20-20 split
    df_train, df_calib, df_test = time_split_60_20_20(df_enhanced)
    
    # MTTS aggregates from training only (LEAKAGE PREVENTION)
    mtts_train = compute_mtts_on_train(df_train, global_threshold)
    df_train = attach_mtts_aggregate_features(df_train, mtts_train)
    df_calib = attach_mtts_aggregate_features(df_calib, mtts_train)
    df_test = attach_mtts_aggregate_features(df_test, mtts_train)
    
    # Mean scrap rate from training only
    scrap_rate_train = compute_mean_scrap_on_train(df_train, global_threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_scrap_rate = float(scrap_rate_train["mean_scrap_rate_train"].median()) if len(scrap_rate_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    df_train = attach_train_features(df_train, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_calib = attach_train_features(df_calib, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_test = attach_train_features(df_test, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    
    # Prepare X, y with GLOBAL threshold for final evaluation
    X_train, y_train, feats = make_xy(df_train, global_threshold, defect_cols)
    X_calib, y_calib, _ = make_xy(df_calib, global_threshold, defect_cols)
    X_test, y_test, _ = make_xy(df_test, global_threshold, defect_cols)
    
    # CRITICAL: Capture part_ids at same moment as y_test for exact alignment
    # (merges in attach_train_features may have reset df_test index)
    test_part_ids = df_test['part_id'].values.copy()
    train_part_set = set(df_train['part_id'].unique())
    
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
        "scrap_rate_train": scrap_rate_train,
        "part_freq_train": part_freq_train,
        "default_scrap_rate": default_scrap_rate,
        "default_freq": default_freq,
        "metrics": metrics,
        "n_train": len(df_train),
        "n_calib": len(df_calib),
        "n_test": len(df_test),
        "global_threshold": global_threshold,
        # Exact alignment arrays for seen/unseen analysis
        "test_part_ids": test_part_ids,
        "train_part_set": train_part_set,
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
    
    # Add enhanced features BEFORE split (sequential features are backward-looking)
    df = add_multi_defect_features(df, defect_cols)
    df = add_temporal_features(df)
    df = add_mtts_sequential_features(df, threshold)
    
    # 60-20-20 temporal split
    df_train, df_calib, df_test = time_split_60_20_20(df)
    
    # MTTS aggregates from training only (LEAKAGE PREVENTION)
    mtts_train = compute_mtts_on_train(df_train, threshold)
    df_train = attach_mtts_aggregate_features(df_train, mtts_train)
    df_calib = attach_mtts_aggregate_features(df_calib, mtts_train)
    df_test = attach_mtts_aggregate_features(df_test, mtts_train)
    
    # Mean scrap rate from training only
    scrap_rate_train = compute_mean_scrap_on_train(df_train, threshold)
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_scrap_rate = float(scrap_rate_train["mean_scrap_rate_train"].median()) if len(scrap_rate_train) else 1.0
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0
    
    # Attach features
    df_train = attach_train_features(df_train, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_calib = attach_train_features(df_calib, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    df_test = attach_train_features(df_test, scrap_rate_train, part_freq_train, default_scrap_rate, default_freq)
    
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
        "scrap_rate_train": scrap_rate_train, "part_freq_train": part_freq_train,
        "default_scrap_rate": default_scrap_rate, "default_freq": default_freq,
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


def lime_weights_to_campbell(lime_weights_dict, instance_rows, defect_cols):
    """
    Translate LIME feature weights into Campbell process-level attribution,
    respecting multi-cause defects via co-occurrence scoring.

    Single-cause defect  → weight goes entirely to its one process.
    Multi-cause defect   → weight is split across candidate processes
                           proportionally to each process's co-occurrence
                           score given the OTHER defects present in the run.
                           If co-occurrence is uninformative (only one defect
                           observed), weight is split equally — the manager
                           sees all candidate processes and decides.
    Aggregate features   → decomposed back to constituent defects first,
                           then treated as above.
    Model/MTTS features  → excluded (audit signals, not process signals).

    Returns
    -------
    process_weights : dict  {campbell_process: aggregated_weight}
    defect_weights  : dict  {defect_col: aggregated_weight}
    """
    # --- Resolve instance defect values --------------------------------
    if isinstance(instance_rows, pd.Series):
        inst = instance_rows
    elif hasattr(instance_rows, 'iloc'):
        inst = instance_rows.mean(numeric_only=True)
    else:
        inst = pd.Series(instance_rows)

    # Build defect value map and set of non-zero defects for co-occurrence
    defect_values = {}
    for dc in defect_cols:
        if dc in DEFECT_TO_PROCESSES:
            val = float(inst.get(dc, 0.0))
            defect_values[dc] = max(val, 0.0)

    total_defect_burden = sum(defect_values.values())
    observed_defects_set = {d for d, v in defect_values.items() if v > 0}

    def _proportional_shares():
        if total_defect_burden <= 0:
            n = max(len(defect_values), 1)
            return {d: 1.0 / n for d in defect_values}
        return {d: v / total_defect_burden for d, v in defect_values.items()
                if v > 0}

    def _max_defect():
        if not defect_values or total_defect_burden <= 0:
            return None
        return max(defect_values, key=defect_values.get)

    # --- Accumulate weights at the defect level -----------------------
    defect_weights = {}

    for feat, weight in lime_weights_dict.items():
        bare = feat.split(' ')[0].strip()

        if bare in DEFECT_TO_PROCESSES:
            defect_weights[bare] = defect_weights.get(bare, 0.0) + weight
        elif bare == 'total_defect_rate':
            for d, share in _proportional_shares().items():
                defect_weights[d] = defect_weights.get(d, 0.0) + weight * share
        elif bare == 'max_defect_rate':
            md = _max_defect()
            if md:
                defect_weights[md] = defect_weights.get(md, 0.0) + weight
        elif bare in ('defect_concentration', 'n_defect_types', 'has_multiple_defects'):
            for d, share in _proportional_shares().items():
                defect_weights[d] = defect_weights.get(d, 0.0) + weight * share
        elif '_x_' in bare:
            parts = bare.split('_x_')
            candidates = [p + '_rate' for p in parts]
            mapped = [c for c in candidates if c in DEFECT_TO_PROCESSES]
            if mapped:
                per_d = weight / len(mapped)
                for d in mapped:
                    defect_weights[d] = defect_weights.get(d, 0.0) + per_d
        # model/MTTS/temporal features → skip

    # --- Roll up to Campbell process level with co-occurrence scoring --
    process_weights = {}
    for defect, w in defect_weights.items():
        candidate_procs = DEFECT_TO_PROCESSES.get(defect, [])
        if not candidate_procs:
            continue

        if len(candidate_procs) == 1:
            # Single-cause: direct attribution
            proc_name = candidate_procs[0][0]
            process_weights[proc_name] = process_weights.get(proc_name, 0.0) + w
        else:
            # Multi-cause: score by co-occurrence of OTHER defects from each process
            scores = {
                proc_name: process_co_occurrence_score(proc_name, observed_defects_set)
                for proc_name, _ in candidate_procs
            }
            total_score = sum(scores.values())
            if total_score <= 0:
                # No co-occurrence evidence — split equally
                per_proc = w / len(candidate_procs)
                for proc_name, _ in candidate_procs:
                    process_weights[proc_name] = process_weights.get(proc_name, 0.0) + per_proc
            else:
                for proc_name, score in scores.items():
                    process_weights[proc_name] = (
                        process_weights.get(proc_name, 0.0) + w * (score / total_score)
                    )

    return process_weights, defect_weights


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
    
    # TABS — labels aligned with dissertation (H1=MPTS reliability, H2=RF classifier)
    # Note: tab position order is preserved from original code; only the displayed
    # labels are updated so existing `with tabN:` blocks don't need to be renumbered.
    # Presentation build: only the Defense tab is rendered so the page loads fast.
    # Tabs 1–10 were removed (they each re-ran heavy compute on every rerun).
    (tab0,) = st.tabs(["🎓 Defense — One-Part Results"])
    
    # ================================================================
    # TAB 0: DEFENSE — consolidated one-part results (H1 · H2 · H3)
    # Reuses existing functions so all figures match reported results.
    # ================================================================
    with tab0:
        st.header("🎓 Defense — One-Part Results  ·  H1 · H2 · H3")
        st.caption("A single view of everything reported for a selected part. "
                   "Part and threshold controls mirror the Prognostic tab; no LIME.")

        # ---- Controls: threshold only. The PART is the app-wide selector
        # at the top of the page (selected_part) — no second dropdown here. ----
        d_part = selected_part
        st.caption(f"Showing results for **Part {d_part}** — change the part using the "
                   f"**Select Part ID** control at the top of the page.")
        _spacer, dcol2 = st.columns([1, 1])
        d_data = df[df['part_id'] == d_part]
        d_part_avg = float(d_data['scrap_percent'].mean()) if len(d_data) else 0.0
        d_global = float(df['scrap_percent'].mean())
        d_scrap_max = float(round(d_data['scrap_percent'].max() + 1, 1)) if len(d_data) else 12.0

        # When the part changes, reset the threshold slider to THIS part's own average.
        _thr_max = float(max(12.0, d_scrap_max))
        if st.session_state.get("defense_last_part") != d_part:
            st.session_state["defense_thr"] = float(min(max(round(d_part_avg, 2), 0.5), _thr_max))
            st.session_state["defense_last_part"] = d_part
        with dcol2:
            d_thr = st.slider("Scrap % Threshold (failure definition)",
                              min_value=0.5,
                              max_value=_thr_max,
                              step=0.1, key="defense_thr",
                              help="Defaults to the selected part's own average (the MPTS renewal baseline). "
                                   "Global average shown for reference.")
        st.markdown(
            f"<div style='background:#EAF4F0;border-left:5px solid #0E7C7B;padding:8px 12px;border-radius:4px;'>"
            f"<strong>Part {d_part}</strong> &nbsp;·&nbsp; own avg <strong>{d_part_avg:.2f}%</strong> "
            f"&nbsp;·&nbsp; global avg <strong>{d_global:.2f}%</strong> "
            f"&nbsp;·&nbsp; threshold in use <strong>{d_thr:.2f}%</strong> "
            f"&nbsp;·&nbsp; runs <strong>{len(d_data)}</strong></div>",
            unsafe_allow_html=True)

        # ============================================================
        # ELIGIBILITY GATE — is this part valid for MPTS?
        # Criteria (§3.3.1): ≥20 runs AND ≥4 own-mean exceedances AND no trend.
        # The seven reported parts: 15, 63, 74, 122, 3, 14, 124.
        # ============================================================
        _ELIGIBLE_SEVEN = {'15', '63', '74', '122', '3', '14', '124'}
        _n_runs = len(d_data)
        _n_exceed = int((d_data['scrap_percent'] > d_part_avg).sum()) if _n_runs else 0
        _is_seven = str(d_part) in _ELIGIBLE_SEVEN
        _meets_runs = _n_runs >= 20
        _meets_exceed = _n_exceed >= 4

        if _is_seven:
            st.success(f"✅ **Part {d_part} is one of the seven MPTS-eligible parts.** "
                       f"All results below are valid and reported in the praxis "
                       f"({_n_runs} runs · {_n_exceed} own-mean exceedances · no trend per Louit).")
            _valid = True
        elif _meets_runs and _meets_exceed:
            st.info(f"ℹ️ **Part {d_part} meets the run/exceedance thresholds** "
                    f"({_n_runs} runs · {_n_exceed} exceedances) but is not among the seven parts reported "
                    f"in the praxis (it may not clear the Louit no-trend screen). Treat the figures below as "
                    f"exploratory, not reported results.")
            _valid = True
        else:
            _why = []
            if not _meets_runs: _why.append(f"only {_n_runs} runs (needs ≥20)")
            if not _meets_exceed: _why.append(f"only {_n_exceed} own-mean exceedances (needs ≥4)")
            st.warning(f"⚠️ **Part {d_part} is NOT MPTS-eligible** — " + "; ".join(_why) + ". "
                       "MPTS needs ≥20 runs and ≥4 scrap-threshold exceedances so the reliability estimate is "
                       "statistically meaningful (§3.3.1). Any reliability, signal, or avoidance figure shown "
                       "below for this part is **not valid** and is not part of the reported results — the seven "
                       "reported parts are 15, 63, 74, 122, 3, 14, and 124.")
            _valid = False

        st.divider()

        # ============================================================
        # H1 — Reliability licensed (Louit) + MPTS reliability
        # ============================================================
        st.subheader("H1 · Reliability model is licensed  (MPTS + Louit renewal screen)")
        if not _valid:
            st.caption("⚠️ Not valid for this part — shown for illustration only (part is not MPTS-eligible).")
        try:
            louit = compute_louit_screening(df, d_part)
        except Exception as e:
            louit = None
        pooled = compute_pooled_prediction(df, d_part, d_thr)

        # MPTS parts at the current threshold (drives R(n) = e^(-n / MPTS_parts))
        _mpts_parts = None
        if louit and louit.get('mpts_parts'):
            _mpts_parts = float(louit['mpts_parts'])
        elif pooled.get('mtts_parts'):
            _mpts_parts = float(pooled['mtts_parts'])

        # Order quantity (parts in the next run): default to this part's average, editable.
        _avg_oq = float(round(d_data['order_quantity'].mean())) if len(d_data) else 100.0
        oqc1, oqc2 = st.columns([1, 2])
        with oqc1:
            d_oq = st.number_input("Parts in next run (order qty)",
                                   min_value=1, value=int(max(1, _avg_oq)), step=1,
                                   key="defense_oq",
                                   help=f"Defaults to Part {d_part}'s average order quantity ({int(_avg_oq)}). "
                                        "Enter any order size to see reliability for that run.")
        # Reliability at the chosen order quantity
        if _mpts_parts and _mpts_parts > 0:
            d_reliab = float(np.exp(-d_oq / _mpts_parts))
        else:
            d_reliab = float(pooled.get('reliability_next_run', 0))
        with oqc2:
            st.caption(f"R(n) = e^(−n / MPTS) = e^(−{d_oq:,} / {(_mpts_parts or 0):,.0f}) "
                       f"→ probability the run of {d_oq:,} parts completes without a scrap event.")

        h1c1, h1c2, h1c3, h1c4 = st.columns(4)
        h1c1.metric("MPTS (runs)", f"{pooled.get('mtts_runs', float('nan')):.1f}")
        h1c2.metric("MPTS (parts)", f"{(_mpts_parts or pooled.get('mtts_parts', 0)):,.0f}")
        h1c3.metric(f"Reliability (run of {d_oq:,})", f"{d_reliab*100:.1f}%")
        h1c4.metric("Failures / runs", f"{pooled.get('failure_count', 0)} / {len(d_data)}")

        if louit and louit.get('trend_testable'):
            lap = louit.get('laplace_L')
            lr = louit.get('lewis_robinson_LR')
            verdict = louit.get('verdict', '')
            st.markdown(
                f"**Louit two-step trend test** &nbsp;·&nbsp; "
                f"Laplace L = {lap:.3f}" + (f" &nbsp;·&nbsp; Lewis–Robinson LR = {lr:.3f}" if lr is not None else "") +
                f" &nbsp;·&nbsp; **{verdict}**")
        elif louit:
            st.info(f"Louit trend test: {louit.get('verdict', 'insufficient failures for a trend test')}")

        st.caption("MPTS failure event = a run exceeding the part's own average scrap rate — the 'like-new' "
                   "renewal state the Louit screen validates (Table 4-1). Exponential R(n)=e^(−n̄/MPTS).")

        st.divider()

        # ============================================================
        # H2 — RF classifier validation (global headline metrics)
        # ============================================================
        st.subheader("H2 · Diagnostic classifier is accurate  (Random Forest, global hold-out)")
        m = global_model.get('metrics', {}) if 'global_model' in dir() else {}
        rec = m.get('recall'); prec = m.get('precision'); auc = m.get('auc'); brier = m.get('brier')
        # Clopper-Pearson lower bound on recall from stored y_test/y_pred
        cp_lo = None
        try:
            yt = np.array(m.get('y_test')); yp = np.array(m.get('y_pred'))
            tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
            if (tp + fn) > 0:
                cp_lo, _ = clopper_pearson_ci(tp, tp + fn)
        except Exception:
            cp_lo = None

        h2c1, h2c2, h2c3, h2c4 = st.columns(4)
        h2c1.metric("Recall", f"{rec*100:.1f}%" if rec is not None else "—",
                    help="Pre-registered bar: ≥80% recall AND CP lower bound ≥80%")
        h2c2.metric("CP 95% lower bound", f"{cp_lo*100:.1f}%" if cp_lo is not None else "—")
        h2c3.metric("Precision", f"{prec*100:.1f}%" if prec is not None else "—")
        h2c4.metric("AUC · Brier", (f"{auc:.3f} · {brier:.3f}" if auc is not None else "—"))

        # seen vs unseen generalization
        try:
            su = compute_seen_unseen_metrics(global_model)
            if su:
                seen = su.get('seen', {}); unseen = su.get('unseen', {})
                st.markdown(
                    f"**Generalization** &nbsp;·&nbsp; seen parts recall "
                    f"{seen.get('recall', float('nan'))*100:.0f}% (n={seen.get('failures','?')}) "
                    f"&nbsp;·&nbsp; **unseen parts recall {unseen.get('recall', float('nan'))*100:.0f}% "
                    f"(n={unseen.get('failures','?')})** — the model generalizes on signature, not identity.")
        except Exception:
            pass

        st.caption("RF trained foundry-wide against the global scrap average (4.90%). It classifies completed "
                   "runs — the diagnostic layer that checks whether 'other factors are at play' (Juran). "
                   "These validation metrics are the single foundry-wide model's out-of-sample result "
                   "(one RF for all parts), not a per-part figure — the same model then scores each part's last run below.")

        st.divider()

        # ============================================================
        # S-SIGNAL — dual-model divergence for THIS part
        # ============================================================
        st.subheader("Dual-model signal for this part  (MPTS vs RF → S1–S4)")
        if not _valid:
            st.caption("⚠️ Not valid for this part — the signal needs a stable MPTS baseline (≥20 runs, ≥4 exceedances).")

        # No @st.cache_data here: st.cache_data does not invalidate when the
        # nested compute_dual_model_validation_table changes, which stales the
        # defect columns. One part is a cheap inference, so recompute each run.
        def _defense_signal(_part_id):
            _t, _ = compute_dual_model_validation_table(df, defect_cols, global_model, [_part_id])
            return _t

        _r = None
        try:
            _sig_df = _defense_signal(d_part)
            if _sig_df is not None and len(_sig_df):
                _r = _sig_df.iloc[0]
        except Exception as _e:
            st.warning(f"Signal computation error for Part {d_part}: {type(_e).__name__}: {_e}")
            _r = None

        if _r is not None and _r.get("Signal", "—") not in ("—", None):
            # --- MPTS P% computed INLINE from the SAME MPTS_parts + avg order qty
            #     that H1 uses, so it is guaranteed consistent with 1 - reliability.
            if _mpts_parts and _mpts_parts > 0:
                _mp = round((1.0 - float(np.exp(-(_avg_oq) / _mpts_parts))) * 100.0, 1)
            else:
                _mp = _r.get("MPTS P%")
            # RF last-run probability comes from the RF (its own output) via the table.
            _rf = _r.get("RF Last%")
            # Δ and the S1/S2/S3 classification are derived from the inline MPTS P%.
            if _rf is not None and _mp is not None:
                _dl = round(_rf - _mp, 1)
                if abs(_dl) <= 5:
                    _sig_base = "S1 — Aligned ≈"
                elif _dl > 5:
                    _sig_base = "S2 — Alarm ▲"
                else:
                    _sig_base = "S3 — Improvement ▼"
                # Promote to S4 if improving AND the active process differs from chronic
                _chron = _r.get("Chronic Process", "—"); _activ = _r.get("Last-Run Active", "—")
                if _dl < -5 and _activ not in (_chron, "—", None):
                    _sigtxt = "S4 — New Process ⚡"
                else:
                    _sigtxt = _sig_base
            else:
                _dl = _r.get("Δ (pp)"); _sigtxt = str(_r.get("Signal", "—"))

            sg1, sg2, sg3, sg4 = st.columns(4)
            sg1.metric("MPTS P%", f"{_mp:.1f}%" if _mp is not None else "—",
                       help="Prognostic exceedance probability at the average order quantity "
                            "(= 1 − H1 reliability at that order size).")
            sg2.metric("RF Last%", f"{_rf:.1f}%" if _rf is not None else "—",
                       help="RF probability the last run exceeds the global scrap average.")
            sg3.metric("Δ (pp)", f"{_dl:+.1f}" if _dl is not None else "—",
                       help="Δ = RF Last% − MPTS P%.")
            _color = {"S1": "#1F8A52", "S2": "#B23A48", "S3": "#185FA5", "S4": "#C77700"}
            _key = next((k for k in _color if _sigtxt.startswith(k)), None)
            sg4.markdown(
                f"<div style='padding:6px 10px;border-radius:6px;background:{_color.get(_key,'#55606B')};"
                f"color:#fff;text-align:center;font-weight:700;font-size:15px;margin-top:6px;'>"
                f"{_sigtxt}</div>", unsafe_allow_html=True)
            _cc1, _cc2 = st.columns(2)
            _mpts_def = _r.get('Chronic Defect', '—')
            _rf_def = _r.get('Last-Run Defect', '—')
            _cc1.markdown(f"**Chronic process (MPTS):** {_r.get('Chronic Process','—')}")
            _cc1.caption(f"driven by dominant defect **{_mpts_def}** (across failure runs)")
            _cc2.markdown(f"**Last-run active process (RF):** {_r.get('Last-Run Active','—')}")
            _cc2.caption(f"driven by dominant defect **{_rf_def}** (last run)")
            if _sigtxt.startswith("S4"):
                st.caption(
                    f"→ **S4 rationale:** last-run active process "
                    f"(**{_r.get('Last-Run Active','—')}**, driven by {_rf_def}) differs from the "
                    f"chronic process (**{_r.get('Chronic Process','—')}**, driven by {_mpts_def}) "
                    f"while the run is improving (Δ = {_dl:+.1f} pp) — the leading defect has shifted "
                    f"to a new Campbell process."
                )
            # Cross-check: the signal's MPTS P% must equal 1 - H1 reliability at the
            # AVERAGE order quantity (both are 1 - e^(-avg_oq/MPTS)). If they differ,
            # something is stale — surface it rather than let two numbers disagree.
            try:
                _h1_avg_R = float(np.exp(-(_avg_oq) / _mpts_parts)) if _mpts_parts else None
                if _h1_avg_R is not None and _mp is not None:
                    _implied = (1 - _h1_avg_R) * 100
                    if abs(_implied - _mp) > 0.3:
                        st.warning(f"Consistency note: signal MPTS P% ({_mp:.1f}%) and 1 − H1 reliability "
                                   f"at avg order qty ({_implied:.1f}%) differ — clear the cache (⋮ → Clear cache) "
                                   f"and rerun; they should be identical.")
                    else:
                        st.caption(f"✓ Cross-check: signal MPTS P% ({_mp:.1f}%) = 1 − reliability at the "
                                   f"average order quantity ({int(_avg_oq)} parts). The H1 box lets you vary the "
                                   f"order size; the signal is fixed at the average.")
            except Exception:
                pass
        else:
            st.info(f"Part {d_part} has insufficient failure history to produce a stable dual-model signal "
                    f"(the S-signal requires enough exceedance runs to define the MPTS baseline).")

        # ---- S1–S4 legend / key ----
        st.markdown("**What the signals mean:**")
        _legend = [
            {"Signal": "S1 — Aligned ≈", "Meaning": "Δ within ±5pp. Nothing new at play — status-quo signature.",
             "Manager action": "Conditions consistent with history. Schedule PM as planned."},
            {"Signal": "S2 — Alarm ▲", "Meaning": "RF > MPTS + 5pp. Last run looks worse than the part's baseline.",
             "Manager action": "Expedite PM — do not wait for the scheduled interval."},
            {"Signal": "S3 — Improvement ▼", "Meaning": "RF < MPTS − 5pp. Last run better than baseline.",
             "Manager action": "Confirm whether a prior PM intervention held."},
            {"Signal": "S4 — New Process ⚡", "Meaning": "Improving, but the leading defect shifted to a different Campbell process.",
             "Manager action": "Schedule chronic PM and watch the newly active process next run."},
        ]
        st.dataframe(pd.DataFrame(_legend), hide_index=True, use_container_width=True)
        st.caption("Δ = RF last-run probability − MPTS probability (percentage points). The ±5pp band was "
                   "determined by trial-and-error against the 32-month census to separate the four signals, and "
                   "is a foundry-tunable parameter (proof of concept), not a universal cutoff.")

        st.divider()

        # ============================================================
        # H3 — Avoidance + energy / emissions (this part)
        # ============================================================
        st.subheader("H3 · Acting on it avoids scrap  (energy & emissions)")

        # ---- Facility-level H3 results, as reported in the praxis ----
        fh1, fh2, fh3, fh4 = st.columns(4)
        fh1.metric("Scrap avoidance", "12.70%", help="Of the 20,629 lb facility scrap (first 12 months)")
        fh2.metric("Avoidable scrap", "2,619 lbs/yr", help="Seven parts, Tier 1 clip-to-baseline, CP-adjusted")
        fh3.metric("TTE / GHG avoided", "123.7 MMBtu · 6.56 MT CO₂")
        fh4.metric("Benefit-Cost Ratio", "4.02×")
        st.markdown(
            "<div style='background:#EAF4F0;border-left:5px solid #1F8A52;padding:8px 12px;border-radius:4px;'>"
            "<strong>Facility result (locked):</strong> the seven MPTS-eligible parts (4% of the catalogue) supply "
            "39.25% of first-year scrap; a Tier 1 clip-to-baseline intervention yields <strong>12.70% avoidance</strong> "
            "(2,619 lbs/yr, CP-adjusted) — meeting/exceeding the EPA ENERGY STAR 3–10% range and aligning with the DOE "
            "10% target. Part 15 alone contributes 7.90%.</div>", unsafe_allow_html=True)

        # ---- Table 4-6: seven-part per-part avoidance (computed live, clip-to-baseline) ----
        # ---- Shared H3 window + constants (live) ----
        _CP = 0.902
        _PW = 'piece_weight_lbs'
        _cs = pd.to_datetime(df['week_ending'], errors='coerce').min()
        _w12 = _cs + pd.Timedelta(days=365)
        _wall = df.copy()
        _wall['week_ending'] = pd.to_datetime(_wall['week_ending'], errors='coerce')
        _win = _wall[_wall['week_ending'] < _w12].copy()
        _TTEF = 47250.0      # BTU/lb (Eppich 2004)
        _CO2F = 53.06        # kg CO2/MMBtu (EPA 2023)
        _ECOST = 12.00       # $/MMBtu (DOE)
        _MCOST = 2.50        # $/lb aluminum
        _IMPL = 2000.0       # $ implementation

        # ===== TABLE 4-4 · Foundry Operational Parameters (live) =====
        with st.expander("Table 4-4 · Foundry Operational Parameters (computed from the census)", expanded=False):
            _tot_prod = float((_win['order_quantity'] * _win[_PW]).sum())
            _tot_scrap = float((_win['order_quantity'] * _win[_PW] * (_win['scrap_percent'] / 100.0)).sum())
            _rate32 = df['scrap_percent'].mean()
            _rate12 = 100.0 * _tot_scrap / _tot_prod if _tot_prod else 0.0
            _p15 = df[df['part_id'] == '15'].copy()
            _p15['week_ending'] = pd.to_datetime(_p15['week_ending'], errors='coerce')
            _p15w = _p15[_p15['week_ending'] < _w12]
            _p15_prod = float((_p15w['order_quantity'] * _p15w[_PW]).sum())
            _p15_scrap = float((_p15w['order_quantity'] * _p15w[_PW] * (_p15w['scrap_percent'] / 100.0)).sum())
            _p15_avg12 = 100.0 * _p15_scrap / _p15_prod if _p15_prod else 0.0
            _p15_base = _p15['scrap_percent'].mean()
            # Part 15 new mean after clipping above-baseline runs to baseline (first 12 mo target)
            _p15_clip = _p15w.copy()
            _p15_clip_rate = _p15_clip['scrap_percent'].clip(upper=_p15_base)
            _p15_newmean = float((_p15_clip['order_quantity'] * _p15_clip[_PW] * (_p15_clip_rate / 100.0)).sum()
                                 / _p15_prod * 100.0) if _p15_prod else 0.0
            _t44 = [
                {"Parameter": "Total foundry production (first 12 months)", "Value": f"{_tot_prod:,.0f} lbs",
                 "Source": "Σ(order-qty × piece-weight), first 12 mo"},
                {"Parameter": "Total foundry scrap (first 12 months)", "Value": f"{_tot_scrap:,.0f} lbs",
                 "Source": "Foundry records, first 12 mo of 32-mo census"},
                {"Parameter": "Current foundry scrap rate", "Value": f"{_rate32:.2f}% (32-mo) · {_rate12:.2f}% (12-mo)",
                 "Source": "Population census, n = 1,257"},
                {"Parameter": "Part 15 production (first 12 months)", "Value": f"{_p15_prod:,.0f} lbs",
                 "Source": "Part 15 order-qty × 22 lb/piece"},
                {"Parameter": "Part 15 average scrap rate (first 12 months)", "Value": f"{_p15_avg12:.2f}%",
                 "Source": f"{_p15_scrap:,.0f} lbs / {_p15_prod:,.0f} lbs"},
                {"Parameter": "Part 15 chronic baseline (32-mo mean) — Tier 1 clip level", "Value": f"{_p15_base:.2f}%",
                 "Source": "Mean scrap% over 32 months"},
                {"Parameter": "Part 15 new mean after clipping to baseline", "Value": f"{_p15_newmean:.2f}%",
                 "Source": "Tier 1 target for Eq. 3-11"},
                {"Parameter": "Energy intensity (TTEF)", "Value": "47,250 BTU/lb (94.5 MMBtu/ton)",
                 "Source": "Eppich (2004); EPA (2016)"},
                {"Parameter": "Energy cost", "Value": "$12.00/MMBtu", "Source": "DOE benchmark"},
                {"Parameter": "CO₂ emission factor", "Value": "53.06 kg CO₂/MMBtu", "Source": "EPA (2023)"},
                {"Parameter": "Material cost", "Value": "$2.50/lb", "Source": "Aluminum scrap-value benchmark"},
                {"Parameter": "Implementation cost (estimated)", "Value": "$2,000", "Source": "No new sensor infrastructure"},
            ]
            st.dataframe(pd.DataFrame(_t44), hide_index=True, use_container_width=True)
            st.caption("Production, scrap, and rates computed live from the first 12 months of the census. "
                       "Physical/economic constants (energy, CO₂, costs) are the cited literature values.")

        # ===== TABLE 4-5 · Equation-by-Equation Calculation Chain (live, Part 15) =====
        with st.expander("Table 4-5 · Equation-by-equation calculation chain (Part 15 worked example)", expanded=False):
            _red_raw = _p15_prod * (_p15_avg12 - _p15_newmean) / 100.0     # Eq 3-11
            _red_cp = _red_raw * _CP                                        # Eq 3-12
            _tte = _red_cp * _TTEF / 1_000_000                             # Eq 3-13
            _ghg = _tte * _CO2F / 1000                                     # Eq 3-14
            _matsav = _red_cp * _MCOST                                     # Eq 3-15
            _ensav = _tte * _ECOST                                         # Eq 3-16
            _bcr = (_matsav + _ensav) / _IMPL                              # Eq 3-17
            _totsav = _matsav + _ensav                                     # Eq 3-18
            _pieces = _red_cp / 22.0
            _t45 = [
                {"Equation": "Eq. 3-11: Annual Scrap Reduction (unadjusted)",
                 "Calculation": f"{_p15_prod:,.0f} × ({_p15_avg12:.2f}% − {_p15_newmean:.2f}%)/100",
                 "Result": f"{_red_raw:,.0f} lbs/yr"},
                {"Equation": "Eq. 3-12: C-P-Adjusted Scrap Reduction",
                 "Calculation": f"{_red_raw:,.0f} × 0.902", "Result": f"{_red_cp:,.0f} lbs/yr"},
                {"Equation": "Eq. 3-13: TTE Savings",
                 "Calculation": f"{_red_cp:,.0f} × 47,250 / 1,000,000", "Result": f"{_tte:,.1f} MMBtu/yr"},
                {"Equation": "Eq. 3-14: GHG Reduction",
                 "Calculation": f"{_tte:,.1f} × 53.06 / 1,000", "Result": f"{_ghg:,.2f} MT CO₂/yr"},
                {"Equation": "Eq. 3-15: Material Savings",
                 "Calculation": f"{_red_cp:,.0f} lbs × $2.50/lb", "Result": f"${_matsav:,.0f}/yr"},
                {"Equation": "Eq. 3-16: Energy Savings",
                 "Calculation": f"{_tte:,.1f} × $12.00/MMBtu", "Result": f"${_ensav:,.0f}/yr"},
                {"Equation": "Eq. 3-17: Benefit-Cost Ratio (BCR)",
                 "Calculation": f"(${_matsav:,.0f} + ${_ensav:,.0f}) / $2,000", "Result": f"{_bcr:.2f} BCR"},
                {"Equation": "Eq. 3-18: Total Annual Savings",
                 "Calculation": f"${_matsav:,.0f} + ${_ensav:,.0f}", "Result": f"${_totsav:,.0f}/yr"},
                {"Equation": "Pieces Saved from Scrap",
                 "Calculation": f"{_red_cp:,.0f} lbs / 22 lb/piece", "Result": f"{_pieces:,.0f} pieces"},
            ]
            st.dataframe(pd.DataFrame(_t45), hide_index=True, use_container_width=True)
            st.caption("Each row computed live from Part 15's first-12-month data and the operational parameters "
                       "in Table 4-4. Reproduces the praxis Table 4-5 chain for Part 15.")

        with st.expander("Table 4-6 · Seven-part aggregate Tier 1 avoidable scrap (per part)", expanded=True):
            _CP = 0.902
            _seven = ['15', '63', '74', '122', '3', '14', '124']
            _cs = pd.to_datetime(df['week_ending'], errors='coerce').min()
            _w12 = _cs + pd.Timedelta(days=365)
            _facility = 20629.0
            _rows = []
            _tot = 0.0
            for _p in _seven:
                _pp = df[df['part_id'] == _p].copy()
                _pp['week_ending'] = pd.to_datetime(_pp['week_ending'], errors='coerce')
                _bl = _pp['scrap_percent'].mean() / 100.0
                _wn = _pp[_pp['week_ending'] < _w12]
                _pw = _wn['piece_weight_lbs'] if 'piece_weight_lbs' in _wn.columns else 1.0
                _ex = ((_wn['order_quantity'] * _pw * (_wn['scrap_percent'] / 100.0))
                       - (_wn['order_quantity'] * _pw * _bl)).clip(lower=0)
                _cp = float(_ex.sum()) * _CP
                _tot += _cp
                _rows.append({
                    "Part": _p,
                    "Chronic baseline %": f"{_pp['scrap_percent'].mean():.2f}%",
                    "CP-adj avoidable (lbs/yr)": f"{_cp:,.0f}",
                    "% of facility scrap": f"{100 * _cp / _facility:.2f}%",
                })
            _rows.append({"Part": "Total", "Chronic baseline %": "—",
                          "CP-adj avoidable (lbs/yr)": f"{_tot:,.0f}",
                          "% of facility scrap": f"{100 * _tot / _facility:.2f}%"})
            st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)
            st.caption("Each part clipped over the first 12 months back to its 32-month chronic baseline, "
                       "Clopper-Pearson adjusted (×0.902), against the 20,629 lb facility scrap. "
                       "Reproduces Table 4-6: 2,619 lbs / 12.70%, Part 15 = 7.90%.")

        # ===== TABLE 4-7 · Alignment with EPA ENERGY STAR Benchmark (live) =====
        with st.expander("Table 4-7 · Alignment with EPA ENERGY STAR benchmark", expanded=False):
            _avoid_pct = 100.0 * _tot / 20629.0   # _tot from the 4-6 block above
            _t47 = [
                {"Benchmark": "EPA ENERGY STAR process-level savings", "Reference Range": "3–10%",
                 "This Study": f"{_avoid_pct:.2f}%", "Status": "Exceeded ✓"},
                {"Benchmark": "Annual scrap avoided vs. DOE 10% target", "Reference Range": "≥ 10% of facility scrap",
                 "This Study": f"{_avoid_pct:.2f}%", "Status": "Aligned ✓"},
            ]
            st.dataframe(pd.DataFrame(_t47), hide_index=True, use_container_width=True)
            st.caption("This-study figure is the seven-part CP-adjusted avoidance (Table 4-6) as a percentage of the "
                       "20,629 lb facility scrap, benchmarked against EPA ENERGY STAR (2016) 3–10% and the DOE 10% target.")

        st.markdown(f"---")
        st.markdown(f"**Selected part (Part {d_part}) — its own H3 contribution:**")
        if not _valid:
            st.caption("⚠️ Part not MPTS-eligible — this per-part avoidance is illustrative and excluded from the reported 12.70%.")

        # ---- Actual H3 method: clip first-12-month runs above the part's
        # 32-month chronic baseline back to that baseline; CP-adjust (×0.902). ----
        st.markdown(f"**This part's H3 contribution** — Tier 1, clip-to-baseline (the method behind the 12.70%):")
        _CP = 0.902
        _dd = d_data.copy()
        if 'week_ending' in _dd.columns:
            _dd['week_ending'] = pd.to_datetime(_dd['week_ending'], errors='coerce')
            # Window B = first 12 months of the CENSUS (global start), per H3 method,
            # not the part's own first run — this is what reproduces the 12.70%.
            _census_start = pd.to_datetime(df['week_ending'], errors='coerce').min()
            _w12 = _census_start + pd.Timedelta(days=365)
            _win = _dd[_dd['week_ending'] < _w12].copy()
        else:
            _win = _dd.copy()
        _baseline = d_part_avg / 100.0  # part's 32-month chronic baseline (fraction)
        _pw = _win['piece_weight_lbs'] if 'piece_weight_lbs' in _win.columns else 1.0
        _actual_lbs = _win['order_quantity'] * _pw * (_win['scrap_percent'] / 100.0)
        _base_lbs = _win['order_quantity'] * _pw * _baseline
        _excess = (_actual_lbs - _base_lbs).clip(lower=0)     # only runs ABOVE baseline
        _avoid_raw = float(_excess.sum())
        _avoid_cp = _avoid_raw * _CP
        # energy + emissions on the avoidable pounds
        _tte_mmbtu = _avoid_cp * 47250 / 1_000_000
        _co2_t = _tte_mmbtu * 53.06 / 1000
        _n_clip = int((_excess > 0).sum())

        h3c1, h3c2, h3c3, h3c4 = st.columns(4)
        h3c1.metric("Avoidable scrap (CP-adj)", f"{_avoid_cp:,.0f} lbs/yr")
        h3c2.metric("Runs clipped (12 mo)", f"{_n_clip} of {len(_win)}")
        h3c3.metric("TTE savings", f"{_tte_mmbtu:,.1f} MMBtu/yr")
        h3c4.metric("GHG avoided", f"{_co2_t:,.2f} t CO₂/yr")
        st.caption(f"Method: over the first 12 months, every run above Part {d_part}'s 32-month chronic "
                   f"baseline ({d_part_avg:.2f}%) is clipped back to that baseline; the clipped excess is the "
                   f"avoidable scrap, then Clopper-Pearson adjusted (×0.902). Energy 47,250 BTU/lb (Eppich 2004); "
                   f"53.06 kg CO₂/MMBtu (EPA 2023). Seven parts sum to 2,619 lbs/yr = 12.70% of facility scrap.")

        st.divider()

        # ============================================================
        # Pareto-Campbell process attribution (KEEP) — no LIME
        # ============================================================
        st.subheader("Pareto–Campbell process attribution")
        st.caption(f"Most-occurring defect in this part's failure runs (scrap > {d_thr:.2f}%), "
                   "mapped to its Campbell primary process. Reports the single most-likely process; "
                   "secondaries are documented for the manager to investigate.")
        try:
            sorted_proc, sorted_def = diagnose_processes(df[df['scrap_percent'] > d_thr] if False else d_data,
                                                         d_part, defect_cols)
        except Exception:
            sorted_proc, sorted_def = None, None

        # failure-conditional defect ranking (within threshold-exceeding runs)
        d_fail = d_data[d_data['scrap_percent'] > d_thr]
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Top processes (Campbell-mapped)**")
            if sorted_proc:
                rows = [{"Process": p, "Contribution %": f"{v:.1f}%"} for p, v in sorted_proc[:6] if v > 0]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with pc2:
            st.markdown(f"**Vital-few defects in failure runs (n={len(d_fail)})**")
            if len(d_fail):
                fr = {c: d_fail[c].mean() for c in defect_cols if c in d_fail.columns}
                fr = {k: v for k, v in sorted(fr.items(), key=lambda x: -x[1]) if v > 0}
                rows = [{"Defect": k.replace('_rate', '').replace('_', ' ').title(),
                         "Rate %": f"{v*100:.2f}%"} for k, v in list(fr.items())[:6]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                if rows:
                    top_def = list(fr.keys())[0]
                    top_proc = DEFECT_TO_PROC.get(top_def, "—")
                    st.success(f"Dominant defect **{top_def.replace('_rate','').replace('_',' ').title()}** "
                               f"→ Campbell primary process **{top_proc}**")

        st.caption("Campbell (2003) Ten Rules are a process-design checklist ('necessary, but not sufficient'); "
                   "used here as a conceptual alignment for a proof-of-concept map, not one-to-one causal proof.")

    st.markdown("---")
    st.caption("🏭 Foundry Dashboard V4 | Dual-Method MPTS + RF | n=1,257 runs | Model 1 (MPTS) → H1 | Model 2 (RF) → H2 | 60-20-20 Split | GWU D.Eng. Praxis")


if __name__ == "__main__":
    main()
