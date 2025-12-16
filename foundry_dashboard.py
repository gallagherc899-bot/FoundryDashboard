# ---------------------------------------------------------------
# 🧪 Foundry Scrap Risk Dashboard — Fixed Version (Dec 2025)
# Compatible with Python 3.13 + scikit-learn 1.8.0 + Streamlit 1.52
# ---------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon
from sklearn import __version__ as skl_version
from packaging import version

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

# ---------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

# ---------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)
TOP_K_PARETO = 8
USE_RATE_COLS_PERMANENT = True

# ---------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("#", "num", regex=False)
    )
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed).copy()

    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"]).copy()

    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(
            (df["scrap%"].clip(lower=0) / 100.0) * df["order_quantity"]
        ).astype(int)

    df = df.sort_values("week_ending").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac=0.60, calib_frac=0.20):
    n = len(df)
    t_end = int(train_frac * n)
    c_end = int((train_frac + calib_frac) * n)

    df_train = df.iloc[:t_end].copy()
    df_calib = df.iloc[t_end:c_end].copy()
    df_test = df.iloc[c_end:].copy()

    # Prevent part leakage
    train_parts = set(df_train.part_id.unique())
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)].copy()
    calib_parts = set(df_calib.part_id.unique())
    df_test = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))].copy()
    return df_train, df_calib, df_test


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    t = df_train.copy()
    t["scrap_flag"] = (t["scrap%"] > thr_label).astype(int)
    mtbf = t.groupby("part_id").agg(
        total_runs=("scrap%", "count"),
        failures=("scrap_flag", "sum")
    )
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]


def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s


def make_xy(df, thr_label: float, use_rate_cols: bool):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Missing expected features in DataFrame: {missing}")
    X = df[feats].copy()
    y = (df["scrap%"] > thr_label).astype(int)
    return X, y, feats

# ---------------------------------------------------------------
# Model Training + Safe Calibration
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators: int):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)

    has_both = (y_calib.sum() > 0) and (y_calib.sum() < len(y_calib))
    if not has_both:
        st.warning("⚠️ Calibration skipped — only one class in calibration set.")
        return rf, rf, "uncalibrated"

    method = "isotonic" if len(y_calib) > 500 else "sigmoid"
    try:
        cv_value = 3  # use cross-validation instead of prefit for sklearn >=1.6
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv=cv_value).fit(X_calib, y_calib)
        return rf, cal, f"calibrated ({method}, cv={cv_value})"
    except Exception as e:
        st.error(f"Calibration failed ({type(e).__name__}: {e}). Using uncalibrated RF.")
        return rf, rf, "uncalibrated"

# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Risk Definition")
thr_label = st.sidebar.slider("Scrap % Threshold (label & MTTFscrap)", 1.0, 15.0, 6.50, 0.5)

st.sidebar.header("🧪 Model Validation")
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation (slower)", value=True)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# ---------------------------------------------------------------
# Load and Prepare Data
# ---------------------------------------------------------------
df = load_and_clean(csv_path)
st.title("🧠 Foundry Scrap Risk Dashboard — Actionable Insights")
st.caption("Random Forest + calibrated probabilities • per-part reliability & Pareto analysis")

# 1️⃣ Split data
df_train, df_calib, df_test = time_split(df)

# 2️⃣ Compute reliability and frequency features
mtbf_train = compute_mtbf_on_train(df_train, thr_label)
part_freq_train = df_train["part_id"].value_counts(normalize=True)
default_mtbf = float(mtbf_train["mttf_scrap"].median())
default_freq = float(part_freq_train.median())

df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test  = attach_train_features(df_test,  mtbf_train, part_freq_train, default_mtbf, default_freq)

# 3️⃣ Build feature matrices
X_train, y_train, feats = make_xy(df_train, thr_label, USE_RATE_COLS_PERMANENT)
X_calib, y_calib, _     = make_xy(df_calib, thr_label, USE_RATE_COLS_PERMANENT)

# 4️⃣ Train + calibrate
rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, DEFAULT_ESTIMATORS)

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
st.success(f"✅ Model trained successfully with {method}.")
st.write(f"**Features used:** {feats}")
st.write(f"Training samples: {len(X_train):,}, Calibration samples: {len(X_calib):,}")
