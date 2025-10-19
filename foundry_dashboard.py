# streamlit_app.py
# Run: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample

# -----------------------------
# Config & Constants
# -----------------------------
RANDOM_STATE = 42
INITIAL_THRESHOLD = 5.0   # label threshold for scrap% to define positive class
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20         # test will be the remainder (0.20)
DEFAULT_ESTIMATORS = 150  # lower default for faster startup
MIN_SAMPLES_LEAF = 2

st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize columns
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
    )
    # expected columns (case-normalized)
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed).copy()

    # ensure numeric types
    for col in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"])

    # (Optional) If pieces_scrapped not present, estimate it (best-effort) for the table
    if "pieces_scrapped" not in df.columns:
        est = np.round((df["scrap%"].clip(lower=0) / 100.0) * df["order_quantity"]).astype(int)
        df["pieces_scrapped"] = est

    # sort by time for time-aware splitting
    df = df.sort_values("week_ending").reset_index(drop=True)
    return df

def time_based_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, calib_frac=CALIB_FRAC):
    """Chronological split: train -> calib -> test. Also avoid part_id overlap across splits."""
    n = len(df)
    train_end = int(train_frac * n)
    calib_end = int((train_frac + calib_frac) * n)

    df_train = df.iloc[:train_end].copy()
    df_calib = df.iloc[train_end:calib_end].copy()
    df_test  = df.iloc[calib_end:].copy()

    # ensure group separation (drop any parts that appear in earlier splits)
    train_parts = set(df_train.part_id.unique())
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)].copy()
    calib_parts = set(df_calib.part_id.unique())
    df_test  = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))].copy()

    return df_train, df_calib, df_test

def compute_mtbf_on_train(df_train: pd.DataFrame, threshold=INITIAL_THRESHOLD) -> pd.DataFrame:
    temp = df_train.copy()
    temp["scrap_flag"] = temp["scrap%"] > threshold
    mtbf = temp.groupby("part_id").agg(
        total_runs=("scrap%", "count"),
        failures=("scrap_flag", "sum"),
    )
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]

def attach_train_features(df_sub: pd.DataFrame, mtbf_train: pd.DataFrame, part_freq: pd.Series, default_mtbf: float, default_freq: float):
    sub = df_sub.merge(mtbf_train, on="part_id", how="left")
    sub["mttf_scrap"] = sub["mttf_scrap"].fillna(default_mtbf)

    # frequency encoding of part_id learned on TRAIN ONLY
    sub = sub.merge(part_freq.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    sub["part_freq"] = sub["part_freq"].fillna(default_freq)

    return sub

def make_features(df: pd.DataFrame):
    # Features we keep (no raw label-encoded part_id!)
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    X = df[feats].copy()
    y = (df["scrap%"] > INITIAL_THRESHOLD).astype(int)
    return X, y, feats

def _train_and_calibrate_inner(
    X_train, y_train, X_calib, y_calib,
    n_estimators=DEFAULT_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Prefer isotonic if calib set is reasonably large and has both classes
    has_both = (y_calib.sum() > 0) and (y_calib.sum() < len(y_calib))
    method = "isotonic" if len(y_calib) >= 1000 and has_both else "sigmoid"
    try:
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv="prefit")
        cal.fit(X_calib, y_calib)
    except Exception:
        # Fallback to sigmoid if isotonic failed for any reason
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit")
        cal.fit(X_calib, y_calib)
        method = "sigmoid"

    return rf, cal, method

@st.cache_resource(show_spinner=True)
def train_cached(
    X_train, y_train, X_calib, y_calib,
    n_estimators=DEFAULT_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF
):
    """Heavy training is cached so UI interactions don't retrigger it."""
    return _train_and_calibrate_inner(
        X_train, y_train, X_calib, y_calib,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf
    )

def cost_threshold(c_fp: float, c_fn: float) -> float:
    # Avoid divide-by-zero
    denom = (c_fp + c_fn)
    if denom <= 0:
        return 0.5
    return float(c_fp / denom)

def bootstrap_probability_interval(
    X_row: pd.DataFrame,
    df_train: pd.DataFrame,
    feats: list,
    y_train: pd.Series,
    mtbf_train: pd.DataFrame,
    base_part_freq: pd.Series,
    n_boot: int = 20,
    alpha: float = 0.10,
):
    """
    Lightweight bootstrap around the model to derive a lower/upper bound for predicted probability.
    To keep it snappy, we reduce trees in bootstrap models and use quick sigmoid calibration.
    """
    probs = []
    default_mtbf = mtbf_train["mttf_scrap"].median()
    default_freq = base_part_freq.median() if len(base_part_freq) else 0.0

    for b in range(n_boot):
        boot_df = resample(df_train, replace=True, n_samples=len(df_train), random_state=RANDOM_STATE + b)
        # Recompute mtbf + freq on bootstrap sample
        mtbf_b = compute_mtbf_on_train(boot_df, threshold=INITIAL_THRESHOLD)
        freq_b = boot_df["part_id"].value_counts(normalize=True)

        # Train smaller RF for speed
        X_b, y_b, _ = make_features(
            attach_train_features(
                boot_df, mtbf_b, freq_b, default_mtbf, default_freq
            )
        )
        rf_b = RandomForestClassifier(
            n_estimators=max(DEFAULT_ESTIMATORS // 2, 80),
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=RANDOM_STATE + b,
            n_jobs=-1
        ).fit(X_b, y_b)

        cal_b = CalibratedClassifierCV(estimator=rf_b, method="sigmoid", cv=3)
        cal_b.fit(X_b, y_b)

        p = cal_b.predict_proba(X_row[feats])[0, 1]
        probs.append(p)

    probs = np.array(probs)
    lo = np.quantile(probs, alpha/2)
    hi = np.quantile(probs, 1 - alpha/2)
    return probs.mean(), lo, hi

# -----------------------------
# App UI
# -----------------------------
st.title("🧪 Foundry Scrap Risk Dashboard")
st.caption("Fast startup • Cached training • Cost-based decisions • Optional uncertainty & SHAP")

# File selector & performance toggles
st.sidebar.header("Data & Performance")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")
n_estimators_ui = st.sidebar.slider("RandomForest trees", min_value=80, max_value=600, value=DEFAULT_ESTIMATORS, step=20)
enable_uncertainty = st.sidebar.chec_
