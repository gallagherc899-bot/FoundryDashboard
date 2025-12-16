# ---------------------------------------------------------------
# 🧪 Foundry Scrap Risk Dashboard — Final Fixed Version (Dec 2025)
# Compatible with: Python 3.13, scikit-learn 1.8.0, Streamlit 1.52
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
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
USE_RATE_COLS_PERMANENT = True

S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)

# ---------------------------------------------------------------
# Utility Functions
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
        raise ValueError(f"Missing required columns: {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed).copy()
    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(
            (df["scrap%"].clip(lower=0) / 100.0) * df["order_quantity"]
        ).astype(int)

    return df


def time_split(df: pd.DataFrame, train_frac=0.6, calib_frac=0.2):
    n = len(df)
    t_end = int(train_frac * n)
    c_end = int((train_frac + calib_frac) * n)

    df_train = df.iloc[:t_end].copy()
    df_calib = df.iloc[t_end:c_end].copy()
    df_test = df.iloc[c_end:].copy()

    train_parts = set(df_train.part_id.unique())
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)].copy()
    calib_parts = set(df_calib.part_id.unique())
    df_test = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))].copy()

    return df_train, df_calib, df_test


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    t = df_train.copy()
    t["scrap_flag"] = (t["scrap%"] > thr_label).astype(int)
    mtbf = t.groupby("part_id").agg(total_runs=("scrap%", "count"), failures=("scrap_flag", "sum"))
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"].fillna(mtbf["total_runs"], inplace=True)
    return mtbf[["mttf_scrap"]]


def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"].fillna(default_mtbf, inplace=True)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"].fillna(default_freq, inplace=True)
    return s


def make_xy(df, thr_label: float, use_rate_cols: bool):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Missing expected features: {missing}")
    X = df[feats].copy()
    y = (df["scrap%"] > thr_label).astype(int)
    return X, y, feats


@st.cache_resource(show_spinner=True)
def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators: int):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)

    if y_calib.sum() == 0 or y_calib.sum() == len(y_calib):
        st.warning("⚠️ Calibration skipped — only one class in calibration set.")
        return rf, rf, "uncalibrated"

    method = "isotonic" if len(y_calib) > 500 else "sigmoid"
    try:
        cv_value = 3
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv=cv_value).fit(X_calib, y_calib)
        return rf, cal, f"calibrated ({method}, cv={cv_value})"
    except Exception as e:
        st.error(f"Calibration failed ({type(e).__name__}: {e}). Using uncalibrated RF.")
        return rf, rf, "uncalibrated"

# ---------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Parameters")
thr_label = st.sidebar.slider("Scrap % Threshold", 1.0, 15.0, 6.5, 0.5)
n_est = st.sidebar.slider("Number of Trees (n_estimators)", 50, 300, 180, 10)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# ---------------------------------------------------------------
# Load and Prepare Data
# ---------------------------------------------------------------
df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df)

# Compute part-level features
mtbf_train = compute_mtbf_on_train(df_train, thr_label)
part_freq_train = df_train["part_id"].value_counts(normalize=True)
default_mtbf = float(mtbf_train["mttf_scrap"].median())
default_freq = float(part_freq_train.median())

df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

X_train, y_train, feats = make_xy(df_train, thr_label, USE_RATE_COLS_PERMANENT)
X_calib, y_calib, _ = make_xy(df_calib, thr_label, USE_RATE_COLS_PERMANENT)

rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)

# ---------------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

# ---------------------------------------------------------------
# 🔮 PREDICT TAB
# ---------------------------------------------------------------
with tab1:
    st.subheader("🔮 Predict Scrap Risk by Part")
    if len(df_test) == 0:
        st.warning("Not enough data for prediction window.")
    else:
        X_test, y_test, _ = make_xy(df_test, thr_label, USE_RATE_COLS_PERMANENT)
        preds = cal_model.predict_proba(X_test)[:, 1]
        df_test["predicted_scrap_prob"] = preds
        df_test["predicted_scrap_flag"] = (preds > 0.5).astype(int)
        st.metric("Average Predicted Scrap Probability", f"{df_test['predicted_scrap_prob'].mean():.3f}")

        st.dataframe(df_test[["part_id", "week_ending", "scrap%", "predicted_scrap_prob"]]
                     .sort_values("predicted_scrap_prob", ascending=False)
                     .head(20)
                     .style.background_gradient(cmap="Reds", subset=["predicted_scrap_prob"]))

        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=df_test.to_csv(index=False).encode("utf-8"),
            file_name="predicted_scrap_risks.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------
# 📏 VALIDATION TAB
# ---------------------------------------------------------------
with tab2:
    st.subheader("📏 Rolling 6–2–1 Validation")
    try:
        X_test, y_test, _ = make_xy(df_test, thr_label, USE_RATE_COLS_PERMANENT)
        preds = cal_model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5))
        st.metric("Brier Score", f"{brier:.4f}")
        st.metric("Accuracy", f"{acc:.3f}")

        cm = confusion_matrix(y_test, (preds > 0.5))
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(cm, index=["Actual OK", "Actual Scrap"], columns=["Pred OK", "Pred Scrap"]))

        st.text("Classification Report:")
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation could not be performed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
