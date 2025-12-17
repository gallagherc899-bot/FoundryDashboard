import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
DEFAULT_THRESHOLD = 6.5


# ------------------------------------------------------
# Data loading and cleaning
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0)

    # Flatten multi-index headers (if Excel exported with extra rows)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if pd.notna(x)]).strip()
            for col in df.columns.values
        ]

    # Normalize headers deeply (handles hidden chars & extra spaces)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[\u00A0\u200B\t]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.replace(r"[^\w\s%]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )

    # Drop duplicates
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique()
        st.warning(f"⚠ Found duplicate columns: {', '.join(dupes)}. Keeping first occurrence.")
        df = df.loc[:, ~df.columns.duplicated()]

    # Smart renaming to match foundry dataset
    rename_map = {}
    for col in df.columns:
        c = col.replace(" ", "_")
        if re.search(r"part.?id|work.?order", c):
            rename_map[col] = "part_id"
        elif re.search(r"order.?quantity|order.?qty|pieces?$|qty", c):
            rename_map[col] = "order_quantity"
        elif re.search(r"piece.?weight", c):
            rename_map[col] = "piece_weight_lbs"
        elif re.search(r"scrap.?%|scrap.?percent|scrap.?rate|scrap$", c):
            rename_map[col] = "scrap%"
        elif re.search(r"week.?ending|week.?end.?date", c):
            rename_map[col] = "week_ending"

    df.rename(columns=rename_map, inplace=True)

    # Verify essentials
    required = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing critical columns: {missing}")
        st.write("🔍 Columns after cleaning:", list(df.columns))
        st.stop()

    # Type conversions
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending"])

    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"]).copy()

    # Fill pieces_scrapped if missing
    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(df["order_quantity"] * df["scrap%"].clip(lower=0) / 100).astype(int)

    # Normalize defect rate columns
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in defect_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    st.info(
        f"✅ Loaded {len(df):,} records with {len(df.columns)} columns. "
        f"Detected {len(defect_cols)} defect-type rate columns."
    )

    df.sort_values("week_ending", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------
def time_split(df, train_ratio=0.75, calib_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df.iloc[:train_end], df.iloc[train_end:calib_end], df.iloc[calib_end:]


def compute_mtbf_on_train(df_train, thr_label):
    df_mtbf = df_train.groupby("part_id", dropna=False)["scrap%"].mean().reset_index()
    df_mtbf["mttf_scrap"] = (1 / (df_mtbf["scrap%"].clip(lower=1e-5) / 100)).round(2)
    return df_mtbf


def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s


def make_xy(df, thr_label, use_rate_cols):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        rate_feats = [c for c in df.columns if c.endswith("_rate")]
        feats += rate_feats

    missing = [f for f in feats if f not in df.columns]
    if missing:
        st.warning(f"⚠ Missing features auto-filled: {missing}")
        for f in missing:
            df[f] = 0.0

    X = df[feats].copy()
    y = (df["scrap%"] > thr_label).astype(int)
    return X, y, feats


@st.cache_resource(show_spinner=True)
def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    if y_calib.sum() == 0 or y_calib.sum() == len(y_calib):
        return rf, rf, "uncalibrated"

    method = "isotonic" if len(y_calib) > 500 else "sigmoid"
    try:
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv=3).fit(X_calib, y_calib)
        return rf, cal, f"calibrated ({method})"
    except Exception:
        return rf, rf, "uncalibrated"


# ------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % Threshold", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate defect features", True)
n_est = st.sidebar.slider("Number of Trees", 50, 300, DEFAULT_ESTIMATORS, 10)


# ------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------
if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df)
mtbf_train = compute_mtbf_on_train(df_train, thr_label)
part_freq_train = df_train["part_id"].value_counts(normalize=True)
default_mtbf = float(mtbf_train["mttf_scrap"].median())
default_freq = float(part_freq_train.median())
df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)
X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)


# ------------------------------------------------------
# Dashboard Tabs
# ------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

# -------------------------------
# Prediction Tab
# -------------------------------
with tab1:
    st.subheader("🔮 Predict Scrap Risk and Reliability")

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        part_id = st.text_input("Part ID", value="Unknown")
    with col1:
        order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    with col2:
        piece_weight = st.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0, step=0.1)
    with col3:
        cost_per_part = st.number_input("Cost per Part ($)", min_value=0.1, value=10.0, step=0.1)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame(
                [[part_id, order_qty, piece_weight, default_mtbf, default_freq]],
                columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
            )
            input_X = input_df.drop(columns=["part_id"])
            prob = cal_model.predict_proba(input_X)[0, 1]
            adjusted_prob = min(max(prob, 0), 1)
            exp_scrap = order_qty * adjusted_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1 - adjusted_prob

            st.markdown(f"### 🧩 Prediction Results for Part **{part_id}**")
            st.metric("Predicted Scrap Risk", f"{adjusted_prob*100:.2f}%")
            st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
            st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
            st.metric("Reliability", f"{reliability*100:.2f}%")

            # 📊 Historical Pareto (Top 10 Defect Types)
            st.markdown("#### 📊 Historical Pareto (Top 10 Defect Types by Actual Defects)")
            defect_cols = [c for c in df_train.columns if c.endswith("_rate")]
            if len(defect_cols) > 0:
                defect_summary = {}
                for c in defect_cols:
                    defect_summary[c.replace("_rate", "").replace("_", " ").title()] = (
                        df_train["order_quantity"] * df_train[c]
                    ).sum()
                hist = (
                    pd.DataFrame(list(defect_summary.items()), columns=["Defect Type", "Historical Defects"])
                    .sort_values("Historical Defects", ascending=False)
                    .head(10)
                )
                st.dataframe(hist)
            else:
                st.warning("⚠ No defect columns detected.")

            # 🔮 Predicted Pareto
            st.markdown("#### 🔮 Predicted Pareto (Top 10 Defect Types by Expected Defects)")
            defect_pred = {}
            for c in defect_cols:
                defect_pred[c.replace("_rate", "").replace("_", " ").title()] = (
                    df_test["order_quantity"] * df_test[c]
                ).sum()
            pareto = (
                pd.DataFrame(list(defect_pred.items()), columns=["Defect Type", "Predicted Defects"])
                .sort_values("Predicted Defects", ascending=False)
                .head(10)
            )
            st.dataframe(pareto)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Validation Tab
# -------------------------------
with tab2:
    st.subheader("📏 Rolling 6–2–1 Validation")
    try:
        X_test, y_test, _ = make_xy(df_test, thr_label, use_rate_cols)
        preds = cal_model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5))
        st.metric("Brier Score", f"{brier:.4f}")
        st.metric("Accuracy", f"{acc:.3f}")

        cm = confusion_matrix(y_test, (preds > 0.5))
        st.write(pd.DataFrame(cm,
                              index=["Actual OK", "Actual Scrap"],
                              columns=["Pred OK", "Pred Scrap"]))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
