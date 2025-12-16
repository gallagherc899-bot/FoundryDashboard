# Foundry Scrap Risk Dashboard — Complete Version (Defect-Type Pareto Fixed)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------
# Streamlit Setup
# -------------------------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
DEFAULT_THRESHOLD = 6.5

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize columns
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.replace("_$", "", regex=True)
    )

    rename_map = {
        "work_order_num": "work_order",
        "order_quantity": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "scrap_": "scrap%",
        "week_ending": "week_ending",
        "piece_weight_lbs": "piece_weight_lbs",
        "part_id": "part_id",
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing critical columns: {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending"])
    for col in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"]).copy()

    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(df["order_quantity"] * df["scrap%"].clip(lower=0) / 100).astype(int)

    df.sort_values("week_ending", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def time_split(df, train_ratio=0.75, calib_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    df_train = df.iloc[:train_end].copy()
    df_calib = df.iloc[train_end:calib_end].copy()
    df_test = df.iloc[calib_end:].copy()
    return df_train, df_calib, df_test


def compute_mtbf_on_train(df_train, thr_label):
    df_mtbf = df_train.groupby("part_id")["scrap%"].mean().reset_index()
    df_mtbf.rename(columns={"scrap%": "mttf_scrap"}, inplace=True)
    df_mtbf["mttf_scrap"] = np.where(df_mtbf["mttf_scrap"] <= thr_label, 1.0, df_mtbf["mttf_scrap"])
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
    method = "sigmoid"
    cal = CalibratedClassifierCV(estimator=rf, method=method, cv=3).fit(X_calib, y_calib)
    return rf, cal, f"calibrated ({method}, cv=3)"

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % Threshold (Label & MTTF)", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
prior_shift = st.sidebar.checkbox("Enable prior shift", True)
guard = st.sidebar.slider("Prior-shift guard (pp tolerance)", 0, 50, 20, 5)
manual_qh = st.sidebar.checkbox("Manual Quick-Hook override", False)
manual_s = st.sidebar.number_input("Manual s", value=1.0, step=0.05, disabled=not manual_qh)
manual_g = st.sidebar.number_input("Manual γ", value=0.5, step=0.05, disabled=not manual_qh)
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation", True)
n_est = st.sidebar.slider("Number of Trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# -------------------------------------------------
# Data & Model Prep
# -------------------------------------------------
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

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

# -------------------------------------------------
# Prediction Tab
# -------------------------------------------------
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

    mttf_val = default_mtbf
    part_freq_val = default_freq
    input_df = pd.DataFrame(
        [[part_id, order_qty, piece_weight, mttf_val, part_freq_val]],
        columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
    )

    if st.button("Predict"):
        try:
            input_features = input_df.drop(columns=["part_id"]).copy()
            for col in feats:
                if col not in input_features.columns:
                    input_features[col] = 0.0
            input_features = input_features[feats]

            prob = cal_model.predict_proba(input_features)[0, 1]
            s = manual_s if manual_qh else 1.0
            g = manual_g if manual_qh else 0.5
            adjusted_prob = min(max(prob * s * (1 + g / 10), 0), 1)
            exp_scrap = order_qty * adjusted_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1 - adjusted_prob

            st.markdown(f"### 🧩 Prediction Results for Part **{part_id}**")
            st.metric("Predicted Scrap Risk (raw)", f"{prob*100:.2f}%")
            st.metric("Adjusted Scrap Risk", f"{adjusted_prob*100:.2f}%")
            st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
            st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
            st.metric("MTTF Scrap", f"{mttf_val:.1f}")
            st.metric("Reliability", f"{reliability*100:.2f}%")

            # 📊 Historical Pareto by Defect Type
            st.markdown("#### 📊 Historical Pareto (Top 10 Defect Types by Actual Defects)")
            defect_cols = [c for c in df_train.columns if c.endswith("_rate")]
            if len(defect_cols) == 0:
                st.warning("⚠ No defect-type rate columns found in dataset.")
            else:
                defect_summary = {
                    col.replace("_rate", "").replace("_", " ").title():
                    (df_train["order_quantity"] * df_train[col]).sum()
                    for col in defect_cols
                }
                hist = (
                    pd.DataFrame(list(defect_summary.items()), columns=["Defect Type", "historical_defects"])
                    .sort_values("historical_defects", ascending=False)
                    .head(10)
                )
                hist["share_%"] = hist["historical_defects"] / hist["historical_defects"].sum() * 100
                hist["cumulative_%"] = hist["share_%"].cumsum()
                st.dataframe(hist)

            # 🔮 Predicted Pareto by Expected Defects
            st.markdown("#### 🔮 Predicted Pareto (Top 10 Defect Types by Expected Defects)")
            if len(defect_cols) > 0:
                df_test["pred_prob"] = cal_model.predict_proba(make_xy(df_test, thr_label, use_rate_cols)[0])[:, 1]
                predicted_summary = {
                    col.replace("_rate", "").replace("_", " ").title():
                    (df_test["order_quantity"] * df_test["pred_prob"] * df_test[col]).sum()
                    for col in defect_cols
                }
                pareto = (
                    pd.DataFrame(list(predicted_summary.items()), columns=["Defect Type", "expected_defects"])
                    .sort_values("expected_defects", ascending=False)
                    .head(10)
                )
                pareto["share_%"] = pareto["expected_defects"] / pareto["expected_defects"].sum() * 100
                pareto["cumulative_%"] = pareto["share_%"].cumsum()
                st.dataframe(pareto)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# Validation Tab
# -------------------------------------------------
with tab2:
    st.subheader("📏 Rolling 6–2–1 Validation")
    try:
        X_test, y_test, _ = make_xy(df_test, thr_label, use_rate_cols)
        preds = cal_model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5))
        st.metric("Brier Score", f"{brier:.4f}")
        st.metric("Accuracy", f"{acc:.3f}")
        try:
            w, p = wilcoxon(y_test, preds)
            st.metric("Wilcoxon p-value", f"{p:.4f}")
        except Exception:
            st.warning("Wilcoxon test not applicable on this dataset.")
        cm = confusion_matrix(y_test, (preds > 0.5))
        st.write(pd.DataFrame(cm, index=["Actual OK", "Actual Scrap"], columns=["Pred OK", "Pred Scrap"]))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
