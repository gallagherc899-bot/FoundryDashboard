# Foundry Scrap Risk Dashboard — Stable Final Build (Auto Handles Defects & Part IDs)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import wilcoxon

# -------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
DEFAULT_THRESHOLD = 6.5

# -------------------------------------------------
# Data Loading & Cleaning
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize header names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )

    # Rename key columns
    rename_map = {
        "work_order": "part_id",
        "work_order_number": "part_id",
        "work_order_#": "part_id",
        "part_id": "part_id",
        "order_quantity": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "scrap": "scrap%",
        "scrap_percent": "scrap%",
        "scrap_percentage": "scrap%",
        "scrap_": "scrap%",
        "week_ending": "week_ending",
        "piece_weight_lbs": "piece_weight_lbs",
        "piece_weight": "piece_weight_lbs",
    }
    df.rename(columns=rename_map, inplace=True)

    # Fill missing key columns
    required = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    # Type conversions
    df["week_ending"] = pd.to_datetime(df.get("week_ending", pd.NaT), errors="coerce")
    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute pieces scrapped if missing
    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(df["order_quantity"] * df["scrap%"].clip(lower=0) / 100).astype(float)

    df = df.dropna(subset=["week_ending"])
    df.sort_values("week_ending", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Detect defect columns robustly
    rate_cols = [c for c in df.columns if "rate" in c and not c.startswith("scrap")]
    st.info(f"✅ Detected {len(rate_cols)} defect columns: {', '.join(rate_cols)}")

    return df

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def time_split(df, train_ratio=0.75, calib_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:calib_end].copy(), df.iloc[calib_end:].copy()

def compute_mtbf_on_train(df_train, thr_label):
    # 🩹 Defensive fix for part_id
    if "part_id" not in df_train.columns:
        raise ValueError("❌ 'part_id' column missing from dataset.")
    if isinstance(df_train["part_id"], pd.DataFrame):
        st.warning("⚠ 'part_id' had multiple columns. Flattening.")
        df_train["part_id"] = df_train["part_id"].iloc[:, 0]

    df_train["part_id"] = df_train["part_id"].astype(str).fillna("unknown")

    df_mtbf = (
        df_train.groupby("part_id", dropna=False)["scrap%"]
        .mean(numeric_only=True)
        .reset_index()
    )
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
        feats += [c for c in df.columns if "rate" in c and not c.startswith("scrap")]
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    X = df[feats].copy()
    y = (df["scrap%"] > thr_label).astype(int)
    return X, y, feats

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
    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
    return rf, cal, "calibrated (sigmoid, cv=3)"

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % Threshold (Label & MTTF)", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of Trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation", True)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# -------------------------------------------------
# Data Prep
# -------------------------------------------------
df = load_and_clean(csv_path)

# 🩹 Fallback part_id logic
if "part_id" not in df.columns:
    poss = [c for c in df.columns if "work" in c and "order" in c]
    if poss:
        df["part_id"] = df[poss[0]]
        st.info(f"✅ Using '{poss[0]}' as part_id column.")
    else:
        df["part_id"] = "unknown"

if isinstance(df["part_id"], pd.DataFrame):
    df["part_id"] = df["part_id"].iloc[:, 0]
df["part_id"] = df["part_id"].astype(str)

# Split + Train
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
    part_id = col0.text_input("Part ID", value="Unknown")
    order_qty = col1.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = col2.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0, step=0.1)
    cost_per_part = col3.number_input("Cost per Part ($)", min_value=0.1, value=10.0, step=0.1)

    input_df = pd.DataFrame(
        [[part_id, order_qty, piece_weight, default_mtbf, default_freq]],
        columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
    )

    if st.button("Predict"):
        try:
            X_input = input_df.drop(columns=["part_id"])
            for col in feats:
                if col not in X_input.columns:
                    X_input[col] = 0.0
            X_input = X_input[feats]
            prob = cal_model.predict_proba(X_input)[0, 1]
            adj_prob = np.clip(prob, 0, 1)
            exp_scrap = order_qty * adj_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1 - adj_prob

            st.markdown(f"### 🧩 Prediction Results for Part **{part_id}**")
            st.metric("Predicted Scrap Risk (raw)", f"{prob*100:.2f}%")
            st.metric("Adjusted Scrap Risk", f"{adj_prob*100:.2f}%")
            st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
            st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
            st.metric("MTTF Scrap", f"{default_mtbf:.1f}")
            st.metric("Reliability", f"{reliability*100:.2f}%")

            # Historical Pareto
            defect_cols = [c for c in df.columns if "rate" in c and not c.startswith("scrap")]
            if len(defect_cols) == 0:
                st.warning("⚠ No defect-type rate columns found.")
            else:
                st.markdown("#### 📊 Historical Pareto (Top 10 Defect Types)")
                hist = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Historical Defects": [(df_train["order_quantity"] * df_train[c]).sum() for c in defect_cols]
                    })
                    .sort_values("Historical Defects", ascending=False)
                    .head(10)
                )
                hist["Share (%)"] = hist["Historical Defects"] / hist["Historical Defects"].sum() * 100
                st.dataframe(hist)
                st.bar_chart(hist.set_index("Defect Type")["Historical Defects"])

                # Predicted Pareto
                st.markdown("#### 🔮 Predicted Pareto (Top 10 Expected Defects)")
                df_test["pred_prob"] = cal_model.predict_proba(make_xy(df_test, thr_label, use_rate_cols)[0])[:, 1]
                pred = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Expected Defects": [(df_test["order_quantity"] * df_test[c] * df_test["pred_prob"]).sum() for c in defect_cols]
                    })
                    .sort_values("Expected Defects", ascending=False)
                    .head(10)
                )
                pred["Share (%)"] = pred["Expected Defects"] / pred["Expected Defects"].sum() * 100
                st.dataframe(pred)
                st.bar_chart(pred.set_index("Defect Type")["Expected Defects"])

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
        st.metric("Brier Score", f"{brier_score_loss(y_test, preds):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, (preds > 0.5)):.3f}")
        st.write(pd.DataFrame(confusion_matrix(y_test, (preds > 0.5)),
                              index=["Actual OK", "Actual Scrap"],
                              columns=["Pred OK", "Pred Scrap"]))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
