# Foundry Scrap Risk Dashboard — Complete Updated Script
# Drop-in replacement for your current foundry_dashboard.py

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

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2


# -------------------------------
# Data loading and cleaning
# -------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    # Try reading CSV robustly
    df = pd.read_csv(csv_path, header=0)

    # --- Detect multi-index header (extra blank rows above header) ---
    if isinstance(df.columns, pd.MultiIndex):
        st.warning("⚠ Multi-index header detected. Flattening column names.")
        df.columns = [
            "_".join([str(x) for x in col if pd.notna(x)]).strip()
            for col in df.columns.values
        ]

    # --- Clean up and normalize headers ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[\u00A0\u200B\t]+", "", regex=True)  # remove non-breaking / invisible chars
        .str.replace(r"\s+", " ", regex=True)  # collapse spaces
        .str.lower()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )

    # Drop duplicate columns if any remain
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique()
        st.warning(f"⚠ Found duplicate columns: {', '.join(dupes)}. Keeping first occurrence.")
        df = df.loc[:, ~df.columns.duplicated()]

    # Rename key columns to match model expectations
    rename_map = {
        "work_order_#": "work_order",
        "order_quantity": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "scrap_": "scrap%",
        "week_ending": "week_ending",
        "piece_weight_lbs": "piece_weight_lbs",
        "part_id": "part_id",
    }
    df.rename(columns=rename_map, inplace=True)

    # Verify critical columns exist
    required = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing critical columns: {missing}")

    # Convert types safely
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending"])
    for col in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"]).copy()

    # Derive pieces_scrapped if missing
    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round(df["order_quantity"] * df["scrap%"].clip(lower=0) / 100).astype(int)

    df.sort_values("week_ending", inplace=True)
    df.reset_index(drop=True, inplace=True)

    st.info(f"✅ Loaded {len(df):,} records and {len(df.columns)} columns successfully.")
    return df



# -------------------------------
# Helper functions
# -------------------------------
def time_split(df, train_ratio=0.75, calib_ratio=0.1):
    # Temporal split assuming df is already time-ordered by week_ending ascending
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:calib_end], df_sorted.iloc[calib_end:]


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    # Part-level mean scrap% as a proxy for MTTF-like signal; threshold floors to 1.0
    df_train = df_train.copy()
    df_train["part_id"] = df_train["part_id"].astype(str)
    # Defensive: ensure scrap% is numeric
    df_train["scrap%"] = pd.to_numeric(df_train["scrap%"], errors="coerce")

    grp = (
        df_train.groupby("part_id", dropna=False)["scrap%"]
        .mean(numeric_only=True)
        .reset_index()
    )
    grp.rename(columns={"scrap%": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp


def attach_train_features(df_sub: pd.DataFrame,
                          mtbf_train: pd.DataFrame,
                          part_freq_train: pd.Series,
                          default_mtbf: float,
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
    # Ensure all features exist
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    X = df[feats].copy()
    y = (pd.to_numeric(df["scrap%"], errors="coerce") > thr_label).astype(int)
    return X, y, feats


def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    # If calibration set is degenerate, skip calibration gracefully
    pos = int(y_calib.sum())
    if pos == 0 or pos == len(y_calib):
        return rf, rf, "uncalibrated"

    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
    return rf, cal, "calibrated (sigmoid, cv=3)"


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("📂 Data source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model controls")
thr_label = st.sidebar.slider("Scrap % threshold (Label & MTTF)", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation", True)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()


# -------------------------------
# Data preparation
# -------------------------------
df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df)

# Compute part-level signals from train
mtbf_train = compute_mtbf_on_train(df_train, thr_label)
part_freq_train = df_train["part_id"].value_counts(normalize=True)

# Defaults for unseen parts
default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

# Attach features
df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

# Train and calibrate
X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)


# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])


# -------------------------------
# Predict tab
# -------------------------------
with tab1:
    st.subheader("🔮 Predict scrap risk and reliability")

    c0, c1, c2, c3 = st.columns(4)
    part_id_input = c0.text_input("Part ID", value="Unknown")
    order_qty = c1.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = c2.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = c3.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

    input_df = pd.DataFrame(
        [[part_id_input, order_qty, piece_weight, default_mtbf, default_freq]],
        columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
    )

    if st.button("Predict"):
        try:
            X_input = input_df.drop(columns=["part_id"])
            # Ensure feature parity with training
            for col in feats:
                if col not in X_input.columns:
                    X_input[col] = 0.0
            X_input = X_input[feats]

            prob = float(cal_model.predict_proba(X_input)[0, 1])
            adj_prob = np.clip(prob, 0.0, 1.0)
            exp_scrap = order_qty * adj_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1.0 - adj_prob

            st.markdown(f"### 🧩 Prediction results for part **{part_id_input}**")
            st.metric("Predicted Scrap Risk (raw)", f"{prob*100:.2f}%")
            st.metric("Adjusted Scrap Risk", f"{adj_prob*100:.2f}%")
            st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
            st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
            st.metric("MTTF Scrap (default)", f"{default_mtbf:.1f}")
            st.metric("Reliability", f"{reliability*100:.2f}%")

            # Pareto — Historical
            defect_cols = [c for c in df.columns if c.endswith("_rate")]
            if len(defect_cols) == 0:
                st.warning("⚠ No defect-type rate columns found.")
            else:
                st.markdown("#### 📊 Historical Pareto (Top 10 defect types by actual defects)")
                hist = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Historical Defects": [(df_train["order_quantity"] * df_train[c]).sum() for c in defect_cols]
                    })
                    .sort_values("Historical Defects", ascending=False)
                    .head(10)
                )
                total_hist = hist["Historical Defects"].sum()
                hist["Share (%)"] = np.where(total_hist > 0, hist["Historical Defects"] / total_hist * 100, 0.0)
                st.dataframe(hist)
                st.bar_chart(hist.set_index("Defect Type")["Historical Defects"])

                # Pareto — Predicted
                st.markdown("#### 🔮 Predicted Pareto (Top 10 defect types by expected defects)")
                df_test_local = df_test.copy()
                df_test_local["pred_prob"] = cal_model.predict_proba(make_xy(df_test_local, thr_label, use_rate_cols)[0])[:, 1]
                pred = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Expected Defects": [(df_test_local["order_quantity"] * df_test_local[c] * df_test_local["pred_prob"]).sum()
                                             for c in defect_cols]
                    })
                    .sort_values("Expected Defects", ascending=False)
                    .head(10)
                )
                total_pred = pred["Expected Defects"].sum()
                pred["Share (%)"] = np.where(total_pred > 0, pred["Expected Defects"] / total_pred * 100, 0.0)
                st.dataframe(pred)
                st.bar_chart(pred.set_index("Defect Type")["Expected Defects"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -------------------------------
# Validation tab
# -------------------------------
with tab2:
    st.subheader("📏 Rolling 6–2–1 validation")
    try:
        X_test, y_test, _ = make_xy(df_test, thr_label, use_rate_cols)
        preds = cal_model.predict_proba(X_test)[:, 1]
        st.metric("Brier Score", f"{brier_score_loss(y_test, preds):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, (preds > 0.5)):.3f}")
        st.write(pd.DataFrame(
            confusion_matrix(y_test, (preds > 0.5)),
            index=["Actual OK", "Actual Scrap"],
            columns=["Pred OK", "Pred Scrap"]
        ))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
