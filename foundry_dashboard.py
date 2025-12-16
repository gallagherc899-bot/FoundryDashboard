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
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
USE_RATE_COLS_PERMANENT = True
DEFAULT_THRESHOLD = 6.5

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
        raise ValueError(f"Missing columns: {missing}")
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

def time_split(df, train_frac=0.6, calib_frac=0.2):
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

def compute_mtbf_on_train(df_train, thr_label):
    t = df_train.copy()
    t["scrap_flag"] = (t["scrap%"] > thr_label).astype(int)
    mtbf = t.groupby("part_id").agg(total_runs=("scrap%", "count"), failures=("scrap_flag", "sum"))
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]

def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"].fillna(default_mtbf, inplace=True)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"].fillna(default_freq, inplace=True)
    return s

def make_xy(df, thr_label, use_rate_cols):
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
        cv_value = 3
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv=cv_value).fit(X_calib, y_calib)
        return rf, cal, f"calibrated ({method}, cv={cv_value})"
    except Exception:
        return rf, rf, "uncalibrated"

st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
reset_defaults = st.sidebar.checkbox("Reset to Recommended Defaults", value=False)
if reset_defaults:
    thr_label = DEFAULT_THRESHOLD
    use_rate_cols = True
    prior_shift = True
    guard = 20
    manual_qh = False
    manual_s = 1.0
    manual_g = 0.5
    run_validation = True
else:
    thr_label = st.sidebar.slider("Scrap % Threshold (Label & MTTF)", 1.0, 15.0, 6.5, 0.5)
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
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

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
            prob = cal_model.predict_proba(input_df.drop(columns=["part_id"]))[0, 1]
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

            st.markdown("#### 📊 Historical Pareto (Top 10 Parts)")
            hist = (
                df_train.groupby("part_id")["scrap%"].mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={"scrap%": "hist_mean_rate"})
            )
            hist["share_%"] = hist["hist_mean_rate"] / hist["hist_mean_rate"].sum() * 100
            hist["cumulative_%"] = hist["share_%"].cumsum()
            st.dataframe(hist)

            st.markdown("#### 🔮 Predicted Pareto (Top 10 Parts)")
            Xt, yt, _ = make_xy(df_test, thr_label, use_rate_cols)
            df_test["pred_prob"] = cal_model.predict_proba(Xt)[:, 1]
            pareto = (
                df_test.groupby("part_id")["pred_prob"].mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            pareto["Δprob (pp)"] = pareto["pred_prob"].diff().fillna(0)
            pareto["share_%"] = pareto["pred_prob"] / pareto["pred_prob"].sum() * 100
            st.dataframe(pareto)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


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
        st.write(pd.DataFrame(cm,
                              index=["Actual OK", "Actual Scrap"],
                              columns=["Pred OK", "Pred Scrap"]))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
