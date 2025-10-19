# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42
INITIAL_THRESHOLD = 5.0       # scrap% threshold for classification
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
DEFAULT_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 2
GLOBAL_TARGET_SCRAP = 0.065   # 6.5% target global mean

st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

# -----------------------------
# Utility Functions
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed)
    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"])
    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round((df["scrap%"] / 100.0) * df["order_quantity"]).astype(int)
    df = df.sort_values("week_ending").reset_index(drop=True)
    return df

def time_split(df, train_frac=TRAIN_FRAC, calib_frac=CALIB_FRAC):
    n = len(df)
    train_end = int(train_frac * n)
    calib_end = int((train_frac + calib_frac) * n)
    df_train = df.iloc[:train_end].copy()
    df_calib = df.iloc[train_end:calib_end].copy()
    df_test = df.iloc[calib_end:].copy()
    train_parts = set(df_train.part_id.unique())
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)]
    calib_parts = set(df_calib.part_id.unique())
    df_test = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))]
    return df_train, df_calib, df_test

def compute_mtbf(df_train, thr=INITIAL_THRESHOLD):
    temp = df_train.copy()
    temp["scrap_flag"] = temp["scrap%"] > thr
    mtbf = temp.groupby("part_id").agg(total_runs=("scrap%", "count"),
                                       failures=("scrap_flag", "sum"))
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]

def attach_features(df_sub, mtbf_train, part_freq, def_mtbf, def_freq):
    sub = df_sub.merge(mtbf_train, on="part_id", how="left")
    sub["mttf_scrap"] = sub["mttf_scrap"].fillna(def_mtbf)
    sub = sub.merge(part_freq.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    sub["part_freq"] = sub["part_freq"].fillna(def_freq)
    return sub

def make_xy(df):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    X = df[feats].copy()
    y = (df["scrap%"] > INITIAL_THRESHOLD).astype(int)
    return X, y, feats

@st.cache_resource(show_spinner=True)
def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators=150):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)
    has_both = (y_calib.sum() > 0) and (y_calib.sum() < len(y_calib))
    method = "isotonic" if has_both and len(y_calib) > 500 else "sigmoid"
    try:
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv="prefit").fit(X_calib, y_calib)
    except Exception:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        method = "sigmoid"
    return rf, cal, method

def compute_rfm(df_train):
    frac = (df_train["scrap%"] > INITIAL_THRESHOLD).mean()
    return max(0.05, min(frac, 0.25))

def compute_iconf(y_true, y_pred):
    if len(y_true) == 0:
        return 0.85
    brier = brier_score_loss(y_true, y_pred)
    conf = 1 - min(brier / 0.25, 1.0)
    return max(0.6, min(conf, 0.95))

# -----------------------------
# New per-part scaling
# -----------------------------
def compute_part_baselines(df_train):
    part_baseline = df_train.groupby("part_id")["scrap%"].mean() / 100.0
    part_baseline = part_baseline.clip(upper=0.25)
    global_mean = part_baseline.mean()
    part_scale = (part_baseline / global_mean).fillna(1.0)
    return part_baseline, part_scale, global_mean

def apply_dynamic_correction(part_id, predicted_proba, rfm, iconf, part_scale, part_baseline):
    part_adj_factor = part_scale.get(part_id, 1.0)
    corrected = predicted_proba * rfm * iconf * part_adj_factor
    corrected = np.clip(corrected, 0, 1)
    return corrected

# -----------------------------
# UI
# -----------------------------
st.title("🧪 Foundry Scrap Risk Dashboard")
st.caption("Per-part calibration • Dynamic RFM/IConf • Historical scrap context")

st.sidebar.header("Data & Model Settings")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")
n_estimators = st.sidebar.slider("RandomForest Trees", 80, 600, DEFAULT_ESTIMATORS, 20)
use_manual = st.sidebar.checkbox("Use manual RFM/IConf", value=False)
rfm_manual = st.sidebar.slider("Manual RFM", 0.05, 0.30, 0.15, 0.01)
iconf_manual = st.sidebar.slider("Manual IConf", 0.60, 0.95, 0.85, 0.01)

if not os.path.exists(csv_path):
    st.error("CSV not found.")
    st.stop()

df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df)

mtbf_train = compute_mtbf(df_train)
default_mtbf = mtbf_train["mttf_scrap"].median()
part_freq_train = df_train["part_id"].value_counts(normalize=True)
default_freq = part_freq_train.median() if len(part_freq_train) else 0.0

df_train_f = attach_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib_f = attach_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test_f = attach_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

X_train, y_train, FEATURES = make_xy(df_train_f)
X_calib, y_calib, _ = make_xy(df_calib_f)
X_test, y_test, _ = make_xy(df_test_f)

rf_model, calibrated_model, calib_method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators)
p_test = calibrated_model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
rfm_est = compute_rfm(df_train)
iconf_est = compute_iconf(y_test, p_test)
if use_manual:
    rfm_est, iconf_est = rfm_manual, iconf_manual

part_baseline, part_scale, global_mean = compute_part_baselines(df_train)

# -----------------------------
# Interactive Prediction
# -----------------------------
st.subheader("Scrap Risk Estimation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    part_ids = sorted(df["part_id"].unique())
    selected_part = st.selectbox("Select Part ID", part_ids)
with col2:
    quantity = st.number_input("Order Quantity", 1, 10000, 351)
with col3:
    weight = st.number_input("Piece Weight (lbs)", 0.1, 100.0, 4.0)
with col4:
    cost_per_part = st.number_input("Cost per Part ($)", 0.01, 10.0, 0.01)

mttf_value = mtbf_train.loc[selected_part, "mttf_scrap"] if selected_part in mtbf_train.index else default_mtbf
part_freq_value = float(part_freq_train.get(selected_part, default_freq))
input_row = pd.DataFrame([[quantity, weight, mttf_value, part_freq_value]], columns=FEATURES)

if st.button("Predict", use_container_width=True, type="primary"):
    base_p = calibrated_model.predict_proba(input_row)[0, 1]
    corrected_p = apply_dynamic_correction(selected_part, base_p, rfm_est, iconf_est, part_scale, part_baseline)
    expected_scrap_count = int(round(corrected_p * quantity))
    expected_loss = round(expected_scrap_count * cost_per_part, 2)

    hist_avg = part_baseline.get(selected_part, np.nan)
    n_obs = int(df_train[df_train["part_id"] == selected_part].shape[0])

    st.metric("Predicted Scrap Risk (raw)", f"{base_p*100:.2f}%")
    st.metric("Adjusted Scrap Risk (calibrated)", f"{corrected_p*100:.2f}%")
    st.metric("Expected Scrap Count", f"{expected_scrap_count} parts")
    st.metric("Expected Loss", f"${expected_loss:.2f}")

    st.markdown(f"**Historical Scrap Avg:** {hist_avg*100:.2f}%  ({n_obs} runs)")
    if corrected_p > hist_avg:
        st.warning("⬆️ Prediction above historical average")
    elif corrected_p < hist_avg:
        st.success("⬇️ Prediction below historical average")
    else:
        st.info("≈ Equal to historical average")

# -----------------------------
# Diagnostics
# -----------------------------
st.subheader("Model Diagnostics")
try:
    test_brier = brier_score_loss(y_test, p_test) if len(p_test) else np.nan
except Exception:
    test_brier = np.nan
st.write(f"Calibration: **{calib_method}**, Brier: {test_brier:.4f}")
st.caption("Per-part RFM/IConf calibration aligns predictions with each part’s historical scrap trend.")
