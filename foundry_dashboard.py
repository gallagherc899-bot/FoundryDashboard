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

# ---------- Page config ----------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Config & Constants
# -----------------------------
RANDOM_STATE = 42
INITIAL_THRESHOLD = 5.0
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
DEFAULT_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 2
TARGET_HISTORICAL_SCRAP = 0.065   # 6.5% historical scrap target

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
    )
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed).copy()

    for col in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"])

    if "pieces_scrapped" not in df.columns:
        est = np.round((df["scrap%"].clip(lower=0) / 100.0) * df["order_quantity"]).astype(int)
        df["pieces_scrapped"] = est

    df = df.sort_values("week_ending").reset_index(drop=True)
    return df

def time_split(df, train_frac=TRAIN_FRAC, calib_frac=CALIB_FRAC):
    n = len(df)
    train_end = int(train_frac * n)
    calib_end = int((train_frac + calib_frac) * n)
    df_train = df.iloc[:train_end].copy()
    df_calib = df.iloc[train_end:calib_end].copy()
    df_test  = df.iloc[calib_end:].copy()
    train_parts = set(df_train.part_id.unique())
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)].copy()
    calib_parts = set(df_calib.part_id.unique())
    df_test  = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))].copy()
    return df_train, df_calib, df_test

def mtbf_train_only(df_train, thr=INITIAL_THRESHOLD):
    t = df_train.copy()
    t["scrap_flag"] = t["scrap%"] > thr
    mtbf = t.groupby("part_id").agg(total_runs=("scrap%", "count"),
                                    failures=("scrap_flag", "sum"))
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]

def attach_feats(df_sub, mtbf_tr, part_freq, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_tr, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s

def make_xy(df):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    X = df[feats].copy()
    y = (df["scrap%"] > INITIAL_THRESHOLD).astype(int)
    return X, y, feats

def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators, min_leaf):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)
    has_both = (y_calib.sum() > 0) and (y_calib.sum() < len(y_calib))
    method = "isotonic" if len(y_calib) >= 1000 and has_both else "sigmoid"
    try:
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv="prefit").fit(X_calib, y_calib)
    except Exception:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        method = "sigmoid"
    return rf, cal, method

@st.cache_resource(show_spinner=True)
def train_cached(X_train, y_train, X_calib, y_calib, n_estimators, min_leaf):
    return train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators, min_leaf)

def cost_threshold(c_fp, c_fn):
    d = c_fp + c_fn
    return 0.5 if d <= 0 else float(c_fp / d)

# -----------------------------
# New: Dynamic RFM + IConf estimation
# -----------------------------
def compute_rfm(df_train):
    """Estimate Failure Mode Ratio dynamically as fraction of detectable scrap modes."""
    scrap_flags = (df_train["scrap%"] > INITIAL_THRESHOLD).astype(int)
    rfm = scrap_flags.sum() / len(scrap_flags)
    return max(0.05, min(rfm, 0.25))  # bound within [0.05, 0.25]

def compute_iconf(y_true, y_pred):
    """Estimate IConf dynamically from model reliability (1 - normalized Brier)."""
    try:
        brier = brier_score_loss(y_true, y_pred)
        # Normalize vs max theoretical variance (0.25 for binary balanced)
        conf = 1 - min(brier / 0.25, 1.0)
        return max(0.6, min(conf, 0.95))
    except Exception:
        return 0.85

def apply_dynamic_correction(predicted_proba, rfm, iconf, historical_target=TARGET_HISTORICAL_SCRAP):
    """Apply RFM and IConf correction to bring average prediction near historical target."""
    corrected = predicted_proba * rfm * iconf
    # Normalize to align the global mean with target scrap rate
    scale_factor = historical_target / max(np.mean(corrected), 1e-6)
    corrected *= scale_factor
    return np.clip(corrected, 0, 1)

# -----------------------------
# App UI
# -----------------------------
st.title("🧪 Foundry Scrap Risk Dashboard")
st.caption("Dynamic PdM-style correction • Cached training • RFM/IConf scaling for realistic scrap risk")

# Sidebar
st.sidebar.header("Data & Model Settings")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")
n_estimators_ui = st.sidebar.slider("RandomForest trees", 80, 600, DEFAULT_ESTIMATORS, 20)
rfm_manual = st.sidebar.slider("Manual RFM override (optional)", 0.05, 0.30, 0.15, 0.01)
iconf_manual = st.sidebar.slider("Manual IConf override (optional)", 0.60, 0.95, 0.85, 0.01)
enable_manual = st.sidebar.checkbox("Use manual RFM/IConf overrides", value=False)

st.sidebar.header("Decision Cost Settings")
default_cfp_factor = st.sidebar.slider("False-alarm cost as % of unit cost", 0, 100, 25, 5)

# Load data
if not os.path.exists(csv_path):
    st.error(f"CSV not found: {csv_path}")
    st.stop()

df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df, TRAIN_FRAC, CALIB_FRAC)

mtbf_tr = mtbf_train_only(df_train, thr=INITIAL_THRESHOLD)
default_mtbf = mtbf_tr["mttf_scrap"].median()
part_freq_tr = df_train["part_id"].value_counts(normalize=True)
default_freq = part_freq_tr.median() if len(part_freq_tr) else 0.0

df_train_f = attach_feats(df_train, mtbf_tr, part_freq_tr, default_mtbf, default_freq)
df_calib_f = attach_feats(df_calib, mtbf_tr, part_freq_tr, default_mtbf, default_freq)
df_test_f  = attach_feats(df_test,  mtbf_tr, part_freq_tr, default_mtbf, default_freq)

X_train, y_train, FEATURES = make_xy(df_train_f)
X_calib, y_calib, _ = make_xy(df_calib_f)
X_test,  y_test,  _ = make_xy(df_test_f)

rf_model, calibrated_model, calib_method = train_cached(X_train, y_train, X_calib, y_calib, n_estimators_ui, MIN_SAMPLES_LEAF)

# Compute dynamic RFM/IConf
rfm_est = compute_rfm(df_train)
p_test = calibrated_model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
iconf_est = compute_iconf(y_test, p_test)

if enable_manual:
    rfm_est = rfm_manual
    iconf_est = iconf_manual

st.sidebar.markdown(f"**RFM (auto):** {rfm_est:.3f}  **IConf (auto):** {iconf_est:.3f}")

# -----------------------------
# Interactive Prediction
# -----------------------------
st.subheader("Scrap Risk Estimation")

c1, c2, c3, c4 = st.columns([2,2,2,2])
with c1:
    part_ids = sorted(df["part_id"].unique().tolist())
    selected_part = st.selectbox("Select Part ID", part_ids)
with c2:
    quantity = st.number_input("Order Quantity", min_value=1, step=1, value=351)
with c3:
    weight = st.number_input("Piece Weight (lbs)", min_value=0.01, step=0.01, value=4.00)
with c4:
    cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01, step=0.01, value=0.01)

C_FP = (default_cfp_factor / 100.0) * float(cost_per_part)
C_FN = float(cost_per_part) * float(quantity)
t_star = cost_threshold(C_FP, C_FN)

mttf_value = mtbf_tr.loc[selected_part, "mttf_scrap"] if selected_part in mtbf_tr.index else default_mtbf
part_freq_value = float(part_freq_tr.get(selected_part, default_freq))
input_row = pd.DataFrame([[quantity, weight, mttf_value, part_freq_value]], columns=FEATURES)

if st.button("Predict", type="primary", use_container_width=True):
    base_p = calibrated_model.predict_proba(input_row)[0, 1]
    corrected_p = apply_dynamic_correction(base_p, rfm_est, iconf_est, TARGET_HISTORICAL_SCRAP)
    action_flag = corrected_p >= t_star

    expected_scrap_count = int(round(corrected_p * quantity))
    expected_loss = round(expected_scrap_count * cost_per_part, 2)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Predicted Scrap Risk (raw)", f"{base_p*100:.2f}%")
    with k2: st.metric("Adjusted Scrap Risk (RFM·IConf)", f"{corrected_p*100:.2f}%")
    with k3: st.metric("Expected Scrap Count", f"{expected_scrap_count} parts")
    with k4: st.metric("Expected Loss", f"${expected_loss:.2f}")

    st.write(f"RFM = {rfm_est:.3f}, IConf = {iconf_est:.3f}, Target = {TARGET_HISTORICAL_SCRAP*100:.1f}%")
    st.success("Decision: ⚠️ ACT" if action_flag else "Decision: ✅ NO ACTION")

# -----------------------------
# Diagnostics
# -----------------------------
st.subheader("Model Diagnostics")
try:
    test_brier = brier_score_loss(y_test, p_test) if len(p_test) else np.nan
except Exception:
    test_brier = np.nan
st.write(f"Calibration: **{calib_method}**, Brier: {test_brier:.4f}")
st.caption("Adjusted probability scales predictions to realistic 6.5 % average scrap rate using dynamic RFM and IConf factors.")

