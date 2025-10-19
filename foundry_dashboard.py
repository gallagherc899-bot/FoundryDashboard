# streamlit_app.py
# Run: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample

import shap

# -----------------------------
# Config & Constants
# -----------------------------
RANDOM_STATE = 42
INITIAL_THRESHOLD = 5.0   # label threshold for scrap% to define positive class
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20         # test will be the remainder (0.20)
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 2
USE_SMOTE = False         # we prefer class_weight="balanced" to keep priors intact

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
    # - part_id, week_ending, scrap%, order_quantity, piece_weight_lbs
    # - optional: pieces_scrapped
    if "week_ending" not in df.columns:
        raise ValueError("Column 'week_ending' is required in the CSV.")
    if "part_id" not in df.columns:
        raise ValueError("Column 'part_id' is required in the CSV.")
    if "scrap%" not in df.columns:
        raise ValueError("Column 'scrap%' is required in the CSV.")
    if "order_quantity" not in df.columns:
        raise ValueError("Column 'order_quantity' is required in the CSV.")
    if "piece_weight_lbs" not in df.columns:
        raise ValueError("Column 'piece_weight_lbs' is required in the CSV.")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    # drop rows missing key fields
    df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"]).copy()

    # ensure numeric types
    for col in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"])

    # (Optional) If pieces_scrapped not present, estimate it (best-effort) for the table
    if "pieces_scrapped" not in df.columns:
        # estimate from % when reasonable
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

def train_and_calibrate(X_train, y_train, X_calib, y_calib):
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
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
    To keep it snappy, we reduce trees in bootstrap models.
    """
    probs = []
    # Precompute defaults
    default_mtbf = mtbf_train["mttf_scrap"].median()
    default_freq = base_part_freq.median()

    for b in range(n_boot):
        boot_df = resample(df_train, replace=True, n_samples=len(df_train), random_state=RANDOM_STATE + b)
        # Recompute mtbf + freq on bootstrap sample
        mtbf_b = compute_mtbf_on_train(boot_df, threshold=INITIAL_THRESHOLD)
        freq_b = boot_df["part_id"].value_counts(normalize=True)

        # Train small RF for speed
        X_b, y_b, _ = make_features(
            attach_train_features(
                boot_df, mtbf_b, freq_b, default_mtbf, default_freq
            )
        )
        rf_b = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced", random_state=RANDOM_STATE + b, n_jobs=-1
        ).fit(X_b, y_b)

        # For prob, use Platt (sigmoid) small calibrator to stabilize tails
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
st.caption("Leakage-safe training • Cost-based decisions • Confidence-aware flags")

# File selector
st.sidebar.header("Data")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")
enable_uncertainty = st.sidebar.checkbox("Enable uncertainty band (bootstrap)", value=True)
n_boot = st.sidebar.slider("Bootstrap models (uncertainty)", min_value=10, max_value=100, value=20, step=5)

# Business cost settings
st.sidebar.header("Decision Cost Settings")
default_cfp_factor = st.sidebar.slider(
    "False-alarm handling cost as % of unit cost (per part)",
    min_value=0, max_value=100, value=25, step=5
)
# Help text
st.sidebar.caption(
    "C_FP ≈ (percent of per-part cost) × cost_per_part. "
    "C_FN ≈ cost_per_part × order quantity (loss if defect occurs and you didn't act)."
)

if not os.path.exists(csv_path):
    st.error(f"CSV not found: {csv_path}")
    st.stop()

# Load data
df = load_and_clean(csv_path)

# Time-based split
df_train, df_calib, df_test = time_based_split(df, TRAIN_FRAC, CALIB_FRAC)

if len(df_calib) == 0 or len(df_test) == 0:
    st.warning("Calibration or test split is empty after group separation. Consider adjusting split fractions or checking data coverage.")
# Train-only MTBF
mtbf_train = compute_mtbf_on_train(df_train, threshold=INITIAL_THRESHOLD)
default_mtbf = mtbf_train["mttf_scrap"].median()

# Train-only PART frequency encoding
part_freq_train = df_train["part_id"].value_counts(normalize=True)
default_freq = part_freq_train.median() if len(part_freq_train) else 0.0

# Attach train-time features to each split
df_train_f = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib_f = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test_f  = attach_train_features(df_test,  mtbf_train, part_freq_train, default_mtbf, default_freq)

# Build features/labels
X_train, y_train, FEATURES = make_features(df_train_f)
X_calib, y_calib, _ = make_features(df_calib_f)
X_test,  y_test,  _ = make_features(df_test_f)

# Train + calibrate
rf_model, calibrated_model, calib_method = train_and_calibrate(X_train, y_train, X_calib, y_calib)

# Quick calibration sanity metric
if len(X_test) > 0:
    p_test = calibrated_model.predict_proba(X_test)[:, 1]
    try:
        test_brier = brier_score_loss(y_test, p_test)
    except Exception:
        test_brier = np.nan
else:
    test_brier = np.nan

# -----------------------------
# Interactive Prediction
# -----------------------------
st.subheader("Scrap Risk Estimation")

part_ids = sorted(df["part_id"].unique().tolist())
selected_part = st.selectbox("Select Part ID", part_ids)

quantity = st.number_input("Order Quantity", min_value=1, step=1, value=351)
weight = st.number_input("Piece Weight (lbs)", min_value=0.01, step=0.01, value=4.00)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01, step=0.01, value=0.01)

# Cost-based threshold
C_FP = (default_cfp_factor / 100.0) * float(cost_per_part)
C_FN = float(cost_per_part) * float(quantity)
t_star = cost_threshold(C_FP, C_FN)

# Prepare the single-row input using train-derived features
# Use train MTBF if present; else default
if selected_part in mtbf_train.index:
    mttf_value = mtbf_train.loc[selected_part, "mttf_scrap"]
else:
    mttf_value = default_mtbf

if selected_part in part_freq_train.index:
    part_freq_value = float(part_freq_train.loc[selected_part])
else:
    part_freq_value = default_freq

input_row = pd.DataFrame(
    [[quantity, weight, mttf_value, part_freq_value]],
    columns=FEATURES
)

if st.button("Predict"):
    # Predicted probability
    base_p = calibrated_model.predict_proba(input_row)[0, 1]

    # Optional uncertainty interval via bootstrap
    if enable_uncertainty:
        mean_p, lo_p, hi_p = bootstrap_probability_interval(
            input_row, df_train, FEATURES, y_train, mtbf_train, part_freq_train,
            n_boot=n_boot, alpha=0.10
        )
    else:
        mean_p, lo_p, hi_p = base_p, np.nan, np.nan

    # Decision: act if conservative (lower) bound exceeds threshold (if enabled), else base prob
    if enable_uncertainty and not np.isnan(lo_p):
        decision_prob = lo_p  # conservative
    else:
        decision_prob = base_p

    action_flag = (decision_prob >= t_star)

    expected_scrap_count = int(round(base_p * quantity))
    expected_loss = round(expected_scrap_count * cost_per_part, 2)

    st.metric("Predicted Scrap Risk (p)", f"{base_p * 100:.2f}%")
    if enable_uncertainty:
        st.caption(f"90% probability band: [{lo_p*100:.2f}%, {hi_p*100:.2f}%]")

    st.write(f"**Cost-based decision threshold (t\*):** {t_star:.4f}")
    st.write(f"**Decision (conservative):** {'⚠️ ACT' if action_flag else '✅ NO ACTION'}")

    st.write(f"Expected Scrap Count: **{expected_scrap_count}** parts")
    st.write(f"Expected Financial Loss: **${expected_loss:.2f}**")

    # -----------------------------
    # SHAP Analysis (on RF only)
    # -----------------------------
    st.subheader("🔍 SHAP Analysis")
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_row)
        # shap_values for RF is list [class0, class1]; we use class1
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_val = shap_values[1][0]
        else:
            shap_val = np.array(shap_values)[0]

        shap_df = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP Value": shap_val,
            "Impact (|SHAP|)": np.abs(shap_val)
        }).sort_values(by="Impact (|SHAP|)", ascending=False)
        st.dataframe(shap_df, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP analysis failed: {e}")

    # -----------------------------
    # Benchmark Table (UI-consistent weight)
    # -----------------------------
    st.subheader("📊 Benchmark Scrap Prediction Table")
    bench = df.head(10).copy()

    # Attach train features (so the model can score consistently)
    bench = attach_train_features(bench, mtbf_train, part_freq_train, default_mtbf, default_freq)
    # Build feature matrix and score
    benchX = bench[FEATURES]
    bench["predicted_prob"] = calibrated_model.predict_proba(benchX)[:, 1]
    bench["predicted_scrap_pounds"] = bench["predicted_prob"] * bench["order_quantity"] * weight

    # Actual scrap pounds — prefer measured pieces_scrapped if available
    if "pieces_scrapped" in bench.columns and bench["pieces_scrapped"].notna().any():
        bench["actual_scrap_pounds"] = bench["pieces_scrapped"].fillna(0) * weight
    else:
        # fallback from % (best-effort)
        bench["actual_scrap_pounds"] = (bench["scrap%"].clip(lower=0) / 100.0) * bench["order_quantity"] * weight

    bench["abs_error"] = (bench["actual_scrap_pounds"] - bench["predicted_scrap_pounds"]).abs()

    st.dataframe(
        bench[["part_id", "order_quantity", "scrap%", "predicted_prob",
               "actual_scrap_pounds", "predicted_scrap_pounds", "abs_error"]],
        use_container_width=True
    )

# -----------------------------
# Model Diagnostics
# -----------------------------
st.subheader("Model Diagnostics")
st.write(f"Calibration method: **{calib_method}**")
if not np.isnan(test_brier):
    st.write(f"Brier score (test): **{test_brier:.5f}** (lower is better)")

with st.expander("Feature schema used by the model"):
    st.json({"features": FEATURES})

with st.expander("Notes & Tips"):
    st.markdown("""
- **No SMOTE** by default to keep class priors realistic. If you must oversample, oversample **train only**, never calibration.
- We recompute **MTTF** and **part frequency** on **train only**, then join to calib/test — avoids leakage.
- **Cost-based threshold** replaces the naive 0.5 rule. Tune *False-Alarm %* in the sidebar to reflect your process cost.
- **Uncertainty band** helps avoid false alarms by requiring the **lower** bound to exceed the threshold.
- SHAP is computed on the **raw forest**; calibration is only for probability scaling.
""")
