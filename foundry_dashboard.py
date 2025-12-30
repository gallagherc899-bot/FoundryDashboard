# ============================================================
# 🏭 Foundry Scrap Risk Dashboard
# Dynamic 6–2–1 retraining + similarity expansion + calibration-safe feature importances
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

# ============================================================
# Data loading and cleaning
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != "None"]).strip() for col in df.columns.values]

    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )

    forbidden_cols = [c for c in df.columns if "work_order" in c]
    if forbidden_cols:
        df.drop(columns=forbidden_cols, inplace=True, errors="ignore")
        st.warning(f"⚠ Dropped disallowed columns: {forbidden_cols}")

    rename_map = {
        "order_quantity": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "scrap_percent": "scrap_percent",
        "scrap": "scrap_percent",
        "scrap_": "scrap_percent",
        "sellable": "sellable",
        "heats": "heats",
        "week_ending": "week_ending",
        "piece_weight_lbs": "piece_weight_lbs",
        "part_id": "part_id",
    }
    df.rename(columns=rename_map, inplace=True)

    if "part_id" not in df.columns:
        for c in df.columns:
            if "part" in c and "id" in c:
                df.rename(columns={c: "part_id"}, inplace=True)
                break
    if "part_id" not in df.columns:
        st.error("❌ No 'Part ID' column found — please add one to your CSV.")
        st.stop()

    if isinstance(df["part_id"], pd.DataFrame):
        df["part_id"] = df["part_id"].iloc[:, 0]
    df["part_id"] = df["part_id"].astype(str).replace({"nan": "unknown", "": "unknown"}).str.strip()

    num_cols = [c for c in df.columns if c.endswith("_rate") or "scrap" in c or "weight" in c or "quantity" in c]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "week_ending" in df.columns:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
        df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    st.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns. Using 'Part ID' as the unique identifier.")
    return df

# ============================================================
# Helper functions
# ============================================================
def time_split(df, train_ratio=0.6, calib_ratio=0.2):
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:calib_end], df_sorted.iloc[calib_end:]

def compute_mtbf_on_train(df_train, thr_label):
    grp = df_train.groupby("part_id", dropna=False)["scrap_percent"].mean().reset_index()
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp

def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s

def make_xy(df, thr_label, use_rate_cols):
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0)
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    y = (df["scrap_percent"] > thr_label).astype(int)
    if y.nunique() < 2:
        st.warning(
            f"⚠ Not enough label diversity after applying scrap threshold {thr_label}%. "
            f"All samples are of one class — model cannot learn risk."
        )
        st.stop()
    X = df[feats]
    return X, y, feats

def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    pos = int(y_calib.sum())
    if pos == 0 or pos == len(y_calib):
        return rf, rf, "uncalibrated"
    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
    return rf, cal, "calibrated (sigmoid, cv=3)"

def safe_feature_importances(model):
    """Safely extract feature importances from calibrated or uncalibrated models."""
    base_model = getattr(model, "base_estimator", model)
    if hasattr(base_model, "feature_importances_"):
        return base_model.feature_importances_
    return np.zeros(1)

# ============================================================
# Dynamic part-based data expansion
# ============================================================
def prepare_part_data(df: pd.DataFrame, target_part: str, min_samples: int = 30) -> pd.DataFrame:
    df_part = df[df["part_id"] == target_part].copy()
    base_weight = df_part["piece_weight_lbs"].mean() if not df_part.empty else df["piece_weight_lbs"].median()
    df_candidate = df.copy()
    weight_tol, defect_tol, max_tol = 0.01, 0.01, 0.15
    iteration = 0
    defect_cols = [c for c in df.columns if c.endswith("_rate")]

    while len(df_part) < min_samples and weight_tol <= max_tol:
        iteration += 1
        lower, upper = base_weight * (1 - weight_tol), base_weight * (1 + weight_tol)
        df_weight = df_candidate[df_candidate["piece_weight_lbs"].between(lower, upper)]
        if defect_cols:
            for col in defect_cols:
                base_def = df_part[col].mean() if not df_part.empty else df_weight[col].mean()
                low, high = base_def * (1 - defect_tol), base_def * (1 + defect_tol)
                df_weight = df_weight[(df_weight[col] >= low) & (df_weight[col] <= high)]
        df_part = pd.concat([df_part, df_weight]).drop_duplicates(subset=df.columns)
        weight_tol += 0.005
        defect_tol += 0.005

    st.info(f"✅ Using {len(df_part)} samples for Part {target_part} "
            f"(±{weight_tol*100:.1f}% weight, ±{defect_tol*100:.1f}% defects, {iteration} iterations)")
    return df_part

# ============================================================
# Streamlit UI
# ============================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold (Label & MTTF)", 1.0, 15.0, 6.5, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

st.title("🏭 Foundry Scrap Risk Dashboard (Dynamic Auto 6–2–1)")

part_id_input = st.text_input("Part ID", value="Unknown")
order_qty = st.number_input("Order Quantity", min_value=1, value=100)
piece_weight = st.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

if st.button("Predict"):
    try:
        st.info("🔁 Re-training model (6–2–1) with part-specific dataset...")

        df = load_and_clean(csv_path)
        df_part = prepare_part_data(df, part_id_input, min_samples=30)
        if df_part.empty:
            st.error("❌ No suitable data found for this part.")
            st.stop()

        # Auto-expansion fallback
        if df_part["scrap_percent"].nunique() < 2:
            st.warning(f"⚠ Only one scrap label found. Expanding ±25%...")
            df_part = prepare_part_data(df, part_id_input, min_samples=30)
            if df_part["scrap_percent"].nunique() < 2:
                st.error("❌ Still only one scrap label found — not enough diversity.")
                st.stop()
            else:
                st.success(f"✅ Additional data found ({len(df_part)} samples).")

        df_train, df_calib, df_test = time_split(df_part)
        mtbf_train = compute_mtbf_on_train(df_train, thr_label)
        part_freq_train = df_train["part_id"].value_counts(normalize=True)
        default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
        default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

        df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

        X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
        X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
        rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)

        st.success(f"✅ Model retrained ({method}) using {len(X_train)} samples.")

        # Safe feature importances
        importances = safe_feature_importances(cal_model)
        if len(importances) == len(feats):
            fi_df = pd.DataFrame({"Feature": feats, "Importance": importances}).sort_values(
                "Importance", ascending=False
            )
            st.write("### 🔍 Feature Importances")
            st.dataframe(fi_df)

        # Prediction
        mtbf_val = mtbf_train["mttf_scrap"].mean()
        freq_val = part_freq_train.mean()

        input_df = pd.DataFrame(
            [[part_id_input, order_qty, piece_weight, mtbf_val, freq_val]],
            columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
        )
        X_input = input_df.drop(columns=["part_id"])
        for col in feats:
            if col not in X_input.columns:
                X_input[col] = 0.0
        X_input = X_input[feats]

        try:
            prob = float(cal_model.predict_proba(X_input)[0, 1])
        except Exception:
            prob = 0.0

        adj_prob = np.clip(prob, 0.0, 1.0)
        exp_scrap = order_qty * adj_prob
        exp_loss = exp_scrap * cost_per_part
        reliability = 1.0 - adj_prob

        st.markdown(f"### 🧩 Prediction Results for Part **{part_id_input}**")
        st.metric("Predicted Scrap Risk", f"{adj_prob*100:.2f}%")
        st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
        st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
        st.metric("Reliability", f"{reliability*100:.2f}%")

        st.success("🔁 Full pipeline executed successfully.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
