# ============================================================
# 🏭 Foundry Scrap Risk Dashboard (Full Auto 6–2–1)
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
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2

# ============================================================
# Data loading and cleaning
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Loads and cleans foundry scrap data.
    Only 'Part ID' is used as the unique identifier (no work order mappings).
    """
    df = pd.read_csv(csv_path)

    # Flatten any multi-row headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != "None"]).strip()
            for col in df.columns.values
        ]

    # Normalize headers (spaces -> underscores, lowercase)
    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )

    # 🔒 Explicitly disallow any "work_order" columns
    forbidden_cols = [c for c in df.columns if "work_order" in c]
    if forbidden_cols:
        df.drop(columns=forbidden_cols, inplace=True, errors="ignore")
        st.warning(f"⚠ Dropped disallowed columns: {forbidden_cols}")

    # Canonical renames
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

    # 🧩 Ensure 'part_id' exists
    if "part_id" not in df.columns:
        for c in df.columns:
            if "part" in c and "id" in c:
                df.rename(columns={c: "part_id"}, inplace=True)
                break
    if "part_id" not in df.columns:
        st.error("❌ No 'Part ID' column found — please add one to your CSV.")
        st.stop()

    # Handle duplicate part_id columns
    if (df.columns == "part_id").sum() > 1:
        first_idx = np.where(df.columns == "part_id")[0][0]
        keep_mask = np.ones(len(df.columns), dtype=bool)
        dup_idx = np.where(df.columns == "part_id")[0][1:]
        keep_mask[dup_idx] = False
        df = df.loc[:, keep_mask]
        st.warning("⚠ Multiple 'Part ID' columns detected — keeping the first.")

    # Guarantee part_id is 1D and string
    if isinstance(df["part_id"], pd.DataFrame):
        df["part_id"] = df["part_id"].iloc[:, 0]
    df["part_id"] = (
        df["part_id"].astype(str).replace({"nan": "unknown", "": "unknown"}).str.strip()
    )

    # Numeric conversions
    num_cols = [
        c
        for c in df.columns
        if c.endswith("_rate")
        or "scrap" in c
        or "weight" in c
        or "quantity" in c
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Convert dates
    if "week_ending" in df.columns:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
        df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    st.info(
        f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns. Using 'Part ID' as the unique identifier."
    )
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
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    y = (df["scrap_percent"] > thr_label).astype(int)
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

# ============================================================
# Similarity search
# ============================================================
@st.cache_data(show_spinner=False)
def find_similar_parts(df, target_weight, defect_cols, min_rows=30):
    """Iteratively widen similarity window for weight and defects until enough data are found."""
    if df.empty or target_weight <= 0:
        return pd.DataFrame()

    weight_tol = 0.01  # ±1%
    defect_tol = 0.01
    max_tol = 0.10
    similar = pd.DataFrame()

    while len(similar) < min_rows and weight_tol <= max_tol:
        lower = target_weight * (1 - weight_tol)
        upper = target_weight * (1 + weight_tol)
        similar = df[df["piece_weight_lbs"].between(lower, upper)]
        weight_tol += 0.005

    if not similar.empty and defect_cols:
        mean_defects = df[defect_cols].mean()
        defect_tol = 0.01
        while len(similar) < min_rows and defect_tol <= max_tol:
            mask = (similar[defect_cols].sub(mean_defects).abs() <= defect_tol).all(axis=1)
            similar = similar[mask]
            defect_tol += 0.005

    return similar

# ============================================================
# Streamlit Sidebar
# ============================================================
st.sidebar.header("📂 Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model Controls")
thr_label = st.sidebar.slider("Scrap % threshold (Label & MTTF)", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()

# ============================================================
# Main Prediction Panel
# ============================================================
st.title("🏭 Foundry Scrap Risk Dashboard (Full Auto 6–2–1)")

part_id_input = st.text_input("Part ID", value="Unknown")
order_qty = st.number_input("Order Quantity", min_value=1, value=100)
piece_weight = st.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

if st.button("Predict"):
    try:
        st.info("🔁 Re-training model (6–2–1 split) and refreshing features...")

        # 1️⃣ Load and prepare
        df = load_and_clean(csv_path)
        df_train, df_calib, df_test = time_split(df)
        mtbf_train = compute_mtbf_on_train(df_train, thr_label)
        part_freq_train = df_train["part_id"].value_counts(normalize=True)
        default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
        default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

        df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
        df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

        # 2️⃣ Train fresh model
        X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
        X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
        rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)

        st.success(f"✅ Model retrained ({method}) using {len(X_train)} samples.")

        # 3️⃣ Handle part lookup or similarity
        defect_cols = [c for c in df_train.columns if c.endswith("_rate")]
        if part_id_input in mtbf_train["part_id"].values:
            mtbf_val = mtbf_train.loc[mtbf_train["part_id"] == part_id_input, "mttf_scrap"].values[0]
            freq_val = part_freq_train.get(part_id_input, default_freq)
        else:
            similar_df = find_similar_parts(df_train, piece_weight, defect_cols)
            if len(similar_df) > 0:
                mtbf_val = similar_df["scrap_percent"].mean()
                freq_val = similar_df["part_id"].value_counts(normalize=True).mean()
                st.info(f"⚙️ Using {len(similar_df)} similar parts for estimation.")
                st.dataframe(similar_df.head(10))
            else:
                mtbf_val = default_mtbf
                freq_val = default_freq
                st.warning("⚠ No similar parts found — using global defaults.")

        # 4️⃣ Predict
        input_df = pd.DataFrame(
            [[part_id_input, order_qty, piece_weight, mtbf_val, freq_val]],
            columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
        )
        X_input = input_df.drop(columns=["part_id"])
        for col in feats:
            if col not in X_input.columns:
                X_input[col] = 0.0
        X_input = X_input[feats]

        prob = float(cal_model.predict_proba(X_input)[0, 1])
        adj_prob = np.clip(prob, 0.0, 1.0)
        exp_scrap = order_qty * adj_prob
        exp_loss = exp_scrap * cost_per_part
        reliability = 1.0 - adj_prob

        # 5️⃣ Display results
        st.markdown(f"### 🧩 Prediction Results for Part **{part_id_input}**")
        st.metric("Predicted Scrap Risk", f"{adj_prob*100:.2f}%")
        st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
        st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
        st.metric("Reliability", f"{reliability*100:.2f}%")
        st.metric("MTTF Scrap", f"{mtbf_val:.2f}")
        st.metric("Part Frequency", f"{freq_val:.4f}")

        st.success("🔁 Full pipeline (6–2–1 + similarity search) executed successfully.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
