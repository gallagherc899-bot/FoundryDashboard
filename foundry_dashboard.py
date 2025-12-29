# ================================================================
# 🏭 Aluminum Foundry Scrap Analytics Dashboard (Enhanced Logic)
# ================================================================
# This version is visually identical to the Working Dashboard 12.28.25
# but integrates the Campbell 9-Process correlation model internally.
# It automatically computes 9 process indices derived from your defect
# columns, then silently adds them as features to the ML model.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss
)
import warnings
import os

# ================================================================
# Streamlit setup (must be first Streamlit call)
# ================================================================
st.set_page_config(page_title="Aluminum Foundry Scrap Analytics Dashboard", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
# 1️⃣ Data Loading & Cleaning
# ================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")

    # Normalize column headers (strip spaces, lowercase, underscores)
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Map known columns to consistent internal names
    rename_map = {
        "work_order_": "part_id",
        "work_order": "part_id",
        "work_order_number": "part_id",
        "work_order_#": "part_id",
        "order_quantity": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "total_scrap_weight_(lbs)": "total_scrap_weight_lbs",
        "scrap%": "scrap_percent",
        "scrap": "scrap_percent",
        "sellable": "sellable",
        "heats": "heats",
        "week_ending": "week_ending",
        "piece_weight_lbs": "piece_weight_lbs",
        "piece_weight_(lbs)": "piece_weight_lbs",
        "part_id": "part_id"
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert data types
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(0.0)
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce").fillna(0.0)

    # Drop missing dates
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    # Identify defect rate columns (those ending with '_rate')
    defect_cols = [c for c in df.columns if c.endswith("_rate")]

    return df, defect_cols

df, defect_cols = load_data()

# ================================================================
# 2️⃣ Compute Campbell Process Indices (ENHANCEMENT)
# ================================================================
# Each index = mean of associated defect rates
# Defects are matched to your dataset's available defect columns

process_mapping = {
    "Sand_System_Index": ["sand_rate", "gas_porosity_rate", "runout_rate"],
    "Core_Making_Index": ["core_rate", "crush_rate", "missrun_rate"],
    "Pattern_Maintenance_Index": ["shift_rate", "short_pour_rate", "cut_into_rate"],
    "Mold_Assembly_Index": ["runout_rate", "crush_rate", "tear_up_rate"],
    "Melting_Alloy_Index": ["dross_rate", "gas_porosity_rate", "shrink_rate"],
    "Pouring_Index": ["short_pour_rate", "missrun_rate", "tear_up_rate"],
    "Solidification_Index": ["shrink_rate", "shrink_porosity_rate", "gas_porosity_rate"],
    "Shakeout_Index": ["over_grind_rate", "bent_rate", "gouged_rate"],
    "Finishing_Index": ["dirty_pattern_rate", "failed_zyglo_rate",
                        "outside_process_scrap_rate", "zyglo_rate"]
}

# Create process indices only for defects that exist in your dataset
for proc, cols in process_mapping.items():
    existing = [c for c in cols if c in df.columns]
    if existing:
        df[proc] = df[existing].mean(axis=1)
    else:
        df[proc] = 0.0  # fallback if no defects found for that process

# Combine all process index columns into a list
process_indices = list(process_mapping.keys())

# ================================================================
# 3️⃣ Rolling 6–2–1 Split (same as original)
# ================================================================
def rolling_splits(df, weeks_train=6, weeks_val=2, weeks_test=1):
    weeks = sorted(df["week_ending"].unique())
    total = len(weeks) - (weeks_train + weeks_val + weeks_test) + 1
    for i in range(total):
        yield (
            df[df["week_ending"].isin(weeks[i : i + weeks_train])].copy(),
            df[df["week_ending"].isin(weeks[i + weeks_train : i + weeks_train + weeks_val])].copy(),
            df[df["week_ending"].isin(weeks[i + weeks_train + weeks_val : i + weeks_train + weeks_val + weeks_test])].copy(),
        )

# ================================================================
# 4️⃣ Train & Evaluate Function (now includes process indices)
# ================================================================
def train_and_evaluate(df, threshold):
    results = []
    features = defect_cols + process_indices  # <- include both defects and process indices

    for train, val, test in rolling_splits(df):
        for d in [train, val, test]:
            d["Label"] = (d["scrap_percent"] > threshold).astype(int)

        X_train, y_train = train[features], train["Label"]
        X_val, y_val     = val[features], val["Label"]
        X_test, y_test   = test[features], test["Label"]

        rf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Calibration for probability accuracy
        try:
            cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
            cal.fit(X_val, y_val)
            model = cal
        except ValueError:
            model = rf

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        results.append({
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "brier": brier_score_loss(y_test, probs),
        })
    return pd.DataFrame(results), rf

# ================================================================
# 5️⃣ Streamlit UI (identical to original)
# ================================================================
st.title("🏭 Aluminum Foundry Scrap Analytics Dashboard")

with st.sidebar:
    st.header("🔧 Manager Input Controls")
    part_id = st.text_input("Enter Part ID")
    order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    weight = st.number_input("Piece Weight (lbs)", min_value=0.0, value=10.0)
    cost = st.number_input("Cost per Part ($)", min_value=0.0, value=50.0)
    threshold = st.slider("Scrap% Threshold", 0.0, 5.0, 2.5, 0.5)
    predict = st.button("🔮 Predict")

tab1, tab2 = st.tabs(["📈 Dashboard", "📊 Validation (6–2–1)"])

# ================================================================
# 6️⃣ Prediction Logic (unchanged externally, enhanced internally)
# ================================================================
if predict:
    with st.spinner("⏳ Training enhanced model..."):
        df_part = df.copy() if not part_id else df[df["part_id"].astype(str).str.contains(str(part_id), case=False, na=False)]
        if df_part.empty:
            st.warning(f"No data found for Part ID '{part_id}'. Using full dataset.")
            df_part = df.copy()

        results, model = train_and_evaluate(df_part, threshold)

        # Predict scrap risk
        scrap_pred = model.predict_proba(df_part[defect_cols + process_indices])[:, 1].mean() * 100
        expected_scrap = order_qty * (scrap_pred / 100)
        loss = expected_scrap * cost
        mtts = (len(df_part) / (expected_scrap + 1)) * 7

        # Pareto calculations (unchanged UI)
        pareto_hist = df_part[defect_cols].mean().sort_values(ascending=False)
        pareto_pred = pd.Series(model.feature_importances_, index=defect_cols + process_indices).sort_values(ascending=False)

        st.session_state.update({
            "results": results,
            "pareto_hist": pareto_hist,
            "pareto_pred": pareto_pred,
            "scrap_pred": scrap_pred,
            "loss": loss,
            "mtts": mtts
        })
    st.success("✅ Prediction Complete!")

# ================================================================
# 7️⃣ Tab 1 — Dashboard View
# ================================================================
with tab1:
    st.header("📈 Scrap Risk & Pareto Dashboard")
    if "results" in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Scrap %", f"{st.session_state.scrap_pred:.2f}%")
        col2.metric("Expected Scrap Cost", f"${st.session_state.loss:,.2f}")
        col3.metric("MTTS (days)", f"{st.session_state.mtts:.1f}")

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Historical Pareto (Observed)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_pred.head(15).plot(kind="bar", ax=ax, color="seagreen")
            ax.set_title("Predicted Pareto (Enhanced Model)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

# ================================================================
# 8️⃣ Tab 2 — Rolling Validation
# ================================================================
with tab2:
    st.header("📊 Rolling 6–2–1 Validation Results")
    if "results" in st.session_state:
        val_df = st.session_state.results
        st.dataframe(val_df.describe().T.style.format("{:.3f}"))

        fig, ax = plt.subplots(figsize=(6,3))
        val_df[["accuracy", "recall", "precision", "f1"]].plot(ax=ax)
        ax.set_title("Rolling Validation Performance (Enhanced)")
        ax.set_xlabel("Rolling Window #")
        ax.set_ylabel("Score")
        st.pyplot(fig)

st.caption("© 2025 Foundry Analytics | Enhanced Predictive Engine (Internal Only)")
