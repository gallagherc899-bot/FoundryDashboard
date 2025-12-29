# ================================================================
# 🏭 Aluminum Foundry Scrap Analytics Dashboard (Enhanced Logic v2)
# ================================================================
# Identical UI to "Working Dashboard 12.28.25"
# Internally enhanced with Campbell 9-Process correlations.
# Includes defensive fix for single-class thresholds.
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

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
# 1️⃣ Data Loading & Cleaning
# ================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")

    # Normalize headers: lower case, underscores, no symbols
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Map consistent names for key fields
    rename_map = {
        "work_order": "part_id",
        "work_order_#": "part_id",
        "order_quantity": "order_quantity",
        "scrap": "scrap_percent",
        "scrap_percent": "scrap_percent",
        "scrap%": "scrap_percent",
        "week_ending": "week_ending",
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure proper types
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    # Identify defect columns dynamically
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    return df, defect_cols

df, defect_cols = load_data()

# ================================================================
# 2️⃣ Campbell Process Index Calculation (ENHANCED LOGIC)
# ================================================================
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

# Compute indices if corresponding defect columns exist
for proc, cols in process_mapping.items():
    existing = [c for c in cols if c in df.columns]
    df[proc] = df[existing].mean(axis=1) if existing else 0.0

process_indices = list(process_mapping.keys())

# ================================================================
# 3️⃣ Rolling 6–2–1 Validation Split
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
# 4️⃣ Model Training & Evaluation (with IndexError Fix)
# ================================================================
def train_and_evaluate(df, threshold):
    results = []
    features = defect_cols + process_indices  # include both defect + process features

    for train, val, test in rolling_splits(df):
        # Apply label threshold
        for d in [train, val, test]:
            d["Label"] = (d["scrap_percent"] > threshold).astype(int)

        X_train, y_train = train[features], train["Label"]
        X_val, y_val     = val[features], val["Label"]
        X_test, y_test   = test[features], test["Label"]

        # Defensive: skip if y_train has only one class
        if len(np.unique(y_train)) < 2:
            st.warning("⚠️ Training set contains only one class at this threshold. Try lowering Scrap%.")
            return pd.DataFrame(), None

        rf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Calibration for smoother probability prediction
        try:
            cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
            cal.fit(X_val, y_val)
            model = cal
        except ValueError:
            model = rf

        # Fix: handle single-class predict_proba outputs
        probs = model.predict_proba(X_test)
        if probs.shape[1] == 1:
            probs = np.zeros(len(X_test))
        else:
            probs = probs[:, 1]

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
# 5️⃣ Streamlit UI (Identical to Original)
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
# 6️⃣ Prediction Logic (Enhanced Internals)
# ================================================================
if predict:
    with st.spinner("⏳ Training enhanced predictive model..."):
        df_part = df.copy() if not part_id else df[df["part_id"].astype(str).str.contains(str(part_id), case=False, na=False)]
        if df_part.empty:
            st.warning(f"No data found for Part ID '{part_id}'. Using full dataset.")
            df_part = df.copy()

        results, model = train_and_evaluate(df_part, threshold)

        if model is not None and not results.empty:
            scrap_pred = model.predict_proba(df_part[defect_cols + process_indices])
            if scrap_pred.shape[1] == 1:
                scrap_pred = np.zeros(len(df_part))
            else:
                scrap_pred = scrap_pred[:, 1]

            scrap_pred = scrap_pred.mean() * 100
            expected_scrap = order_qty * (scrap_pred / 100)
            loss = expected_scrap * cost
            mtts = (len(df_part) / (expected_scrap + 1)) * 7

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
# 7️⃣ Dashboard Tab
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
# 8️⃣ Validation Tab
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

st.caption("© 2025 Foundry Analytics | Internal Enhanced Logic (Stable v2)")
