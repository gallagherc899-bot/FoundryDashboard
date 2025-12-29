# ================================================================
# 🧠 Aluminum Foundry Scrap Analytics Dashboard (Doctoral Version)
# ---------------------------------------------------------------
# Author: [Your Name]
# Based on Campbell (2003), Juran (1999), Taguchi (2004), AFS (2015)
# ---------------------------------------------------------------
# Description:
# This Streamlit dashboard analyzes aluminum greensand foundry scrap
# performance using SPC principles and a multivariate machine learning
# approach integrating process-defect relationships.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from tqdm import tqdm

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    df["Week_Ending"] = pd.to_datetime(df["Week_Ending"], errors="coerce")
    df = df.sort_values("Week_Ending").dropna(subset=["Week_Ending"]).fillna(0)
    defect_cols = [c for c in df.columns if c.lower().endswith("rate")]
    return df, defect_cols

df, defect_cols = load_data()

# ------------------------------------------------
# Define Campbell’s process-defect meta-feature groups
# ------------------------------------------------
process_groups = {
    "Sand_System_Index": ["Sand_Rate", "Gas_Porosity_Rate", "Runout_Rate"],
    "Core_Making_Index": ["Core_Rate", "Crush_Rate", "Shrink_Porosity_Rate"],
    "Melting_Index": ["Dross_Rate", "Gas_Porosity_Rate", "Shrink_Rate", "Shrink_Porosity_Rate"],
    "Pouring_Index": ["Missrun_Rate", "Short_Pour_Rate", "Dross_Rate", "Tear_Up_Rate"],
    "Solidification_Index": ["Shrink_Rate", "Shrink_Porosity_Rate", "Gas_Porosity_Rate"],
    "Finishing_Index": ["Over_Grind_Rate", "Bent_Rate", "Gouged_Rate", "Shift_Rate"],
}

# Add meta-feature columns
for name, cols in process_groups.items():
    present = [c for c in cols if c in df.columns]
    df[name] = df[present].mean(axis=1) if present else 0.0

# ------------------------------------------------
# Rolling 6–2–1 window split generator
# ------------------------------------------------
def rolling_splits(df, weeks_train=6, weeks_val=2, weeks_test=1):
    weeks = sorted(df["Week_Ending"].unique())
    total_rolls = len(weeks) - (weeks_train + weeks_val + weeks_test) + 1
    splits = []
    for i in range(total_rolls):
        train_w = weeks[i : i + weeks_train]
        val_w   = weeks[i + weeks_train : i + weeks_train + weeks_val]
        test_w  = weeks[i + weeks_train + weeks_val : i + weeks_train + weeks_val + weeks_test]
        splits.append((
            df[df["Week_Ending"].isin(train_w)],
            df[df["Week_Ending"].isin(val_w)],
            df[df["Week_Ending"].isin(test_w)],
        ))
    return splits

# ------------------------------------------------
# Train and evaluate rolling windows
# ------------------------------------------------
def train_and_evaluate(df, threshold, use_meta=False):
    results = []
    splits = rolling_splits(df)
    feature_cols = defect_cols.copy()
    if use_meta:
        feature_cols += list(process_groups.keys())

    for roll, (train, val, test) in enumerate(tqdm(splits, desc=f"Rolling thr={threshold}")):
        train, val, test = train.copy(), val.copy(), test.copy()

        train["Label"] = (train["Scrap_"] > threshold).astype(int)
        val["Label"]   = (val["Scrap_"] > threshold).astype(int)
        test["Label"]  = (test["Scrap_"] > threshold).astype(int)

        X_train, y_train = train[feature_cols], train["Label"]
        X_val, y_val     = val[feature_cols], val["Label"]
        X_test, y_test   = test[feature_cols], test["Label"]

        rf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        try:
            if len(np.unique(y_val)) < 2 or y_val.value_counts().min() < 3:
                raise ValueError("Not enough samples to calibrate")
            cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
            cal.fit(X_val, y_val)
            model = cal
        except ValueError:
            model = rf

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        results.append({
            "roll": roll + 1,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "brier": brier_score_loss(y_test, probs),
        })

    return pd.DataFrame(results)

# ------------------------------------------------
# Streamlit Dashboard UI
# ------------------------------------------------
st.set_page_config("Aluminum Foundry Scrap Analytics Dashboard", layout="wide")
st.title("🏭 Aluminum Foundry Scrap Analytics Dashboard")
st.markdown("""
This dashboard integrates Statistical Process Control (SPC) with a multivariate Machine Learning model 
to identify and predict casting defects based on both individual defect rates and interconnected foundry processes.
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Manager Input Controls")
    part_id = st.text_input("Enter Part ID")
    order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    cost = st.number_input("Cost per Part ($)", min_value=0.0, value=50.0)
    threshold = st.slider("Scrap% Threshold", 0.0, 5.0, 2.5, 0.5)
    predict = st.button("🔮 Predict Scrap Performance")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Manager Dashboard", "📊 Research Comparison", "⚙️ Enhanced Manager Dashboard"])

# ------------------------------------------------
# Prediction logic
# ------------------------------------------------
if predict:
    with st.spinner("Training models for selected Part ID..."):
        df_part = df[df["Part_ID"].astype(str).str.contains(str(part_id), case=False, na=False)]
        if df_part.empty:
            st.warning(f"No records found for Part ID '{part_id}'. Using entire dataset.")
            df_part = df.copy()

        # Re-run rolling evaluation
        base_results = train_and_evaluate(df_part, threshold, use_meta=False)
        enh_results = train_and_evaluate(df_part, threshold, use_meta=True)

        # Aggregate
        base_summary = base_results.mean().to_dict()
        enh_summary = enh_results.mean().to_dict()

        # Train final models for visualization
        df_part["Label"] = (df_part["Scrap_"] > threshold).astype(int)
        rf_base = RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42)
        rf_enh  = RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42)
        rf_base.fit(df_part[defect_cols], df_part["Label"])
        rf_enh.fit(df_part[defect_cols + list(process_groups.keys())], df_part["Label"])

        # Compute predictions
        scrap_pred_base = rf_base.predict_proba(df_part[defect_cols])[:, 1].mean() * 100
        scrap_pred_enh  = rf_enh.predict_proba(df_part[defect_cols + list(process_groups.keys())])[:, 1].mean() * 100

        expected_scrap_base = order_qty * (scrap_pred_base / 100)
        expected_scrap_enh  = order_qty * (scrap_pred_enh / 100)

        loss_base = expected_scrap_base * cost
        loss_enh  = expected_scrap_enh * cost

        # Pareto charts
        pareto_hist = df_part[defect_cols].mean().sort_values(ascending=False)
        pareto_pred_base = pd.Series(rf_base.feature_importances_, index=defect_cols).sort_values(ascending=False)
        pareto_pred_enh  = pd.Series(rf_enh.feature_importances_, index=defect_cols + list(process_groups.keys())).sort_values(ascending=False)

        # Save to session for other tabs
        st.session_state.update({
            "base_summary": base_summary,
            "enh_summary": enh_summary,
            "pareto_hist": pareto_hist,
            "pareto_pred_base": pareto_pred_base,
            "pareto_pred_enh": pareto_pred_enh,
            "scrap_pred_base": scrap_pred_base,
            "scrap_pred_enh": scrap_pred_enh,
            "loss_base": loss_base,
            "loss_enh": loss_enh
        })
    st.success("✅ Prediction complete! Scroll down to view results.")

# ------------------------------------------------
# Tab 1 – Baseline
# ------------------------------------------------
with tab1:
    st.header("📈 Manager Dashboard — Baseline Model")
    if "base_summary" in st.session_state:
        st.metric("Predicted Scrap %", f"{st.session_state.scrap_pred_base:.2f}%")
        st.metric("Expected Scrap Cost", f"${st.session_state.loss_base:,.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Historical Pareto (Observed)")
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Observed Defect Rates")
            ax.set_ylabel("Mean Rate (%)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        with col2:
            st.subheader("Predicted Pareto (Baseline)")
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_pred_base.head(15).plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Model Feature Importance (Baseline)")
            ax.set_ylabel("Importance")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        st.write("This tab reflects traditional SPC univariate control aligned with Juran (1999) and AFS (2015).")

# ------------------------------------------------
# Tab 2 – Research Comparison
# ------------------------------------------------
with tab2:
    st.header("📊 Research Comparison — Baseline vs Enhanced")
    if "base_summary" in st.session_state:
        comp_df = pd.DataFrame([
            {"Model":"Baseline", **st.session_state.base_summary},
            {"Model":"Enhanced", **st.session_state.enh_summary},
        ])
        st.dataframe(comp_df.style.format("{:.3f}"))
        st.markdown("#### Feature Importance (Enhanced Model)")
        fig, ax = plt.subplots(figsize=(8,4))
        st.session_state.pareto_pred_enh.head(20).plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Top 20 Feature Importances — Enhanced Model")
        st.pyplot(fig)

# ------------------------------------------------
# Tab 3 – Enhanced Manager Dashboard
# ------------------------------------------------
with tab3:
    st.header("⚙️ Enhanced Manager Dashboard — Process-Aware Model")
    if "enh_summary" in st.session_state:
        st.metric("Predicted Scrap % (Enhanced)", f"{st.session_state.scrap_pred_enh:.2f}%")
        st.metric("Expected Scrap Cost (Enhanced)", f"${st.session_state.loss_enh:,.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Historical Pareto (Observed)")
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Observed Defect Rates")
            st.pyplot(fig)

        with col2:
            st.subheader("Predicted Pareto (Enhanced Model)")
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_pred_enh.head(15).plot(kind="bar", ax=ax, color="mediumseagreen")
            ax.set_title("Feature Importance (Process-Aware)")
            st.pyplot(fig)

        st.markdown("### 🧩 Process–Defect Mapping (Enhanced Insight)")
        mapping_data = []
        for defect in defect_cols:
            linked_procs = [proc for proc, dcols in process_groups.items() if defect in dcols]
            if linked_procs:
                proc_means = df[linked_procs].mean().mean()
                mapping_data.append({
                    "Defect": defect,
                    "Associated Processes": ", ".join(linked_procs),
                    "Avg Process Index": round(proc_means, 3),
                })
        st.dataframe(pd.DataFrame(mapping_data))
        st.write("This mapping reveals how multivariate process interactions contribute to recurring defect patterns, "
                 "supporting Campbell (2003) and DOE Best Practice (2004) findings.")
