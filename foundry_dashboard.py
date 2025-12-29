import streamlit as st

# ✅ Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Aluminum Foundry Scrap Analytics Dashboard",
    layout="wide"
)

# ================================================================
# 🧠 Aluminum Foundry Scrap Analytics Dashboard (Doctoral Version)
# ================================================================
# Author: [Your Name]
# Based on Campbell (2003), Juran (1999), Taguchi (2004), AFS (2015)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss
)
from tqdm import tqdm

# ------------------------------------------------
# Load and prepare data
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
# Define Campbell process-defect groups
# ------------------------------------------------
process_groups = {
    "Sand_System_Index": ["Sand_Rate", "Gas_Porosity_Rate", "Runout_Rate"],
    "Core_Making_Index": ["Core_Rate", "Crush_Rate", "Shrink_Porosity_Rate"],
    "Melting_Index": ["Dross_Rate", "Gas_Porosity_Rate", "Shrink_Rate", "Shrink_Porosity_Rate"],
    "Pouring_Index": ["Missrun_Rate", "Short_Pour_Rate", "Dross_Rate", "Tear_Up_Rate"],
    "Solidification_Index": ["Shrink_Rate", "Shrink_Porosity_Rate", "Gas_Porosity_Rate"],
    "Finishing_Index": ["Over_Grind_Rate", "Bent_Rate", "Gouged_Rate", "Shift_Rate"],
}

for name, cols in process_groups.items():
    present = [c for c in cols if c in df.columns]
    df[name] = df[present].mean(axis=1) if present else 0.0

# ------------------------------------------------
# Rolling 6–2–1 split generator
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
# Train and evaluate model
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
            n_estimators=180, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        try:
            if len(np.unique(y_val)) < 2 or y_val.value_counts().min() < 3:
                raise ValueError
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
# Streamlit UI
# ------------------------------------------------
st.title("🏭 Aluminum Foundry Scrap Analytics Dashboard")
st.markdown("""
This dashboard integrates Statistical Process Control (SPC) and Machine Learning (Random Forest)
to predict and analyze aluminum casting defects.  
It models how **multivariate process interactions** influence scrap outcomes (Campbell, 2003).
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Manager Input Controls")
    part_id = st.text_input("Enter Part ID")
    order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    weight = st.number_input("Piece Weight (lbs)", min_value=0.0, value=10.0)
    cost = st.number_input("Cost per Part ($)", min_value=0.0, value=50.0)
    threshold = st.slider("Scrap% Threshold", 0.0, 5.0, 2.5, 0.5)
    predict = st.button("🔮 Predict Scrap Performance")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Manager Dashboard", "📊 Research Comparison", "⚙️ Enhanced Manager Dashboard"])

# ------------------------------------------------
# Run Prediction
# ------------------------------------------------
if predict:
    with st.spinner("Running rolling-window training... please wait (~1 min)..."):
        df_part = df[df["Part_ID"].astype(str).str.contains(str(part_id), case=False, na=False)]
        if df_part.empty:
            df_part = df.copy()
            st.warning(f"No records found for Part ID '{part_id}'. Using full dataset instead.")

        base_res = train_and_evaluate(df_part, threshold, use_meta=False)
        enh_res  = train_and_evaluate(df_part, threshold, use_meta=True)

        df_part["Label"] = (df_part["Scrap_"] > threshold).astype(int)
        rf_base = RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42)
        rf_enh  = RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42)
        rf_base.fit(df_part[defect_cols], df_part["Label"])
        rf_enh.fit(df_part[defect_cols + list(process_groups.keys())], df_part["Label"])

        scrap_pred_base = rf_base.predict_proba(df_part[defect_cols])[:, 1].mean() * 100
        scrap_pred_enh  = rf_enh.predict_proba(df_part[defect_cols + list(process_groups.keys())])[:, 1].mean() * 100

        expected_scrap_base = order_qty * (scrap_pred_base / 100)
        expected_scrap_enh  = order_qty * (scrap_pred_enh / 100)
        loss_base = expected_scrap_base * cost
        loss_enh  = expected_scrap_enh * cost
        mtts_base = (len(df_part) / (expected_scrap_base + 1)) * 7
        mtts_enh  = (len(df_part) / (expected_scrap_enh + 1)) * 7

        pareto_hist = df_part[defect_cols].mean().sort_values(ascending=False)
        pareto_base = pd.Series(rf_base.feature_importances_, index=defect_cols).sort_values(ascending=False)
        pareto_enh  = pd.Series(rf_enh.feature_importances_, index=defect_cols + list(process_groups.keys())).sort_values(ascending=False)

        st.session_state.update({
            "base_res": base_res, "enh_res": enh_res,
            "pareto_hist": pareto_hist, "pareto_base": pareto_base, "pareto_enh": pareto_enh,
            "scrap_pred_base": scrap_pred_base, "scrap_pred_enh": scrap_pred_enh,
            "loss_base": loss_base, "loss_enh": loss_enh,
            "mtts_base": mtts_base, "mtts_enh": mtts_enh
        })
    st.success("✅ Prediction completed successfully.")

# ------------------------------------------------
# TAB 1 — Manager Dashboard (Baseline)
# ------------------------------------------------
with tab1:
    st.header("📈 Baseline Model")
    if "base_res" in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Scrap %", f"{st.session_state.scrap_pred_base:.2f}%")
        col2.metric("Expected Scrap Cost", f"${st.session_state.loss_base:,.2f}")
        col3.metric("MTTS (days)", f"{st.session_state.mtts_base:.1f}")

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Historical Pareto (Observed)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_base.head(15).plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Predicted Pareto (Baseline)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

# ------------------------------------------------
# TAB 2 — Research Comparison
# ------------------------------------------------
with tab2:
    st.header("📊 Baseline vs Enhanced Comparison")
    if "base_res" in st.session_state:
        df_comp = pd.DataFrame([
            {"Model": "Baseline", **st.session_state.base_res.mean().to_dict()},
            {"Model": "Enhanced", **st.session_state.enh_res.mean().to_dict()},
        ])
        st.dataframe(df_comp.style.format("{:.3f}"))

        fig, ax = plt.subplots(figsize=(8,4))
        st.session_state.pareto_enh.head(20).plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Feature Importance — Enhanced Model")
        st.pyplot(fig)

# ------------------------------------------------
# TAB 3 — Enhanced Manager Dashboard
# ------------------------------------------------
with tab3:
    st.header("⚙️ Process-Aware Model (Enhanced)")
    if "enh_res" in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Scrap %", f"{st.session_state.scrap_pred_enh:.2f}%")
        col2.metric("Expected Scrap Cost", f"${st.session_state.loss_enh:,.2f}")
        col3.metric("MTTS (days)", f"{st.session_state.mtts_enh:.1f}")

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Historical Pareto")
            st.pyplot(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5,3))
            st.session_state.pareto_enh.head(15).plot(kind="bar", ax=ax, color="mediumseagreen")
            ax.set_title("Predicted Pareto (Enhanced)")
            st.pyplot(fig)

        st.markdown("### 🧩 Process–Defect Mapping Table")
        mapping = []
        for defect in defect_cols:
            linked = [p for p, ds in process_groups.items() if defect in ds]
            if linked:
                avg_idx = df[linked].mean().mean()
                mapping.append({
                    "Defect": defect,
                    "Associated Processes": ", ".join(linked),
                    "Avg Process Index": round(avg_idx, 3)
                })
        st.dataframe(pd.DataFrame(mapping))
