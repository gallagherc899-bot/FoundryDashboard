# ================================================================
# 🏭 Foundry Scrap Prediction Dashboard — Baseline + Enhanced
# Author: [Your Name]
# Based on Campbell (2003), Juran (1999), Taguchi (2004)
# Date: 2025-12-29
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
)

# ------------------------------------------------------------
# 0️⃣ Setup
# ------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`")
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy")

st.set_page_config(
    page_title="Foundry Scrap Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 1️⃣ Dashboard Header
# ------------------------------------------------------------
st.title("🏭 Foundry Scrap Prediction Dashboard — Baseline vs Enhanced")
st.markdown("""
This dashboard integrates **Statistical Process Control (SPC)** concepts with  
**Campbell’s (2003)** multivariate process–defect relationships to model and predict  
scrap events in aluminum greensand foundries.  
Compare the traditional *baseline defect-only model* to the *process-aware enhanced model*  
and analyze how predicted defect priorities shift in the Pareto analysis.
""")

# ------------------------------------------------------------
# 2️⃣ File Upload
# ------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload your Foundry Dataset (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    if "Week_Ending" in df.columns:
        df["Week_Ending"] = pd.to_datetime(df["Week_Ending"], errors="coerce")
        df = df.sort_values("Week_Ending").dropna(subset=["Week_Ending"])
    df = df.fillna(0)

    if "Part_ID" in df.columns:
        df["Part_ID"] = df["Part_ID"].replace({"nan": "unknown", "": "unknown"})

    defect_cols = [c for c in df.columns if c.lower().endswith("rate")]
    st.success(f"✅ Dataset Loaded: {len(df)} records, {len(defect_cols)} defect rate columns detected.")
else:
    st.warning("⬆️ Please upload a dataset to continue.")
    st.stop()

# ------------------------------------------------------------
# 3️⃣ Campbell Process Groups
# ------------------------------------------------------------
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

process_cols = list(process_groups.keys())

# ------------------------------------------------------------
# 4️⃣ Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")
threshold = st.sidebar.slider("Scrap % Threshold", 0.0, 5.0, 2.5, 0.5)
show_pareto = st.sidebar.checkbox("Show Pareto Comparison", value=True)

# ------------------------------------------------------------
# 5️⃣ Train Models (Baseline & Enhanced)
# ------------------------------------------------------------
y = (df["Scrap_"] > threshold).astype(int)

# Baseline Model
X_base = df[defect_cols]
rf_base = RandomForestClassifier(
    n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_base.fit(X_base, y)

# Enhanced Model
X_enh = df[defect_cols + process_cols]
rf_enh = RandomForestClassifier(
    n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_enh.fit(X_enh, y)

# ------------------------------------------------------------
# 6️⃣ Evaluate Models
# ------------------------------------------------------------
def evaluate_model(df, model, X_cols):
    y_true = (df["Scrap_"] > threshold).astype(int)
    probs = model.predict_proba(df[X_cols])[:, 1]
    preds = (probs > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Brier": brier_score_loss(y_true, probs)
    }

results = pd.DataFrame([
    evaluate_model(df, rf_base, defect_cols),
    evaluate_model(df, rf_enh, defect_cols + process_cols)
], index=["Baseline", "Enhanced"])

st.subheader("📈 Model Performance Comparison")
st.dataframe(results.style.highlight_max(axis=0, color="lightgreen"))

# ------------------------------------------------------------
# 7️⃣ Pareto Comparison
# ------------------------------------------------------------
if show_pareto:
    st.subheader("📊 Baseline vs Enhanced Predicted Pareto")

    df["Baseline_Prob"] = rf_base.predict_proba(df[defect_cols])[:, 1]
    df["Enhanced_Prob"] = rf_enh.predict_proba(df[defect_cols + process_cols])[:, 1]

    def pareto_data(prob_col, label):
        defect_contrib = df[defect_cols].multiply(df[prob_col], axis=0).mean().sort_values(ascending=False)
        pareto = defect_contrib.reset_index()
        pareto.columns = ["Defect", "Weighted_Impact"]
        pareto["Model"] = label
        return pareto

    pareto_base = pareto_data("Baseline_Prob", "Baseline")
    pareto_enh = pareto_data("Enhanced_Prob", "Enhanced")
    combined_pareto = pd.concat([pareto_base, pareto_enh])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=combined_pareto,
        x="Weighted_Impact", y="Defect", hue="Model",
        palette=["#66c2a5", "#fc8d62"], ax=ax
    )
    plt.title("Baseline vs Enhanced Predicted Pareto Comparison")
    plt.xlabel("Predicted Weighted Scrap Impact")
    plt.ylabel("Defect Type")
    plt.legend(title="Model")
    plt.grid(alpha=0.3)
    st.pyplot(fig)

    # Delta Comparison Table
    comparison = (
        pareto_enh.set_index("Defect")["Weighted_Impact"]
        .subtract(pareto_base.set_index("Defect")["Weighted_Impact"], fill_value=0)
        .sort_values(ascending=False)
        .reset_index()
    )
    comparison.columns = ["Defect", "Change_in_Impact"]

    st.markdown("### 🔄 Change in Defect Impact (Enhanced − Baseline)")
    st.dataframe(comparison.style.background_gradient(cmap="coolwarm", axis=0))

    st.info("""
    This analysis compares predicted defect contributions between models.
    Defects with **positive ΔImpact** are those whose likelihood is better explained  
    by **multivariate process interactions**, while negative changes indicate defects  
    dominated by **single-process variation** (SPC-suitable).
    """)

# ------------------------------------------------------------
# 8️⃣ Feature Importances
# ------------------------------------------------------------
st.markdown("### 🔍 Top 15 Feature Importances (Enhanced Model)")
importances = pd.Series(rf_enh.feature_importances_, index=X_enh.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances.values[:15], y=importances.index[:15], palette="viridis", ax=ax)
plt.title("Feature Importance — Enhanced Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
st.pyplot(fig)

st.caption("Model trained using RandomForest (180 estimators, class_weight='balanced').")
st.caption("Process indices computed per Campbell (2003), *Castings Practice: The Ten Rules of Castings.*")
