# ---------------------------------------------------------
# Aluminum Foundry Scrap Analytics Dashboard v8.7
# Enhanced Campbell Logic | Dual Pareto + Defect–Process Map
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

st.set_page_config(page_title="Aluminum Foundry Scrap Analytics Dashboard", layout="wide")

# ---------------------------------------------------------
# CAMPBELL DEFECT–PROCESS MAPPING (From your provided outline)
# ---------------------------------------------------------
campbell_mapping = {
    "Sand System and Preparation": ["sand_rate", "dirty_pattern_rate", "crush_rate", "runout_rate", "gas_porosity_rate"],
    "Core Making": ["core_rate", "gas_porosity_rate", "shrink_porosity_rate", "crush_rate"],
    "Pattern Making and Maintenance": ["shift_rate", "bent_rate", "dirty_pattern_rate"],
    "Mold Making and Assembly": ["shift_rate", "runout_rate", "missrun_rate", "short_pour_rate", "gas_porosity_rate"],
    "Melting and Alloy Treatment": ["dross_rate", "gas_porosity_rate", "shrink_rate", "shrink_porosity_rate", "gouged_rate"],
    "Pouring and Mold Filling": ["missrun_rate", "short_pour_rate", "dross_rate", "tear_up_rate"],
    "Solidification and Cooling": ["shrink_rate", "shrink_porosity_rate", "gas_porosity_rate", "missrun_rate"],
    "Shakeout and Cleaning": ["tear_up_rate", "over_grind_rate", "sand_rate"],
    "Inspection and Finishing": ["failed_zyglo_rate", "zyglo_rate", "outside_process_scrap_rate"]
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("🛠️ **Manager Input Controls**")
part_id = st.sidebar.text_input("Enter Part ID (matches 'Part ID' column exactly)", "")
order_qty = st.sidebar.number_input("Order Quantity", min_value=1, value=100)
piece_weight = st.sidebar.number_input("Piece Weight (lbs)", min_value=0.1, value=10.0)
cost_per_part = st.sidebar.number_input("Cost per Part ($)", min_value=0.01, value=10.0)
threshold = st.sidebar.slider("Scrap% Threshold", min_value=0.0, max_value=5.0, value=2.5)
st.sidebar.caption("📊 Scrap Risk = probability that actual scrap% exceeds this threshold.")
predict_btn = st.sidebar.button("🔮 Predict")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df = load_data()

# ---------------------------------------------------------
# Helper: Find Similar Parts if Limited Data
# ---------------------------------------------------------
def find_similar_parts(df, target_part, tolerance=1.0, max_iterations=5):
    part_data = df[df["part_id"].astype(str) == str(target_part)]
    if len(part_data) > 3:
        return part_data, 0

    ref_weight = part_data["piece_weight_(lbs)"].mean() if not part_data.empty else df["piece_weight_(lbs)"].mean()
    ref_scrap = part_data["scrap%"].mean() if not part_data.empty else df["scrap%"].mean()

    for i in range(max_iterations):
        low, high = ref_scrap - tolerance, ref_scrap + tolerance
        similar = df[(df["piece_weight_(lbs)"].between(ref_weight - 1, ref_weight + 1)) &
                     (df["scrap%"].between(low, high))]
        if len(similar) >= 10:
            return similar, i + 1
        tolerance += 0.5
    return df.sample(min(20, len(df))), max_iterations

# ---------------------------------------------------------
# Train / Predict
# ---------------------------------------------------------
def train_and_evaluate(df_part):
    features = [col for col in df_part.columns if "rate" in col and "scrap" not in col]
    X = df_part[features]
    y = (df_part["scrap%"] > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_pred)
    }

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return metrics, importance

# ---------------------------------------------------------
# Main Prediction Logic
# ---------------------------------------------------------
if predict_btn:
    if part_id:
        df_part = df[df["part_id"].astype(str) == str(part_id)]
        similar_df, expand_steps = find_similar_parts(df, part_id)

        if len(df_part) == 0:
            st.warning("⚠️ No direct matches for this Part ID. Using similar parts for prediction.")
            df_part = similar_df

        metrics, importance = train_and_evaluate(df_part)

        # ------------------------------
        # Pareto Charts
        # ------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Historical (Observed)
        historical = df_part[[col for col in df_part.columns if "rate" in col]].mean().sort_values(ascending=False)
        ax[0].bar(historical.index, historical.values, color="steelblue")
        ax[0].set_title("Historical Pareto (Observed)")
        ax[0].tick_params(axis="x", rotation=90)

        # Predicted (Enhanced ML-PHM)
        ax[1].bar(importance["Feature"], importance["Importance"], color="seagreen")
        ax[1].set_title("Predicted Pareto (Enhanced ML-PHM)")
        ax[1].tick_params(axis="x", rotation=90)

        st.pyplot(fig)

        # ------------------------------
        # Defect–Process Influence Table
        # ------------------------------
        process_rows = []
        for process, defects in campbell_mapping.items():
            matching = importance[importance["Feature"].isin(defects)]
            importance_mean = matching["Importance"].mean() if not matching.empty else 0.0
            influence_pct = importance_mean / importance["Importance"].sum() * 100 if importance["Importance"].sum() > 0 else 0.0
            defect_list = ", ".join(defects)
            process_rows.append([defect_list, process, importance_mean, influence_pct])

        process_df = pd.DataFrame(process_rows, columns=["Defect(s)", "Process", "Importance", "Influence_%"])
        process_df = process_df.sort_values("Importance", ascending=False)

        st.markdown("### 🧱 Defect–Process Influence (Campbell PHM)")
        st.dataframe(process_df, use_container_width=True)

        # ------------------------------
        # Model Performance Metrics
        # ------------------------------
        st.markdown("### 📈 Model Performance (ML-PHM Framework v8.7)")
        perf = pd.DataFrame(metrics.items(), columns=["Metric", "Score"]).set_index("Metric")
        st.table(perf)

        st.success("✅ Prediction Complete! ML-PHM successfully analyzed this part.")
    else:
        st.warning("Please enter a valid Part ID.")
