import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# Initial threshold for MTTFscrap calculation
initial_threshold = 5.0
df["scrap_flag"] = df["scrap%"] > initial_threshold
mtbf_df = df.groupby("part_id").agg(
    total_runs=("scrap%", "count"),
    failures=("scrap_flag", "sum")
)
mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
mtbf_df["mttf_scrap"] = mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"])
df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

# Encode part_id
df["part_id_encoded"] = LabelEncoder().fit_transform(df["part_id"])

# Define features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train models
rf_pre = RandomForestClassifier(random_state=42)
rf_pre.fit(X_train, y_train)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_post = RandomForestClassifier(random_state=42)
rf_post.fit(X_resampled, y_resampled)

# Streamlit UI
st.title("🧪 Foundry Scrap Risk & Reliability Dashboard")
model_choice = st.radio("Select Model Version", ["Pre-SMOTE", "Post-SMOTE"])
model = rf_pre if model_choice == "Pre-SMOTE" else rf_post

st.subheader("🔍 Scrap Risk Prediction")
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)
threshold = st.slider("Scrap % Threshold", min_value=1.0, max_value=10.0, value=5.0)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01, step=0.01)

if st.button("Predict Scrap Risk"):
    part_known = selected_part != "New"
    part_id_input = int(float(selected_part)) if part_known else None

    if part_known:
        mttf_value = mtbf_df.loc[part_id_input, "mttf_scrap"] if part_id_input in mtbf_df.index else 1.0
        input_data = pd.DataFrame([[quantity, weight, part_id_input, mttf_value]], columns=features)
        input_data["part_id_encoded"] = LabelEncoder().fit_transform(input_data["part_id_encoded"])

        predicted_class = model.predict(input_data)[0]
        predicted_proba = model.predict_proba(input_data)[0][1]

        part_df = df[df["part_id"] == part_id_input]
        N = len(part_df)
        failures = (part_df["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")

        st.success(f"✅ Known Part ID: {part_id_input}")
        st.metric(f"{model_choice} Predicted Scrap Risk", f"{round(predicted_proba * 100, 2)}%")
        st.caption("This prediction reflects what the algorithm sees right now, based on features like weight, quantity, and part history.")

        st.metric("MTTFscrap", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
        st.write(f"Failures above threshold: **{failures}** out of **{N}** runs")

        lambda_scrap = 1 / mtbf_scrap if mtbf_scrap != float("inf") else 0
        reliability_next_run = np.exp(-lambda_scrap * 1)
        reliability_display = f"{round(reliability_next_run * 100, 2)}%" if lambda_scrap > 0 else "100%"
        st.metric("Reliability for Next Run", reliability_display)
        st.caption("This tells you how likely it is that the part won’t scrap in the next run — based on past performance. \( R(t) = e^{-\\lambda t} \), where \( \\lambda = 1 / \text{MTTFscrap} \).")

        expected_scrap_count = round(predicted_proba * quantity)
        expected_loss = round(expected_scrap_count * cost_per_part, 2)
        st.subheader("💰 Financial Impact")
        st.write(f"At a threshold of {threshold}%, the model predicts {round(predicted_proba * 100, 2)}% scrap.")
        st.write(f"For {quantity} parts at ${cost_per_part} each, this results in an expected loss of **${expected_loss}**.")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * 100 + fp * 20
        st.write(f"Estimated Cost Impact (FN=$100, FP=$20): **${cost}**")

        st.subheader("📊 Confusion Matrix")
        z = [[tp, fn], [fp, tn]]
        x_labels = ['Predicted Scrap', 'Predicted Non-Scrap']
        y_labels = ['Actual Scrap', 'Actual Non-Scrap']
        fig = ff.create_annotated_heatmap(z, x=x_labels, y=y_labels, colorscale='Blues', showscale=True)
        fig.update_layout(title=f'Confusion Matrix: {model_choice} Model')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Pareto Risk Drivers")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_all = explainer.shap_values(input_data)
            shap_values_single = shap_values_all[1][0] if isinstance(shap_values_all, list) else shap_values_all[0]
            shap_values_single = np.array(shap_values_single).flatten()
            total_shap = np.sum(np.abs(shap_values_single))

            contributions = []
            for i, feature in enumerate(input_data.columns):
                shap_val = shap_values_single[i]
                percent = round((abs(shap_val) / total_shap) * 100, 2)
                direction = "↑" if shap_val > 0 else "↓"
                contributions.append((feature, percent, shap_val, direction))

            contributions.sort(key=lambda x: x[1], reverse=True)
            cumulative = 0
            pareto_features = []
            for item in contributions:
                pareto_features.append(item)
                cumulative += item[1]
                if cumulative >= 80:
                    break

            st.write("**Top Features Contributing to 80% of Scrap Risk:**")
            for feature, percent, shap_val, direction in pareto_features:
                st.write(f"- {feature}: {percent}% ({direction} SHAP = {round(shap_val, 3)})")

        except Exception as e:
            st.error(f"SHAP panel failed: {e}")

# 📊 Model Comparison Table
st.subheader("🧮 Model Configuration Comparison Table")
comparison_data = {
    "Configuration": [
        "No SMOTE, No SHAP",
        "No SMOTE, With SHAP",
        "With SMOTE, No SHAP",
        "With SMOTE, With SHAP"
    ],
    "Accuracy (TP + TN / Total)": ["70.0%", "70
