import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import altair as alt
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# Encode features
features = ["order_quantity", "piece_weight_lbs", "part_id"]
X = df[features].copy()
y = (df["scrap%"] > 5.0).astype(int)
X["part_id"] = LabelEncoder().fit_transform(X["part_id"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Pre-SMOTE model
rf_pre = RandomForestClassifier(random_state=42)
rf_pre.fit(X_train, y_train)

# Train Post-SMOTE model
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_post = RandomForestClassifier(random_state=42)
rf_post.fit(X_resampled, y_resampled)

# UI Header
st.title("🧪 Foundry Scrap Risk & Reliability Dashboard")
st.markdown("Compare Pre-SMOTE and Post-SMOTE predictions, interpret SHAP values, and evaluate cost impact.")

# Model toggle
model_choice = st.radio("Select Model Version", ["Pre-SMOTE", "Post-SMOTE"])
model = rf_pre if model_choice == "Pre-SMOTE" else rf_post

# Scrap Prediction Panel
st.subheader("🔍 Scrap Risk Prediction")
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)
threshold = st.slider("Scrap % Threshold for Failure (MTBF)", min_value=1.0, max_value=10.0, value=5.0)
st.write(f"Current failure threshold: **{threshold}%**")

# Prediction button
if st.button("Predict Scrap Risk"):
    part_known = selected_part != "New"
    part_id_input = int(float(selected_part)) if part_known else None

    if part_known:
        input_data = pd.DataFrame([[quantity, weight, part_id_input]], columns=features)
        input_data["part_id"] = LabelEncoder().fit_transform(input_data["part_id"])
        predicted_class = model.predict(input_data)[0]
        predicted_proba = model.predict_proba(input_data)[0][1]

        part_df = df[df["part_id"] == part_id_input]
        N = len(part_df)
        failures = (part_df["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")

        st.success(f"✅ Known Part ID: {part_id_input}")
        st.metric(f"{model_choice} Predicted Scrap Risk", f"{round(predicted_proba * 100, 2)}%")
        st.metric("MTBFscrap", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
        st.write(f"Failures above threshold: **{failures}** out of **{N}** runs")

        # Cost-weighted evaluation
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * 100 + fp * 20
        st.write(f"Estimated Cost Impact (FN=$100, FP=$20): **${cost}**")

        # Confusion Matrix Visualization
        st.subheader("📊 Confusion Matrix")
        z = [[tp, fn], [fp, tn]]
        x_labels = ['Predicted Scrap', 'Predicted Non-Scrap']
        y_labels = ['Actual Scrap', 'Actual Non-Scrap']
        fig = ff.create_annotated_heatmap(z, x=x_labels, y=y_labels, colorscale='Blues', showscale=True)
        fig.update_layout(title=f'Confusion Matrix: {model_choice} Model')
        st.plotly_chart(fig, use_container_width=True)

        if model_choice == "Post-SMOTE":
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

            pareto_features = []
            cumulative = 0
            for item in contributions:
                pareto_features.append(item)
                cumulative += item[1]
                if cumulative >= 80:
                    break

            st.write("**Top Features Contributing to 80% of Scrap Risk:**")
            for feature, percent, shap_val, direction in pareto_features:
                st.write(f"- {feature}: {percent}% ({direction} SHAP = {round(shap_val, 3)})")

            chart_data = pd.DataFrame({
                "Feature": [f for f, _, _, _ in pareto_features],
                "Contribution (%)": [p for _, p, _, _ in pareto_features]
            })
            st.bar_chart(chart_data.set_index("Feature"))

        except Exception as e:
            st.error(f"Pareto panel failed: {e}")

    st.subheader("🔍 Likely Defects (Pareto 80%)")
    try:
        defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
        defect_data = df[df["part_id"] == part_id_input][defect_cols].sum()
        total_defects = defect_data.sum()

        if total_defects == 0:
            st.info("No defect data available for this part.")
        else:
            sorted_defects = defect_data.sort_values(ascending=False)
            cumulative = 0
            top_defects = []

            for defect, count in sorted_defects.items():
                percent = round((count / total_defects) * 100, 2)
                cumulative += percent
                likelihood = "High" if percent > 40 else "Medium" if percent > 20 else "Low"
                top_defects.append((defect.replace("_rate", "").replace("_", " ").title(), likelihood, f"{percent}% of total defects"))
                if cumulative >= 80:
                    break

            for defect, likelihood, reason in top_defects:
                st.write(f"- **{defect}**: {likelihood} likelihood ({reason})")

            chart_df = pd.DataFrame({
                "Defect Type": [d for d, _, _ in top_defects],
                "Contribution (%)": [float(r.split('%')[0]) for _, _, r in top_defects]
            })
            st.bar_chart(chart_df.set_index("Defect Type"))

    except Exception as e:
        st.error(f"Defect panel failed: {e}")

# MTBF Trend Visualization
st.markdown("---")
st.subheader("📈 MTBFscrap Trend by Week")
date_range = st.date_input("Select Time Window", value=[df["week_ending"].min(), df["week_ending"].max()])
selected_parts = st.multiselect("Select up to 5 Part IDs", options=sorted(df["part_id"].unique()), max_selections=5)

if selected_parts and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

