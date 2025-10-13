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
            # Compute SHAP values
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
            # Compute SHAP values
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
