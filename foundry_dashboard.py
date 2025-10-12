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

        # SHAP summary (Post-SMOTE only)
        if model_choice == "Post-SMOTE":
            st.subheader("🔎 SHAP Feature Importance (Post-SMOTE)")
            # SHAP summary (Post-SMOTE only)
        if model_choice == "Post-SMOTE":
            st.subheader("🔎 SHAP Feature Importance (Post-SMOTE)")

    # Validate input data
    expected_cols = ['order_quantity', 'piece_weight_lbs', 'part_id']
    missing_cols = [col for col in expected_cols if col not in X_test.columns]
    
    if X_test.empty:
        st.warning("⚠️ SHAP cannot be computed: X_test is empty.")
    elif missing_cols:
        st.warning(f"⚠️ SHAP cannot be computed: Missing columns in X_test: {missing_cols}")
    else:
        try:
            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Plot SHAP summary
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP plot failed to render: {e}")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig)

# MTBF Trend Visualization
st.markdown("---")
st.subheader("📈 MTBFscrap Trend by Week")
date_range = st.date_input("Select Time Window", value=[df["week_ending"].min(), df["week_ending"].max()])
selected_parts = st.multiselect("Select up to 5 Part IDs", options=sorted(df["part_id"].unique()), max_selections=5)

if selected_parts and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = df[(df["week_ending"] >= start_date) & (df["week_ending"] <= end_date) & (df["part_id"].isin(selected_parts))]
    grouped = filtered.groupby(["part_id", "week_ending"])
    mtbf_data = []
    for (pid, week), group in grouped:
        N = len(group)
        failures = (group["scrap%"] > threshold).sum()
        mtbf = N / failures if failures > 0 else float("inf")
        mtbf_data.append({"Part ID": pid, "Week": week, "MTBFscrap": mtbf})
    mtbf_df = pd.DataFrame(mtbf_data)

    chart = alt.Chart(mtbf_df).mark_line(point=True).encode(
        x="Week:T",
        y=alt.Y("MTBFscrap:Q", title="MTBF (Runs per Failure)"),
        color="Part ID:N",
        tooltip=["Part ID", "Week", "MTBFscrap"]
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

# Defect Count Bar Chart
st.subheader("📊 Total Defect Counts by Part")
if selected_parts and len(date_range) == 2:
    defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
    filtered = df[(df["week_ending"] >= start_date) & (df["week_ending"] <= end_date) & (df["part_id"].isin(selected_parts))]
    defect_counts = filtered.groupby("part_id")[defect_cols].sum().reset_index()
    melted = defect_counts.melt(id_vars="part_id", var_name="Defect Type", value_name="Count")
    bar_chart = alt.Chart(melted).mark_bar().encode(
        x="part_id:N",
        y="Count:Q",
        color="Defect Type:N",
        tooltip=["part_id", "Defect Type", "Count"]
    ).properties(width=800, height=400)
    st.altair_chart(bar_chart, use_container_width=True)
