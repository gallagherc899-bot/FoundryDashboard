import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# MTTFscrap Calculation
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

# Features and Target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)

# Split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

# SMOTE + Training
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Calibrate model
calibrated_model = CalibratedClassifierCV(estimator=rf_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

# UI
st.title("🧪 Foundry Scrap Risk & Reliability Dashboard")
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
    part_id_input = int(float(selected_part)) if part_known else -1
    mttf_value = mtbf_df.loc[part_id_input, "mttf_scrap"] if part_known and part_id_input in mtbf_df.index else 1.0
    part_id_encoded = LabelEncoder().fit(df["part_id"]).transform([part_id_input])[0] if part_known else -1
    input_data = pd.DataFrame([[quantity, weight, part_id_encoded, mttf_value]], columns=features)

    predicted_proba = calibrated_model.predict_proba(input_data)[0][1]
    st.metric("Calibrated Scrap Risk", f"{round(predicted_proba * 100, 2)}%")

    if part_known:
        part_df = df[df["part_id"] == part_id_input]
        N = len(part_df)
        failures = (part_df["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")
        lambda_scrap = 1 / mtbf_scrap if mtbf_scrap != float("inf") else 0
        reliability = np.exp(-lambda_scrap)
        st.metric("MTTFscrap", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs/failure")
        st.metric("Reliability for Next Run", f"{round(reliability * 100, 2)}%")

    expected_scrap_count = round(predicted_proba * quantity)
    expected_loss = round(expected_scrap_count * cost_per_part, 2)
    st.subheader("💰 Financial Impact")
    st.write(f"Expected loss: **${expected_loss}** at {round(predicted_proba * 100, 2)}% predicted scrap.")

    y_pred = calibrated_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    st.subheader("📊 Confusion Matrix")
    fig = ff.create_annotated_heatmap(
        z=[[tp, fn], [fp, tn]],
        x=['Predicted Scrap', 'Predicted Non-Scrap'],
        y=['Actual Scrap', 'Actual Non-Scrap'],
        colorscale='Blues'
    )
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Feature Contributions (SHAP)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_data)
    shap_val = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    shap_df = pd.DataFrame({
        'Feature': features,
        'SHAP Value': shap_val,
        'Impact': np.abs(shap_val)
    }).sort_values(by="Impact", ascending=False)
    st.dataframe(shap_df[["Feature", "SHAP Value"]])
