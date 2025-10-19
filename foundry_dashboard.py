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

# === Load & Clean Data ===
df = pd.read_csv("anonymized_parts.csv")
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("(", "", regex=False)
              .str.replace(")", "", regex=False))
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# === Feature Engineering ===
initial_threshold = 5.0
df["scrap_flag"] = df["scrap%"] > initial_threshold
mtbf_df = (df.groupby("part_id")
           .agg(total_runs=("scrap%", "count"),
                failures=("scrap_flag", "sum")))
mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
mtbf_df["mttf_scrap"] = mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"])
df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

# Encode part_id
le = LabelEncoder()
df["part_id_encoded"] = le.fit_transform(df["part_id"])

# Define features & target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)

# === Train / Calibration / Test Split ===
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)  # 60/20/20

# === Model Training ===
# Non‑SMOTE case
rf_pre = RandomForestClassifier(random_state=42)
rf_pre.fit(X_train, y_train)

# SMOTE case
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_post = RandomForestClassifier(random_state=42)
rf_post.fit(X_resampled, y_resampled)

# === Calibration (Platt Scaling) ===
calibrated_model_post = CalibratedClassifierCV(estimator=rf_post, method='sigmoid', cv="prefit")
calibrated_model_post.fit(X_calib, y_calib)

# === Optimized Scrap Prediction Weights (Benchmarking) ===
optimal_weights = [0.1283, 2.2272]  # [order_quantity, piece_weight_lbs]
benchmark_df = df.head(20).copy()
benchmark_df["actual_scrap_pounds"] = (benchmark_df["order_quantity"]
                                       * benchmark_df["piece_weight_lbs"]
                                       * (benchmark_df["scrap%"] / 100))
benchmark_df["optimized_predicted_scrap"] = (
    benchmark_df["order_quantity"] * optimal_weights[0] +
    benchmark_df["piece_weight_lbs"] * optimal_weights[1])
benchmark_df["lower_bound"] = benchmark_df["actual_scrap_pounds"] * 0.90
benchmark_df["upper_bound"] = benchmark_df["actual_scrap_pounds"] * 1.10
benchmark_df["within_tolerance"] = (
    (benchmark_df["optimized_predicted_scrap"] >= benchmark_df["lower_bound"])
    & (benchmark_df["optimized_predicted_scrap"] <= benchmark_df["upper_bound"])
)

# === Streamlit UI ===
st.title("🧪 Foundry Scrap Risk & Reliability Dashboard")

st.sidebar.subheader("Model & Data Options")
use_smote = st.sidebar.checkbox("Use Post‑SMOTE Calibrated Model", value=True)
model_choice_label = "Post‑SMOTE + Calibrated" if use_smote else "Pre‑SMOTE (Uncalibrated)"
model = calibrated_model_post if use_smote else rf_pre

st.subheader(f"🔍 Scrap Risk Prediction ({model_choice_label})")
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)
threshold = st.slider("Scrap % Threshold", min_value=1.0, max_value=10.0, value=initial_threshold)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01, step=0.01)

if st.button("Predict Scrap Risk"):
    part_known = (selected_part != "New")
    part_id_input = int(float(selected_part)) if part_known else None

    if part_known:
        mttf_value = mtbf_df.loc[part_id_input, "mttf_scrap"] if part_id_input in mtbf_df.index else 1.0
        encoded_id = (le.transform([part_id_input])[0]
                      if part_id_input in le.classes_ else -1)
    else:
        mttf_value = 1.0
        encoded_id = -1

    input_row = pd.DataFrame([[quantity, weight, encoded_id, mttf_value]], columns=features)

    if use_smote:
        predicted_proba = model.predict_proba(input_row)[0][1]
    else:
        predicted_proba = model.predict_proba(input_row)[0][1]

    st.metric("Predicted Scrap Risk", f"{round(predicted_proba * 100, 2)}%")

    if part_known:
        subset = df[df["part_id"] == part_id_input]
        N = len(subset)
        failures = (subset["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")
        lambda_scrap = 1 / mtbf_scrap if mtbf_scrap != float("inf") else 0
        reliability = np.exp(-lambda_scrap * 1)

        st.metric("MTTFscrap (runs per failure)",
                  f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)}")
        st.metric("Reliability for Next Run", f"{round(reliability * 100, 2)}%")

    expected_scrap_count = round(predicted_proba * quantity)
    expected_loss = round(expected_scrap_count * cost_per_part, 2)
    st.subheader("💰 Financial Impact")
    st.write(f"Expected scrap count: **{expected_scrap_count}** out of {quantity} parts.")
    st.write(f"Estimated loss: **${expected_loss}** at a cost per part of ${cost_per_part}.")

    # Confusion matrix on test set
    if use_smote:
        y_pred = calibrated_model_post.predict(X_test)
    else:
        y_pred = rf_pre.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    st.subheader("📊 Confusion Matrix: "+model_choice_label)
    fig = ff.create_annotated_heatmap(
        z=[[tp, fn], [fp, tn]],
        x=['Predicted Scrap', 'Predicted Non‑Scrap'],
        y=['Actual Scrap', 'Actual Non‑Scrap'],
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Interpretation
    st.subheader("🔍 Feature Impact (SHAP)")
    explainer = shap.TreeExplainer(rf_post if use_smote else rf_pre)
    shap_values = explainer.shap_values(input_row)
    if isinstance(shap_values, list):  
        shap_array = shap_values[1]
    else:
        shap_array = shap_values
    shap_val = shap_array.ravel()
    shap_df = pd.DataFrame({
        'Feature': features,
        'SHAP Value': shap_val,
        'Impact': np.abs(shap_val)
    }).sort_values(by="Impact", ascending=False)

    st.dataframe(shap_df[['Feature', 'SHAP Value']])

# === Optionally show benchmarking table ===
if st.sidebar.checkbox("Show benchmark optimization table"):
    st.subheader("📋 Benchmark: Optimized vs Actual Scrap (First 20 Rows)")
    st.dataframe(benchmark_df)
