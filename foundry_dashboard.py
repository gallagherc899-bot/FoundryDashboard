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
le = LabelEncoder()
df["part_id_encoded"] = le.fit_transform(df["part_id"])

# Define features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)

# Split into train, calibration, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)  # 60/20/20

# Train and calibrate model
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_post = RandomForestClassifier(random_state=42)
rf_post.fit(X_resampled, y_resampled)
calibrated_model = CalibratedClassifierCV(estimator=rf_post, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

# Optimized scrap prediction weights
optimal_weights = [0.1283, 2.2272]  # [order_quantity, piece_weight_lbs]
benchmark_df = df.head(20).copy()
benchmark_df['actual_scrap_pounds'] = benchmark_df['order_quantity'] * benchmark_df['piece_weight_lbs'] * (benchmark_df['scrap%'] / 100)
benchmark_df['optimized_predicted_scrap'] = (
    benchmark_df['order_quantity'] * optimal_weights[0] + benchmark_df['piece_weight_lbs'] * optimal_weights[1]
)
benchmark_df['lower_bound'] = benchmark_df['actual_scrap_pounds'] * 0.90
benchmark_df['upper_bound'] = benchmark_df['actual_scrap_pounds'] * 1.10
benchmark_df['within_tolerance'] = (
    (benchmark_df['optimized_predicted_scrap'] >= benchmark_df['lower_bound']) &
    (benchmark_df['optimized_predicted_scrap'] <= benchmark_df['upper_bound'])
)

# Streamlit UI
st.title("🧪 Foundry Scrap Risk Dashboard")
st.subheader("Scrap Risk Estimation")

part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Order Quantity", min_value=1)
weight = st.number_input("Piece Weight (lbs)", min_value=0.1)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01)

if st.button("Predict Scrap Risk"):
    part_known = selected_part != "New"
    part_id_input = int(float(selected_part)) if part_known else -1
    mttf_value = mtbf_df.loc[part_id_input, "mttf_scrap"] if part_known and part_id_input in mtbf_df.index else df["mttf_scrap"].mean()
    part_encoded = le.transform([part_id_input])[0] if part_known else 0

    input_data = pd.DataFrame([[quantity, weight, part_encoded, mttf_value]], columns=features)
    predicted_class = calibrated_model.predict(input_data)[0]
    predicted_proba = calibrated_model.predict_proba(input_data)[0][1]

    expected_scrap_count = round(predicted_proba * quantity)
    expected_loss = round(expected_scrap_count * cost_per_part, 2)

    st.metric("Predicted Scrap Risk", f"{round(predicted_proba * 100, 2)}%")
    st.write(f"Expected Scrap Count: {expected_scrap_count} parts")
    st.write(f"Expected Financial Loss: ${expected_loss}")

    # SHAP analysis
    try:
        explainer = shap.TreeExplainer(calibrated_model.base_estimator)
        shap_val = explainer.shap_values(input_data)[1][0]
        shap_df = pd.DataFrame({
            'Feature': features,
            'SHAP Value': shap_val,
            'Impact': np.abs(shap_val)
        }).sort_values(by="Impact", ascending=False)

        st.subheader("Top Risk Drivers")
        for i, row in shap_df.iterrows():
            direction = "↑" if row['SHAP Value'] > 0 else "↓"
            st.write(f"- {row['Feature']}: {round(row['Impact'], 3)} ({direction})")
    except Exception as e:
        st.error(f"SHAP analysis failed: {e}")
